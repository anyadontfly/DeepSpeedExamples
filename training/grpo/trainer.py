# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GRPO training loop for multi-turn tool-calling rollouts.

Each step:

  1. **Rollout.** The hybrid engine switches to inference mode and the policy
     solves each rank's prompts, calling the code interpreter between turns
     (:class:`~deepspeed.runtime.rollout.HybridEngineRollout`).
  2. **Reward.** Every trajectory gets the strict-box answer score, and its
     advantage is the score normalized within its prompt's group.
  3. **Update.** One optimizer step over all trajectories, accumulated over
     micro-batches: PPO-clip with clip-higher on the generated tokens, averaged
     over every generated token across ranks, or per trajectory first with
     ``algorithm.loss_aggregation = "sample_mean"``. Tool output is masked out.

With one update per rollout, the old policy is the current one, so the old
log-probs are the current ones detached; the importance ratio is 1 and the
loss reduces to the policy gradient.
"""

import os
import re
import time

import psutil
import torch
import torch.distributed as torch_dist
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

import metrics as M
from config import GRPOConfig
from reward import retool_reward
from deepspeed.runtime.rollout import RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.base import RolloutRecord


def _is_rank_zero() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _mean_across_ranks(values: dict) -> dict:
    """Average each metric over ranks with one all-reduce."""
    names = sorted(values)
    tensor = torch.tensor([float(values[name]) for name in names], device=get_accelerator().current_device_name())
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return dict(zip(names, tensor.tolist()))


def _sum_across_ranks(values: dict) -> dict:
    """Sum each metric over ranks with one all-reduce."""
    names = sorted(values)
    tensor = torch.tensor([float(values[name]) for name in names], device=get_accelerator().current_device_name())
    dist.all_reduce(tensor)
    return dict(zip(names, tensor.tolist()))


def _gather_lists(items: list) -> list:
    """Every rank's ``items`` concatenated, on every rank."""
    gathered = [None] * dist.get_world_size()
    torch_dist.all_gather_object(gathered, items)
    return [item for rank_items in gathered for item in rank_items]


def group_advantages(samples, rewards):
    """GRPO advantages: each reward normalized by the mean and std of its group's rewards."""
    groups = {}
    for sample, reward in zip(samples, rewards):
        groups.setdefault(sample.group_index, []).append(reward)
    stats = {}
    for group_index, group_rewards in groups.items():
        values = torch.tensor(group_rewards)
        stats[group_index] = (values.mean().item(), values.std().item())
    advantages = []
    for sample, reward in zip(samples, rewards):
        mean, std = stats[sample.group_index]
        advantages.append((reward - mean) / (std + 1e-6))
    return advantages


class GRPOTrainer:

    def __init__(
        self,
        cfg: GRPOConfig,
        engine,
        tokenizer,
        rollout: RolloutEngine,
        dataloader,
        eval_rows=None,
        metrics_logger=None,
        span_log=None,
    ):
        self.cfg = cfg
        self.engine = engine
        self.tokenizer = tokenizer
        self.rollout = rollout
        self.dataloader = dataloader
        self.eval_rows = eval_rows or []
        self.metrics_logger = metrics_logger or M.MetricsLogger()
        self.span_log = span_log or M.SpanLog()

        self.device = get_accelerator().current_device_name()
        self.pad_token_id = tokenizer.pad_token_id
        self.step = 0
        self.sampling = SamplingConfig(
            max_new_tokens=cfg.rollout.max_new_tokens,
            max_new_tokens_per_turn=cfg.rollout.max_new_tokens_per_turn,
            temperature=cfg.rollout.temperature,
            n_samples_per_prompt=cfg.rollout.n_samples_per_prompt,
        )
        self.eval_sampling = SamplingConfig(
            max_new_tokens=cfg.eval.max_new_tokens,
            max_new_tokens_per_turn=cfg.eval.max_new_tokens_per_turn,
            temperature=cfg.eval.temperature,
            n_samples_per_prompt=cfg.eval.n_samples_per_prompt,
        )

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def resume(self, checkpoint_dir: str) -> None:
        """Load the latest checkpoint in ``checkpoint_dir``: weights, optimizer state and the step it was saved at."""
        load_path, _ = self.engine.load_checkpoint(checkpoint_dir)
        if load_path is None:
            raise FileNotFoundError(f"No checkpoint to resume from in {checkpoint_dir}")
        # load_path is a file inside the checkpoint's tag directory, and tags are step_<N> (see _save).
        tag = re.search(r"(?:^|/)step_(\d+)(?:/|$)", load_path)
        if tag is None:
            raise ValueError(f"Cannot read the step from checkpoint path {load_path}")
        self.step = int(tag.group(1))
        if _is_rank_zero():
            print(f"[grpo] resumed from {load_path} at step {self.step}", flush=True)

    def train(self) -> None:
        # A resumed run picks up the data order where its checkpoint left off.
        epoch, skip = divmod(self.step, len(self.dataloader))
        while self.step < self.cfg.training.num_rollouts:
            # Reseed the DistributedSampler so each epoch reshuffles each rank's shard.
            self.dataloader.sampler.set_epoch(epoch)
            for rows in self.dataloader:
                if skip:
                    skip -= 1
                    continue
                if self.step >= self.cfg.training.num_rollouts:
                    break
                if self._eval_due():
                    self._log(self.evaluate(), prefix="eval", step_key="eval/step")
                metrics = self._train_step(rows)
                self.step += 1
                if self.step % self.cfg.training.logging_steps == 0:
                    # Rollout and train metrics are numbered by the step that produced them, from 0.
                    self._log(metrics, prefix="train", step_key="rollout/step", step=self.step - 1)
                if self.step % self.cfg.training.save_steps == 0:
                    self._save()
            epoch += 1
        if self.cfg.eval.path and self.cfg.eval.at_end:
            self._log(self.evaluate(), prefix="eval", step_key="eval/step")
        if self.step % self.cfg.training.save_steps:
            self._save()

    def _eval_due(self) -> bool:
        """Evaluate before training starts and every ``eval.interval`` steps."""
        return bool(self.cfg.eval.path) and self.cfg.eval.interval > 0 and self.step % self.cfg.eval.interval == 0

    # ------------------------------------------------------------------
    # Rollout and reward
    # ------------------------------------------------------------------

    def _rollout(self, rows, sampling: SamplingConfig, evaluation: bool = False) -> tuple:
        """Generate trajectories for ``rows`` in chunks, the hybrid engine in inference mode.

        Returns the trajectories and each generate call's profile (empty without ``rollout.enable_profiling``).
        """
        self.engine.eval()
        # Tool-call spans written during this rollout belong to this step and phase.
        self.span_log.context = {"step": self.step, "eval": evaluation}
        samples = []
        profiles = []
        # A call decodes rollout.generate_batch_size trajectories, by default as many as a training step does
        # on this rank, so a step's rows roll out in one call. Evaluation's larger n gets fewer prompts per call.
        trajectories_per_call = self.cfg.rollout.generate_batch_size
        if trajectories_per_call is None:
            trajectories_per_call = self.cfg.training.global_batch_size // dist.get_world_size()
        prompts_per_chunk = max(1, trajectories_per_call // sampling.n_samples_per_prompt)
        for chunk, start in enumerate(range(0, len(rows), prompts_per_chunk)):
            request = RolloutRequest(samples=[
                RolloutRecord(group_index=row["index"], prompt=row["prompt"]) for row in rows[start:start + prompts_per_chunk]
            ])
            samples.extend(self.rollout.generate(request, sampling).samples)
            if self.cfg.rollout.enable_profiling:
                # Covers all turns of this chunk's generate call.
                profile = self.rollout.get_last_profile()
                # Scalars only: the continuous-batching poll trace is a long list.
                timings = {name: value for name, value in profile.items() if isinstance(value, (int, float))}
                self._log({"chunk": chunk, **timings}, prefix="rollout")
                self.span_log.write({"ev": "generate_call", "chunk": chunk, **timings})
                profiles.append(timings)
        return samples, profiles

    def _summaries(self, samples, rewards) -> list:
        """Every rank's per-trajectory summaries (see metrics.sample_summary); also written as sample spans."""
        summaries = [M.sample_summary(sample, reward) for sample, reward in zip(samples, rewards)]
        for summary in summaries:
            self.span_log.write({"ev": "sample", **{k: v for k, v in summary.items() if k != "group_index"}})
        return _gather_lists(summaries)

    def _score(self, samples, rows) -> list:
        """Each sample's reward, also stored on the sample."""
        labels = {row["index"]: row["label"] for row in rows}
        rewards = []
        for sample in samples:
            num_tool_calls = sample.response.count("<tool_call>")
            result = retool_reward(sample.response, labels[sample.group_index], num_tool_calls,
                                   self.cfg.algorithm.tool_call_bonus)
            sample.reward = result["score"]
            rewards.append(result["score"])
        return rewards

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Accuracy over the eval set, averaged over every sample; its prompts are split across ranks."""
        rank, world_size = dist.get_rank(), dist.get_world_size()
        rows = self.eval_rows[rank::world_size]
        start = time.time()
        samples, _ = self._rollout(rows, self.eval_sampling, evaluation=True)
        rewards = self._score(samples, rows)
        # A correct answer scores +1; any wrong one, even with the tool bonus, stays below 0.
        num_correct = sum(reward > 0 for reward in rewards)
        totals = torch.tensor([num_correct, len(rewards)], dtype=torch.float, device=self.device)
        dist.all_reduce(totals)
        metrics = M.eval_metrics(self._summaries(samples, rewards), self.cfg.eval.name)
        return {
            "eval/accuracy": (totals[0] / totals[1].clamp(min=1)).item(),
            "eval/duration_seconds": time.time() - start,
            **metrics,
        }

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def _train_step(self, rows) -> dict:
        accelerator = get_accelerator()
        accelerator.reset_peak_memory_stats()
        step_start = time.time()
        samples, profiles = self._rollout(rows, self.sampling)
        rollout_time = time.time() - step_start
        rewards = self._score(samples, rows)
        advantages = group_advantages(samples, rewards)

        # Rollout and training take turns on the GPU: return each phase's cached memory before the other runs.
        accelerator.empty_cache()
        self.engine.train()
        train_start = time.time()
        loss, update = self._update(samples, advantages)
        train_time = time.time() - train_start
        accelerator.empty_cache()

        metrics = _mean_across_ranks({"loss": loss, "reward": sum(rewards) / len(rewards)})
        metrics.update(self._step_metrics(samples, rewards, profiles, {**update, "loss": loss}, rollout_time, train_time))
        metrics["perf/step_time"] = time.time() - step_start
        return metrics

    def _step_metrics(self, samples, rewards, profiles, update, rollout_time, train_time) -> dict:
        """The step's rollout, train and perf metrics, over all ranks."""
        world_size = dist.get_world_size()
        metrics = M.rollout_metrics(self._summaries(samples, rewards), self.cfg.rollout.max_new_tokens_per_turn)
        totals = _sum_across_ranks(update)
        grad_norm = self.engine.get_global_grad_norm()
        metrics.update({
            # Summed over ranks, the per-rank loss shares are the step loss; there is no KL or entropy term.
            "train/loss": totals["loss"],
            "train/pg_loss": totals["loss"],
            "train/pg_clipfrac": totals["clipped_tokens"] / max(totals["loss_tokens"], 1),
            "train/ppo_kl": totals["log_ratio_sum"] / max(totals["loss_tokens"], 1),
            "train/grad_norm": float(grad_norm) if grad_norm is not None else float("nan"),
            "train/lr": self.engine.get_lr()[0],
        })
        # Every rank rolls out and trains at once: times are averaged over ranks and throughput sums them.
        times = _mean_across_ranks({"rollout": rollout_time, "train": train_time})
        effective = metrics["rollout/response_len/mean"] * metrics["rollout/num_training_samples"]
        raw = metrics["multi_turn/raw_response_length/response_length_mean"] * metrics["rollout/num_training_samples"]
        metrics.update({
            "perf/rollout_time": times["rollout"],
            "perf/tokens_per_gpu_per_sec": raw / times["rollout"] / world_size,
            "perf/effective_tokens_per_gpu_per_sec": effective / times["rollout"] / world_size,
            "perf/longest_sample_tokens_per_sec":
            metrics["multi_turn/raw_response_length/response_length_max"] / times["rollout"],
            # The hybrid engine has no separate log-prob or reference passes: the update is all of training.
            "perf/actor_train_time": times["train"],
            "perf/train_time": times["train"],
            "perf/actor_train_tok_per_s": totals["train_tokens"] / times["train"],
            "perf/gpu_max_memory_allocated_gb": get_accelerator().max_memory_allocated() / 1024**3,
            "perf/process_rss_gb": psutil.Process().memory_info().rss / 1024**3,
            **M.host_memory_gb(),
        })
        metrics.update(M.sum_profiles(profiles))
        return metrics

    def _update(self, samples, advantages) -> tuple:
        """One optimizer step over ``samples``: PPO-clip, averaged over all ranks.

        Returns this rank's share of the step loss (summed over ranks it is the step loss) and this rank's
        counts for the train metrics: loss tokens, clipped tokens, the summed log-ratio and trained tokens.
        """
        sample_mean = self.cfg.algorithm.loss_aggregation == "sample_mean"
        if sample_mean:
            # Each trajectory weighs the same, however many tokens it generated.
            num_units = float(len(samples))
        else:
            # Every generated token across ranks weighs the same.
            num_units = float(sum(sum(sample.loss_mask or []) for sample in samples))
        total_units = torch.tensor(num_units, device=self.device)
        dist.all_reduce(total_units)
        # DeepSpeed averages gradients over ranks, so each rank's sum is scaled by world_size / total.
        loss_scale = dist.get_world_size() / total_units.clamp(min=1).item()

        eps_low, eps_high = self.cfg.algorithm.eps_clip, self.cfg.algorithm.eps_clip_high
        step_loss = 0.0
        update = {"loss_tokens": 0.0, "clipped_tokens": 0.0, "log_ratio_sum": 0.0, "train_tokens": 0.0}
        # micro_batch_size counts trajectories across ranks.
        micro_batch_size = self.cfg.training.micro_batch_size // dist.get_world_size()
        for start in range(0, len(samples), micro_batch_size):
            batch = self._collate(samples[start:start + micro_batch_size], advantages[start:start + micro_batch_size])
            log_probs = self.engine(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    temperature=self.cfg.rollout.temperature)
            mask = batch["loss_mask"]
            advantage = batch["advantages"].unsqueeze(1)

            ratio = torch.exp(log_probs - log_probs.detach())
            unclipped = -advantage * ratio
            clipped = -advantage * ratio.clamp(1 - eps_low, 1 + eps_high)
            token_loss = torch.maximum(unclipped, clipped) * mask
            with torch.no_grad():
                # pg_clipfrac and ppo_kl; with one update per rollout the ratio is 1 and both are 0.
                update["loss_tokens"] += mask.sum().item()
                update["clipped_tokens"] += ((clipped > unclipped).float() * mask).sum().item()
                update["log_ratio_sum"] += (-ratio.detach().log() * mask).sum().item()
                update["train_tokens"] += batch["attention_mask"].sum().item()
            if sample_mean:
                # A trajectory cut before its first turn has no tokens and adds nothing.
                token_loss = token_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = token_loss.sum() * loss_scale
            # The loss is already normalized over the whole step, so it must not
            # be divided again by the gradient accumulation steps.
            self.engine.backward(loss, scale_wrt_gas=False)
            self.engine.step()
            step_loss += loss.item()
        return step_loss, update

    def _collate(self, samples, advantages) -> dict:
        """Right-pad samples into training tensors aligned with the next-token log-probs.

        Rollout samples carry left padding from batched generation, which is
        dropped here. Column ``t`` of ``loss_mask`` describes token ``t + 1``.
        """
        rows = []
        for sample in samples:
            tokens = [token for token, keep in zip(sample.tokens, sample.attention_mask) if keep]
            prompt_len = len(tokens) - sample.response_length
            # A trajectory cut before its first turn has no loss mask.
            loss_mask = [0] * prompt_len + (sample.loss_mask or [])
            rows.append((tokens, loss_mask[1:]))

        width = max(len(tokens) for tokens, _ in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        loss_mask = torch.zeros((len(rows), width - 1))
        for row, (tokens, row_loss_mask) in enumerate(rows):
            input_ids[row, :len(tokens)] = torch.tensor(tokens)
            attention_mask[row, :len(tokens)] = 1
            loss_mask[row, :len(row_loss_mask)] = torch.tensor(row_loss_mask, dtype=torch.float)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "advantages": torch.tensor(advantages, dtype=torch.float),
        }
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------

    def _log(self, metrics: dict, prefix: str, step_key: str = None, step: int = None) -> None:
        """Print ``metrics``; with ``step_key``, also record them (metrics.jsonl, wandb) at ``step``."""
        if not _is_rank_zero():
            return
        values = " ".join(f"{name}={value:.4g}" for name, value in sorted(metrics.items()))
        print(f"[grpo][{prefix}][step {self.step}] {values}", flush=True)
        if step_key is not None:
            self.metrics_logger.log(metrics, step_key, self.step if step is None else step)

    def _save(self) -> None:
        """Save a resumable DeepSpeed checkpoint and a Hugging Face copy of the weights."""
        tag = f"step_{self.step}"
        save_dir = self.cfg.training.save_dir
        self.engine.save_checkpoint(save_dir, tag=tag)
        # ZeRO-2 keeps full weights on every rank, so rank 0 can export them alone.
        if _is_rank_zero():
            hf_dir = os.path.join(save_dir, f"hf_{tag}")
            self.engine.module.model.save_pretrained(hf_dir)
            self.tokenizer.save_pretrained(hf_dir)
            print(f"[grpo] saved checkpoint to {save_dir}/{tag} and {hf_dir}", flush=True)
        dist.barrier()
