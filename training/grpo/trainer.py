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
     over every generated token across ranks. Tool output is masked out.

With one update per rollout, the old policy is the current one, so the old
log-probs are the current ones detached; the importance ratio is 1 and the
loss reduces to the policy gradient.
"""

import os

import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

from config import GRPOConfig
from reward import retool_reward
from deepspeed.runtime.rollout import RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.base import Sample


def _is_rank_zero() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _mean_across_ranks(values: dict) -> dict:
    """Average each metric over ranks with one all-reduce."""
    names = sorted(values)
    tensor = torch.tensor([float(values[name]) for name in names], device=get_accelerator().current_device_name())
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return dict(zip(names, tensor.tolist()))


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
    ):
        self.cfg = cfg
        self.engine = engine
        self.tokenizer = tokenizer
        self.rollout = rollout
        self.dataloader = dataloader

        self.device = get_accelerator().current_device_name()
        self.pad_token_id = tokenizer.pad_token_id
        self.step = 0
        self.sampling = SamplingConfig(
            max_new_tokens=cfg.rollout.max_new_tokens,
            temperature=cfg.rollout.temperature,
            n_samples_per_prompt=cfg.rollout.n_samples_per_prompt,
        )

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def train(self) -> None:
        epoch = 0
        while self.step < self.cfg.training.num_rollouts:
            # Reseed the DistributedSampler so each epoch reshuffles each rank's shard.
            self.dataloader.sampler.set_epoch(epoch)
            for rows in self.dataloader:
                if self.step >= self.cfg.training.num_rollouts:
                    break
                metrics = self._train_step(rows)
                self.step += 1
                if self.step % self.cfg.training.logging_steps == 0:
                    self._log(metrics, prefix="train")
                if self.step % self.cfg.training.save_steps == 0:
                    self._save()
            epoch += 1
        if self.step % self.cfg.training.save_steps:
            self._save()

    # ------------------------------------------------------------------
    # Rollout and reward
    # ------------------------------------------------------------------

    def _rollout(self, rows) -> list:
        """Generate trajectories for ``rows`` in chunks, the hybrid engine in inference mode."""
        self.engine.eval()
        samples = []
        # Each prompt expands to n_samples_per_prompt trajectories.
        prompts_per_chunk = max(1, self.cfg.rollout.generate_batch_size // self.sampling.n_samples_per_prompt)
        for chunk, start in enumerate(range(0, len(rows), prompts_per_chunk)):
            request = RolloutRequest(samples=[
                Sample(group_index=row["index"], prompt=row["prompt"]) for row in rows[start:start + prompts_per_chunk]
            ])
            samples.extend(self.rollout.generate(request, self.sampling).samples)
            if self.cfg.rollout.enable_profiling:
                # Covers all turns of this chunk's generate call.
                profile = self.rollout.get_last_profile()
                timings = {name: value for name, value in profile.items() if value is not None}
                self._log({"chunk": chunk, **timings}, prefix="rollout")
        return samples

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

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def _train_step(self, rows) -> dict:
        samples = self._rollout(rows)
        rewards = self._score(samples, rows)
        advantages = group_advantages(samples, rewards)

        self.engine.train()
        loss = self._update(samples, advantages)
        return _mean_across_ranks({"loss": loss, "reward": sum(rewards) / len(rewards)})

    def _update(self, samples, advantages) -> float:
        """One optimizer step over ``samples``: PPO-clip, token-mean over all ranks.

        Returns this rank's share of the step loss; summed over ranks it is the token-mean loss.
        """
        # Every generated token across ranks weighs the same. DeepSpeed averages
        # gradients over ranks, so each rank's sum is scaled by world_size / tokens.
        num_tokens = torch.tensor(float(sum(sum(sample.loss_mask or []) for sample in samples)), device=self.device)
        dist.all_reduce(num_tokens)
        loss_scale = dist.get_world_size() / num_tokens.clamp(min=1).item()

        eps_low, eps_high = self.cfg.algorithm.eps_clip, self.cfg.algorithm.eps_clip_high
        step_loss = 0.0
        micro_batch_size = self.cfg.training.micro_batch_size_per_gpu
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
            token_loss = torch.maximum(unclipped, clipped)
            loss = (token_loss * mask).sum() * loss_scale
            # The loss is already normalized over the whole step, so it must not
            # be divided again by the gradient accumulation steps.
            self.engine.backward(loss, scale_wrt_gas=False)
            self.engine.step()
            step_loss += loss.item()
        return step_loss

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

    def _log(self, metrics: dict, prefix: str) -> None:
        if not _is_rank_zero():
            return
        values = " ".join(f"{name}={value:.4g}" for name, value in sorted(metrics.items()))
        print(f"[grpo][{prefix}][step {self.step}] {values}", flush=True)

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
