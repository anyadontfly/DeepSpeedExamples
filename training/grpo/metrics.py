# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Rollout quality and performance metrics: ``rollout/*``, ``passrate/*``, ``multi_turn/*``, ``perf/*``, ``train/*``
and ``eval/*``.

``MetricsLogger`` writes each record to ``metrics.jsonl`` and, when a project is given and
``WANDB_API_KEY`` is set, to wandb. ``SpanLog`` writes per-sample, per-generate-call and per-tool-call
records to ``spans.<pid>.jsonl``.
"""

import json
import math
import os
import time
import zlib
from collections import defaultdict

import numpy as np

# A response repeats itself if a 10K-character window compresses more than 10x.
REPETITION_WINDOW_SIZE_CHARS = 10_000
REPETITION_WINDOW_STRIDE_CHARS = 5_000
REPETITION_MAX_WINDOWS = 32
REPETITION_COMPRESSION_RATIO_THRESHOLD = 10.0


def has_repetition(text: str) -> bool:
    """Whether any overlapping window of ``text`` is highly compressible."""
    if len(text) < REPETITION_WINDOW_SIZE_CHARS:
        return False
    final_start = len(text) - REPETITION_WINDOW_SIZE_CHARS
    stride = max(REPETITION_WINDOW_STRIDE_CHARS, math.ceil(final_start / (REPETITION_MAX_WINDOWS - 1)))
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    for start in starts:
        raw = text[start:start + REPETITION_WINDOW_SIZE_CHARS].encode("utf-8")
        if len(raw) / len(zlib.compress(raw, 9)) > REPETITION_COMPRESSION_RATIO_THRESHOLD:
            return True
    return False


def statistics(values) -> dict:
    """Mean, median, max and min of ``values``."""
    values = np.array(values, dtype=float)
    return {
        "mean": values.mean().item(),
        "median": np.median(values).item(),
        "max": values.max().item(),
        "min": values.min().item()
    }


def pass_at_k(rewards_by_group: list[list[float]]) -> dict:
    """Unbiased pass@k for k = 1, 2, 4, ... up to the group size; correct means reward 1."""
    group_size = len(rewards_by_group[0])
    if group_size < 2:
        return {}
    num_correct = [sum(reward == 1 for reward in rewards) for rewards in rewards_by_group]
    metrics = {}
    for k in [2**i for i in range(int(math.log2(group_size)) + 1)]:
        estimates = [1.0 if group_size - c < k else 1.0 - np.prod(1.0 - k / np.arange(group_size - c + 1, group_size + 1))
                     for c in num_correct]
        metrics[f"pass@{k}"] = float(np.mean(estimates))
    return metrics


def zero_std(rewards_by_group: list[list[float]]) -> dict:
    """Zero-std metrics: groups whose samples all got the same reward give GRPO no signal."""
    uniform = [rewards[0] for rewards in rewards_by_group if all(reward == rewards[0] for reward in rewards)]
    counts = defaultdict(int)
    for reward in uniform:
        counts[str(round(reward, 1))] += 1
    metrics = {f"zero_std/count_{reward}": count for reward, count in counts.items()}
    metrics["zero_std/all_zero_percentage"] = sum(reward == 0 for reward in uniform) / len(rewards_by_group)
    metrics["zero_std/all_one_percentage"] = sum(reward == 1 for reward in uniform) / len(rewards_by_group)
    return metrics


def sample_summary(sample, reward: float) -> dict:
    """The per-trajectory fields the metrics need, small enough to gather across ranks."""
    effective = sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length
    return {
        "group_index": sample.group_index,
        "reward": float(reward),
        "status": sample.status.value,
        # Tokens after the prompt, tool output included (the raw response length).
        "response_length": sample.response_length,
        # Tokens the model generated, tool output excluded (the effective response length).
        "effective_response_length": effective,
        "total_length": sum(sample.attention_mask) if sample.attention_mask else len(sample.tokens),
        "num_tool_calls": sample.response.count("<tool_call>"),
        "repetition": has_repetition(sample.response),
    }


def rollout_metrics(summaries: list[dict], max_response_len_per_turn) -> dict:
    """The rollout/*, passrate/* and multi_turn/* metrics of one step's trajectories (all ranks)."""
    groups = defaultdict(list)
    for summary in summaries:
        groups[summary["group_index"]].append(summary["reward"])
    rewards_by_group = list(groups.values())
    effective = [s["effective_response_length"] for s in summaries]
    raw = [s["response_length"] for s in summaries]

    metrics = {
        "rollout/raw_reward": float(np.mean([s["reward"] for s in summaries])),
        "rollout/num_training_samples": len(summaries),
        "rollout/repetition_frac": float(np.mean([s["repetition"] for s in summaries])),
        "rollout/truncated_ratio": float(np.mean([s["status"] == "truncated" for s in summaries])),
        "rollout/tool_calls_per_sample": float(np.mean([s["num_tool_calls"] for s in summaries])),
    }
    metrics.update({f"rollout/response_len/{k}": v for k, v in statistics(effective).items()})
    metrics.update({f"rollout/{k}": v for k, v in zero_std(rewards_by_group).items()})
    metrics.update({f"passrate/{k}": v for k, v in pass_at_k(rewards_by_group).items()})
    raw_stats = statistics(raw)
    effective_stats = statistics(effective)
    for k in ("mean", "max", "min"):
        metrics[f"multi_turn/raw_response_length/response_length_{k}"] = raw_stats[k]
        metrics[f"multi_turn/wo_obs_response_length/response_length_{k}"] = effective_stats[k]
    if max_response_len_per_turn:
        # Compares the whole trajectory's length with the per-turn cap.
        metrics["multi_turn/raw_response_length/response_length_clip_ratio"] = float(
            np.mean([length >= max_response_len_per_turn for length in raw]))
    return metrics


def eval_metrics(summaries: list[dict], name: str) -> dict:
    """The eval/<name>* metrics: mean reward, truncation and response lengths over every eval sample."""
    metrics = {
        f"eval/{name}": float(np.mean([s["reward"] for s in summaries])),
        f"eval/{name}-truncated_ratio": float(np.mean([s["status"] == "truncated" for s in summaries])),
    }
    lengths = [s["effective_response_length"] for s in summaries]
    metrics.update({f"eval/{name}/response_len/{k}": v for k, v in statistics(lengths).items()})
    return metrics


def sum_profiles(profiles: list[dict]) -> dict:
    """One step's generate-call profiles summed into perf/rollout_profile/*; tokens/s over the summed time."""
    totals = defaultdict(float)
    for profile in profiles:
        for name, value in profile.items():
            if name.endswith("_ms") or (name.startswith("num_") and name != "num_samples_per_prompt"):
                totals[name] += value or 0.0
    metrics = {f"perf/rollout_profile/{name}": value for name, value in totals.items()}
    if totals.get("total_ms"):
        metrics["perf/rollout_profile/tokens_per_second"] = totals["num_generated_tokens"] / (totals["total_ms"] / 1000)
    # Continuous batching: the share of decode slots that produced a kept token.
    occupancy = [profile["continuous_decode_occupancy"] for profile in profiles if "continuous_decode_occupancy" in profile]
    if occupancy:
        metrics["perf/rollout_profile/continuous_decode_occupancy"] = float(np.mean(occupancy))
    return metrics


def host_memory_gb() -> dict:
    """System-wide used host memory and swap, from /proc/meminfo."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            name, value = line.split(":", 1)
            info[name] = int(value.split()[0]) / 1024**2
    return {
        "perf/host_memory_used_gb": info["MemTotal"] - info["MemAvailable"],
        "perf/host_swap_used_gb": info.get("SwapTotal", 0.0) - info.get("SwapFree", 0.0),
    }


class MetricsLogger:
    """Writes metric records to ``<log_dir>/metrics.jsonl`` and to wandb; rank 0 only."""

    # wandb x-axes: each metric family is plotted against its own step counter.
    STEP_KEYS = {"eval/": "eval/step", "train/": "train/step"}

    def __init__(self, log_dir: str = "", wandb_project: str = "", wandb_run_name: str = "", config: dict = None):
        self.file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.file = open(os.path.join(log_dir, "metrics.jsonl"), "a", buffering=1)
        self.wandb = None
        if wandb_project and os.environ.get("WANDB_API_KEY"):
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name or None, config=config)
            for step_key in ("rollout/step", "train/step", "eval/step"):
                wandb.define_metric(step_key)
            for prefix in ("rollout/", "passrate/", "multi_turn/", "perf/"):
                wandb.define_metric(f"{prefix}*", step_metric="rollout/step")
            for prefix, step_key in self.STEP_KEYS.items():
                wandb.define_metric(f"{prefix}*", step_metric=step_key)
            self.wandb = wandb

    def log(self, metrics: dict, step_key: str, step: int) -> None:
        record = {step_key: step, **metrics}
        if self.file is not None:
            self.file.write(json.dumps({"time": time.time(), **record}) + "\n")
        if self.wandb is not None:
            self.wandb.log(record)

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
        if self.wandb is not None:
            self.wandb.finish()


class SpanLog:
    """Appends JSON records to ``<log_dir>/spans.<pid>.jsonl``; a no-op without a directory."""

    def __init__(self, log_dir: str = ""):
        self.file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.file = open(os.path.join(log_dir, f"spans.{os.getpid()}.jsonl"), "a", buffering=1)
        # What the records being written belong to; the trainer sets it before each rollout.
        self.context = {}

    def write(self, record: dict) -> None:
        if self.file is not None:
            self.file.write(json.dumps({**record, **self.context}) + "\n")
