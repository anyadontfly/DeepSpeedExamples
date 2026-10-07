# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""GRPO application configuration.

``GRPOConfig`` is loaded from JSON. The ``rollout`` sub-config is consumed
by DeepSpeed's rollout engine; optimizer, ZeRO and hybrid-engine settings
live in the DeepSpeed JSON that ``deepspeed_config`` points to.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from deepspeed.runtime.rollout import RolloutConfig as _BaseRolloutConfig


@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False


@dataclass
class RolloutConfig(_BaseRolloutConfig):
    """Extends DeepSpeed's RolloutConfig with the sampling knobs of training rollouts."""
    multi_turn: bool = True
    max_turns: Optional[int] = 16
    max_context_len: Optional[int] = 16384
    # Generated tokens per trajectory, summed over its turns.
    max_new_tokens: int = 8192
    temperature: float = 1.0
    n_samples_per_prompt: int = 8
    # Trajectories per generate call (whole prompts, at least one); bounds the KV cache of a rollout chunk.
    generate_batch_size: int = 16
    # Read by HybridEngineRollout: time each generate call (see get_last_profile()).
    enable_profiling: bool = False


@dataclass
class ToolConfig:
    timeout: float = 120.0
    memory_limit_gb: int = 4
    max_concurrency: int = 32
    # Long tool outputs are cut so one print cannot overflow the context.
    max_output_chars: int = 4096


@dataclass
class AlgorithmConfig:
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    # Sequence positions per LM-head chunk when computing token log-probs.
    logprob_chunk_size: int = 1024
    # miles' retool reward bonus for tool calls on wrong answers.
    tool_call_bonus: bool = False


@dataclass
class TrainingConfig:
    # Prompts per step across all ranks; each step is one optimizer update.
    rollout_batch_size: int = 32
    micro_batch_size_per_gpu: int = 1
    num_rollouts: int = 3000
    logging_steps: int = 1
    save_steps: int = 1000
    save_dir: str = "./grpo_ckpt"
    seed: int = 42


@dataclass
class DataConfig:
    path: str = ""
    prompt_key: str = "prompt"
    label_key: str = "label"
    shuffle: bool = True


@dataclass
class GRPOConfig:
    model: ModelConfig
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deepspeed_config: str = ""

    @classmethod
    def from_json(cls, path: str) -> "GRPOConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        return cls(
            model=ModelConfig(**raw["model"]),
            rollout=RolloutConfig(**raw.get("rollout", {})),
            tool=ToolConfig(**raw.get("tool", {})),
            algorithm=AlgorithmConfig(**raw.get("algorithm", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            deepspeed_config=raw.get("deepspeed_config", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self, world_size: int) -> None:
        if self.rollout.engine != "hybrid_engine":
            raise ValueError(f"Unknown rollout engine {self.rollout.engine!r}; expected 'hybrid_engine'")
        if self.rollout.n_samples_per_prompt < 2:
            raise ValueError("GRPO needs at least 2 samples per prompt to compare them")
        if self.rollout.temperature <= 0:
            raise ValueError("GRPO samples with temperature > 0")
        if self.training.rollout_batch_size % world_size:
            raise ValueError(f"rollout_batch_size {self.training.rollout_batch_size} is not divisible by the "
                             f"world size {world_size}")
        samples_per_rank = self.training.rollout_batch_size // world_size * self.rollout.n_samples_per_prompt
        if samples_per_rank % self.training.micro_batch_size_per_gpu:
            raise ValueError(f"{samples_per_rank} samples per rank do not split into micro-batches of "
                             f"{self.training.micro_batch_size_per_gpu}")
