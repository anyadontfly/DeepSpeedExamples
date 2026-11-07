# GRPO with Multi-Turn Tool Calling (ReTool)

This example trains a model with GRPO to solve math problems while calling a Python interpreter. Rollout and training share one DeepSpeed **hybrid engine** (ZeRO-2): the policy generates with its own training weights, so there is no separate inference server and no weight sync.

Each step:

1. **Rollout.** Each rank takes its share of the step's prompts. The policy writes `<tool_call>` blocks, and `deepspeed.runtime.rollout.HybridEngineRollout` runs them in a sandboxed `code_interpreter` (`tools.py`) and appends the output as a tool message. This repeats for up to `max_turns` turns. Only the tokens the model generated are trained on; tool output is masked out.
2. **Reward.** DAPO strict-box check (`reward.py`): +1 if the last `\boxed{...}` of the response equals the label, −1 otherwise. Advantages are normalized within each prompt's group of samples.
3. **Update.** One optimizer step per rollout, using PPO-clip with clip-higher (0.2 / 0.28). The loss is averaged over every generated token across ranks (or per trajectory first, see `algorithm.loss_aggregation`), with no KL term. The step is accumulated over micro-batches. The LM head runs in chunks, so full-vocabulary logits of long trajectories are never kept in memory (`policy.py`).

## Install

The multi-turn rollout is not in a DeepSpeed release yet, so install DeepSpeed from source, then:

```bash
pip install -r requirements.txt
```

The `retool_qwen35_4b*` configs use Qwen3.5-4B and segment-KI, which also need a DeepSpeed build with PR #8602.

## Data

The training set is `dapo-math-17k` and the evaluation set is `aime-2024`, in JSONL with a chat-message `prompt` and a string `label`.

```bash
cd training/grpo
bash scripts/prepare_data.sh
```

The script also wraps each AIME prompt in the instructions every `dapo-math-17k` prompt has. The downloaded problems don't ask for a `\boxed{}` answer, which is the only form the scorer accepts, so without them nearly every sample scores −1. If you downloaded `aime-2024` before this step existed, run the script again.

## Run

`scripts/run_retool.sh` takes a config, which defaults to `configs/retool_qwen3_4b_sft.json`, and reads the GPU count from `NUM_GPUS`, which defaults to 8. `training.global_batch_size` must be a multiple of `NUM_GPUS` × `n_samples_per_prompt`, so set `NUM_GPUS` to suit the config.

```bash
# quick check with Qwen3-0.6B, on one GPU
NUM_GPUS=1 bash scripts/run_retool.sh configs/debug_qwen3_0.6b.json

# a fine-tuned Qwen3-4B at a small scale, which already calls the tool, on one or two GPUs
NUM_GPUS=1 bash scripts/run_retool.sh configs/retool_qwen3_4b_sft.json

# the same checkpoint at full scale with AIME evaluation
NUM_GPUS=2 bash scripts/run_retool.sh configs/retool_qwen3_4b_sft_2gpus_compare.json

# base Qwen3-4B
NUM_GPUS=8 bash scripts/run_retool.sh configs/retool_qwen3_4b_base.json

# Qwen3.5-4B on one A100-40GB: optimizer state on the CPU, graph capture and segment-KI decode
NUM_GPUS=1 bash scripts/run_retool.sh configs/retool_qwen35_4b.json

# the same settings on one A40, with metrics files and wandb logging
NUM_GPUS=1 bash scripts/run_retool.sh configs/retool_qwen35_4b_a40.json

# Qwen3.5-4B at full scale on one H100, with the optimizer state on the CPU
NUM_GPUS=1 bash scripts/run_retool.sh configs/retool_qwen35_4b_h100.json

# Qwen3.5-4B at full scale on one H200, with the optimizer state on the GPU
NUM_GPUS=1 bash scripts/run_retool.sh configs/retool_qwen35_4b_h200.json
```

Rank 0 prints the loss, the mean reward and the step's metrics after each step. When `eval.path` is set, it also logs AIME accuracy (over every sample) before training, every `eval.interval` steps and at the end. Checkpoints go to `training.save_dir`: a resumable DeepSpeed checkpoint plus a Hugging Face copy in `hf_step_<N>`.

## Metrics and resuming

- `training.metrics_dir`: writes per-step metrics to `metrics.jsonl`, and per-sample, per-generate-call and per-tool-call spans to `spans.<pid>.jsonl`. Empty disables both.
- `training.wandb_project`, `training.wandb_run_name`: also log the metrics to wandb when `WANDB_API_KEY` is set.
- `training.resume_from`: continues from the latest checkpoint in this directory, with its weights, optimizer state, step count and position in the data order. Random number state is not restored.
- `eval.name`: the metric names, `eval/<name>`; the default is `aime`.
- `eval.at_end`: evaluates once more after the last step; on by default.

## Configuration

`configs/retool_qwen3_4b_sft_2gpus_compare.json` is the full-scale Qwen3-4B setup:

| Setting | Value |
|---|---|
| Trajectories per step | 256 (32 prompts × 8 samples), one optimizer step |
| Sampling | temperature 1.0, no top-p / top-k |
| Turns | at most 16 |
| Optimizer | AdamW, lr 1e-6, betas (0.9, 0.98), weight decay 0.1, grad clip 1.0 |
| Evaluation | AIME 2024 every 20 steps, 16 samples per problem, 16384 tokens per turn |

ZeRO, the optimizer and the hybrid engine are set in `configs/ds_zero2.json`; `configs/ds_zero2_offload.json` also keeps the optimizer state on the CPU (ZeRO-Offload, about 16 bytes of host memory per parameter), for single-GPU runs. The `*_fp32_grad_accum` variants accumulate gradients in fp32. The trainer fills in `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps`, so that one step covers all of a rank's trajectories.

The main knobs:

- `rollout.max_new_tokens`: generated tokens per trajectory, **summed over its turns**.
- `rollout.max_new_tokens_per_turn`: an additional cap on each turn, off by default. `eval` has the same two fields.
- `rollout.max_context_len`: total prompt, response and tool tokens per trajectory. A trajectory that reaches it stops as truncated.
- `training.global_batch_size`: trajectories per step across all ranks, a multiple of ranks × `n_samples_per_prompt` so that each rank rolls out whole prompts. Each rank rolls out its share in one `generate` call, then the step takes one optimizer step over all of them. The call's KV cache is about 2.4 GB per trajectory at 16k tokens for Qwen3-4B. Evaluation decodes the same number of trajectories per call, so with more samples per prompt it takes fewer prompts per call.
- `training.micro_batch_size`: trajectories per forward and backward pass across all ranks, split evenly between them; it defaults to `global_batch_size` (one pass per step). It must divide `global_batch_size`; gradient accumulation covers the rest, so it changes memory and speed, not the update.
- `rollout.generate_batch_size`: trajectories per `generate` call, in whole prompts and at least one; by default a rank's whole share of the step. It bounds the call's KV cache and prefill memory, so lower it if rollout runs out of memory. The training step still covers all of `global_batch_size`.
- `rollout.continuous_batch_size`: decode rows for continuous batching (off by default). Each turn of each trajectory takes a free row and releases it as soon as the turn ends, and tools run while the rows keep decoding, instead of every turn of a call decoding until its longest row ends. Rows are sized by `max_context_len`, so the KV cache is `continuous_batch_size x max_context_len` tokens. Needs a DeepSpeed build with multi-turn continuous batching.
- `rollout.tool_call_parser`: the format of the model's tool calls, `"hermes"` (Qwen2.5, Qwen3; the default) or `"qwen3_coder"` (Qwen3.5).
- `rollout.use_graph_capture`, `rollout.use_segki`: CUDA-graph decode and segment-KI kernels in the hybrid engine rollout. `use_segki` needs a DeepSpeed build with segment-KI (PR #8602).
- `rollout.enable_profiling`: times each `generate` call and adds the timings to the metrics.
- `tool.max_output_chars`: tool output is cut to this length, so that one large print cannot fill the context.
- `algorithm.tool_call_bonus`: an optional reward bonus for calling tools on wrong answers, off by default.
- `algorithm.loss_aggregation`: `"token_mean"` (default) weighs every generated token of the step equally; `"sample_mean"` averages each trajectory over its own tokens first, then over trajectories.

## Notes and limitations

- **The sandbox only screens obvious misuse.** It applies an import allowlist, a pattern filter, a 4 GB memory limit and a timeout. It is not a security boundary, so run untrusted training on isolated machines.
- **Hybrid engine without injected kernels.** DeepSpeed has no inference policy for Qwen3, so without segment-KI, generation is plain Hugging Face `generate` on the training weights. The hybrid engine still builds its inference workspace at startup, which may JIT-compile DeepSpeed's inference op.
- **Rollout speed.** Every turn re-prefills the whole conversation. The rollout also generates each turn to its full token budget, so that ZeRO-3 ranks stay in lockstep. That lockstep isn't needed with ZeRO-2. Expect rollout to dominate step time.
- **ZeRO-2 only.** Under ZeRO-3, ranks with different numbers of turns would run different numbers of parameter gathers and hang.

## Acknowledgement

This example is adapted from the [miles retool_v2](https://github.com/radixark/miles/tree/main/examples/retool_v2) recipe (Copyright 2025-2026 RadixArk, Inc., Apache License 2.0). The code interpreter in `tools.py`, the answer check in `reward.py` and the metric definitions in `metrics.py` are based on miles.
