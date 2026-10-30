# GRPO with Multi-Turn Tool Calling (ReTool)

This example trains a model with GRPO to solve math problems while calling a
Python interpreter, following the
[miles retool_v2](https://github.com/radixark/miles/tree/main/examples/retool_v2)
recipe. Rollout and training share one DeepSpeed **hybrid engine** (ZeRO-2,
torch AdamW): the policy generates with its own training weights, so there
is no separate inference server and no weight sync.

Each step:

1. **Rollout.** Each rank takes its share of the step's prompts. The policy
   writes Hermes-style `<tool_call>` blocks, and
   `deepspeed.runtime.rollout.HybridEngineRollout` runs them in a sandboxed
   `code_interpreter` (`tools.py`) and appends the output as a tool message.
   This repeats for up to `max_turns` turns. Only the tokens the model
   generated are trained on; tool output is masked out.
2. **Reward.** DAPO strict-box check (`reward.py`): +1 if the last
   `\boxed{...}` of the response equals the label, −1 otherwise. Advantages
   are normalized within each prompt's group of samples.
3. **Update.** One optimizer step per rollout, using PPO-clip with clip-higher
   (0.2 / 0.28). The loss is averaged over every generated token across
   ranks (or per trajectory first, see `algorithm.loss_aggregation`), with
   no KL term. The step is accumulated over micro-batches. The
   LM head runs in chunks, so full-vocabulary logits of long trajectories are
   never kept in memory (`policy.py`).

## Install

The multi-turn rollout is not in a DeepSpeed release yet, so install
DeepSpeed from source, then:

```bash
pip install -r requirements.txt
```

## Data

The training set is `dapo-math-17k` and the evaluation set is `aime-2024`,
in miles' JSONL format: a chat-message `prompt` and a string `label`.

```bash
cd training/grpo
bash scripts/prepare_data.sh
```

## Run

```bash
# miles' retool_v2 recipe on its fine-tuned checkpoint
bash scripts/run_retool.sh configs/retool_qwen3_4b_sft_miles.json

# the same checkpoint at a small scale (recommended first: it already calls the tool)
bash scripts/run_retool.sh configs/retool_qwen3_4b_sft.json

# base Qwen3-4B
bash scripts/run_retool.sh configs/retool_qwen3_4b_base.json

# Qwen3.5-4B on one 40GB GPU: optimizer state offloaded to the CPU, graph capture + segment-KI decode
NUM_GPUS=1 bash scripts/run_retool.sh configs/retool_qwen35_4b.json

# Qwen3.5-4B on one H200: miles' recipe size (32 prompts x 8, 8192 tokens per turn, no context cap), Adam on the GPU
NUM_GPUS=1 bash scripts/run_retool.sh configs/retool_qwen35_4b_h200.json

# quick check on one or two GPUs with Qwen3-0.6B
NUM_GPUS=1 bash scripts/run_retool.sh configs/debug_qwen3_0.6b.json
```

Rank 0 logs the loss and the mean reward after each step. When `eval.path`
is set, it also logs AIME accuracy (over every sample) before training, every
`eval.interval` steps and at the end. Checkpoints go to
`training.save_dir`: a resumable DeepSpeed checkpoint plus a Hugging Face copy
in `hf_step_<N>`.

## Configuration

`configs/retool_qwen3_4b_sft_miles.json` follows miles' retool_v2 defaults:

| Setting | Value |
|---|---|
| Trajectories per step | 256 (32 prompts × 8 samples), one optimizer step |
| Sampling | temperature 1.0, no top-p / top-k |
| Turns | at most 16 |
| Optimizer | AdamW, lr 1e-6, betas (0.9, 0.98), weight decay 0.1, grad clip 1.0 |
| Evaluation | AIME 2024 every 20 steps, 16 samples per problem, 16384 tokens |

ZeRO, the optimizer and the hybrid engine are set in `configs/ds_zero2.json`;
`configs/ds_zero2_offload.json` also keeps the optimizer state on the CPU
(ZeRO-Offload, about 16 bytes of host memory per parameter), for single-GPU runs.
The trainer fills in `train_micro_batch_size_per_gpu` and
`gradient_accumulation_steps`, so that one step covers all of a rank's
trajectories.

The main knobs:

- `rollout.max_new_tokens`: generated tokens per trajectory, **summed over
  its turns**.
- `rollout.max_new_tokens_per_turn`: an additional cap on each turn, off by
  default. miles' `rollout_max_response_len` (8192) is this per-turn cap; to
  match it, set this to 8192 and `max_new_tokens` to `max_turns` × 8192.
  `eval` has the same two fields.
- `rollout.max_context_len`: total prompt, response and tool tokens per
  trajectory. A trajectory that reaches it stops as truncated.
- `training.global_batch_size`: trajectories per step across all ranks,
  a multiple of ranks × `n_samples_per_prompt` so that each rank rolls out
  whole prompts. Each rank rolls out its share in one `generate` call, then
  the step takes one optimizer step over all of them. The call's KV cache is
  about 2.4 GB per trajectory at 16k tokens for Qwen3-4B. Evaluation decodes
  the same number of trajectories per call, so with more samples per prompt
  it takes fewer prompts per call.
- `training.micro_batch_size`: trajectories per forward and backward pass
  across all ranks, split evenly between them; it defaults to
  `global_batch_size` (one pass per step). It must divide `global_batch_size`;
  gradient accumulation covers the rest, so it changes memory and speed, not
  the update.
- `rollout.generate_batch_size`: trajectories per `generate` call, in whole
  prompts and at least one; by default a rank's whole share of the step. It
  bounds the call's KV cache and prefill memory, so lower it if rollout runs
  out of memory. The training step still covers all of `global_batch_size`.
- `rollout.continuous_batch_size`: decode rows for continuous batching
  (off by default). Each turn of each trajectory takes a free row and
  releases it as soon as the turn ends, and tools run while the rows keep
  decoding, instead of every turn of a call decoding until its longest row
  ends. Rows are sized by `max_context_len`, so the KV cache is
  `continuous_batch_size x max_context_len` tokens. Needs a DeepSpeed build
  with multi-turn continuous batching.
- `rollout.tool_call_parser`: the format of the model's tool calls,
  `"hermes"` (Qwen2.5, Qwen3; the default) or `"qwen3_coder"` (Qwen3.5).
- `rollout.use_graph_capture`, `rollout.use_segki`: CUDA-graph decode and
  segment-KI kernels in the hybrid engine rollout. `use_segki` needs a
  DeepSpeed build with segment-KI (PR #8602).
- `tool.max_output_chars`: tool output is cut to this length, so that one
  large print cannot fill the context.
- `algorithm.tool_call_bonus`: miles' optional reward bonus for calling
  tools on wrong answers, off by default.
- `algorithm.loss_aggregation`: `"token_mean"` (default) weighs every
  generated token of the step equally; `"sample_mean"` averages each
  trajectory over its own tokens first, then over trajectories, as miles
  does unless `--calculate-per-token-loss` is set.

## Notes and limitations

- **The sandbox only screens obvious misuse.** It applies an import
  allowlist, a pattern filter, a 4 GB memory limit and a timeout, as in
  miles. It is not a security boundary, so run untrusted training on
  isolated machines.
- **Hybrid engine without injected kernels.** DeepSpeed has no inference
  policy for Qwen3, so generation is plain Hugging Face `generate` on the
  training weights. The hybrid engine still builds its inference workspace
  at startup, which may JIT-compile DeepSpeed's inference op.
- **Rollout speed.** Every turn re-prefills the whole conversation. The
  rollout also generates each turn to its full token budget, so that ZeRO-3
  ranks stay in lockstep. That lockstep isn't needed with ZeRO-2. Expect
  rollout to dominate step time.
- **ZeRO-2 only.** Under ZeRO-3, ranks with different numbers of turns would
  run different numbers of parameter gathers and hang.
