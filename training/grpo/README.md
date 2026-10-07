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
   ranks, with no KL term. The step is accumulated over micro-batches. The
   LM head runs in chunks, so full-vocabulary logits of long trajectories are
   never kept in memory (`policy.py`).

## Install

The multi-turn rollout is not in a DeepSpeed release yet, so install
DeepSpeed from source, then:

```bash
pip install -r requirements.txt
```

## Data

The training set is `dapo-math-17k`, in miles' JSONL format: a
chat-message `prompt` and a string `label`.

```bash
cd training/grpo
bash scripts/prepare_data.sh
```

## Run

```bash
# miles' fine-tuned checkpoint (recommended: it already calls the tool)
bash scripts/run_retool.sh configs/retool_qwen3_4b_sft.json

# base Qwen3-4B
bash scripts/run_retool.sh configs/retool_qwen3_4b_base.json

# quick check on one or two GPUs with Qwen3-0.6B
NUM_GPUS=1 bash scripts/run_retool.sh configs/debug_qwen3_0.6b.json
```

Rank 0 logs the loss and the mean reward after each step. Checkpoints go to
`training.save_dir`: a resumable DeepSpeed checkpoint plus a Hugging Face copy
in `hf_step_<N>`.

## Configuration

`configs/retool_qwen3_4b_sft.json` follows miles' defaults:

| Setting | Value |
|---|---|
| Prompts per step × samples per prompt | 32 × 8, one optimizer step |
| Sampling | temperature 1.0, no top-p / top-k |
| Turns | at most 16 |
| Optimizer | AdamW, lr 1e-6, betas (0.9, 0.98), weight decay 0.1, grad clip 1.0 |

ZeRO, the optimizer and the hybrid engine are set in `configs/ds_zero2.json`.
The trainer fills in `train_micro_batch_size_per_gpu` and
`gradient_accumulation_steps`, so that one step covers all of a rank's
trajectories.

The main knobs:

- `rollout.max_new_tokens`: generated tokens per trajectory, **summed over
  its turns**. miles' `rollout_max_response_len` (8192) applies per turn
  instead.
- `rollout.max_context_len`: total prompt, response and tool tokens per
  trajectory. A trajectory that reaches it stops as truncated.
- `rollout.generate_batch_size`: trajectories per `generate` call, in whole
  prompts and at least one. It bounds the KV cache, about 2.4 GB per
  trajectory at 16k tokens for Qwen3-4B, so lower it if rollout runs out
  of memory.
- `tool.max_output_chars`: tool output is cut to this length, so that one
  large print cannot fill the context.
- `algorithm.tool_call_bonus`: miles' optional reward bonus for calling
  tools on wrong answers, off by default.

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
