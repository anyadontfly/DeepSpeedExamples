#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Assumes you're cd'd into training/grpo/.
set -euo pipefail

CONFIG="${1:-configs/retool_qwen3_4b_sft.json}"
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed --num_gpus "${NUM_GPUS}" main.py --config "${CONFIG}"
