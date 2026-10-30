#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Downloads the miles retool training and AIME eval data into data/. Assumes you're cd'd into training/grpo/.
set -euo pipefail

hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir data/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir data/aime-2024
