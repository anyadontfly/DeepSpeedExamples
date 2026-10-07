# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""The policy as DeepSpeed trains it: a causal LM whose forward returns token log-probs.

Full logits of a long rollout are too large to keep (16k tokens x 152k
vocab is 9.7 GB in fp32), so the LM head runs over the sequence in chunks
under activation checkpointing and only each target token's log-prob is
kept. Training goes through ``engine(...)`` rather than a
submodule, because DeepSpeed hooks the engine's forward outputs to prepare
ZeRO for the backward pass.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_log_probs(hidden, lm_head, targets, temperature):
    logits = lm_head(hidden).float() / temperature
    return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class PolicyModel(torch.nn.Module):
    """Wraps a Hugging Face causal LM; ``generate`` is passed through for rollout."""

    def __init__(self, model, chunk_size: int = 1024):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, input_ids, attention_mask, temperature: float = 1.0):
        """Log-prob of each next token under ``softmax(logits / temperature)``.

        Rollout samples from ``softmax(logits / temperature)``, so the
        policy being trained is that tempered distribution.

        Returns:
            ``[B, T - 1]`` log-probs; position ``t`` scores ``input_ids[:, t + 1]``.
        """
        hidden = self.model.get_decoder()(input_ids=input_ids, attention_mask=attention_mask,
                                          use_cache=False).last_hidden_state
        hidden = hidden[:, :-1]
        targets = input_ids[:, 1:]
        lm_head = self.model.get_output_embeddings()
        log_probs = []
        for start in range(0, hidden.shape[1], self.chunk_size):
            end = start + self.chunk_size
            log_probs.append(
                checkpoint(_chunk_log_probs,
                           hidden[:, start:end],
                           lm_head,
                           targets[:, start:end],
                           temperature,
                           use_reentrant=False))
        return torch.cat(log_probs, dim=1)
