# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
"""ProVe-lossless — the proposer verifies its own proposed block (the paper's name).

``ProVeLossless`` is the name used in the paper for the configuration whose verifier is
the proposer itself. It is a thin alias of :class:`MarsExactStream`, the class the
experiments were run with; both names register the same algorithm, so
``--dllm-algorithm ProVeLossless`` and ``--dllm-algorithm MarsExactStream`` are
interchangeable (log lines keep the ``[MarsExactStream]`` tag).

One decoding cycle costs TWO proposer forwards and needs no other model:

1. Forward A proposes a block of ``block_size`` tokens from a window of placeholders.
2. Forward B re-reads the proposed block as real tokens (causal) and yields, for every
   position, the token greedy AR decoding would have produced there. The longest prefix
   on which the two agree is committed, plus the first disagreeing AR token, so every
   cycle emits at least one token.

The output equals greedy AR decoding of the proposer token for token (up to bf16 ties).
Config keys: ``block_size`` (16 for the released proposers) and ``audit``
(``inference/configs/README.md``).
"""
from sglang.srt.dllm.algorithm.mars_exact_stream import MarsExactStream


class ProVeLossless(MarsExactStream):
    """Paper name for MarsExactStream: self-verification, exact greedy-AR output."""


Algorithm = ProVeLossless
