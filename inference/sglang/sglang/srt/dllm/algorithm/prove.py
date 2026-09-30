# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
"""ProVe — proposer-verifier decoding with a small stock verifier (the paper's name).

``ProVe`` is the name used in the paper ("ProVe: Proposer-Verifier Decoding"). It is a
thin alias of :class:`MarsVerifierStream`, the class every experiment was run with; both
names register the same algorithm, so ``--dllm-algorithm ProVe`` and
``--dllm-algorithm MarsVerifierStream`` are interchangeable (log lines keep the
``[MarsVerifierStream]`` tag).

One decoding cycle:

1. The PROPOSER (a MARS block-trained AR model) runs ONE forward over a window of
   ``block_size`` placeholder tokens and proposes a block of tokens.
2. The VERIFIER (a smaller stock AR model served in-process, ``verifier_path``) scores the
   proposed block in ONE forward conditioned on the realized prefix. Position i is
   committed while ``p~(y_i) * q(y_i) >= theta`` (``rule: and``; ``p~`` = verifier
   probability, ``q`` = proposer probability); the first token of the block is the
   proposer's exact next token and is always committed.

Config keys are documented in ``inference/configs/README.md`` (``verifier_path``,
``theta``, ``rule``, ``verifier_temp``, ``audit``, ``inproc_pool_gb`` /
``inproc_mem_fraction``, ...).
"""
from sglang.srt.dllm.algorithm.mars_verifier_stream import MarsVerifierStream


class ProVe(MarsVerifierStream):
    """Paper name for MarsVerifierStream: small stock verifier, gate p~ * q >= theta."""


Algorithm = ProVe
