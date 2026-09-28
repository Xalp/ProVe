# Third-party notices

ProVe's own code — everything outside `inference/sglang/`, plus the files listed as new or
modified in `inference/sglang/PROVE_CHANGES.md` — is released under the Apache License 2.0
(see `LICENSE`). The inference stack bundles code from two projects.

## SGLang — Apache License 2.0

`inference/sglang/` is a fork of SGLang (https://github.com/sgl-project/sglang),
Copyright (c) SGLang Team, licensed under the Apache License, Version 2.0. The full text is
in `inference/sglang/LICENSE` (the same text as the top-level `LICENSE`, apart from the
copyright line). The changes ProVe made to it are listed in `inference/sglang/PROVE_CHANGES.md`.

## I-DLM — BSD 3-Clause License

The sglang fork was taken from the I-DLM repository
(https://github.com/Introspective-Diffusion/I-DLM, commit `a23c1a1`), which adds the
diffusion-LLM serving path that ProVe builds on: `sglang/srt/dllm/config.py`,
`sglang/srt/dllm/mixin/`, `sglang/srt/dllm/algorithm/{base,idlm_blockN,joint_threshold,
low_confidence,fused_verify_kernel}.py`, the dLLM hooks in the scheduler / model runner /
attention backends, and the `install.sh` / `setup_env.sh` layout. Its license:

```
BSD 3-Clause License

Copyright (c) 2025, Together AI

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Models

The released proposer `Xalphinions/MARS-Qwen2.5-7B-blk16` and the stock verifiers
`Qwen/Qwen2.5-3B-Instruct` / `Qwen/Qwen2.5-7B-Instruct` are distributed under their own
licenses on the Hugging Face Hub (Qwen2.5: Apache-2.0 for 3B/7B).
