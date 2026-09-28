# sglang (ProVe fork)

This directory is the SGLang fork bundled with I-DLM
(https://github.com/Introspective-Diffusion/I-DLM, `inference/sglang/`), plus the ProVe
changes listed in [PROVE_CHANGES.md](PROVE_CHANGES.md): the `ProVe` / `ProVeLossless`
decoding algorithms under `sglang/srt/dllm/`.

Install it with `bash ../install.sh` (or `pip install -e .` after installing torch 2.9.1
cu129). Everything user-facing (configs, `serve.sh`, benchmarks) lives one level up; see
[../README.md](../README.md).

License: Apache-2.0 (SGLang) — see [LICENSE](LICENSE). The dLLM serving path under
`sglang/srt/dllm/` (`config.py`, `mixin/`, `algorithm/{base,idlm_blockN,joint_threshold,
low_confidence,fused_verify_kernel}.py` and the dLLM hooks in the scheduler, model runner and
attention backends) originates from I-DLM (BSD-3-Clause, Together AI); its license text and
the attribution are in the top-level `THIRD_PARTY_NOTICES.md`.
