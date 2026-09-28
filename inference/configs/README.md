# Algorithm configs (`--dllm-algorithm-config <yaml>`)

Every key here is read by the algorithm class named with `--dllm-algorithm`. `serve.sh`
picks the yaml for you and fills in the verifier path; use these files directly only when
you launch `python -m sglang.launch_server` yourself, and replace `__VERIFIER__` first: the
server refuses to start while the placeholder is still there.

| file | `--dllm-algorithm` | verifier | when to use |
|---|---|---|---|
| `prove_lossless.yaml` | `ProVeLossless` | the proposer itself | you want the proposer's exact greedy output, faster |
| `prove_3b_theta0.10.yaml` | `ProVe` | stock 3B (`__VERIFIER__`) | strictest lossy setting of the paper grid |
| `prove_3b_theta0.05.yaml` | `ProVe` | stock 3B | the paper's default operating point; `serve.sh --verifier` default |
| `prove_3b_theta0.01.yaml` | `ProVe` | stock 3B | fastest, largest quality drop |
| `prove_7b_theta0.05.yaml` | `ProVe` | stock 7B | smallest quality drop; the verifier's weights need ~16 GB more free memory |
| `mars_tau0.95.yaml` | `MarsThreshold` | none (self-gate on the proposer's confidence) | the MARS baseline |

The 3B and 7B ProVe files carry the same keys; the names say which verifier their comments
and memory notes describe, so a 7B verifier is not launched from a file written for the 3B.

## Keys, in plain words

Common to every algorithm

* `block_size` — how many tokens one proposed block holds, i.e. the window of placeholder
  tokens the proposer fills in one forward. Must equal the block size the proposer was
  trained with: 16 for `MARS-Qwen2.5-7B-blk16` and the 32B proposer. Without the yaml the
  server would default to 7 for a Qwen2 checkpoint, so always keep this key.
* `mask_id` — token id of the placeholder `<|mask|>`; default 151665 (the id in the
  released MARS checkpoints). `tools/prepare_checkpoint.py` prints the id it finds and
  tells you if it differs.

ProVe-lossless (`ProVeLossless` = `MarsExactStream`)

* `audit` — `true` counts, in the `MARSV_LOG=1` statistics, how often the first proposed
  token differs from the AR token. Pure bookkeeping; the output is exact either way.
  (`theta` / `gate` exist for experiments and default to `0` / `"q"`; leave them out.)

ProVe (`ProVe` = `MarsVerifierStream`)

* `verifier_path` — required. A **local directory** holding a stock AR model that shares
  the proposer's tokenizer (Qwen2.5-3B-Instruct or Qwen2.5-7B-Instruct for the Qwen2.5
  proposers). `serve.sh --verifier` writes it into the placeholder `__VERIFIER__`, and
  downloads a Hugging Face id first: the in-process verifier reads the size of its weights
  from the files in this directory, so an id that was not downloaded would be mis-sized.
* `theta` — the gate threshold. At each position of the proposed block the verifier's
  probability `p~` of the proposed token is multiplied by the proposer's own probability `q`;
  the block is committed up to the last position where `p~ * q >= theta`. Lower theta accepts
  more tokens per cycle (faster) and lets more disagreements through (lossier). Paper grid:
  0.10, 0.05, 0.01.
* `rule` — `and` (the paper): gate on `p~ * q`. `ptilde`: gate on `p~` alone.
* `verifier_temp` — temperature applied to the verifier's logits before reading `p~`
  (1.0 in the paper).
* `audit` — `true` reads, for free, whether the proposer's own AR token agrees with each
  committed token and reports the mismatch rate in the `MARSV_LOG=1` statistics.
* `vtok` — `false` in the paper. `"pq"` (or `"v"`) additionally commits a verifier-chosen
  token at the position where the gate stopped; the cycle gains a token but that token comes
  from the verifier.
* `inproc_fp8` — `true` runs the verifier in FP8. Paper: `false` (bf16).
* **Verifier memory.** The verifier runs inside the proposer's process, on the same GPU(s),
  with its own weights, CUDA graphs and KV pool, all carved out of the memory the proposer
  leaves free, i.e. whatever the proposer's `--mem-fraction-static` did not take (on an 80 GB
  GPU with `--mem-fraction-static 0.60` that is roughly 30 GB after the proposer's CUDA graphs).
  Two keys size the KV pool:
  * `inproc_pool_gb` (the shipped configs; also the default, 2.0, when neither key is
    present) — the verifier's KV pool in GB, absolute. 2 GB is about 58k tokens of
    Qwen2.5-3B KV or 37k tokens of Qwen2.5-7B KV, more than a dozen 4096-token requests.
    sglang's fraction is derived from it at start-up against the memory actually free, so one
    value ports across proposer sizes and GPU sizes. What the proposer leaves must cover the
    verifier's weights (about 6.4 GB for the 3B and 15.5 GB for the 7B on one GPU; halved per
    rank at TP=2), this pool and 1 GB of reserve. If it does not, the log says
    `POOL TARGET NOT REACHABLE` with the arithmetic, and if not even the weights fit the server
    stops and names the proposer's `--mem-fraction-static` as the knob to lower.
  * `inproc_mem_fraction` — pins sglang's raw `mem_fraction_static` for the verifier's runner
    instead. sglang applies that fraction to the memory *free when the runner starts* and
    deducts the weights from it: `pool = free_after_proposer x fraction - verifier_weights`.
    The same number therefore means a different pool beside every proposer and on every GPU
    size, and a value whose pool comes out at or below zero stops the server with the
    arithmetic (it used to fall back silently to a slow verifier). The paper's campaign pinned
    it: 3B verifier 0.18 with the 32B proposer at `--mem-fraction-static 0.70`, 7B verifier
    0.32 at 0.55 (TP=2, 80 GB GPUs): `serve.sh --tp 2 --mem-fraction 0.70 --verifier-mem-fraction 0.18`.
    A pinned fraction takes precedence over `inproc_pool_gb`; the environment variable
    `MARSV_INPROC_POOL_GB=<gb>` overrides both.
  * `inproc_reserve_gb` — memory kept free beyond weights and pool for the verifier's
    attention workspace and graph capture (default 1.0; 0.75 was measured to be too little).

  Too small a pool does not fail: the verifier cannot hold long prompts, requests degrade to
  one accepted token per cycle and speed collapses. The log reports it (`CAPACITY`,
  `degraded` counts next to the `cum stats` line).
* `inproc_max_cache` — the longest context (prompt + generation, tokens) the verifier can
  score; default 4096. Raise it for long prompts, together with `inproc_pool_gb`.
* `inproc_verifier` — `true` (default) uses the in-process verifier. `false` falls back to
  a HuggingFace-transformers stepper (`graph_verifier` / `fast_verifier`): tens of
  milliseconds per call instead of about 2.6 ms, only meant for debugging.
* `inproc_fallback` — `false` (default): if the in-process verifier cannot be built the
  server stops with the reason. `true` (or `MARSV_INPROC_FALLBACK=1`) restores the old
  behaviour of logging `[InprocVerifier] unavailable (...)` and continuing on the slow
  transformers stepper.

### Did the verifier come up?

With a verifier the server log must contain

```
[InprocVerifier] injected into dllm algorithm
```

`serve.sh` checks for it once the server answers and prints `in-process verifier: OK`
together with the `pool sizing` line. `grep '\[InprocVerifier\]' serve_<port>.log` shows the
sizing arithmetic. A `[InprocVerifier] unavailable` warning means the verifier fell back to
the slow transformers path (only possible with `inproc_fallback: true`); `POOL TARGET NOT
REACHABLE`, or a `PINNED` line whose predicted pool is at or below zero, means the proposer
left too little memory: lower the proposer's `--mem-fraction-static` (`serve.sh
--mem-fraction`) or drop a pinned `inproc_mem_fraction`.

MARS self-gate (`MarsThreshold`)

* `threshold` — tau. Tokens of the proposed block are committed left to right while the
  proposer's own top-1 probability is at least tau; the first token is always committed.
  0.95 in the paper.
* `temperature` — 0 (greedy) by default; the gate always uses the untempered probability.

Everything not listed is an experiment switch and keeps its default when omitted.
