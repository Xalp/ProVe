# ProVe inference

Proposer-verifier decoding on top of SGLang. A **proposer** (a MARS block-trained AR model)
writes a **proposed block** of 16 tokens in one forward pass; a **verifier** reads that block
in one more forward pass and decides which prefix to commit. Everything an inference-only
user needs is in this directory.

| configuration | verifier | output | `--dllm-algorithm` |
|---|---|---|---|
| **ProVe-lossless** | the proposer itself (two proposer forwards per cycle) | identical to greedy AR decoding of the proposer, token for token (bf16 ties aside) | `ProVeLossless` |
| **ProVe** | a smaller stock AR model (Qwen2.5-3B/7B-Instruct); gate p̃·q ≥ θ | lossy; θ trades speed for quality | `ProVe` |
| MARS self-gate (baseline) | none; τ on the proposer's own top-1 probability | lossy | `MarsThreshold` |
| AR (baseline) | — | greedy AR, same weights | no `--dllm-*` flags |

`ProVe` / `ProVeLossless` are the paper's names; the classes they alias
(`MarsVerifierStream` / `MarsExactStream`) still work and still tag their log lines.
All ProVe configurations decode greedily: compare them against an AR server queried with
`temperature: 0`.

## Prerequisites

* Linux x86_64 with CUDA GPUs and a **CUDA 12.x** stack: a 12.x driver (a 535 driver works
  with the CUDA 12.9 compat package; the wheels are `cu129`) **and the CUDA 12.x toolkit with
  `nvcc`** (`CUDA_HOME` set): flashinfer and the fork's JIT kernels are compiled at first use.
  `install.sh` warns when `nvcc` is missing; the prebuilt `flashinfer-jit-cache` package for
  cu129 is the alternative, which we have not tested.
* The paper's numbers were produced on Hopper-class data-center GPUs. The in-process verifier
  is built on the FlashAttention-3 backend, so Ampere (compute capability 8.0) is **untested**
  for the ProVe (verifier) configuration; ProVe-lossless and the AR baseline use only the
  FlashInfer backend.
* Memory. Only the paper's setting (32B proposer, two 80 GB GPUs) is measured; the other rows
  are estimates from the formula in `configs/README.md` (proposer pool = fraction of the GPU
  minus its weights; verifier = its weights + a 2 GB KV pool + 1 GB reserve out of what the
  proposer leaves free). Check the server log's `[InprocVerifier] pool sizing` line.

  | GPU memory | proposer | verifier | `serve.sh` arguments |
  |---|---|---|---|
  | 24 GB (estimate) | 7B (15 GB of bf16 weights) | none: ProVe-lossless or AR | default `--mem-fraction 0.75` leaves ~3 GB of proposer KV |
  | 40 GB (estimate) | 7B | 3B (6.4 GB of weights) | default `--mem-fraction 0.60` with a verifier |
  | 80 GB (estimate) | 7B | 3B or 7B (15.5 GB of weights) | default |
  | 2 x 80 GB (measured) | 32B, `--tp 2` | 3B or 7B | default, or the paper's pinned split: `--mem-fraction 0.70 --verifier-mem-fraction 0.18` (3B), `--mem-fraction 0.55 --verifier-mem-fraction 0.32` (7B) |
* Python 3.12 (3.10+ should work) with `pip`, or `uv` / `conda` for `setup_env.sh`.
* Disk: about 15 GB for the 7B proposer, 6 GB for the 3B verifier, 30 GB for the wheels.

## Install

```bash
cd inference
bash setup_env.sh          # creates a Python 3.12 env "prove" (uv if present, else conda) and runs install.sh
# or, into an environment you already have:
bash install.sh            # torch 2.9.1 cu129 + pip install -e ./sglang + import check
```

`install.sh` ends with an import check that must print
`OK: ProVe, ProVeLossless and MarsThreshold are registered`. Exact versions are listed in
`requirements-lock.txt`; the fork itself is documented in `sglang/PROVE_CHANGES.md`.

## Serve ProVe-lossless with the public 7B proposer

```bash
hf download Xalphinions/MARS-Qwen2.5-7B-blk16 --local-dir models/MARS-Qwen2.5-7B-blk16
bash serve.sh --proposer models/MARS-Qwen2.5-7B-blk16 --port 30000
```

`serve.sh` does three things and prints `READY http://127.0.0.1:30000` when the server
answers its first request:

1. **Prepares the checkpoint.** MARS checkpoints carry `model_type: a2d-qwen2`; the serving
   stack needs `qwen2`. `tools/prepare_checkpoint.py` writes `models/MARS-Qwen2.5-7B-blk16-qwen2`
   with a rewritten `config.json` and links (never copies) every other file. You can run it
   yourself: `python tools/prepare_checkpoint.py <ckpt> [<out>] [--copy]`.
2. **Writes the algorithm yaml** (from `configs/`, see the table below) into a temp dir.
3. **Launches** `python -m sglang.launch_server` with the paper's flags and waits for it.
   The log goes to `./serve_<port>.log`; the server keeps running after the script returns
   (`kill <pid>` to stop it, the pid is printed).

The server speaks the normal SGLang HTTP API:

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "default", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "A train travels 60 km in 45 minutes. What is its speed in km/h?"}]}'
```

## Serve ProVe with a 3B verifier

```bash
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
bash serve.sh --proposer models/MARS-Qwen2.5-7B-blk16 \
              --verifier models/Qwen2.5-3B-Instruct --theta 0.05 --port 30001
```

`--theta` picks the gate threshold on p̃·q (paper grid 0.10 / 0.05 / 0.01; lower = faster,
lossier) and the matching `configs/prove_3b_theta<theta>.yaml`. The verifier runs **in the
same process** as the proposer, on the same GPU(s), with its own KV pool carved out of the
memory the proposer leaves free: with a verifier `serve.sh` lowers the proposer's
`--mem-fraction` to 0.60 (TP=1; 0.70 at TP≥2), and the yaml asks for a 2 GB verifier pool
(`inproc_pool_gb`, portable across GPUs). Once the server answers, the script prints
`in-process verifier: OK` with the sizing line; without that line ProVe would be running on a
slow fallback (see Troubleshooting). `--verifier` also accepts a Hugging Face id, which is
downloaded first. With a 7B verifier use

```bash
bash serve.sh --proposer <proposer> --verifier models/Qwen2.5-7B-Instruct \
              --config configs/prove_7b_theta0.05.yaml --port 30001
```

(its weights need about 16 GB, so one 80 GB GPU with the 7B proposer, or two with the 32B).
The paper's own runs pinned sglang's fraction instead of using an absolute pool:
`--tp 2 --mem-fraction 0.70 --verifier-mem-fraction 0.18` for the 3B and
`--mem-fraction 0.55 --verifier-mem-fraction 0.32` for the 7B verifier (32B proposer, 80 GB
GPUs); `configs/README.md` explains the two keys. The verifier must share the proposer's
tokenizer; the Qwen2.5 Instruct models do.

## Benchmark AR vs ProVe

`bench/quick_bench.py` sends GSM8K questions one at a time (greedy, chat template applied)
and reports completion tokens per second. Start the servers with `MARSV_LOG=1` so the
algorithm logs its accepted-tokens-per-cycle statistics, either on two GPUs at once or one
after the other on the same GPU:

```bash
# AR baseline, same weights, GPU 0
CUDA_VISIBLE_DEVICES=0 MARSV_LOG=1 bash serve.sh --proposer models/MARS-Qwen2.5-7B-blk16 --algorithm ar --port 30000
python bench/quick_bench.py --port 30000 --tokenizer models/MARS-Qwen2.5-7B-blk16 --n 50 --save ar.jsonl

# ProVe-lossless, GPU 1 (or stop the first server and reuse GPU 0)
CUDA_VISIBLE_DEVICES=1 MARSV_LOG=1 bash serve.sh --proposer models/MARS-Qwen2.5-7B-blk16 --port 30001 --log serve_30001.log
python bench/quick_bench.py --port 30001 --tokenizer models/MARS-Qwen2.5-7B-blk16 --n 50 \
       --save lossless.jsonl --compare ar.jsonl --server-log serve_30001.log
```

The second run prints the throughput, the fraction of outputs identical to the AR run
(`--compare`; expect 100 % up to bf16 ties) and, from the log, the accepted tokens per cycle
(`--server-log`). `quick_bench.py` generates up to 1024 new tokens per prompt (the paper's
cap; accepted tokens per cycle and tok/s both change with it). With the 7B proposer on GSM8K,
this repository's stack measured 433 vs 234 tok/s (1.85×) at 6.13 accepted tokens per cycle
for ProVe-lossless, accuracy 75.1 vs 74.4 (`eval/run_eval.sh --config ar` / `lossless` on
`gsm8k_cot`: greedy, one request at a time, 1024 new tokens). That 7B experiment is **not
reported in the paper**, whose tables use the 32B proposer. `--prompts-file my.jsonl` (one
`{"prompt": ...}` per line) benchmarks your own prompts; `--no-chat-template` sends them raw.

## Configs

| file | `--dllm-algorithm` | needs `--verifier` | notes |
|---|---|---|---|
| `configs/prove_lossless.yaml` | `ProVeLossless` | no | exact greedy output; `serve.sh` default |
| `configs/prove_3b_theta0.10.yaml` | `ProVe` | yes (3B) | strictest lossy setting |
| `configs/prove_3b_theta0.05.yaml` | `ProVe` | yes (3B) | `serve.sh --verifier` default |
| `configs/prove_3b_theta0.01.yaml` | `ProVe` | yes (3B) | fastest, largest quality drop |
| `configs/prove_7b_theta0.05.yaml` | `ProVe` | yes (7B) | the verifier's weights need ~16 GB of free memory |
| `configs/mars_tau0.95.yaml` | `MarsThreshold` | no | MARS self-gate baseline (`--algorithm mars-tau`) |

Every key is explained in [configs/README.md](configs/README.md). `serve.sh --dry-run`
prints the yaml and the launch command without starting anything.

### Launching sglang yourself

```bash
python tools/prepare_checkpoint.py models/MARS-Qwen2.5-7B-blk16     # -> models/MARS-Qwen2.5-7B-blk16-qwen2
sed 's|^verifier_path:.*|verifier_path: "models/Qwen2.5-3B-Instruct"|' configs/prove_3b_theta0.05.yaml > /tmp/prove.yaml
python -m sglang.launch_server --model-path models/MARS-Qwen2.5-7B-blk16-qwen2 \
  --tp-size 1 --port 30001 --host 127.0.0.1 --mem-fraction-static 0.60 --max-running-requests 16 \
  --attention-backend flashinfer --dtype bfloat16 --trust-remote-code \
  --dllm-algorithm ProVe --dllm-algorithm-config /tmp/prove.yaml
```

Use `--dllm-algorithm ProVeLossless --dllm-algorithm-config configs/prove_lossless.yaml` for
the lossless configuration (then `--mem-fraction-static 0.75`), and no `--dllm-*` flags for
the AR baseline. A yaml that still contains `__VERIFIER__` makes the server refuse to start
with a message naming the key. Do **not** pass `--context-length` with a ProVe config (see
troubleshooting). The 32B proposer of the paper was served with `--tp-size 2` and otherwise
the same flags; wait for the server on `/health` (the older `/get_model_info` endpoint logs a
deprecation warning on every call).

## Diagnostics (environment variables of the server)

| variable | effect |
|---|---|
| `MARSV_LOG=1` | at the end of every request, log cumulative statistics: `[MarsExactStream] cum stats: fwd=… tok=… tok/cycle=…` or `[MarsVerifierStream] cum stats: … v_accept=… v_reject=… audit_false_accept=a/b`. `quick_bench.py --server-log` reads them. |
| `MARSV_CYCLE_TIMELINE=1` | per-cycle wall-clock breakdown (`[marsX timeline]` / `[marsvS timeline]`); syncs the GPU, so it slows decoding — attribution only |
| `MARSV_CYCLE_LOG=1` (`MARSV_CYCLE_LOG_MAX`, default 80) | log the first N ProVe cycles in detail |
| `MARSV_PASS_LOG=1` | ProVe-lossless: GPU time of the propose and verify forwards |
| `MARSV_INPROC_POOL_GB=<gb>` | absolute KV-pool size for the in-process verifier (overrides `inproc_pool_gb` and a pinned `inproc_mem_fraction`) |
| `MARSV_INPROC_FALLBACK=1` | if the in-process verifier cannot be built, continue on the slow transformers verifier instead of stopping the server (same as `inproc_fallback: true`) |
| `TAU_CYCLE_LOG=1` | per-cycle log of the MARS self-gate baseline |
| `DLLM_CUSTOM_MASK=1` | explicit attention masks for window forwards; not needed by the shipped configurations |
| `IDLM_TIMING`, `IDLM_STEP_DEBUG`, `IDLM_ACCEPT_LOG`, `DLLM_GRAPH_DEBUG` | I-DLM / graph-replay diagnostics |

## Troubleshooting

* **`RuntimeError: Unknown diffusion LLM: A2DQwen2LMHeadModel`** (or a config-loading error
  mentioning `a2d-qwen2`): the proposer's `config.json` was not rewritten. Run
  `python tools/prepare_checkpoint.py <ckpt>` and serve the `…-qwen2` directory; `serve.sh`
  does this automatically.
* **Server dies with `CUDA error: an illegal memory access` when a ProVe config is used
  together with `--context-length`.** The verifier's context must stay at the model default;
  drop `--context-length` (it is fine for the AR / speculative baselines).
* **Is the verifier really in-process?** With a verifier the server log must contain
  `[InprocVerifier] injected into dllm algorithm`; `serve.sh` checks and prints
  `in-process verifier: OK`. `grep '\[InprocVerifier\]' serve_<port>.log` shows the sizing
  arithmetic (`pool sizing: want … free-for-this-runner … -> mem_fraction_static=…`, or a
  `PINNED` line when `inproc_mem_fraction` is set). A `[InprocVerifier] unavailable` warning
  means ProVe fell back to the slow transformers verifier — far below paper speed; this only
  happens with `inproc_fallback: true`, otherwise the server stops with the reason.
* **Server stops with `[InprocVerifier] could not be built`** (the lines above it say why:
  a pinned `inproc_mem_fraction` whose pool comes out ≤ 0, or `verifier KV pool would be … GB`),
  **or the log warns `POOL TARGET NOT REACHABLE`** (the server runs on, with a smaller pool than
  asked). Either way the proposer left too little memory for the verifier's weights (6.4 GB
  for the 3B, 15.5 GB for the 7B on one GPU) plus its pool. Lower the proposer's share
  (`serve.sh --mem-fraction 0.50`, or less), drop a pinned `inproc_mem_fraction` so
  `inproc_pool_gb` sizes the pool, or use a smaller verifier / larger GPU.
* **ProVe is barely faster than AR, or the log reports `degraded` requests / prompts that do
  not fit.** The verifier's KV pool is too small for your prompts: raise `inproc_pool_gb`
  (`serve.sh --verifier-pool-gb 4`) and, if needed, `inproc_max_cache`, and lower the
  proposer's `--mem-fraction` so that the pool still fits.
* **`ValueError: ProVe / MarsVerifierStream needs a verifier model`**: the yaml has no
  `verifier_path`, or still contains the placeholder `__VERIFIER__`. Pass `--verifier` to
  `serve.sh` (which fills it in), replace it by hand, or use `ProVeLossless`, which needs no
  verifier.
* **Wheels / driver.** Only CUDA 12.x wheels work: `install.sh` pulls torch from the `cu129`
  index. A CUDA 13 torch (the default `pip install torch` on some dates) fails to load on a
  535/12.x driver stack. If `torch.cuda.is_available()` is `False`, reinstall torch from
  `https://download.pytorch.org/whl/cu129`.
* **Wrong tokens or garbage after a few words** with a self-trained proposer: the yaml's
  `block_size` must equal the block size the proposer was trained with, and `mask_id` must be
  the id of `<|mask|>` in its tokenizer (`prepare_checkpoint.py` prints it).
* **`ProVe` is not in the algorithm list** printed by `install.sh`: the registry skips a
  module whose import failed; the warning above the list names the missing package (usually
  a torch / flashinfer version mismatch — reinstall from `requirements-lock.txt`).
* **The first request hangs at TP=2** (rare, seen with a 7B verifier): `serve.sh` treats a
  warm-up request that does not return within 10 minutes as a failed start, stops the server
  and exits 1; run it again.

## Results (single request, greedy, 1024 new tokens)

| proposer | configuration | speed-up vs AR | accuracy change | source |
|---|---|---|---|---|
| 32B (TP=2, 6 benchmarks) | ProVe-lossless | 2.5× on average (4.0× BBH, 1.4× IFEval) | exact | paper |
| 32B | ProVe, 3B verifier, θ = 0.01 | 2.7× | −2.6 points | paper |
| 32B | ProVe, 7B verifier, θ = 0.05 | 2.3× | −0.2 points | paper |
| 7B (public `MARS-Qwen2.5-7B-blk16`), GSM8K | ProVe-lossless | 1.85× (433 vs 234 tok/s), 6.13 accepted tokens per cycle | 75.1 vs 74.4 | measured with this repository; not in the paper |

Full evaluation (lm-evaluation-harness, accuracy and speed from the same generations) is in
`../eval/`; training the proposer is in `../train/`.
