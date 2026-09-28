# Evaluation

Accuracy **and** decode speed of one decoding configuration on the six benchmarks of the paper,
from the same generations, with three commands. This is the exact protocol behind Table 1.

## Protocol

| Benchmark | lm-eval task(s) | shots | n | metric key in `results*.json` |
|---|---|---|---|---|
| HumanEval | `humaneval_instruct` | 0 | 164 | `pass@1,create_test` |
| GSM8K | `gsm8k_cot` | 0 | 1,319 | `exact_match,flexible-extract` |
| GPQA | `gpqa_main_cot_zeroshot` | 0 | 448 | `exact_match,flexible-extract` |
| BBH | `bbh_cot_fewshot_<subtask>` (27 units) | 3 | 6,511 | `exact_match,get-answer` |
| MMLU-Pro | `mmlu_pro_<category>` (14 units) | 0 | 12,032 | `exact_match,custom-extract` |
| IFEval | `ifeval` | 0 | 541 | `prompt_level_strict_acc,none` |

* Full test splits; the 45 units are listed in `tasks.txt` (BBH per subtask, MMLU-Pro per
  category; `aggregate.py` combines them weighted by sample count).
* Harness: the [lm-evaluation-harness fork pinned by MARS](https://github.com/ZHZisZZ/lm-evaluation-harness),
  with `--apply_chat_template --log_samples --batch_size 1`, the scoring the MARS paper used.
* Generation: chat template, **greedy** (temperature 0), **max_new_tokens 1024 for every task**,
  **one request in flight** (the harness runs with batch size 1 and the backend posts one HTTP
  request at a time). The speed numbers are therefore single-request throughput, the setting
  of the paper; the server is the same one `inference/serve.sh` starts.
* Accuracy and speed come from the **same** generations: the backend `lmeval_sgl.py`
  (`--model sgl_timed`) sends each prompt to the running sglang server, returns the text to
  the harness for scoring, and records the completion-token count and latency of every
  request in `speed_stats.json`.

## Install

The harness goes into the inference environment or into a second one:

```bash
pip install "lm-eval[ifeval,math] @ git+https://github.com/ZHZisZZ/lm-evaluation-harness.git"
```

The backend additionally needs `requests` and `transformers` (for the tokenizer / chat
template), both already present in the inference environment. If you use a second environment,
point `EVAL_PYTHON` at its interpreter; the server is started by `inference/serve.sh` with the
inference environment's interpreter (`PYTHON`, default `python`).
Benchmark data is downloaded from the Hugging Face Hub on first use. HumanEval executes the
generated code: `run_eval.sh` sets `HF_ALLOW_CODE_EVAL=1` and passes
`--confirm_run_unsafe_code` for that unit.

## Files

| File | Role |
|---|---|
| `run_eval.sh` | one configuration x all units: starts the server via `inference/serve.sh`, waits for it, loops over `tasks.txt`, stops the server |
| `lmeval_sgl.py` | the `sgl_timed` harness backend (timed HTTP calls to `/generate`) |
| `aggregate.py` | per-benchmark accuracy and pooled tok/s tables, speedup vs a baseline |
| `tasks.txt` | the 45 `task:fewshot` units |

## Reproduce one row of Table 1

Serve and evaluate the AR baseline (1.00x), then the ProVe configuration, then aggregate:

```bash
# 1. plain greedy AR decoding of the proposer -- the baseline every speedup is relative to
bash eval/run_eval.sh --config ar --proposer models/prove-32b --tp 2 --out eval_out

# 2a. ProVe-lossless (the proposer verifies its own proposed block; output == greedy AR)
bash eval/run_eval.sh --config lossless --proposer models/prove-32b --tp 2 --out eval_out

# 2b. ProVe with a small stock verifier: 3B at theta 0.10 / 0.05 / 0.01, or 7B at 0.05
bash eval/run_eval.sh --config prove --proposer models/prove-32b --tp 2 --out eval_out \
     --verifier models/Qwen2.5-3B-Instruct --theta 0.05

# 3. tables
python eval/aggregate.py --root eval_out --baseline ar
```

`--config prove` writes to `eval_out/prove_<verifier>_theta<T>/` (override with `--name`). The
paper's rows: `lossless`; `prove` with `Qwen2.5-3B-Instruct` at theta 0.10, 0.05, 0.01; `prove`
with `Qwen2.5-7B-Instruct` at theta 0.05. By default `serve.sh` gives the verifier an absolute
2 GB KV pool (`inproc_pool_gb`) with the proposer at `--mem-fraction 0.70` (TP=2); the paper's
campaign pinned sglang's fraction instead, which you reproduce with
`--serve-args "--mem-fraction 0.70 --verifier-mem-fraction 0.18"` for the 3B verifier and
`--serve-args "--mem-fraction 0.55 --verifier-mem-fraction 0.32"` for the 7B (see
`inference/configs/README.md`). Check `server.log` for `[InprocVerifier] injected into dllm
algorithm` — `run_eval.sh` fails the start if it is missing. `run_eval.sh` maps
`--config` onto `inference/serve.sh --algorithm ar|lossless|prove` and, for `ar`, adds the
`--context-length 8192` the paper's AR server used (`AR_CONTEXT_LENGTH=` disables it; the ProVe
configurations keep the model's default context, as in the paper). `--serve-args` passes further
`serve.sh` options, `--sglang-args` further `sglang.launch_server` arguments.
Each configuration takes several hours on two GPUs: the 32B AR baseline answers 21,015
requests (3.2 M generated tokens) at ~70 tok/s, about 13 h; ProVe configurations finish in
roughly a third of that.
`--limit 20` runs the first 20 examples of every unit for a smoke test (no `.done` marker is
written and `aggregate.py` marks such units with `*`). A finished unit (`.done`) is skipped when
the command is re-run, so an interrupted configuration resumes where it stopped.

Output layout, one directory per unit:

```
eval_out/<name>/<task>_<fewshot>shot/
    <model_name_sanitized>/results_<timestamp>.json    lm-eval scores (the keys above)
    <model_name_sanitized>/samples_<task>_<timestamp>.jsonl
    speed_stats.json                                  tokens and seconds per request (below)
    lmeval.log                                        harness output
    .done                                             written when the unit finished completely
eval_out/<name>/server.log                            sglang server log (accepted tokens per cycle when MARSV_LOG=1)
eval_out/<name>/serve_stdout.log                      serve.sh output: the algorithm yaml and the launch command
```

`aggregate.py` prints two markdown tables: accuracy per benchmark (BBH and MMLU-Pro weighted by
n, plus the unweighted six-benchmark average the paper reports as Avg) and pooled tok/s per
benchmark with the speedup over `--baseline` (default `ar`); `--detail` adds the per-unit rows,
`--md` / `--json` write them to files.

## BBH stop sequences for stock models

lm-eval's `bbh_cot_fewshot_*` tasks stop a generation at `</s>`, `Q` and `\n\n`. A **stock**
(not fine-tuned) `Qwen2.5-32B-Instruct` writes a blank line before "So the answer is", so its
BBH answers are cut off on 11 of the 27 subtasks and its BBH score reads 40.8 instead of 62.5.
Models fine-tuned on Dolci (the proposers and their AR-SFT checkpoints) never emit the blank
line and are unaffected; the paper's BBH numbers use the harness defaults. To score a stock
model on BBH, pass `--bbh-until-eos`, which adds `--gen_kwargs until=</s>` to the BBH units so
that only the end-of-sequence token stops the generation (`max_new_tokens` still bounds it).
Use the same setting for every model you compare.

## How speed is measured

For every request the backend records

* `sec`: wall-clock time around the HTTP `POST /generate` (tokenized prompt in, finished
  completion out), so prefill, every decode cycle and the HTTP round trip are all included;
* `tokens`: `meta_info.completion_tokens` as reported by the server;
* `prompt_tokens` and `finish` (`stop`, or `length` when the 1024-token cap was hit).

Per unit, `speed_stats.json` holds `n`, `total_tokens`, `total_sec`,
**`tok_per_s_decode = total_tokens / total_sec`** (pooled over requests, not the mean of
per-request rates), `mean_tokens`, `capped` (requests that hit `max_new_tokens`) and the
per-request `records`. Per benchmark, `aggregate.py` pools the same way over all of its units,
and the speedup is the ratio of pooled tok/s between two configurations on the same benchmark;
the Avg speedup is the unweighted mean over the six benchmarks. Because exactly one request is
in flight, tok/s is the inverse of the per-token latency the user of a single stream sees.

Accepted tokens per cycle are not exposed over HTTP; with `MARSV_LOG=1` (the default in
`run_eval.sh`) the server logs cumulative statistics per finished request, and `run_eval.sh`
prints the last line at the end (see also `inference/bench/quick_bench.py --server-log`).
