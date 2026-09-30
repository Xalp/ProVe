#!/usr/bin/env bash
# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
# ======================================================================================
# run_eval.sh -- accuracy AND speed of one decoding configuration on the paper's benchmarks.
#
# Starts the server through inference/serve.sh, waits until it answers, runs every unit of
# --tasks with eval/lmeval_sgl.py (lm-evaluation-harness + the sgl_timed backend: greedy,
# one request in flight, chat template, max_new_tokens 1024), writes
#   <out>/<name>/<task>_<fewshot>shot/{<model>/results*.json, <model>/samples*.jsonl,
#                                      speed_stats.json, lmeval.log, .done}
# and stops the server at the end (also on Ctrl-C or error). Finished units are skipped on rerun.
#
# Usage
#   bash eval/run_eval.sh --config ar       --proposer /path/to/proposer
#   bash eval/run_eval.sh --config lossless --proposer /path/to/proposer
#   bash eval/run_eval.sh --config prove    --proposer /path/to/proposer \
#                         --verifier /path/to/Qwen2.5-3B-Instruct --theta 0.05
#   python eval/aggregate.py --root eval_out --baseline ar
#
# Options (defaults in brackets)
#   --config ar|lossless|prove   ar = greedy AR decoding of the proposer (the 1.00x baseline)
#                                lossless = ProVe-lossless: the proposer verifies its own block
#                                prove = ProVe with a small stock verifier (--verifier, --theta)
#   --proposer PATH      proposer checkpoint; also the harness's tokenizer            [required]
#   --verifier PATH      verifier checkpoint (prove only)   --theta T   gate threshold [0.05]
#   --tp N               tensor-parallel size [2]           --port P    server port [30000]
#   --out ROOT           output root [eval_out]
#   --name N             result directory under ROOT [= --config; prove: prove_<verifier>_theta<T>]
#   --tasks FILE         unit list, one "task:fewshot" per line [eval/tasks.txt]
#   --limit N            first N examples of every unit (smoke tests; no .done marker is written)
#   --max-new-tokens N   generation cap for every task [1024]
#   --bbh-until-eos      stop BBH generations at EOS only (stock, non-fine-tuned models; see README)
#   --attach             do not start or stop a server: use the one already listening on --port
#   --serve-args "..."   extra options for inference/serve.sh, e.g. "--mem-fraction 0.55 --verifier-mem-fraction 0.32"
#                        (required with a 7B verifier), "--config FILE", "--prepared DIR"
#   --sglang-args "..."  extra arguments for sglang.launch_server itself (passed after serve.sh's "--")
#   --ready-timeout S    seconds to wait for the server to answer [1800]
# Environment
#   EVAL_PYTHON        python with lm-eval installed [python]
#   PYTHON             python of the inference environment, used by serve.sh for the server [python]
#   MARSV_LOG          passed to the server; 1 = accepted-tokens statistics per request in its log [1]
#   AR_CONTEXT_LENGTH  --context-length given to the AR baseline server, as in the paper [8192]; empty = model default
# ======================================================================================
set -uo pipefail
usage() { sed -n '3,/^# =====/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=""; PROPOSER=""; VERIFIER=""; THETA=0.05; TP=2; PORT=30000; OUT=eval_out; NAME=""
TASKS=$ROOT/eval/tasks.txt; LIMIT=""; MAXNEW=1024; BBH_EOS=0; ATTACH=0; SERVE_ARGS=""; SGLANG_ARGS=""; READY_TIMEOUT=1800
AR_CONTEXT_LENGTH=${AR_CONTEXT_LENGTH-8192}
EVAL_PYTHON=${EVAL_PYTHON:-python}
export MARSV_LOG=${MARSV_LOG:-1}
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2;;
    --proposer) PROPOSER=$2; shift 2;;
    --verifier) VERIFIER=$2; shift 2;;
    --theta) THETA=$2; shift 2;;
    --tp) TP=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --name) NAME=$2; shift 2;;
    --tasks) TASKS=$2; shift 2;;
    --limit) LIMIT=$2; shift 2;;
    --max-new-tokens) MAXNEW=$2; shift 2;;
    --bbh-until-eos) BBH_EOS=1; shift;;
    --attach) ATTACH=1; shift;;
    --serve-args) SERVE_ARGS=$2; shift 2;;
    --sglang-args) SGLANG_ARGS=$2; shift 2;;
    --ready-timeout) READY_TIMEOUT=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
fatal() { echo "[eval] FATAL: $1" >&2; exit "${2:-1}"; }
case "$CONFIG" in ar|lossless|prove) ;; *) fatal "--config must be ar, lossless or prove (got '$CONFIG')" 2;; esac
[ -n "$PROPOSER" ] || fatal "--proposer is required" 2
[ -d "$PROPOSER" ] && PROPOSER=$(cd "$PROPOSER" && pwd)
if [ "$CONFIG" = prove ]; then
  [ -n "$VERIFIER" ] || fatal "--config prove needs --verifier" 2
  [ -d "$VERIFIER" ] && VERIFIER=$(cd "$VERIFIER" && pwd)
fi
[ -f "$TASKS" ] || fatal "task list $TASKS not found" 2
[ -n "$NAME" ] || { NAME=$CONFIG; [ "$CONFIG" = prove ] && NAME="prove_$(basename "$VERIFIER")_theta$THETA"; }
mkdir -p "$OUT/$NAME" || fatal "cannot create $OUT/$NAME"
OUT=$(cd "$OUT" && pwd)
"$EVAL_PYTHON" -c "import lm_eval, requests, transformers" 2>/dev/null \
  || fatal "EVAL_PYTHON=$EVAL_PYTHON cannot import lm_eval/requests/transformers; see eval/README.md (Install)" 3
UNITS=$(grep -v '^[[:space:]]*#' "$TASKS" | grep -o '^[A-Za-z0-9_]*:[0-9]*')
[ -n "$UNITS" ] || fatal "no task:fewshot units in $TASKS" 2

# ---- how each --config maps onto inference/serve.sh (bash inference/serve.sh --help) ----
# ar: the same weights served without any --dllm-* flag; the paper also passed --context-length 8192
# to the AR baseline (never to the ProVe configurations, whose verifier keeps the model default).
serve_flags() {
  case "$CONFIG" in
    ar)       echo "--algorithm ar" ;;
    lossless) echo "--algorithm lossless" ;;
    prove)    echo "--algorithm prove --verifier $VERIFIER --theta $THETA" ;;
  esac
}
if [ "$CONFIG" = ar ] && [ -n "$AR_CONTEXT_LENGTH" ] && [[ " $SGLANG_ARGS " != *" --context-length "* ]]; then
  SGLANG_ARGS="--context-length $AR_CONTEXT_LENGTH $SGLANG_ARGS"
fi

port_open() { curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
SERVER_LOG=$OUT/$NAME/server.log
SPID=""
stop_server() {
  trap - EXIT INT TERM
  [ -n "$SPID" ] || return 0
  echo "[eval] stopping the server (process group $SPID) @ $(date)"
  kill -- -"$SPID" 2>/dev/null; kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null   # the server serve.sh detached, if it left the group
  local i
  for i in $(seq 1 30); do port_open || break; sleep 2; done
  if port_open; then kill -9 -- -"$SPID" 2>/dev/null; pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null; fi
  SPID=""
  echo "[eval] server stopped"
}
wait_ready() {
  local t0; t0=$(date +%s)
  while :; do
    if port_open && curl -sf --max-time 600 -X POST "http://127.0.0.1:$PORT/generate" \
         -H 'Content-Type: application/json' \
         -d '{"input_ids":[9707],"sampling_params":{"temperature":0,"max_new_tokens":1}}' >/dev/null 2>&1; then
      return 0
    fi
    if [ -n "$SPID" ] && ! kill -0 "$SPID" 2>/dev/null; then
      # serve.sh has returned: 0 = READY (it waited itself), anything else = the server failed
      wait "$SPID" 2>/dev/null; src=$?
      if [ "$src" -ne 0 ] || ! port_open; then
        echo "[eval] serve.sh exited with status $src; its output:"; tail -20 "$OUT/$NAME/serve_stdout.log"
        echo "[eval] tail of the server log $SERVER_LOG:"; tail -30 "$SERVER_LOG" 2>/dev/null; return 1
      fi
    fi
    [ $(( $(date +%s) - t0 )) -lt "$READY_TIMEOUT" ] \
      || { echo "[eval] server not ready after ${READY_TIMEOUT}s; tail of $SERVER_LOG:"; tail -30 "$SERVER_LOG"; return 1; }
    sleep 5
  done
}

echo "[eval] config=$CONFIG name=$NAME proposer=$PROPOSER verifier=${VERIFIER:-none} theta=$THETA tp=$TP port=$PORT"
echo "[eval] out=$OUT/$NAME tasks=$TASKS max_new_tokens=$MAXNEW limit=${LIMIT:-none} @ $(date)"
if [ "$ATTACH" = 1 ]; then
  port_open || fatal "--attach given but nothing answers on port $PORT" 4
  echo "[eval] using the server already listening on port $PORT"
else
  [ -f "$ROOT/inference/serve.sh" ] || fatal "$ROOT/inference/serve.sh not found" 4
  port_open && fatal "port $PORT is already in use; pick another --port or use --attach" 4
  SERVE_CMD=(bash "$ROOT/inference/serve.sh" --proposer "$PROPOSER" --tp "$TP" --port "$PORT" --log "$SERVER_LOG"
             --timeout "$READY_TIMEOUT" $(serve_flags) $SERVE_ARGS)
  [ -n "$SGLANG_ARGS" ] && SERVE_CMD+=(-- $SGLANG_ARGS)
  echo "[eval] starting the server: ${SERVE_CMD[*]}"
  # a new session: serve.sh and the server it detaches share the process group $SPID, which
  # stop_server kills as a whole; serve.sh itself exits 0 once the server answered a request
  setsid "${SERVE_CMD[@]}" > "$OUT/$NAME/serve_stdout.log" 2>&1 &
  SPID=$!
  trap stop_server EXIT INT TERM
  wait_ready || exit 5
  echo "[eval] server ready @ $(date)"
  if [ "$CONFIG" = prove ] && ! grep -aq '\[InprocVerifier\] injected into dllm algorithm' "$SERVER_LOG"; then
    echo "[eval] the server log has no '[InprocVerifier] injected into dllm algorithm' line: ProVe would run on the slow fallback verifier."
    echo "[eval] grep '\[InprocVerifier\]' $SERVER_LOG for the reason (see inference/configs/README.md); refusing to measure."
    exit 5
  fi
fi

n_ok=0; n_fail=0; n_skip=0
for u in $UNITS; do
  task=${u%%:*}; fs=${u##*:}
  UDIR=$OUT/$NAME/${task}_${fs}shot
  if [ -f "$UDIR/.done" ]; then echo "[eval] $NAME/$task already done, skipping"; n_skip=$((n_skip + 1)); continue; fi
  rm -rf "$UDIR"; mkdir -p "$UDIR"
  EXTRA=""
  [ "$task" = humaneval_instruct ] && EXTRA="--confirm_run_unsafe_code"
  [ "$BBH_EOS" = 1 ] && case "$task" in bbh_*) EXTRA="$EXTRA --gen_kwargs until=</s>";; esac
  [ -n "$LIMIT" ] && EXTRA="$EXTRA --limit $LIMIT"
  echo "[eval] $NAME/$task ($fs-shot) start @ $(date)"
  # shellcheck disable=SC2086
  ( cd "$ROOT/eval" && HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=True TOKENIZERS_PARALLELISM=false \
      PYTHONPATH="$ROOT/eval${PYTHONPATH:+:$PYTHONPATH}" \
      "$EVAL_PYTHON" lmeval_sgl.py --model sgl_timed \
        --model_args "port=$PORT,ckpt=$PROPOSER,max_new_tokens=$MAXNEW,stats=$UDIR/speed_stats.json" \
        --tasks "$task" --num_fewshot "$fs" --apply_chat_template --log_samples --batch_size 1 \
        --output_path "$UDIR" $EXTRA ) > "$UDIR/lmeval.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$(find "$UDIR" -name 'results*.json' | head -1)" ] || [ ! -f "$UDIR/speed_stats.json" ]; then
    echo "[eval] $NAME/$task FAILED (lm-eval rc=$rc); tail of $UDIR/lmeval.log:"; tail -20 "$UDIR/lmeval.log"
    [ -n "$SPID" ] && ! port_open && { echo "[eval] the server is gone; tail of $SERVER_LOG:"; tail -20 "$SERVER_LOG"; exit 6; }
    n_fail=$((n_fail + 1)); continue
  fi
  [ -z "$LIMIT" ] && touch "$UDIR/.done"
  "$EVAL_PYTHON" - "$UDIR" "$NAME/$task" <<'PY'
import glob, json, sys
d, tag = sys.argv[1], sys.argv[2]
r = json.load(open(sorted(glob.glob(d + "/**/results*.json", recursive=True))[-1]))["results"]
for task, m in r.items():
    keep = {k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float)) and "stderr" not in k}
    print(f"[eval] {tag} accuracy: {json.dumps(keep)}")
s = json.load(open(d + "/speed_stats.json"))
print(f"[eval] {tag} speed: {s['tok_per_s_decode']:.1f} tok/s over {s['n']} requests, "
      f"mean {s['mean_tokens']:.0f} tokens, {s['capped']} hit max_new_tokens")
PY
  n_ok=$((n_ok + 1))
done

if [ -n "$SPID" ] && [ -f "$SERVER_LOG" ]; then
  # accepted tokens per cycle are only reported in the server log (MARSV_LOG=1): show the last cumulative line
  grep -a "cum stats" "$SERVER_LOG" | tail -1 | sed 's/^/[eval] server: /'
fi
echo "[eval] $NAME finished: $n_ok units ok, $n_fail failed, $n_skip already done @ $(date)"
echo "[eval] aggregate with:  python $ROOT/eval/aggregate.py --root $OUT --baseline ar"
[ $n_fail -eq 0 ] || exit 7
exit 0
