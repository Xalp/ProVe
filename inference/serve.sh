#!/usr/bin/env bash
# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
# serve.sh — launch an sglang server for a MARS proposer in one of four decoding modes.
#
#   ProVe-lossless (default, no verifier):  bash serve.sh --proposer <dir>
#   ProVe with a small stock verifier:      bash serve.sh --proposer <dir> --verifier <dir> [--theta 0.05]
#   MARS self-gate baseline (tau):          bash serve.sh --proposer <dir> --algorithm mars-tau [--tau 0.95]
#   plain AR baseline (same weights):       bash serve.sh --proposer <dir> --algorithm ar
#
# Options
#   --proposer PATH        MARS proposer checkpoint (required). An a2d config is converted
#                          into <PATH>-qwen2 by tools/prepare_checkpoint.py (weights linked, not copied).
#   --verifier PATH        stock AR verifier: a local directory, or a Hugging Face id such as
#                          Qwen/Qwen2.5-3B-Instruct (downloaded to the HF cache first).
#                          Given -> ProVe; absent -> ProVe-lossless.
#   --theta F              ProVe gate threshold on p~*q (default 0.05; paper grid 0.10 / 0.05 / 0.01).
#                          Picks configs/prove_3b_theta<F>.yaml; any other value uses the 0.05 file with theta replaced.
#   --tau F                MarsThreshold gate (default 0.95), only with --algorithm mars-tau
#   --algorithm NAME       prove | lossless | mars-tau | ar   (overrides the --verifier rule)
#   --config FILE          use this algorithm yaml instead of the one picked from configs/
#                          (verifier_path is still filled in; --theta/--tau then only apply if given)
#   --verifier-pool-gb F   verifier KV pool in GB, absolute (yaml inproc_pool_gb; default 2.0, portable)
#   --verifier-mem-fraction F   pin sglang's raw fraction for the verifier instead (yaml inproc_mem_fraction;
#                          pool = free_after_proposer * F - verifier_weights). Paper, 32B at TP=2 on 80 GB GPUs:
#                          3B verifier 0.18 with --mem-fraction 0.70; 7B verifier 0.32 with --mem-fraction 0.55.
#   --tp N                 tensor-parallel size (default 1; the 32B proposer needs 2)
#   --port N               (default 30000)          --host H   (default 127.0.0.1)
#   --mem-fraction F       --mem-fraction-static for the proposer. Default: 0.75 without a verifier;
#                          with --verifier 0.60 at TP=1 and 0.70 at TP>=2, leaving room for the verifier.
#   --max-running-requests N   (default 16)
#   --prepared PATH        where to write the converted checkpoint (default <proposer>-qwen2)
#   --log FILE             server log (default ./serve_<port>.log)
#   --timeout SEC          how long to wait for the server to come up (default 1800)
#   --no-wait              return right after launching
#   --dry-run              print the yaml and the launch command; launch nothing
#   -- ARGS...             everything after -- is passed to sglang.launch_server verbatim
#
# Environment: PYTHON=<interpreter> (default: python); MARSV_LOG=1 makes the server log
# cumulative accept statistics at the end of every request (bench/quick_bench.py reads them).
# With a verifier the server log must contain "[InprocVerifier] injected into dllm algorithm";
# the script checks for it once the server is up.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
export PYTHONNOUSERSITE=1   # never let packages in ~/.local shadow the environment (install.sh does the same)

PROPOSER=""; VERIFIER=""; THETA=""; TAU=""; ALGO=""; CONFIG=""; VMEM=""; VPOOL=""
TP=1; PORT=30000; HOST=127.0.0.1; MF=""; MAXREQ=16; PREPARED=""; LOG=""
TIMEOUT=1800; WAIT=1; DRY=0
EXTRA=()

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

need_arg() { [ $# -ge 2 ] || { echo "serve.sh: $1 needs a value" >&2; exit 2; }; }

while [ $# -gt 0 ]; do
  case "$1" in
    --proposer)  need_arg "$@"; PROPOSER="$2"; shift 2;;
    --verifier)  need_arg "$@"; VERIFIER="$2"; shift 2;;
    --theta)     need_arg "$@"; THETA="$2"; shift 2;;
    --tau)       need_arg "$@"; TAU="$2"; shift 2;;
    --algorithm) need_arg "$@"; ALGO="$2"; shift 2;;
    --config)    need_arg "$@"; CONFIG="$2"; shift 2;;
    --verifier-mem-fraction) need_arg "$@"; VMEM="$2"; shift 2;;
    --verifier-pool-gb)      need_arg "$@"; VPOOL="$2"; shift 2;;
    --tp)        need_arg "$@"; TP="$2"; shift 2;;
    --port)      need_arg "$@"; PORT="$2"; shift 2;;
    --host)      need_arg "$@"; HOST="$2"; shift 2;;
    --mem-fraction) need_arg "$@"; MF="$2"; shift 2;;
    --max-running-requests) need_arg "$@"; MAXREQ="$2"; shift 2;;
    --prepared)  need_arg "$@"; PREPARED="$2"; shift 2;;
    --log)       need_arg "$@"; LOG="$2"; shift 2;;
    --timeout)   need_arg "$@"; TIMEOUT="$2"; shift 2;;
    --no-wait)   WAIT=0; shift;;
    --dry-run)   DRY=1; shift;;
    -h|--help)   usage; exit 0;;
    --)          shift; EXTRA=("$@"); break;;
    *)           echo "serve.sh: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

[ -n "$PROPOSER" ] || { echo "serve.sh: --proposer is required" >&2; usage >&2; exit 2; }
[ -d "$PROPOSER" ] || { echo "serve.sh: proposer directory not found: $PROPOSER" >&2; exit 2; }
PROPOSER="${PROPOSER%/}"
is_number() { case "$1" in ''|*[!0-9.]*|.|*.*.*) return 1;; *) return 0;; esac; }
for v in THETA:--theta TAU:--tau VMEM:--verifier-mem-fraction VPOOL:--verifier-pool-gb; do
  name="${v%%:*}"; opt="${v#*:}"
  [ -z "${!name}" ] || is_number "${!name}" || { echo "serve.sh: $opt must be a number (got '${!name}')" >&2; exit 2; }
done
case "$TP" in ''|*[!0-9]*) echo "serve.sh: --tp must be an integer (got '$TP')" >&2; exit 2;; esac

# ---- 1. algorithm ----------------------------------------------------------------
if [ -z "$ALGO" ]; then
  if [ -n "$VERIFIER" ]; then ALGO=prove; else ALGO=lossless; fi
fi
case "$ALGO" in
  prove|lossless|mars-tau|ar) ;;
  *) echo "serve.sh: --algorithm must be prove | lossless | mars-tau | ar (got '$ALGO')" >&2; exit 2;;
esac
if [ "$ALGO" = prove ] && [ -z "$VERIFIER" ]; then
  echo "serve.sh: ProVe needs --verifier <stock AR model dir or HF id> (or use --algorithm lossless)" >&2; exit 2
fi
if [ "$ALGO" != prove ] && [ -n "$VERIFIER" ]; then
  echo "[serve] note: --verifier is ignored for --algorithm $ALGO" >&2
fi
if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then
  echo "serve.sh: --config file not found: $CONFIG" >&2; exit 2
fi
# proposer memory share: leave room for an in-process verifier (its weights + KV pool come
# out of what the proposer does not take)
if [ -z "$MF" ]; then
  if [ "$ALGO" = prove ]; then
    if [ "$TP" -le 1 ]; then MF=0.60; else MF=0.70; fi
  else
    MF=0.75
  fi
  MF_NOTE=" (default for $ALGO at tp=$TP)"
else
  MF_NOTE=""
fi

# ---- 2. checkpoint: a2d config -> qwen2 config, weights linked -------------------
model_type_of() {
  "$PY" - "$1" <<'PYEOF'
import json, os, sys
try:
    print(json.load(open(os.path.join(sys.argv[1], "config.json"))).get("model_type", ""))
except Exception:
    print("")
PYEOF
}
mt="$(model_type_of "$PROPOSER")"
if [ "$mt" = "qwen2" ]; then
  SERVE_PATH="$PROPOSER"
else
  SERVE_PATH="${PREPARED:-${PROPOSER}-qwen2}"
  if [ "$DRY" = 1 ]; then
    echo "[serve] would run: $PY $SCRIPT_DIR/tools/prepare_checkpoint.py $PROPOSER $SERVE_PATH"
  else
    "$PY" "$SCRIPT_DIR/tools/prepare_checkpoint.py" "$PROPOSER" "$SERVE_PATH"
  fi
fi

# ---- 3. verifier: must be a local directory (the verifier's size is read from its files) --
if [ "$ALGO" = prove ] && [ ! -d "$VERIFIER" ]; then
  if [ "$DRY" = 1 ]; then
    echo "[serve] note: --verifier '$VERIFIER' is not a local directory; a real run downloads it as a Hugging Face id (huggingface_hub.snapshot_download)" >&2
  else
    echo "[serve] --verifier '$VERIFIER' is not a local directory; downloading it as a Hugging Face id" >&2
    VERIFIER="$("$PY" - "$VERIFIER" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PYEOF
)" || { echo "serve.sh: could not download the verifier; pass a local directory (hf download <id> --local-dir <dir>)" >&2; exit 2; }
    echo "[serve] verifier: $VERIFIER" >&2
  fi
fi

# ---- 4. algorithm yaml -----------------------------------------------------------
TMP="$(mktemp -d "${TMPDIR:-/tmp}/prove_serve.XXXXXX")"
YAML="$TMP/algorithm.yaml"
DLLM_ARGS=()
set_key() {  # $1 = yaml, $2 = key, $3 = value: replace the key's line or append it
  local esc; esc="$(printf '%s' "$3" | sed 's/[&|\\]/\\&/g')"
  if grep -q "^$2:" "$1"; then sed -i "s|^$2:.*|$2: $esc|" "$1"; else printf '%s: %s\n' "$2" "$3" >> "$1"; fi
}
case "$ALGO" in
  prove)
    if [ -n "$CONFIG" ]; then
      src="$CONFIG"
    else
      [ -n "$THETA" ] || THETA=0.05
      tf="$(printf '%.2f' "$THETA")"
      src="$SCRIPT_DIR/configs/prove_3b_theta${tf}.yaml"
      if [ ! -f "$src" ]; then
        echo "[serve] note: no configs/prove_3b_theta${tf}.yaml for --theta $THETA; using the theta 0.05 file with theta: $THETA" >&2
        src="$SCRIPT_DIR/configs/prove_3b_theta0.05.yaml"
      fi
    fi
    cp "$src" "$YAML"
    set_key "$YAML" verifier_path "\"$VERIFIER\""
    if [ -n "$THETA" ]; then set_key "$YAML" theta "$THETA"; fi
    if [ -n "$VPOOL" ]; then set_key "$YAML" inproc_pool_gb "$VPOOL"; fi
    if [ -n "$VMEM" ]; then
      set_key "$YAML" inproc_mem_fraction "$VMEM"
      echo "[serve] note: inproc_mem_fraction $VMEM pins sglang's raw fraction for the verifier (it takes precedence over inproc_pool_gb)" >&2
    fi
    if grep -v '^[[:space:]]*#' "$YAML" | grep -q "__VERIFIER__"; then echo "serve.sh: verifier placeholder not substituted" >&2; exit 1; fi
    DLLM_ARGS=(--dllm-algorithm ProVe --dllm-algorithm-config "$YAML");;
  lossless)
    cp "${CONFIG:-$SCRIPT_DIR/configs/prove_lossless.yaml}" "$YAML"
    DLLM_ARGS=(--dllm-algorithm ProVeLossless --dllm-algorithm-config "$YAML");;
  mars-tau)
    cp "${CONFIG:-$SCRIPT_DIR/configs/mars_tau0.95.yaml}" "$YAML"
    if [ -n "$TAU" ]; then set_key "$YAML" threshold "$TAU"; fi
    DLLM_ARGS=(--dllm-algorithm MarsThreshold --dllm-algorithm-config "$YAML");;
  ar)
    rm -f "$YAML";;
esac

# ---- 5. launch command (the paper's flags; no --context-length for the ProVe configs) --
CMD=("$PY" -m sglang.launch_server
     --model-path "$SERVE_PATH" --tp-size "$TP" --port "$PORT" --host "$HOST"
     --mem-fraction-static "$MF" --max-running-requests "$MAXREQ"
     --attention-backend flashinfer --dtype bfloat16 --trust-remote-code
     ${DLLM_ARGS[@]+"${DLLM_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})

echo "[serve] algorithm: $ALGO   proposer: $SERVE_PATH   tp=$TP port=$PORT mem-fraction=$MF$MF_NOTE"
if [ -f "$YAML" ]; then echo "[serve] algorithm yaml ($YAML, comments stripped):"; grep -v '^[[:space:]]*#' "$YAML" | sed 's/^/    /'; fi
printf '[serve] command:'; printf ' %q' "${CMD[@]}"; printf '\n'
if [ "$DRY" = 1 ]; then rm -rf "$TMP"; exit 0; fi

LOG="${LOG:-$PWD/serve_${PORT}.log}"
PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOG" 2>&1 &
SPID=$!
echo "[serve] pid $SPID, log $LOG"
if [ "$WAIT" = 0 ]; then exit 0; fi

# ---- 6. wait until /health answers and a 1-token request completes -----------------
probe() {  # $1 = url, $2 = timeout seconds
  if command -v curl >/dev/null 2>&1; then
    curl -sf --max-time "$2" "$1" >/dev/null 2>&1
  else
    "$PY" - "$1" "$2" <<'PYEOF'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])).read()
except Exception:
    sys.exit(1)
PYEOF
  fi
}
warmup() {  # one greedy token through /generate
  local body='{"input_ids":[9707],"sampling_params":{"temperature":0,"max_new_tokens":1}}'
  if command -v curl >/dev/null 2>&1; then
    curl -sf --max-time 600 -X POST "http://$HOST:$PORT/generate" \
      -H 'Content-Type: application/json' -d "$body" >/dev/null 2>&1
  else
    "$PY" - "http://$HOST:$PORT/generate" "$body" <<'PYEOF'
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], data=sys.argv[2].encode(),
                             headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=600).read()
except Exception:
    sys.exit(1)
PYEOF
  fi
}
check_verifier() {  # ProVe only: the in-process verifier must have been built
  [ "$ALGO" = prove ] || return 0
  if grep -q '^inproc_verifier: *false' "$YAML" 2>/dev/null; then return 0; fi
  if grep -aq '\[InprocVerifier\] injected into dllm algorithm' "$LOG"; then
    echo "[serve] in-process verifier: OK ($(grep -a 'pool sizing' "$LOG" | tail -1 | sed 's/.*\[InprocVerifier\] //'))"
  else
    echo "[serve] WARNING: the log has no '[InprocVerifier] injected into dllm algorithm' line." >&2
    echo "[serve]          ProVe is running on the slow transformers verifier (far below paper speed)." >&2
    echo "[serve]          grep '\[InprocVerifier\]' $LOG for the reason; see inference/README.md (Troubleshooting)." >&2
  fi
}

start=$(date +%s)
while :; do
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "[serve] server process exited during start-up; last lines of $LOG:" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
  fi
  if probe "http://$HOST:$PORT/health" 10; then
    if warmup; then
      echo "READY  http://$HOST:$PORT  (pid $SPID, log $LOG)"
      check_verifier
      echo "[serve] stop with: kill $SPID"
      exit 0
    fi
    echo "[serve] server answered /health but the warm-up request failed; stopping pid $SPID (see $LOG)" >&2
    kill "$SPID" 2>/dev/null || true
    exit 1
  fi
  if [ $(( $(date +%s) - start )) -ge "$TIMEOUT" ]; then
    echo "[serve] server not ready after ${TIMEOUT}s (pid $SPID still running; kill it before retrying); see $LOG" >&2
    exit 1
  fi
  sleep 3
done
