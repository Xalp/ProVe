#!/usr/bin/env bash
# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
# install.sh — install the ProVe inference stack into the CURRENT Python environment.
#   1. torch 2.9.1 (+ torchvision, torchaudio) from the CUDA 12.9 wheel index
#   2. the bundled sglang fork (editable) with its pinned dependencies
#   3. an import check: ProVe / ProVeLossless / MarsThreshold must be registered
# Usage:  bash install.sh [--no-torch]        Env: PYTHON=<interpreter> (default: python)
# Needs a CUDA 12.x driver stack (cu12x wheels only; CUDA 13 wheels do not work on a 535 driver)
# and the CUDA 12.x toolkit with nvcc (CUDA_HOME): flashinfer / sgl JIT kernels compile at first use.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
export PYTHONNOUSERSITE=1   # never let packages in ~/.local shadow the environment

INSTALL_TORCH=1
for a in "$@"; do
  case "$a" in
    --no-torch) INSTALL_TORCH=0;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "install.sh: unknown argument $a" >&2; exit 2;;
  esac
done

"$PY" -c 'import sys; assert sys.version_info[:2] >= (3, 10), sys.version' \
  || { echo "install.sh: Python >= 3.10 required (3.12 recommended); got $("$PY" --version 2>&1)" >&2; exit 1; }
echo "[install] using $("$PY" -c 'import sys; print(sys.executable)')"

"$PY" -m pip install --upgrade pip
if [ "$INSTALL_TORCH" = 1 ]; then
  echo "[install] torch 2.9.1 (cu129)"
  "$PY" -m pip install "torch==2.9.1" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
fi

echo "[install] sglang fork (editable) + pinned dependencies"
"$PY" -m pip install -e "$SCRIPT_DIR/sglang"
"$PY" -m pip install "pyyaml>=6" "datasets>=2.0" requests

NVCC="${CUDA_HOME:+$CUDA_HOME/bin/nvcc}"
if [ -n "$NVCC" ] && [ -x "$NVCC" ]; then
  echo "[install] nvcc: $NVCC ($("$NVCC" --version | grep -o 'release [0-9.]*'))"
elif command -v nvcc >/dev/null 2>&1; then
  echo "[install] nvcc: $(command -v nvcc) ($(nvcc --version | grep -o 'release [0-9.]*')); set CUDA_HOME to its toolkit dir"
else
  echo "[install] WARNING: no nvcc found (CUDA_HOME=${CUDA_HOME:-unset}). flashinfer and the fork's JIT kernels compile" >&2
  echo "[install]          at first use and need the CUDA 12.x toolkit: install it and export CUDA_HOME=/usr/local/cuda" >&2
  echo "[install]          (untested alternative: pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129)" >&2
fi

echo "[install] import check"
"$PY" - <<'PYEOF'
import torch
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  cuda available: {torch.cuda.is_available()}")
import sglang
from sglang.srt.dllm.algorithm import algo_name_to_cls
names = sorted(algo_name_to_cls)
print("dllm algorithms:", names)
missing = [n for n in ("ProVe", "ProVeLossless", "MarsThreshold") if n not in names]
if missing:
    raise SystemExit(f"ProVe algorithms failed to register: {missing} (see the import warnings above)")
import sglang.srt.dllm.algorithm.prove  # noqa: F401
print("OK: ProVe, ProVeLossless and MarsThreshold are registered")
PYEOF

echo
echo "Done. Next: bash serve.sh --proposer <MARS checkpoint dir>   (see README.md)"
