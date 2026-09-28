#!/usr/bin/env bash
# setup_env.sh — create a fresh Python 3.12 environment for ProVe inference and install into it.
# Uses uv when available (fast), otherwise conda; then runs install.sh.
# Usage:  bash setup_env.sh [--uv | --conda] [--name NAME]      (default NAME: prove)
#   uv:    virtualenv at inference/.venv-<NAME>   ->  source inference/.venv-<NAME>/bin/activate
#   conda: environment <NAME>                     ->  conda activate <NAME>
# Prerequisites: a CUDA 12.x driver stack, ~30 GB of disk for the wheels + one 7B proposer.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME=prove
TOOL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --uv) TOOL=uv; shift;;
    --conda) TOOL=conda; shift;;
    --name) [ $# -ge 2 ] || { echo "--name needs a value" >&2; exit 2; }; NAME="$2"; shift 2;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "setup_env.sh: unknown argument $1" >&2; exit 2;;
  esac
done

if [ -z "$TOOL" ]; then
  if command -v uv >/dev/null 2>&1; then TOOL=uv
  elif command -v conda >/dev/null 2>&1; then TOOL=conda
  else echo "setup_env.sh: neither uv nor conda found; install one (https://docs.astral.sh/uv/) or use install.sh in an existing env" >&2; exit 1
  fi
fi

if [ "$TOOL" = uv ]; then
  VENV="$SCRIPT_DIR/.venv-$NAME"
  if [ ! -d "$VENV" ]; then
    echo "[setup] creating $VENV (Python 3.12) with uv"
    uv venv --python 3.12 --seed "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  PYTHON="$VENV/bin/python" bash "$SCRIPT_DIR/install.sh"
  echo
  echo "Activate with:  source $VENV/bin/activate"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$NAME"; then
    echo "[setup] creating conda env $NAME (Python 3.12)"
    conda create -n "$NAME" python=3.12 -y
  fi
  conda activate "$NAME"
  PYTHON=python bash "$SCRIPT_DIR/install.sh"
  echo
  echo "Activate with:  conda activate $NAME"
fi
