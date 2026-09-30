#!/usr/bin/env python3
# Copyright 2026 The ProVe Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a MARS proposer checkpoint for serving, without copying its weights.

MARS checkpoints (e.g. Xalphinions/MARS-Qwen2.5-7B-blk16) ship a config.json that says
    "architectures": ["A2DQwen2LMHeadModel"], "model_type": "a2d-qwen2"
The weights are ordinary Qwen2 weights (the placeholder token <|mask|> lives inside the
vocab), so the serving stack only needs the config to say
    "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"
This tool writes a new directory holding a rewritten config.json and, for every other
file, a symlink to the original (falling back to a hardlink, and to a copy with --copy).
Running it again on the same output directory is a no-op apart from refreshing config.json.

Usage
  python prepare_checkpoint.py <checkpoint_dir> [<out_dir>] [--copy] [--force]

  <out_dir> defaults to <checkpoint_dir>-qwen2. If the checkpoint is already serving-ready
  and no <out_dir> is given, nothing is written. serve.sh calls this for you.
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter

SERVING_ARCH = "Qwen2ForCausalLM"
SERVING_TYPE = "qwen2"
MASK_TOKEN = "<|mask|>"
DEFAULT_MASK_ID = 151665  # what sglang/srt/dllm/config.py assumes for Qwen2ForCausalLM


def is_serving_ready(cfg):
    return cfg.get("model_type") == SERVING_TYPE and cfg.get("architectures") == [SERVING_ARCH]


def is_mars_config(cfg):
    return "a2d" in str(cfg.get("model_type", "")).lower() or any(
        "A2D" in a for a in cfg.get("architectures", []) or []
    )


def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def link_one(src, dst, copy):
    """Materialise dst from src: symlink -> hardlink -> copy (only with --copy).

    Returns the method used, or "kept" when dst already resolves to src (idempotent).
    """
    if os.path.lexists(dst):
        if same_file(src, dst):
            return "kept"
        if os.path.isfile(src) and os.path.isfile(dst) and not os.path.islink(dst):
            s, d = os.stat(src), os.stat(dst)
            if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime):
                return "kept"  # an earlier --copy
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
        return "symlink"
    except OSError:
        pass
    if os.path.isfile(src):
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    if not copy:
        raise SystemExit(
            f"cannot symlink or hardlink {src} -> {dst} on this filesystem; "
            f"re-run with --copy to copy the files instead"
        )
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copy"


def find_mask_id(ckpt):
    """Return the id of <|mask|> from added_tokens.json or tokenizer.json, else None."""
    p = os.path.join(ckpt, "added_tokens.json")
    if os.path.isfile(p):
        try:
            d = json.load(open(p))
            if MASK_TOKEN in d:
                return int(d[MASK_TOKEN])
        except (OSError, ValueError):
            pass
    p = os.path.join(ckpt, "tokenizer.json")
    if os.path.isfile(p):
        try:
            for t in json.load(open(p)).get("added_tokens", []):
                if t.get("content") == MASK_TOKEN:
                    return int(t["id"])
        except (OSError, ValueError):
            pass
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="MARS checkpoint directory (a2d config)")
    ap.add_argument("out", nargs="?", help="output directory (default: <checkpoint>-qwen2)")
    ap.add_argument("--copy", action="store_true",
                    help="copy files when neither symlinks nor hardlinks are possible")
    ap.add_argument("--force", action="store_true",
                    help="write the output even if the checkpoint is already serving-ready")
    args = ap.parse_args()

    src = os.path.abspath(args.checkpoint.rstrip("/"))
    cfg_path = os.path.join(src, "config.json")
    if not os.path.isfile(cfg_path):
        sys.exit(f"{cfg_path}: not found (is {src} a HuggingFace checkpoint directory?)")
    cfg = json.load(open(cfg_path))
    out = os.path.abspath(args.out.rstrip("/")) if args.out else src + "-qwen2"

    if is_serving_ready(cfg) and not args.force and not args.out:
        print(f"{src}: config.json already says {SERVING_ARCH} / {SERVING_TYPE}; "
              f"serve it directly (nothing written).")
        return 0
    if out == src:
        sys.exit("output directory must differ from the checkpoint directory")
    if not is_mars_config(cfg) and not is_serving_ready(cfg):
        print(f"warning: {src} does not look like a MARS (a2d) checkpoint "
              f"(model_type={cfg.get('model_type')!r}); rewriting the config anyway.",
              file=sys.stderr)

    os.makedirs(out, exist_ok=True)
    new_cfg = dict(cfg)
    new_cfg["architectures"] = [SERVING_ARCH]
    new_cfg["model_type"] = SERVING_TYPE
    # Remote-code pointers of the training-time class (auto_map -> configuration_a2d.py /
    # modeling_a2d.py) must not survive: with --trust-remote-code transformers would look
    # for those files in the served directory instead of using the built-in Qwen2 classes.
    # Standard keys such as tokenizer_class are kept.
    dropped = [k for k, v in new_cfg.items()
               if k == "auto_map"
               or (k != "architectures" and isinstance(v, str) and "a2d" in v.lower())]
    for k in dropped:
        new_cfg.pop(k)
    if dropped:
        print(f"note: dropped training-time (remote-code) config keys {dropped}", file=sys.stderr)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2)
        f.write("\n")

    counts = Counter()
    for name in sorted(os.listdir(src)):
        if name == "config.json":
            continue
        counts[link_one(os.path.join(src, name), os.path.join(out, name), args.copy)] += 1

    mask_id = find_mask_id(src)
    if mask_id is None:
        print(f"warning: no {MASK_TOKEN} token found in the tokenizer files; a ProVe proposer "
              f"must have one (is this a MARS checkpoint?)", file=sys.stderr)
    elif mask_id != DEFAULT_MASK_ID:
        print(f"note: {MASK_TOKEN} has id {mask_id}, not the default {DEFAULT_MASK_ID}; "
              f"add `mask_id: {mask_id}` to the algorithm yaml.", file=sys.stderr)

    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"{out}: config.json rewritten ({SERVING_ARCH} / {SERVING_TYPE}); other files: {summary}")
    if mask_id is not None:
        print(f"{MASK_TOKEN} id = {mask_id}")
    print(f"serve with: --model-path {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
