#!/usr/bin/env python3
"""quick_bench.py — single-stream benchmark against a running ProVe / sglang server.

Sends N prompts ONE AT A TIME (greedy, temperature 0) to the server's /generate endpoint and
reports throughput (completion tokens per second), mean completion tokens and latency.
Optionally it saves every output (--save) so two servers can be compared token for token
(--compare, e.g. AR vs ProVe-lossless), and parses the server log for the algorithm's
accepted-tokens-per-cycle statistics (--server-log; the server must run with MARSV_LOG=1).

Prompts come from the GSM8K test split (`datasets`, downloaded on first use) or from a jsonl
file with a "prompt" field per line. By default the proposer's chat template is applied,
using the tokenizer at --tokenizer (or the server's model path).

Examples
  # AR baseline on :30000, ProVe-lossless on :30001 (both started with serve.sh, MARSV_LOG=1)
  python quick_bench.py --port 30000 --tokenizer models/MARS-Qwen2.5-7B-blk16 --save ar.jsonl
  python quick_bench.py --port 30001 --tokenizer models/MARS-Qwen2.5-7B-blk16 --save lossless.jsonl \
      --compare ar.jsonl --server-log serve_30001.log
"""
import argparse
import json
import re
import sys
import time
from collections import Counter

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("quick_bench.py needs the `requests` package (pip install requests)")

CUM_RE = re.compile(r"\[(?P<cls>[A-Za-z_]+)\] cum stats: (?P<kv>.*)")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z_/]*)=([0-9]+(?:\.[0-9]+)?)")
AUDIT_RE = re.compile(r"audit_false_accept=([0-9]+)/([0-9]+)")
_LOG_PATH = [None]   # set in main() so report_log can re-read the log for verifier warnings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--n", type=int, default=50, help="number of prompts (default 50)")
    p.add_argument("--offset", type=int, default=0, help="skip this many prompts first")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="generation cap per prompt (default 1024, the paper's setting; accepted "
                        "tokens per cycle and tok/s both change with it)")
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k"],
                   help="built-in prompt source (default: openai/gsm8k test split)")
    p.add_argument("--prompts-file",
                   help="jsonl with a 'prompt' (or question/text/input) field per line; overrides --dataset")
    p.add_argument("--tokenizer",
                   help="tokenizer dir or HF id for the chat template (default: the server's model path)")
    p.add_argument("--chat-template", dest="chat_template", action="store_true", default=True,
                   help="wrap each prompt in the tokenizer's chat template (default)")
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false",
                   help="send the raw prompt")
    p.add_argument("--timeout", type=float, default=3600, help="per-request HTTP timeout (s)")
    p.add_argument("--save", help="write one json line per request (prompt, output, tokens, seconds)")
    p.add_argument("--compare",
                   help="results jsonl of another run; report the fraction of identical outputs")
    p.add_argument("--server-log",
                   help="server log to parse for accepted tokens per cycle (server needs MARSV_LOG=1)")
    return p.parse_args()


# ----------------------------------------------------------------------------- prompts
def load_prompts(args):
    if args.prompts_file:
        prompts = []
        with open(args.prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if isinstance(d, str):
                    prompts.append(d)
                    continue
                for k in ("prompt", "question", "text", "input"):
                    if k in d:
                        prompts.append(d[k])
                        break
                else:
                    sys.exit(f"{args.prompts_file}: each line needs a 'prompt' "
                             f"(or question/text/input) field")
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("the built-in gsm8k prompts need `datasets` (pip install datasets), "
                     "or pass --prompts-file")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [r["question"] for r in ds]
    sel = prompts[args.offset:args.offset + args.n]
    if not sel:
        sys.exit("no prompts selected (check --n / --offset)")
    return sel


def server_model_path(args):
    # /model_info is the current endpoint; /get_model_info (older sglang) logs a deprecation
    # warning on every call in this fork, so it is only the fallback
    last = None
    for ep in ("/model_info", "/get_model_info"):
        try:
            r = requests.get(f"http://{args.host}:{args.port}{ep}", timeout=30)
            r.raise_for_status()
            info = r.json()
            return info.get("tokenizer_path") or info.get("model_path")
        except Exception as e:  # noqa: BLE001 - try the next endpoint
            last = e
    raise last


def load_tokenizer(args):
    """Tokenizer for the chat template / token-level compare; None when not needed."""
    path = args.tokenizer
    if not path:
        try:
            path = server_model_path(args)
        except Exception as e:
            if args.chat_template:
                sys.exit(f"could not query /model_info ({e}); pass --tokenizer <dir>")
            return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        if args.chat_template:
            sys.exit(f"could not load a tokenizer from {path!r} ({e}); "
                     f"pass --tokenizer <dir> or --no-chat-template")
        print(f"warning: no tokenizer ({e}); sending raw text, string-level compare only",
              file=sys.stderr)
        return None


# ----------------------------------------------------------------------------- requests
def generate(args, prompt, tok):
    if tok is not None:
        if args.chat_template:
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        ids = tok(text, add_special_tokens=False)["input_ids"]
        payload = {"input_ids": ids}
    else:
        ids = None
        payload = {"text": prompt}
    payload["sampling_params"] = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens}
    t0 = time.perf_counter()
    r = requests.post(f"http://{args.host}:{args.port}/generate", json=payload,
                      timeout=args.timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    d = d[0] if isinstance(d, list) else d
    mi = d.get("meta_info") or {}
    fr = mi.get("finish_reason")
    return {
        "output": d.get("text", ""),
        "tokens": int(mi.get("completion_tokens") or 0),
        "prompt_tokens": int(mi.get("prompt_tokens") or (len(ids) if ids else 0)),
        "sec": dt,
        "finish": fr.get("type", "") if isinstance(fr, dict) else str(fr or ""),
    }


# ----------------------------------------------------------------------------- server log
def parse_cum_stats(path):
    """{algorithm class: {stat: value}} from the LAST '[<Class>] cum stats:' line per class.

    Real formats (MARSV_LOG=1, written at request cleanup):
      [MarsExactStream] cum stats: fwd=N tok=N tok/fwd=X cycles=N tok/cycle=X match=N reject=N ...
      [MarsVerifierStream] cum stats: fwd=N tok=N tok/fwd=X v_accept=N v_reject=N audit_false_accept=A/B (F) v_time=Tms/Ccalls
      [MarsThreshold] cum stats: fwd=N tok=N tok/fwd=X ...
    ProVe / ProVeLossless are subclasses and log under the parent class name.
    """
    out = {}
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = CUM_RE.search(line)
                if not m:
                    continue
                kv = {k: float(v) for k, v in KV_RE.findall(m.group("kv"))}
                a = AUDIT_RE.search(m.group("kv"))
                if a:
                    kv["audit_mismatch"], kv["audit_pairs"] = float(a.group(1)), float(a.group(2))
                out[m.group("cls")] = kv
    except FileNotFoundError:
        return {}
    return out


VERIFIER_LINES = {
    "injected": "[InprocVerifier] injected into dllm algorithm",
    "unavailable": "[InprocVerifier] unavailable",
    "not_reachable": "POOL TARGET NOT REACHABLE",
    "starved": "NUMERIC SELF-TEST FAILED",
}


def check_verifier_log(path):
    """Warn when a ProVe server did not get its in-process verifier (or a starved one)."""
    found = {k: False for k in VERIFIER_LINES}
    pinned = None
    try:
        with open(path, errors="replace") as f:
            for line in f:
                for k, needle in VERIFIER_LINES.items():
                    if needle in line:
                        found[k] = True
                if "PINNED at" in line:
                    pinned = line.strip()
    except FileNotFoundError:
        return
    if found["unavailable"]:
        print("WARNING: the server log says '[InprocVerifier] unavailable': ProVe ran on the slow "
              "transformers verifier, far below paper speed (lower the proposer's --mem-fraction "
              "or drop a pinned inproc_mem_fraction; see inference/configs/README.md)")
    if found["not_reachable"]:
        print("WARNING: the server log says 'POOL TARGET NOT REACHABLE': the verifier's KV pool is "
              "smaller than asked; long prompts degrade to one accepted token per cycle")
    if found["starved"]:
        print("WARNING: the server log reports a failed verifier numeric self-test")
    if pinned and not found["injected"]:
        print(f"WARNING: {pinned}")


def report_log(before, after):
    if not after:
        print("server log: no '[<Algorithm>] cum stats:' lines found — start the server with "
              "MARSV_LOG=1 (an AR server without --dllm-algorithm writes none)")
        return
    if "MarsVerifierStream" in after:
        check_verifier_log(_LOG_PATH[0])
    for cls, kv in after.items():
        b = before.get(cls, {})
        d = {k: kv[k] - b.get(k, 0.0) for k in kv}
        if b and any(v < 0 for v in d.values()):  # server restarted between reads
            d, b = dict(kv), {}
        label = "this run" if b else "cumulative since server start"
        tok, fwd, cyc = d.get("tok", 0.0), d.get("fwd", 0.0), d.get("cycles")
        parts = [f"tokens={tok:.0f}", f"proposer_forwards={fwd:.0f}"]
        if cyc:
            parts.append(f"cycles={cyc:.0f}")
            parts.append(f"accepted tokens/cycle={tok / cyc:.2f}")
        elif fwd:
            parts.append(f"accepted tokens/cycle={tok / fwd:.2f} (one proposer forward per cycle)")
        if fwd:
            parts.append(f"tokens/forward={tok / fwd:.2f}")
        if "v_accept" in d or "v_reject" in d:
            acc, rej = d.get("v_accept", 0.0), d.get("v_reject", 0.0)
            parts.append(f"verifier accepted/rejected positions={acc:.0f}/{rej:.0f}")
        if d.get("audit_pairs"):
            parts.append(f"audit mismatches={d.get('audit_mismatch', 0):.0f}/{d['audit_pairs']:.0f} "
                         f"({d.get('audit_mismatch', 0) / d['audit_pairs']:.2%})")
        print(f"server log [{cls}] ({label}): " + "  ".join(parts))


# ----------------------------------------------------------------------------- compare
def compare(results, other_path, tok):
    other = {}
    with open(other_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                other[d["idx"]] = d
    pairs = [(r, other[r["idx"]]) for r in results if r["idx"] in other]
    if not pairs:
        print(f"compare: no overlapping prompt indices with {other_path}")
        return
    same = sum(r["output"] == o["output"] for r, o in pairs)
    print(f"compare vs {other_path}: {same}/{len(pairs)} identical outputs ({same / len(pairs):.1%})")
    if tok is None:
        return
    agree = total = 0
    first_diff = []
    for r, o in pairs:
        a = tok(r["output"], add_special_tokens=False)["input_ids"]
        b = tok(o["output"], add_special_tokens=False)["input_ids"]
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        agree += k
        total += max(len(a), len(b))
        if a != b:
            first_diff.append((r["idx"], k))
    print(f"  token-level prefix agreement {agree}/{total} ({agree / max(1, total):.1%})"
          + (f"; first differences (idx, token position): {first_diff[:10]}" if first_diff else ""))


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    prompts = load_prompts(args)
    tok = load_tokenizer(args) if (args.chat_template or args.tokenizer or args.compare) else None
    try:
        print(f"server http://{args.host}:{args.port}  model {server_model_path(args)}")
    except Exception as e:
        sys.exit(f"server not reachable at http://{args.host}:{args.port} ({e})")
    _LOG_PATH[0] = args.server_log
    before = parse_cum_stats(args.server_log) if args.server_log else {}

    results = []
    t_start = time.perf_counter()
    for i, prompt in enumerate(prompts):
        r = generate(args, prompt, tok)
        r.update(idx=args.offset + i, prompt=prompt)
        results.append(r)
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            tok_so_far = sum(x["tokens"] for x in results)
            sec_so_far = sum(x["sec"] for x in results)
            print(f"  {i + 1}/{len(prompts)}  {tok_so_far / max(1e-9, sec_so_far):.1f} tok/s so far",
                  flush=True)
    wall = time.perf_counter() - t_start

    n = len(results)
    tot_tok = sum(r["tokens"] for r in results)
    tot_sec = sum(r["sec"] for r in results)
    print()
    print(f"requests            {n}")
    print(f"completion tokens   {tot_tok}")
    print(f"seconds             {tot_sec:.1f} in requests, {wall:.1f} wall")
    print(f"tok/s               {tot_tok / max(1e-9, tot_sec):.1f}   (completion tokens / request seconds; "
          f"{tot_tok / max(1e-9, wall):.1f} over wall time)")
    print(f"mean tokens/request {tot_tok / n:.1f}")
    print(f"mean latency        {tot_sec / n:.2f} s")
    print(f"finish reasons      {dict(Counter(r['finish'] for r in results))}")

    if args.save:
        with open(args.save, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"saved {n} results to {args.save}")
    if args.compare:
        compare(results, args.compare, tok)
    if args.server_log:
        time.sleep(1.0)  # let the last request's cleanup line reach the log
        report_log(before, parse_cum_stats(args.server_log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
