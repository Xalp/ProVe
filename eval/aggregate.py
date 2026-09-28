#!/usr/bin/env python
"""Aggregate run_eval.sh outputs into per-benchmark accuracy and speed tables (paper Table 1 layout).

Walks <root>/<config>/<task>_<fs>shot/ and reads the lm-eval results*.json (searched recursively:
the harness nests it under the sanitized model name) and the sgl_timed speed_stats.json.

    python eval/aggregate.py --root eval_out [--configs ar,lossless,prove_...] [--baseline ar]
                             [--md table.md] [--json table.json] [--detail]

Accuracy is the harness metric of each task family (METRIC_KEYS below); BBH subtasks and MMLU-Pro
categories are combined weighted by their sample count. Speed is pooled tok/s = total completion
tokens / total in-flight seconds over every request of the benchmark; speedup = tok/s divided by
the --baseline config's tok/s on the same units. Units without a .done marker (interrupted or
--limit smoke runs) are still read but marked with *.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

# exact metric keys the harness writes for each task family (first present key wins)
METRIC_KEYS = [
    (r"^ifeval$",         ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"]),
    (r"^gsm8k",           ["exact_match,flexible-extract", "exact_match,strict-match"]),
    (r"^gpqa",            ["exact_match,flexible-extract", "exact_match,strict-match"]),
    (r"^humaneval",       ["pass@1,create_test", "pass@1,none"]),
    (r"^mmlu_pro",        ["exact_match,custom-extract"]),
    (r"^bbh_cot_fewshot", ["exact_match,get-answer"]),
]
BENCH = [  # display name, task regex, number of units in tasks.txt
    ("HumanEval", r"^humaneval", 1), ("GSM8K", r"^gsm8k", 1), ("GPQA", r"^gpqa", 1),
    ("BBH", r"^bbh_cot_fewshot", 27), ("MMLU-Pro", r"^mmlu_pro", 14), ("IFEval", r"^ifeval", 1),
]


def bench_of(task):
    for name, rx, _ in BENCH:
        if re.match(rx, task):
            return name
    return task


def pick_metric(task, metrics):
    for rx, keys in METRIC_KEYS:
        if re.match(rx, task):
            for k in keys:
                if k in metrics:
                    return k, metrics[k]
    for k, v in metrics.items():  # fallback: first accuracy-like number
        if isinstance(v, (int, float)) and "stderr" not in k and re.search(r"acc|exact_match|pass", k):
            return k, v
    return None, None


def read_units(root, cfg):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, cfg, "*shot"))):
        unit = os.path.basename(d)
        task = re.sub(r"_\d+shot$", "", unit)
        rfs = sorted(glob.glob(os.path.join(d, "**", "results*.json"), recursive=True))
        if not rfs:
            continue
        res = json.load(open(rfs[-1]))
        m = res["results"].get(task) or next(iter(res["results"].values()))
        key, val = pick_metric(task, m)
        ns = res.get("n-samples", {}).get(task)
        n = ns.get("effective", ns.get("original")) if isinstance(ns, dict) else None
        if n is None:
            sf = sorted(glob.glob(os.path.join(d, "**", "samples_*.jsonl"), recursive=True))
            n = sum(1 for _ in open(sf[-1])) if sf else 0
        sp_path = os.path.join(d, "speed_stats.json")
        sp = json.load(open(sp_path)) if os.path.exists(sp_path) else {}
        rows.append(dict(cfg=cfg, task=task, bench=bench_of(task), metric=key, acc=val, n=n,
                         tokens=sp.get("total_tokens"), sec=sp.get("total_sec"),
                         tok_s=sp.get("tok_per_s_decode"), mean_tok=sp.get("mean_tokens"),
                         capped=sp.get("capped"), done=os.path.exists(os.path.join(d, ".done"))))
    return rows


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="output root passed to run_eval.sh --out")
    ap.add_argument("--configs", default="", help="comma-separated config names (default: every subdir of root)")
    ap.add_argument("--baseline", default="ar", help="config whose tok/s defines 1.00x (default: ar)")
    ap.add_argument("--md", default="", help="also write the markdown to this file")
    ap.add_argument("--json", default="", help="also write the per-unit rows and aggregates as JSON")
    ap.add_argument("--detail", action="store_true", help="append the per-unit table")
    a = ap.parse_args()

    if a.configs:
        cfgs = [c for c in a.configs.split(",") if c]
    else:
        cfgs = sorted(c for c in os.listdir(a.root) if glob.glob(os.path.join(a.root, c, "*shot")))
        if a.baseline in cfgs:  # baseline first
            cfgs.remove(a.baseline); cfgs.insert(0, a.baseline)
    if not cfgs:
        sys.exit(f"no <config>/<task>_<fs>shot directories under {a.root}")

    rows = [r for cfg in cfgs for r in read_units(a.root, cfg)]
    agg = defaultdict(dict)  # bench -> cfg -> aggregate
    for name, _, _ in BENCH:
        for cfg in cfgs:
            rs = [r for r in rows if r["bench"] == name and r["cfg"] == cfg and r["acc"] is not None]
            if not rs:
                continue
            n = sum(r["n"] or 0 for r in rs)
            toks = sum(r["tokens"] or 0 for r in rs)
            sec = sum(r["sec"] or 0 for r in rs)
            agg[name][cfg] = dict(acc=(sum(r["acc"] * (r["n"] or 0) for r in rs) / n if n else None), n=n,
                                  units=len(rs), tok_s=(toks / sec if sec else None),
                                  mean_tok=(toks / n if n else None), metric=rs[0]["metric"],
                                  partial=not all(r["done"] for r in rs))

    lines = [f"## Accuracy (%; harness metric per family, BBH/MMLU-Pro weighted by n) -- root {a.root}", "",
             "| Benchmark | metric | " + " | ".join(cfgs) + " |", "|---|---|" + "---|" * len(cfgs)]
    for name, _, expected in BENCH:
        if name not in agg:
            continue
        metric = next(iter(agg[name].values()))["metric"]
        cells = []
        for cfg in cfgs:
            v = agg[name].get(cfg)
            if not v or v["acc"] is None:
                cells.append("-"); continue
            note = "" if v["units"] == expected else f", {v['units']}/{expected} units"
            cells.append(f"{100 * v['acc']:.1f} (n={v['n']}{note}){'*' if v['partial'] else ''}")
        lines.append(f"| {name} | {metric} | " + " | ".join(cells) + " |")
    # unweighted mean over the six benchmarks (the paper's Avg), only when all six are present
    avg = []
    for cfg in cfgs:
        vals = [agg[b][cfg]["acc"] for b, _, _ in BENCH if cfg in agg.get(b, {}) and agg[b][cfg]["acc"] is not None]
        avg.append(f"{100 * sum(vals) / len(vals):.1f}" if len(vals) == len(BENCH) else "-")
    lines.append("| Avg (6) | | " + " | ".join(avg) + " |")

    lines += ["", f"## Decode speed, pooled tok/s (speedup vs `{a.baseline}` on the same benchmark)", "",
              "| Benchmark | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
    ratios = defaultdict(list)
    for name, _, _ in BENCH:
        if name not in agg:
            continue
        base = agg[name].get(a.baseline, {}).get("tok_s")
        cells = []
        for cfg in cfgs:
            v = agg[name].get(cfg)
            if not v or not v["tok_s"]:
                cells.append("-"); continue
            s = f"{v['tok_s']:.1f}"
            if base:
                s += f" ({v['tok_s'] / base:.2f}x)"; ratios[cfg].append(v["tok_s"] / base)
            cells.append(s)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| Avg speedup | " + " | ".join(
        f"{sum(ratios[c]) / len(ratios[c]):.2f}x" if len(ratios[c]) == len(BENCH) else "-" for c in cfgs) + " |")
    if any(v["partial"] for b in agg.values() for v in b.values()):
        lines.append("\n\\* at least one unit has no .done marker (interrupted run or --limit smoke test)")

    if a.detail:
        lines += ["", "## Per-unit detail", "", "| unit | config | metric | acc | n | tok/s | mean tok | capped |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r["bench"], r["task"], cfgs.index(r["cfg"]))):
            acc = "-" if r["acc"] is None else f"{100 * r['acc']:.1f}"
            lines.append(f"| {r['task']} | {r['cfg']}{'' if r['done'] else '*'} | {r['metric']} | {acc} | {r['n']} | "
                         f"{fmt(r['tok_s'])} | {fmt(r['mean_tok'], 0)} | {r['capped'] if r['capped'] is not None else '-'} |")

    md = "\n".join(lines)
    print(md)
    if a.md:
        open(a.md, "w").write(md + "\n")
    if a.json:
        json.dump({"configs": cfgs, "baseline": a.baseline, "rows": rows, "agg": agg}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
