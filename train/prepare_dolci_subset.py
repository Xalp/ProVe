#!/usr/bin/env python
"""Build the stage-2 training subset of Dolci-Instruct-SFT used for the ProVe proposers.

Stage 2 (block training, B=16) runs on a seed-42 shuffled 12.5 % subset of
allenai/Dolci-Instruct-SFT: 269,014 of the 2,152,112 training rows. The subset is
shuffled BEFORE slicing because Dolci is stored ordered by source (its last ~200k rows are
all "Tool Use"), so a contiguous head slice would silently drop whole domains.

    python train/prepare_dolci_subset.py --out data/dolci_12p5

writes <out>/train.parquet, which train_proposer.sh (stage 2) passes to MARS as
--dataset_args <out>.  A parquet directory is what the public MARS data loader reads
(datasets.load_dataset(<dir>)); --save-to-disk writes Arrow via save_to_disk instead, but
that format is only readable with load_from_disk and is NOT accepted by --dataset_args.

Per-domain counts of the resulting subset are printed so the run can be checked against
the numbers in train/README.md.
"""
import argparse
import collections
import os
import sys

from datasets import load_dataset

DOLCI = "allenai/Dolci-Instruct-SFT"
DOLCI_TOTAL_ROWS = 2_152_112  # train split at the time of the paper


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (train.parquet is written inside it)")
    ap.add_argument("--dataset", default=DOLCI, help="HF dataset id or local path (default: %(default)s)")
    ap.add_argument("--revision", default=None, help="optional dataset revision to pin")
    ap.add_argument("--fraction", type=float, default=0.125, help="fraction of rows to keep (default: 0.125)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42)")
    ap.add_argument("--save-to-disk", action="store_true",
                    help="write Arrow with save_to_disk instead of train.parquet (see module docstring)")
    ap.add_argument("--no-assert-total", action="store_true",
                    help="do not require the Dolci train split to have exactly %d rows" % DOLCI_TOTAL_ROWS)
    args = ap.parse_args()

    kw = {"split": "train"}
    if args.revision:
        kw["revision"] = args.revision
    ds = load_dataset(args.dataset, **kw)
    n_total = len(ds)
    print(f"[dolci] loaded {args.dataset}: {n_total:,} rows, columns {ds.column_names}")
    if args.dataset == DOLCI and not args.no_assert_total:
        assert n_total == DOLCI_TOTAL_ROWS, (
            f"expected {DOLCI_TOTAL_ROWS:,} rows in {DOLCI} train, found {n_total:,}: the dataset changed "
            "upstream, so a seed-42 shuffle would no longer give the paper's subset. Pin a revision "
            "with --revision, or pass --no-assert-total to accept a different subset.")

    n_keep = int(n_total * args.fraction)  # 269,014 for the paper's 12.5 %
    sub = ds.shuffle(seed=args.seed).select(range(n_keep))
    print(f"[dolci] kept the first {n_keep:,} rows of a seed-{args.seed} shuffle ({100 * args.fraction:g} %)")

    if "domain" in sub.column_names:
        counts = collections.Counter(d if d else "(none)" for d in sub["domain"])
        print("[dolci] rows per domain:")
        for dom, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {dom:<14s} {c:>8,d}")

    os.makedirs(args.out, exist_ok=True)
    if args.save_to_disk:
        sub.save_to_disk(args.out)
        print(f"[dolci] wrote Arrow (save_to_disk) to {args.out}  (read it with datasets.load_from_disk)")
    else:
        path = os.path.join(args.out, "train.parquet")
        sub.to_parquet(path)
        print(f"[dolci] wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB); "
              f"use it as STAGE2_DATASET={args.out} for train_proposer.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
