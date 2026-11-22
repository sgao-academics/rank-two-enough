"""The comparison Table 5 was missing: the explicit block against its rank-two fit,
on the SAME cells.

Table 5 (tab:blocks) reports the explicit state x action block, 230 coefficients,
at 0.9216 on 60 cells of the primary configuration. Table 6 (tab:rank) reports the
rank-two bilinear model, 139 coefficients, at 0.9309 on 180 cells of three
configurations. The two families were never scored on a common tally, so the claim
"139 matches 560" has never been checked against the cheaper comparison that a
reader will actually ask for: does compressing the block from 198 to 62 terms cost
anything?

This script answers it by construction. Every model below is fitted and scored on
the identical (configuration, scene seed, partition) cell, so the difference is a
paired one and the spread of the difference is measured rather than inferred from
two marginal spreads.

Fits, all with the shared 32-term linear skeleton:
    main              32    linear base only
    sxa              230    explicit 22 x 9 block            (tab:blocks row 2)
    sxa+axa          275    explicit block + plan curvature  (tab:blocks row 3)
    sxa+sxs+axa      528    the whole degree-2 expansion     (tab:blocks row 4)
    LRB r1           108
    LRB r2           139    matches 560 in tab:rank
    LRB r3           170
    LRB r2, alpha 3  139    a shrinkage the block was not given at r=2

Usage
    python _lrb_vs_bilinear.py --smoke          # one cell, prints the fits
    python _lrb_vs_bilinear.py                  # 3 configs x 12 seeds x 5 splits
    python _lrb_vs_bilinear.py --configs m3 --seeds 0 1 --splits 0 1

Writes logs/lrb_vs_bilinear.json with per-cell values, marginal summaries and
paired differences against the 139-coefficient model.
"""

import argparse
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import numpy as np                      # noqa: E402
import metrics as M                     # noqa: E402
import predictors as P                  # noqa: E402
import report as R                      # noqa: E402
import run_suite as RS                  # noqa: E402
import split as SP                      # noqa: E402
from lowrank_bilinear import LowRankBilinear, split_parts   # noqa: E402

spec = M.load_spec()
eps = float(spec["noise_floor"]["deterministic_fallback"])


class Combo(P.Ridge):
    """Same class as _block_combo.py: the design that produced tab:blocks."""

    kind = "baseline"

    def __init__(self, sxa=True, sxs=False, axa=False, alpha=1.0):
        super().__init__(alpha=alpha)
        self.sxa, self.sxs, self.axa = bool(sxa), bool(sxs), bool(axa)
        tag = [n for n, f in (("sxa", sxa), ("sxs", sxs), ("axa", axa)) if f]
        self.name = "+".join(tag) if tag else "main"
        self.n_cols = None

    def _design(self, tab):
        p = split_parts(tab)
        n, k = p.main.shape[0], p.main.shape[1]
        s = p.state.reshape(-1, p.state.shape[-1])
        a = p.action.reshape(-1, p.action.shape[-1])
        blocks = [p.main.reshape(-1, p.main.shape[-1])]
        ds, da = s.shape[1], a.shape[1]
        if self.sxa:
            blocks.append((s[:, :, None] * a[:, None, :]).reshape(-1, ds * da))
        if self.sxs:
            ii, jj = np.triu_indices(ds)
            blocks.append(s[:, ii] * s[:, jj])
        if self.axa:
            ii, jj = np.triu_indices(da)
            blocks.append(a[:, ii] * a[:, jj])
        x = np.concatenate(blocks, axis=1)
        self.n_cols = x.shape[1] - 1
        return x.reshape(n, k, -1)


MODELS = [
    ("main (32)",          lambda: Combo(sxa=False)),
    ("sxa (230)",          lambda: Combo(sxa=True)),
    ("sxa+axa (275)",      lambda: Combo(sxa=True, axa=True)),
    ("sxa+sxs+axa (528)",  lambda: Combo(sxa=True, sxs=True, axa=True)),
    ("LRB r1 (108)",       lambda: LowRankBilinear(rank=1, alpha=1.0, n_iter=8, init="svd", plan_quad=True)),
    ("LRB r2 (139)",       lambda: LowRankBilinear(rank=2, alpha=1.0, n_iter=8, init="svd", plan_quad=True)),
    ("LRB r3 (170)",       lambda: LowRankBilinear(rank=3, alpha=1.0, n_iter=8, init="svd", plan_quad=True)),
    ("LRB r2 a3 (139)",    lambda: LowRankBilinear(rank=2, alpha=3.0, n_iter=8, init="svd", plan_quad=True)),
]

PAIR_AGAINST = "LRB r2 (139)"


def dar_of(pred, ds, tr, te):
    res = P.evaluate_predictor(pred, ds, convention="C", train_ids=tr,
                               test_ids=te, spec=spec, eps=eps, n_boot=0)
    return float(R.summarize(res["report"], pred.name)["dar"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["m3", "H4", "H8"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
    ap.add_argument("--splits", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="lrb_vs_bilinear.json")
    args = ap.parse_args()

    if args.smoke:
        args.configs, args.seeds, args.splits = ["m3"], [0], [0]

    per_cell = []          # list of dicts: cfg, seed, split, {label: dar}
    t0 = time.time()
    n_cell = 0
    for cfg in args.configs:
        for seed in args.seeds:
            path = RS.ds_path(cfg, seed)
            if not os.path.exists(path):
                continue
            ds = json.load(open(path, encoding="utf-8"))
            for sp in args.splits:
                rows = SP.assign(ds["episodes"], seed=sp)
                ids = {"train": [], "val": [], "test": []}
                for row in rows:
                    ids[row["split"]].append(row["episode_id"])
                if not (ids["train"] and ids["test"]):
                    continue
                cell = {"cfg": cfg, "seed": seed, "split": sp}
                for label, fac in MODELS:
                    cell[label] = dar_of(fac(), ds, ids["train"], ids["test"])
                per_cell.append(cell)
                n_cell += 1
            print("  [%s s%02d] %d cells, %.0fs"
                  % (cfg, seed, n_cell, time.time() - t0), flush=True)

    labels = [l for l, _f in MODELS]
    print("\n" + "=" * 78)
    print("marginal DAR, %d cells, convention C, held-out templates" % n_cell)
    print("=" * 78)
    print("%-22s %8s %8s %8s" % ("model", "DAR", "sd", "n"))
    for l in labels:
        v = [c[l] for c in per_cell]
        print("%-22s %8.4f %8.3f %8d"
              % (l, float(np.mean(v)), float(np.std(v, ddof=1)), len(v)))

    print("\n" + "=" * 78)
    print("paired difference against %s (same cell, so the spread is the"
          % PAIR_AGAINST)
    print("spread of the difference itself, not two marginal spreads)")
    print("=" * 78)
    print("%-22s %10s %9s %9s %10s %6s" %
          ("model", "mean diff", "sd diff", "se", "t", "n"))
    paired = {}
    for l in labels:
        if l == PAIR_AGAINST:
            continue
        d = np.array([c[l] - c[PAIR_AGAINST] for c in per_cell])
        sd = float(np.std(d, ddof=1))
        se = sd / np.sqrt(len(d))
        t = float(np.mean(d) / se) if se > 0 else float("nan")
        paired[l] = {"mean": float(np.mean(d)), "sd": sd, "se": se,
                     "t": t, "n": int(len(d))}
        print("%-22s %+10.4f %9.4f %9.4f %+10.2f %6d"
              % (l, float(np.mean(d)), sd, se, t, len(d)))

    print("\n" + "=" * 78)
    print("by configuration, mean DAR")
    print("=" * 78)
    print("%-22s %s" % ("model", " ".join("%8s" % c for c in args.configs)))
    by_cfg = {}
    for l in labels:
        row = []
        for cfg in args.configs:
            v = [c[l] for c in per_cell if c["cfg"] == cfg]
            row.append(float(np.mean(v)) if v else float("nan"))
            if v:
                by_cfg.setdefault(l, {})[cfg] = [float(np.mean(v)),
                                                 float(np.std(v, ddof=1)), len(v)]
        print("%-22s %s" % (l, " ".join("%8.4f" % x for x in row)))

    # -- persist ------------------------------------------------------------ #
    out = {"cells": n_cell, "configs": args.configs,
           "seeds": args.seeds, "splits": args.splits,
           "marginal": {l: [float(np.mean([c[l] for c in per_cell])),
                            float(np.std([c[l] for c in per_cell], ddof=1)),
                            n_cell] for l in labels},
           "by_config": by_cfg,
           "paired_vs": PAIR_AGAINST,
           "paired": paired,
           "per_cell": per_cell}
    path = os.path.join(RS.ROOT, "logs", args.out)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote logs/%s  (%d cells, %.0fs)"
          % (args.out, n_cell, time.time() - t0))


if __name__ == "__main__":
    main()
