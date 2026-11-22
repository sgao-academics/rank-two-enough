"""LRB at full scale, with the rank chosen on the VALIDATION templates.

Two things this script refuses to do:

  * pick r by looking at test DAR. The suite already holds out two templates as
    `val`; the rank and the shrinkage are selected there, and only the selected
    configuration is scored on `test`.
  * report a single cell. Every number is a mean over 12 dataset seeds x 5
    template partitions, the same unit the rest of the paper uses.

Also on the curve: rank 1..9 so that the "where does the rank stop paying"
question is answered by measurement, not by the sweep's small grid.
"""

import io
import json
import os
import sys
import time
from statistics import mean, stdev

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import report as R           # noqa: E402
import run_suite as RS       # noqa: E402
import split as SP           # noqa: E402
from lowrank_bilinear import LowRankBilinear   # noqa: E402

spec = M.load_spec()
eps = float(spec["noise_floor"]["deterministic_fallback"])

import argparse          # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--configs", nargs="+", default=["m3", "H4", "H8"])
_ap.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
_ap.add_argument("--splits", nargs="+", type=int, default=list(range(5)))
_ap.add_argument("--out", default="lrb_final.json")
# The rank cannot exceed min(d_obs, d_action) = min(22, 3 * plan_len): the
# unconstrained block W is (22 x 3m), so its SVD has at most 3m singular triplets
# and `svd` returns min(self.rank, 3m) of them. At plan_len=3 that bound is 9 and
# the default sweep hits it exactly; at plan_len=1 it is 3 and asking for r=4
# raises inside fit(). Kept as a flag rather than inferred, so that a sweep asked
# for explicitly is honoured and a sweep asked for beyond the bound is an error.
_ap.add_argument("--ranks", nargs="+", type=int, default=list(range(1, 10)))
_ARGS = _ap.parse_args()

CONFIGS = tuple(_ARGS.configs)
SEEDS = list(_ARGS.seeds)
SPLITS = list(_ARGS.splits)
RANKS = tuple(_ARGS.ranks)
ALPHAS = (1.0,)          # fixed by the sweep; kept as a tuple so the val search is
                         # still a search over r rather than a hard-coded answer

REF = [("G0b summed aligned push", lambda: P.HeuristicAlignedSum()),
       ("G1 global direction", lambda: P.GlobalDirection()),
       ("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1)),
       ("B4 ridge poly2", lambda: P.RidgePoly(alpha=1.0))]


def dar_of(pred, ds, tr, te):
    res = P.evaluate_predictor(pred, ds, convention="C", train_ids=tr,
                               test_ids=te, spec=spec, eps=eps, n_boot=0)
    return float(R.summarize(res["report"], res["name"])["dar"])


acc_val, acc_test, chosen = {}, {}, {}
curve = {}
t0 = time.time()
n_cells = 0

for cfg in CONFIGS:
    for seed in SEEDS:
        path = RS.ds_path(cfg, seed)
        if not os.path.exists(path):
            continue
        ds = json.load(open(path, encoding="utf-8"))
        for sp in SPLITS:
            rows = SP.assign(ds["episodes"], seed=sp)
            ids = {"train": [], "val": [], "test": []}
            for row in rows:
                ids[row["split"]].append(row["episode_id"])
            if not (ids["val"] and ids["test"]):
                continue

            # -- rank selection on the validation templates ------------------ #
            best = None
            for r in RANKS:
                for al in ALPHAS:
                    v = dar_of(LowRankBilinear(rank=r, alpha=al, n_iter=8,
                                               seed=0, init="svd",
                                               plan_quad=True),
                               ds, ids["train"], ids["val"])
                    acc_val.setdefault((cfg, r), []).append(v)
                    if best is None or v > best[0]:
                        best = (v, r, al)
            _v, r_star, al_star = best
            chosen.setdefault(r_star, 0)
            chosen[r_star] += 1

            # -- the selected model, scored once on test --------------------- #
            pred = LowRankBilinear(rank=r_star, alpha=al_star, n_iter=8,
                                   seed=0, init="svd", plan_quad=True)
            ck = "LRB(val) r%d" % r_star
            acc_test.setdefault(ck, []).append(dar_of(pred, ds, ids["train"],
                                                      ids["test"]))

            # -- the rank curve on test, for the figure ---------------------- #
            for r in RANKS:
                v = dar_of(LowRankBilinear(rank=r, alpha=1.0, n_iter=8,
                                           seed=0, init="svd",
                                           plan_quad=True),
                           ds, ids["train"], ids["test"])
                curve.setdefault((cfg, r), []).append(v)

            for label, fac in REF:
                acc_test.setdefault(label, []).append(
                    dar_of(fac(), ds, ids["train"], ids["test"]))
            n_cells += 1
        print("  [%s s%02d] %.0fs" % (cfg, seed, time.time() - t0), flush=True)

print("\n" + "=" * 78)
print("rank selected on validation templates")
print("=" * 78)
tot = sum(chosen.values())
for r in sorted(chosen):
    print("  r=%d chosen in %2d of %d partitions (%.0f%%)"
          % (r, chosen[r], tot, 100.0 * chosen[r] / tot))

print("\n" + "=" * 78)
print("test DAR, mean over %d cells" % n_cells)
print("=" * 78)
print("  %-26s %8s %8s %6s" % ("model", "DAR", "sd", "n"))
for label in acc_test:
    v = acc_test[label]
    sd = stdev(v) if len(v) > 1 else 0.0
    print("  %-26s %8.4f %8.4f %6d" % (label, mean(v), sd, len(v)))

print("\n" + "=" * 78)
print("rank curve on test (alpha=1.0), mean by config")
print("=" * 78)
print("  %-6s %s" % ("cfg", " ".join("%7s" % ("r%d" % r) for r in RANKS)))
for cfg in CONFIGS:
    vals = [mean(curve[(cfg, r)]) if (cfg, r) in curve else float("nan")
            for r in RANKS]
    print("  %-6s %s" % (cfg, " ".join("%7.4f" % v for v in vals)))

def _summ(v):
    return [mean(v), stdev(v) if len(v) > 1 else 0.0, len(v)]


out = {"chosen": chosen, "test": {k: _summ(v) for k, v in acc_test.items()},
       "rank_curve": {"%s|r%d" % (c, r): _summ(v)
                      for (c, r), v in curve.items()},
       "val": {"%s|r%d" % (c, r): _summ(v) for (c, r), v in acc_val.items()},
       "cells": n_cells}
with open(os.path.join(RS.ROOT, "logs", _ARGS.out), "w",
          encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote logs/%s  (%d cells, %.0fs)"
      % (_ARGS.out, n_cells, time.time() - t0))
