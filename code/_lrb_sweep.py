"""Rank and shrinkage sweep for LRB, against the two models that matter.

Three questions, in order:
  1. Does spectral initialisation fix the local-minimum problem?
  2. At which (r, alpha) does LRB first match T2's explicit 230-term block?
  3. Does any rank beat B4 at equal or lower cost?

Small grid on purpose: m3, 3 dataset seeds x 2 template splits, so that a wrong
setting is found in seconds rather than after an hour.
"""
import os

import io
import json
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

RANKS = (1, 2, 3, 4, 6, 8)
ALPHAS = (0.1, 1.0, 10.0)
FIXED = [("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1)),
         ("B4 ridge poly2", lambda: P.RidgePoly(alpha=1.0)),
         ("G1 global direction", lambda: P.GlobalDirection())]

acc = {}
t0 = time.time()
cells = 0

for seed in (0, 1, 2):
    ds = json.load(open(RS.ds_path("m3", seed), encoding="utf-8"))
    for sp in (0, 1):
        rows = SP.assign(ds["episodes"], seed=sp)
        ids = {"train": [], "val": [], "test": []}
        for row in rows:
            ids[row["split"]].append(row["episode_id"])
        for label, fac in FIXED:
            res = P.evaluate_predictor(fac(), ds, convention="C",
                                       train_ids=ids["train"],
                                       test_ids=ids["test"], spec=spec,
                                       eps=eps, n_boot=0)
            acc.setdefault(label, []).append(
                float(R.summarize(res["report"], label)["dar"]))
            cells += 1
        for r in RANKS:
            for al in ALPHAS:
                pred = LowRankBilinear(rank=r, alpha=al, n_iter=12,
                                       seed=0, init="svd")
                res = P.evaluate_predictor(pred, ds, convention="C",
                                           train_ids=ids["train"],
                                           test_ids=ids["test"], spec=spec,
                                           eps=eps, n_boot=0)
                label = "LRB r%d a%g" % (r, al)
                acc.setdefault(label, []).append(
                    float(R.summarize(res["report"], label)["dar"]))
                cells += 1
        print("  [s%02d sp%d] %.0fs" % (seed, sp, time.time() - t0), flush=True)

print("\n%-20s %8s %8s" % ("model", "DAR", "sd"))
for label in [f[0] for f in FIXED]:
    v = acc[label]
    print("%-20s %8.4f %8.4f" % (label, mean(v), stdev(v)))

print("\n%-20s %6s %8s %8s %8s" % ("LRB", "params", "DAR", "sd", "vs T2"))
for r in RANKS:
    for al in ALPHAS:
        label = "LRB r%d a%g" % (r, al)
        v = acc[label]
        nprm = 32 + 31 * r
        print("%-20s %6d %8.4f %8.4f %+8.4f"
              % (label, nprm, mean(v), stdev(v),
                 mean(v) - mean(acc["B4 ridge poly2"])))

print("\n%d cells in %.0fs" % (cells, time.time() - t0))
with open(RS.ROOT + r"\logs\lrb_sweep.json", "w", encoding="utf-8") as fh:
    json.dump({k: [mean(v), stdev(v), len(v)] for k, v in acc.items()}, fh,
              indent=1)
print("wrote lrb_sweep.json")
