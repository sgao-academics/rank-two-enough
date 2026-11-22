"""Train on convention C, score on convention C and O -- at the rank-two model the
paper actually reports.

Why this script exists
----------------------
`_lrb_transfer.py` scores `LowRankBilinear(rank=2)` without `plan_quad`, which is
the 94-coefficient variant: the interaction block alone, no plan-curvature terms.
`Table 4` reports the 139-coefficient variant (`plan_quad=True`), and the
560-coefficient quadratic it is compared against carries the curvature. Pairing a
block that lacks 45 terms against an expansion that has them is not a comparison of
architectures, it is a comparison of feature sets, and the gap it produces (0.0098)
is mostly the 45 terms.

So the transfer experiment is rerun here on the same model as Table 4: rank two
with plan curvature, 139 coefficients. Nothing else changes -- same seeds, same
splits, same train convention, same scoring convention.
"""

import io
import json
import os
import statistics
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import report as R           # noqa: E402
import run_suite as RS       # noqa: E402
import split as SP           # noqa: E402
from lowrank_bilinear import LowRankBilinear   # noqa: E402

spec = M.load_spec()
eps = float(spec["noise_floor"]["deterministic_fallback"])

# The two models the section compares, both at n_iter=8 -- the setting Table 4 uses.
MODELS = [
    ("LRB r2 (139)", lambda: LowRankBilinear(rank=2, alpha=1.0, n_iter=8,
                                             seed=0, init="svd", plan_quad=True)),
    ("B4 ridge poly2 (560)", lambda: P.RidgePoly(alpha=1.0)),
]

res = {}
for seed in range(12):
    path = RS.ds_path("m3", seed)
    if not os.path.exists(path):
        print("  ! missing", os.path.basename(path))
        continue
    ds = json.load(io.open(path, encoding="utf-8"))
    for sp in range(5):
        rows = SP.assign(ds["episodes"], seed=sp)
        ids = {"train": [], "val": [], "test": []}
        for row in rows:
            ids[row["split"]].append(row["episode_id"])
        for label, fac in MODELS:
            for conv in ("C", "O"):
                r = P.evaluate_predictor(fac(), ds, convention=conv,
                                         train_ids=ids["train"],
                                         test_ids=ids["test"], spec=spec,
                                         eps=eps, n_boot=0,
                                         train_convention="C")
                dar = float(R.summarize(r["report"], label)["dar"])
                res.setdefault((label, conv), []).append(dar)
    print("  [s%02d] done" % seed, flush=True)

out = {}
for (label, conv), vals in sorted(res.items()):
    out.setdefault(label, {})[conv] = [statistics.mean(vals),
                                       statistics.stdev(vals), len(vals)]

print("\nconvention transfer at the reported model (train on C):")
for label in sorted(out):
    c = out[label]["C"]
    o = out[label]["O"]
    print("  %-22s  C %.4f (sd %.4f)   O %.4f (sd %.4f)   shift %+.4f"
          % (label, c[0], c[1], o[0], o[1], o[0] - c[0]))
    print("  %-22s  n=%d cells" % ("", c[2]))

lr, qu = "LRB r2 (139)", "B4 ridge poly2 (560)"
if lr in out and qu in out:
    gapC = out[qu]["C"][0] - out[lr]["C"][0]
    gapO = out[qu]["O"][0] - out[lr]["O"][0]
    print("\nquadratic minus rank-two:   C %+.4f   O %+.4f   change %+.4f"
          % (gapC, gapO, gapO - gapC))

dst = os.path.join(os.path.dirname(CODE), "05_Logs", "lrb_transfer139.json")
with io.open(dst, "w", encoding="utf-8") as fh:
    json.dump({"cells": 180, "train_convention": "C", "models": out}, fh, indent=1)
print("\nwrote", dst)
