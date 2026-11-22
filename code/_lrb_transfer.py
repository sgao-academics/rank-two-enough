"""Train on convention C, score on convention O.

The suite already emits rows["O"] for every model, so this is the same protocol
with the low-rank model added. The question it answers: does a 94-parameter
low-rank block transfer better than a 560-parameter quadratic, or does the
explicit block's extra freedom buy robustness instead of variance?
"""

import io
import json
import os
import sys
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

MODELS = [("G1 global direction", lambda: P.GlobalDirection()),
          ("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1)),
          ("LRB r2", lambda: LowRankBilinear(rank=2, alpha=1.0, n_iter=12,
                                             seed=0, init="svd")),
          ("B4 ridge poly2", lambda: P.RidgePoly(alpha=1.0))]

res = {}
for seed in range(12):
    path = RS.ds_path("m3", seed)
    if not os.path.exists(path):
        continue
    ds = json.load(open(path, encoding="utf-8"))
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
                res.setdefault((label, conv), []).append(
                    float(R.summarize(r["report"], label)["dar"]))
    print("  [s%02d]" % seed, flush=True)

print("\n%-22s %10s %10s %10s" % ("model", "train C/test C",
                                  "train C/test O", "change"))
out = {}
for label, _f in MODELS:
    vc, vo = res[(label, "C")], res[(label, "O")]
    print("%-22s %10.4f %10.4f %+10.4f"
          % (label, mean(vc), mean(vo), mean(vo) - mean(vc)))
    out[label] = {"C": [mean(vc), stdev(vc), len(vc)],
                  "O": [mean(vo), stdev(vo), len(vo)]}

with open(os.path.join(RS.ROOT, "logs", "lrb_transfer.json"), "w",
          encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote logs/lrb_transfer.json")
