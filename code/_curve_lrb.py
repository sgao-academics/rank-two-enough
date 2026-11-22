"""Does the 94-parameter low-rank block keep its score where the 560-parameter
quadratic loses nine points?

The full-data comparison says LRB r2 (94 parameters) reaches 0.9285 against
B4's 0.9302 (560 parameters). Equality at the top of the data curve is a
parameter-efficiency result, not a wins-column result -- unless the parameter
count also buys something the paper can point at. The candidate is the same one
measured for the frame in section 6.4: at 10% of the training split B4 falls to
0.815 while M1 holds 0.905. If a 94-parameter LEARNED model also holds, the
claim becomes "learned low-rank interactions inherit the frame's sample
efficiency" rather than "the hand-built frame is efficient".

Every cell is a closed-form solve on data that already exists. No simulation.
"""

import argparse
import io
import json
import os
import sys
import time
from statistics import mean, stdev

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import report as R           # noqa: E402
import run_suite as RS       # noqa: E402
import split as SP           # noqa: E402
from lowrank_bilinear import LowRankBilinear   # noqa: E402

FRACS = (0.10, 0.25, 0.50, 1.00)
MODELS = [("G0b summed aligned push", lambda: P.HeuristicAlignedSum()),
          ("G1 global direction", lambda: P.GlobalDirection()),
          ("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1)),
          ("LRB r2+pq", lambda: LowRankBilinear(rank=2, alpha=1.0, n_iter=15,
                                                seed=0, init="svd",
                                                plan_quad=True)),
          ("B4 ridge poly2", lambda: P.RidgePoly(alpha=1.0))]


def sub_train(ids, frac):
    """First n sorted ids -- run_suite._sub_train's rule, kept identical."""
    ordered = sorted(ids)
    n = max(1, int(round(frac * len(ordered))))
    return ordered[:n]


ap = argparse.ArgumentParser()
ap.add_argument("--configs", nargs="+", default=["m3", "H4", "H8"])
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
ap.add_argument("--splits", nargs="+", type=int, default=list(range(5)))
ap.add_argument("--out", default="lrb_lowdata.json")
args = ap.parse_args()

spec = M.load_spec()
eps = float(spec["noise_floor"]["deterministic_fallback"])
acc = {}
t0 = time.time()

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
            for frac in FRACS:
                tr = sub_train(ids["train"], frac)
                for label, fac in MODELS:
                    res = P.evaluate_predictor(fac(), ds, convention="C",
                                               train_ids=tr,
                                               test_ids=ids["test"],
                                               spec=spec, eps=eps, n_boot=0)
                    dar = float(R.summarize(res["report"], label)["dar"])
                    acc.setdefault((cfg, frac, label), []).append(dar)
        print("  [%s s%02d] %.0fs" % (cfg, seed, time.time() - t0), flush=True)

print("\n" + "=" * 80)
print("DAR vs training size (convention C)")
print("=" * 80)
out = {}
for cfg in args.configs:
    keys = [k for k in acc if k[0] == cfg]
    if not keys:
        continue
    print("\n[%s]" % cfg)
    print("  %-22s %8s %8s %8s %8s %9s"
          % ("model", "10%", "25%", "50%", "100%", "drop"))
    for label, _f in MODELS:
        vals = []
        for frac in FRACS:
            v = acc.get((cfg, frac, label))
            vals.append(mean(v) if v else float("nan"))
            out.setdefault(cfg, {}).setdefault(label, {})["%.2f" % frac] = \
                [mean(v) if v else float("nan"),
                 stdev(v) if v and len(v) > 1 else 0.0, len(v) if v else 0]
        drop = vals[3] - vals[0] if vals[0] == vals[0] else float("nan")
        print("  %-22s %8.4f %8.4f %8.4f %8.4f %+9.4f"
              % (label, vals[0], vals[1], vals[2], vals[3], drop))
    lr = [acc.get((cfg, f, "LRB r2+pq")) for f in FRACS]
    b4 = [acc.get((cfg, f, "B4 ridge poly2")) for f in FRACS]
    if all(lr) and all(b4):
        print("  LRB - B4: " + "  ".join("%+.4f" % (mean(a) - mean(b))
                                         for a, b in zip(lr, b4)))

with open(os.path.join(RS.ROOT, "logs", args.out), "w", encoding="utf-8") as fh:
    json.dump({"fracs": list(FRACS), "result": out}, fh, indent=1)
print("\nwrote logs/%s  (%.0fs)" % (args.out, time.time() - t0))
