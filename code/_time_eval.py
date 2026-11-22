"""Time one evaluation at the suite's bootstrap setting (is 2000 affordable?)."""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M      # noqa: E402
import predictors as P   # noqa: E402
import sampler as S      # noqa: E402
import split as SP       # noqa: E402


def main() -> int:
    ds = S.load(os.path.join(HERE, os.pardir, "data", "suite",
                             "scene2_spec13", "ds_m3_s00_e96.json"))
    rows = SP.assign(ds["episodes"], seed=0)
    ids = {"train": [], "val": [], "test": []}
    for r in rows:
        ids[r["split"]].append(r["episode_id"])
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])
    for nb in (200, 2000):
        t = time.time()
        r = P.evaluate_predictor(P.StructuredPlan(level=3, alpha=0.1), ds,
                                 convention="C", train_ids=ids["train"],
                                 test_ids=ids["test"], spec=spec, eps=eps, n_boot=nb)
        m = r["report"]["metrics"]
        print("n_boot=%-5d %.1fs  DAR=%.4f CI=%s  tau=%s"
              % (nb, time.time() - t, m["dar"]["value"], m["dar"]["ci95"],
                 m["tau"]["value"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
