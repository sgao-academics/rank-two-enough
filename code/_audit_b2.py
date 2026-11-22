"""Is B2 a fair linear baseline, or a mis-regularised one?

B2 scores 0.557 while a ZERO-parameter heuristic scores 0.891. Two readings:

  (a) "a linear model cannot do this task"  -- which would make B2 a legitimate
      baseline and the structural story stronger;
  (b) "the linear model was never given a chance" -- because the design is NOT
      row-centred, the 22 observation columns (which are CONSTANT across the 5 arms
      of a step) carry the entire level of Y, so ridge spends its shrinkage on the
      9 plan columns where the only ranking information lives.

(b) would be a desk-reject-level flaw: the paper's linear rung would be a straw man.
The test is trivial and decisive -- refit the SAME features, SAME alpha, with the
design centred within each row. Ranking is invariant to a row-constant offset, so a
fair linear model must lose nothing from centring; if centring lifts it from 0.557 to
~0.91, then (b) is the answer and B2 must be reported as mis-scaled, not as a limit.
"""
from __future__ import annotations

import io
import json
import os
import sys
from statistics import mean, stdev

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import report as R           # noqa: E402
import split as SP           # noqa: E402
import run_suite as RS       # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")


class RidgeCentred(P.Ridge):
    """B2c: identical to B2, design centred within each row."""

    name = "B2c ridge centred (fair linear rung)"

    def _design(self, tab):
        x = super()._design(tab)
        return x - x.mean(axis=1, keepdims=True)


class RidgePolyCentred(P.RidgePoly):
    """B4c: B4 with the same centring, to check the quadratic is not relying on it."""

    name = "B4c ridge poly2 centred"

    def _design(self, tab):
        x = super(P.RidgePoly, self)._design(tab)
        x = x - x.mean(axis=1, keepdims=True)
        n, k, f = x.shape
        flat = x.reshape(-1, f)
        ii, jj = np.triu_indices(f)
        quad = flat[:, ii] * flat[:, jj]
        return np.concatenate([flat, quad], axis=-1).reshape(n, k, -1)


def main():
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])
    models = (("B2 ridge surrogate", lambda: P.Ridge()),
              ("B2c ridge centred", lambda: RidgeCentred()),
              ("G1 global direction", lambda: P.GlobalDirection()),
              ("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1)),
              ("B4 ridge poly2", lambda: P.RidgePoly()),
              ("B4c poly2 centred", lambda: RidgePolyCentred()))
    acc = {}
    for cfg in ("m3",):
        for seed in range(12):
            path = RS.ds_path(cfg, seed)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                ds = json.load(fh)
            for sp in range(5):
                rows = SP.assign(ds["episodes"], seed=sp)
                ids = {"train": [], "val": [], "test": []}
                for r in rows:
                    ids[r["split"]].append(r["episode_id"])
                if not (ids["train"] and ids["test"]):
                    continue
                for label, factory in models:
                    res = P.evaluate_predictor(
                        factory(), ds, convention="C", train_ids=ids["train"],
                        test_ids=ids["test"], spec=spec, eps=eps, n_boot=0)
                    acc.setdefault(label, []).append(
                        float(R.summarize(res["report"], res["name"])["dar"]))
            print("  seed %02d done" % seed, flush=True)

    print("\n" + "=" * 66)
    print("LINEAR RUNG AUDIT  (m3, 12 dataset seeds x 5 template splits)")
    print("=" * 66)
    print("%-26s %8s %8s %8s" % ("model", "DAR", "sd", "n"))
    for label, _f in models:
        v = acc.get(label, [])
        if not v:
            continue
        print("%-26s %8.4f %8.4f %8d"
              % (label, mean(v), stdev(v) if len(v) > 1 else 0.0, len(v)))

    b2, b2c = mean(acc["B2 ridge surrogate"]), mean(acc["B2c ridge centred"])
    print("\ncentring lifts the linear rung by %+.4f (%.1f%% -> %.1f%%)"
          % (b2c - b2, 100 * b2, 100 * b2c))
    print("=> B2 was %s" % ("MIS-SCALED; (b) holds" if b2c - b2 > 0.05
                            else "genuinely weak; (a) holds"))
    with open(os.path.join(RS.ROOT, "logs", "audit_b2.json"), "w",
              encoding="utf-8") as fh:
        json.dump({k: v for k, v in acc.items()}, fh, indent=1)
    print("wrote logs/audit_b2.json")


if __name__ == "__main__":
    main()
