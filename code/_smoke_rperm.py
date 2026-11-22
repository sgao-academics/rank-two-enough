"""Does the R-perm candidate set actually produce an ORDER-ONLY task?

Success criteria, all measurable on a small dataset:
  1. every counterfactual arm has the same total aligned push as the factual plan
     (so the rank-1 heuristic cannot discriminate),
  2. the truth is NOT degenerate (order changes the outcome),
  3. the rank-1 heuristic G0b lands near chance while a model that sees the sequence
     can still rank -- if neither can, the task is order-blind and R-perm fails too.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import sampler as S          # noqa: E402
from run_pilot import make_factors  # noqa: E402


def main() -> int:
    import mujoco_adapter as MU
    rng = np.random.default_rng(0)
    facs = [make_factors(rng, i) for i in range(24)]
    cfg = S.SamplerConfig(K=5, H=6, plan_len=3, T=15, rule="R-perm", seed=0)
    ds = S.generate(MU.MuJoCoPush3D(substeps=50), facs, cfg=cfg,
                    ids=["r%02d" % i for i in range(24)])

    steps = [st for ep in ds["episodes"] for st in ep["steps"]]
    u = P._unit(-P.features_from(ds).obs[:, 12:14])
    plans = np.array([st["plans"] for st in steps])
    y = np.array([st["y_C"] for st in steps])
    fact = plans[:, [0]]
    d = plans - fact
    du = d[..., 0] * u[:, None, None, 0] + d[..., 1] * u[:, None, None, 1]
    sums = du.sum(axis=2)
    print("total aligned push per arm: |mean over arms| = %.2e   sd across arms = %.2e"
          % (float(np.abs(sums.mean(axis=1)).mean()), float(np.std(sums, axis=1).mean())))

    deg = float(np.mean([st["degenerate_C"] for st in steps]))
    spread = float(np.mean(y.max(axis=1) - y.min(axis=1)))
    print("degenerate rate = %.3f   mean across-arm spread of y_C = %.4f"
          % (deg, spread))
    print("mean |y_C| = %.4f  (phi_ref %.4f)" % (float(np.mean(np.abs(y))),
                                                 ds["meta"]["phi_ref"]))

    sp = {}
    # split by episode order (cheap): first 16 train, 4 val, 4 test
    ids = [ep["episode_id"] for ep in ds["episodes"]]
    tr, va, te = ids[:16], ids[16:20], ids[20:]
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])

    rows = [
        ("G0 perturbation magnitude", P.HeuristicMagnitude()),
        ("G0b summed aligned push", P.HeuristicAlignedSum()),
        ("G1 global direction", P.GlobalDirection()),
        ("M1 L3 a0.1", P.StructuredPlan(level=3, alpha=0.1)),
        ("R1 one-step oracle", P.Myopic()),
    ]
    def f(m, k):
        v = (m.get(k) or {}).get("value")
        return "   -  " if v is None else "%6.4f" % float(v)

    print("\n%-30s %6s %8s %6s %6s" % ("ranker", "DAR", "regret", "zero", "usable"))
    for label, pred in rows:
        r = P.evaluate_predictor(pred, ds, convention="C", train_ids=tr,
                                 test_ids=te, spec=spec, eps=eps, n_boot=0)
        m = r["report"]["metrics"]
        print("%-30s %s %s %s %s"
              % (label, f(m, "dar"), f(m, "regret_norm"), f(m, "zero_regret_rate"),
                 f(m, "n_units_usable")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
