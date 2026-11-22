"""Fast API smoke for the predictors added for the suite. No dataset generation.

Catches shape/plumbing errors in seconds instead of after a 90-second generation.
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

DS = os.path.abspath(os.path.join(HERE, os.pardir, "data",
                                  "pilot_rsamp_m3_H6_48ep.json"))


def main() -> int:
    ds = S.load(DS)
    sp = {}
    for ep in ds["episodes"]:
        sp.setdefault(ep["split"], []).append(ep["episode_id"])
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])

    tests = [
        ("B3 ridge first-step only", P.Ridge(plan_steps=1), None),
        ("B4 ridge poly2", P.RidgePoly(), None),
        ("L1 learned one-step (linear)", P.LearnedOneStepLinear(), "S"),
        ("L2 learned one-step (frame)", P.LearnedOneStepFrame(), "S"),
        ("M1 L0", P.StructuredPlan(level=0), None),
        ("M1 L3", P.StructuredPlan(level=3), None),
        ("M1 L3 transfer", P.StructuredPlan(level=3), "C"),
        ("C2 shuffled", P.StructuredPlanShuffled(level=3), None),
    ]
    ok = True
    for label, pred, tc in tests:
        try:
            res = P.evaluate_predictor(pred, ds, convention="C",
                                       train_ids=sp["train"], test_ids=sp["test"],
                                       spec=spec, eps=eps, n_boot=0,
                                       train_convention=tc)
            m = res["report"]["metrics"]
            print("%-32s DAR=%.4f regret=%.4f nMAE=%.3f  tc=%s"
                  % (label, m["dar"]["value"], m["regret_norm"]["value"],
                     m["nmae"]["value"], res["info"]["train_convention"]))
        except Exception as exc:                                  # noqa: BLE001
            ok = False
            print("%-32s ERROR %s: %s" % (label, type(exc).__name__, exc))

    # feature-count sanity: poly must be strictly richer than linear
    ft = P.features_from(ds, sp["train"])
    print("\ndesign widths: linear=%d  poly2=%d  M1-L3=%d"
          % (P.Ridge()._design(ft).shape[-1], P.RidgePoly()._design(ft).shape[-1],
             P._plan_design(ft, 3).shape[-1]))
    # the level block must not be able to move a ranking: it is arm-constant
    X = P._plan_design(ft, 3)
    lev = X[:, :, :4]
    const_across_arms = np.allclose(lev, lev[:, [0], :])
    print("level block arm-constant:", const_across_arms)
    return 0 if ok and const_across_arms else 1


if __name__ == "__main__":
    raise SystemExit(main())
