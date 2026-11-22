"""Is M1 identical to G1 at the same alpha?

After the deviation blocks are centred, the level block (row-constant) is orthogonal
to them, so the ridge solution for the deviation coefficients is the same whether or
not the level block is present. M1 therefore reduces to "one global direction over
the frame-projected deviation features" -- which is exactly what G1 is. If that is
true, then "M1 vs G1" is not a comparison of two models and must never be reported
as one; the informative contrast is between FEATURE MAPS (level 0 vs 3 vs raw).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import predictors as P   # noqa: E402
import sampler as S      # noqa: E402

DS = os.path.abspath(os.path.join(HERE, os.pardir, "data",
                                  "pilot_rsamp_m3_H6_48ep.json"))


def main() -> int:
    ds = S.load(DS)
    sp = {}
    for ep in ds["episodes"]:
        sp.setdefault(ep["split"], []).append(ep["episode_id"])
    ft_tr = P.features_from(ds, sp["train"])
    y_tr = P.truth_matrix(ds, "C", sp["train"])
    ft_te = P.features_from(ds, sp["test"])

    for alpha in (0.1, 1.0, 10.0):
        m1 = P.StructuredPlan(level=3, alpha=alpha).fit(ft_tr, y_tr)
        g1 = P.GlobalDirection(alpha=alpha).fit(ft_tr, y_tr)
        pm = m1.predict(ft_te)
        pg = g1.predict(ft_te)
        dm = pm - pm[:, [0]]
        dg = pg - pg[:, [0]]
        print("alpha=%-5g  max|pred diff|=%.2e  rank-identical=%.3f  "
              "max|Delta diff|=%.2e"
              % (alpha, float(np.abs(pm - pg).max()),
                 float(np.mean([np.array_equal(np.argsort(a), np.argsort(b))
                                for a, b in zip(pm, pg)])),
                 float(np.abs(dm - dg).max())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
