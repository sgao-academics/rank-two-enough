"""Does the SEQUENCE of a plan matter at all, once its total push is fixed?

The suite is about to spend hours comparing structural models. Before that, one
question decides whether the comparison can mean anything:

    if I hold sum(aligned push) fixed and only reorder the plan, does the truth
    move?

If it does not, the task's ranking is a function of one scalar, no amount of
"multi-step reasoning" exists to be demonstrated, and a 0-parameter heuristic is the
correct model. Measured three ways:

  1. out-of-sample R^2 of Delta y on the single scalar sum(du)
  2. out-of-sample R^2 on all 11 structural features (how much is left)
  3. among arm pairs with nearly equal sum(du), how often does the truth differ by
     more than 10% of the typical |Delta y|? (order-sensitive pairs)
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


def feats(tab):
    obs = tab.obs
    u = P._unit(-obs[:, 12:14])
    d = tab.plans - tab.plans[:, [0]]
    du = d[..., 0] * u[:, None, None, 0] + d[..., 1] * u[:, None, None, 1]
    return du


def r2(y, yhat, m):
    r = y - yhat
    return 1.0 - float((r[m] ** 2).sum() / (y[m] ** 2).sum())


def main() -> int:
    ds = S.load(DS)
    sp = {}
    for ep in ds["episodes"]:
        sp.setdefault(ep["split"], []).append(ep["episode_id"])

    out = {}
    for part in ("train", "test"):
        ft = P.features_from(ds, sp[part])
        y = P.truth_matrix(ds, "C", sp[part])
        du = feats(ft)
        # within-step contrasts, the only thing a ranking can use
        yc = y - y.mean(axis=1, keepdims=True)
        s = du.sum(axis=2)
        sc = s - s.mean(axis=1, keepdims=True)
        X = P._plan_design(ft, 3)[:, :, 4:]
        Xc = (X - X.mean(axis=1, keepdims=True)).reshape(-1, X.shape[-1])
        out[part] = dict(y=y, yc=yc, sc=sc, Xc=Xc, du=du)

    yt, yct = out["test"]["y"], out["test"]["yc"]
    # -- 1. one scalar, fitted on train ---------------------------------- #
    str_ = out["train"]["sc"].reshape(-1)
    ytr = out["train"]["yc"].reshape(-1)
    beta = float((str_ @ ytr) / (str_ @ str_))
    pred1 = beta * out["test"]["sc"]
    print("scalar  sum(du): beta=%.4f  test R2(within-step)=%.4f"
          % (beta, r2(yct.reshape(-1), pred1.reshape(-1), slice(None))))

    # -- 2. full structural design, fitted on train ---------------------- #
    m1 = P.StructuredPlan(level=3, alpha=0.1).fit(P.features_from(ds, sp["train"]),
                                                  P.truth_matrix(ds, "C", sp["train"]))
    ft_te = P.features_from(ds, sp["test"])
    pt = m1.predict(ft_te)
    pct = pt - pt.mean(axis=1, keepdims=True)
    print("M1 (11 features): test R2(within-step)=%.4f"
          % r2(yct.reshape(-1), pct.reshape(-1), slice(None)))

    # -- 3. order-sensitive arm pairs ------------------------------------ #
    sc = out["test"]["sc"]
    yc = out["test"]["yc"]
    k = sc.shape[1]
    scale = float(np.median(np.abs(yc)))
    close, differing = 0, 0
    diffs = []
    for i in range(sc.shape[0]):
        for a in range(1, k):
            for b in range(a + 1, k):
                if abs(sc[i, a] - sc[i, b]) <= 0.02 * max(np.ptp(sc[i]), 1e-9):
                    close += 1
                    d = abs(yc[i, a] - yc[i, b])
                    diffs.append(d)
                    differing += int(d > 0.10 * scale)
    print("\nvirtually-equal sum(du) pairs: %d  of which truth differs by >10%% of "
          "the median |Delta y|: %d (%.1f%%)"
          % (close, differing, 100.0 * differing / max(close, 1)))
    if diffs:
        print("  median |Delta y| gap within those pairs = %.4f  (typical |Delta y| "
              "= %.4f)" % (float(np.median(diffs)), scale))

    # the same, without the 2% tolerance: nearest-neighbour pairs
    print("\nfor context: median |sum(du)| across arms = %.4f, sd = %.4f"
          % (float(np.median(np.abs(sc))), float(np.std(sc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
