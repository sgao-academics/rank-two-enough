"""Row-level verification: is the arm permutation doing what the comment says?

Prints, for the first training step, the feature row BEFORE and AFTER the
permutation, the permutation itself, and then the correlation between the shuffled
model's *predicted* arm deltas and the true arm deltas on the test split. If those
correlate, the shuffle is not destroying the plan-outcome pairing, whatever the
cosine of the raw weight vectors says.
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
    ft = P.features_from(ds, sp["train"])
    y = P.truth_matrix(ds, "C", sp["train"])
    x = P._plan_design(ft, 3)

    rng = np.random.default_rng(12345)
    perm = np.argsort(rng.random(y.shape), axis=1)
    xp = np.take_along_axis(x, perm[:, :, None], axis=1)
    print("perm[0] =", perm[0], " identity?", list(perm[0]) == list(range(5)))
    print("x[0,:,9] (cum_u)      =", np.round(x[0, :, 9], 4))
    print("xp[0,:,9] (after perm)=", np.round(xp[0, :, 9], 4))
    print("y[0]                  =", np.round(y[0], 5))
    print("y[0] permuted by perm =", np.round(y[0][perm[0]], 5))

    # fit on the permuted design
    class Shuf(P.StructuredPlan):
        pass

    p = Shuf(level=3, alpha=0.1)
    p.fit(ft, y)
    # p.fit permutes internally with seed 0; refit manually to keep control
    mu = xp.reshape(-1, x.shape[-1]).mean(0)
    sd = xp.reshape(-1, x.shape[-1]).std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = (xp.reshape(-1, x.shape[-1]) - mu) / sd
    tgt = y.reshape(-1)
    a = z.T @ z + 0.1 * np.eye(z.shape[1])
    b = z.T @ (tgt - tgt.mean())
    w = np.concatenate([[tgt.mean()], np.linalg.solve(a, b)])

    ft_te = P.features_from(ds, sp["test"])
    xt = P._plan_design(ft_te, 3)
    zt = (xt.reshape(-1, xt.shape[-1]) - mu) / sd
    pred = (w[0] + zt @ w[1:]).reshape(xt.shape[0], xt.shape[1])
    yt = P.truth_matrix(ds, "C", sp["test"])
    dp = pred - pred[:, [0]]
    dt = yt - yt[:, [0]]
    print("\ncorr(Delta yhat, Delta y) on test = %+.4f"
          % float(np.corrcoef(dp.reshape(-1), dt.reshape(-1))[0, 1]))
    print("mean sign agreement (DAR, manual)=%.4f"
          % float(np.mean(np.sign(dp) == np.sign(dt))))
    print("within-step sd of prediction: %.3e  (true %.3e)"
          % (float(np.mean(np.std(pred, axis=1))), float(np.mean(np.std(yt, axis=1)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
