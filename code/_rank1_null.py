"""Null control for the rank measurement.

The fitted W is a 22x9 block of ridge coefficients. Its singular spectrum can look
"structured" for the trivial reason that 198 coefficients fitted to 5000 rows carry
their own sampling noise, and a noise matrix of that shape also has a dominant
singular value. So the share (0.502) is only meaningful against the spectrum of a W
fitted to data in which the state x action relation has been destroyed.

Null: permute the K outcomes within each step before fitting (the C1 intervention).
The marginals of Y are untouched, only the arm -> outcome correspondence is broken.

If the null's share is ~0.5 as well, the measurement says nothing and must be
dropped from the paper rather than reported as "the interaction is rank 3".
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

import predictors as P       # noqa: E402
import split as SP           # noqa: E402
import run_suite as RS       # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")


class ObsXPlan(P.Ridge):
    name = "T2 main + obs x plan"

    def _parts(self, tab):
        n, k, m, a = tab.plans.shape
        d = tab.obs.shape[1]
        is_fact = np.zeros((n, k, 1))
        is_fact[:, 0, 0] = 1.0
        obs = np.broadcast_to(tab.obs[:, None, :], (n, k, d))
        plan = tab.plans.reshape(n, k, m * a)
        return obs, plan, is_fact, d, m * a

    def _design(self, tab):
        o, p, f, do, dp = self._parts(tab)
        n, k, _ = o.shape
        cross = (o[:, :, :, None] * p[:, :, None, :]).reshape(n, k, do * dp)
        return np.concatenate([o, p, f, cross], axis=-1)


def spectrum(mdl, ft):
    o, p, f, do, dp = mdl._parts(ft)
    off = do + dp + 1
    W = mdl.w[1:][off:off + do * dp].reshape(do, dp)
    sv = np.linalg.svd(W, compute_uv=False)
    e = sv ** 2
    return float(e[0] / e.sum()), float((e.sum() ** 2) / (e ** 2).sum())


def main():
    real, null = [], []
    for seed in range(12):
        path = RS.ds_path("m3", seed)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            ds = json.load(fh)
        rows = SP.assign(ds["episodes"], seed=0)
        ids = [r["episode_id"] for r in rows if r["split"] == "train"]
        if not ids:
            continue
        ft = P.features_from(ds, ids)
        y = P.truth_matrix(ds, "C", ids)
        real.append(spectrum(ObsXPlan().fit(ft, y), ft))

        rng = np.random.default_rng(1000 + seed)
        perm = np.argsort(rng.random(y.shape), axis=1)
        y_null = np.take_along_axis(y, perm, axis=1)
        null.append(spectrum(ObsXPlan().fit(ft, y_null), ft))
        print("  seed %02d  real share=%.3f rank=%.2f | null share=%.3f rank=%.2f"
              % (seed, real[-1][0], real[-1][1], null[-1][0], null[-1][1]))

    rs = [x[0] for x in real]
    ns = [x[0] for x in null]
    rr = [x[1] for x in real]
    nr = [x[1] for x in null]
    print("\n" + "=" * 64)
    print("RANK MEASUREMENT vs ITS NULL (arm-shuffled outcomes)")
    print("=" * 64)
    print("  top share   real %.3f +- %.3f   null %.3f +- %.3f"
          % (mean(rs), stdev(rs), mean(ns), stdev(ns)))
    print("  eff rank    real %.2f          null %.2f"
          % (mean(rr), mean(nr)))
    gap = mean(rs) - mean(ns)
    print("\n  gap = %+.3f  =>  %s"
          % (gap, "measurement is INFORMATIVE" if gap > 0.10
             else "measurement is INDISTINGUISHABLE FROM NOISE -- drop it"))
    with open(os.path.join(RS.ROOT, "logs", "rank1_null.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"real_share": rs, "null_share": ns,
                   "real_rank": rr, "null_rank": nr}, fh, indent=1)


if __name__ == "__main__":
    main()
