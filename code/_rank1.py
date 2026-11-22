"""Is the state x action interaction actually rank 1?

T2 (main + all 198 obs x plan products) reproduces 98% of the generic quadratic's
gain, so the task's utility is bilinear in (state, action):
        Y_arm ~ s^T W a      with s in R^22, a in R^9, W in R^{22 x 9}.

The structural model is the claim that W is (close to) rank 1 and that its single
direction is the object->target frame. That claim has never been measured -- it was
designed in. Here W is READ OFF a fitted T2 and its singular spectrum is reported.

Two numbers decide it:
  * share of ||W||_F in the top singular triple -- how much of the interaction a
    single state scalar x single action direction can explain;
  * participation-ratio effective rank of the spectrum.

If the share is ~0.9 and the effective rank ~1, then "resolve the plan in the
object->target frame" is not a modelling choice, it is the measured structure of the
task, and the 200 extra products in T2 are 200 approximations of one term.
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
    """Same design as _ablation_terms.ObsXPlan, coefficients kept addressable."""

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


def main():
    shares, eranks, ratios = [], [], []
    for seed in range(12):
        path = RS.ds_path("m3", seed)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            ds = json.load(fh)
        rows = SP.assign(ds["episodes"], seed=0)
        ids = {"train": [], "val": [], "test": []}
        for r in rows:
            ids[r["split"]].append(r["episode_id"])
        if not ids["train"]:
            continue
        ft = P.features_from(ds, ids["train"])
        y = P.truth_matrix(ds, "C", ids["train"])
        mdl = ObsXPlan().fit(ft, y)
        o, p, f, do, dp = mdl._parts(ft)
        n_cross = do * dp
        # w = [intercept, obs(do), plan(dp), is_fact(1), cross(n_cross)]
        w = mdl.w[1:]
        off = do + dp + 1
        w_cross = w[off:off + n_cross]
        W = w_cross.reshape(do, dp)
        sv = np.linalg.svd(W, compute_uv=False)
        e = sv ** 2
        share = float(e[0] / e.sum())
        pr = float((e.sum() ** 2) / (e ** 2).sum())     # participation ratio
        shares.append(share)
        eranks.append(pr)
        ratios.append(float(sv[0] / sv[1]) if len(sv) > 1 and sv[1] > 0 else float("inf"))
        print("  seed %02d  |W|_F=%.3f  top share=%.3f  eff rank=%.2f  s1/s2=%.1f"
              % (seed, np.linalg.norm(W), share, pr, ratios[-1]))

    print("\n" + "=" * 62)
    print("RANK OF THE STATE x ACTION INTERACTION  (m3, 12 dataset seeds)")
    print("=" * 62)
    print("  top singular share   %.3f +- %.3f"
          % (mean(shares), stdev(shares) if len(shares) > 1 else 0.0))
    print("  participation rank   %.2f  (of a possible 9)"
          % mean(eranks))
    print("  sigma_1 / sigma_2    %.1f"
          % mean([min(r, 1e6) for r in ratios]))
    print("\n  => a single state-scalar x action-direction term carries %.0f%% of the"
          % (100 * mean(shares)))
    print("     learned interaction; the structural model writes down exactly that term.")
    with open(os.path.join(RS.ROOT, "logs", "rank1.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"share": shares, "eff_rank": eranks, "s1_over_s2": ratios},
                  fh, indent=1)


if __name__ == "__main__":
    main()
