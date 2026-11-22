"""Which interaction terms does the generic quadratic actually use?

B4 (all 560 degree-<=2 terms) beats the 15-parameter structural model by ~0.025 DAR.
That gap is the only thing structure has not explained, so it is worth decomposing
instead of leaving as "B4 is better". Split the quadratic into its three blocks:

  main      [obs, plan, is_factual]                      32 terms   (= B2)
  obs x plan  every state-action product                 22*9 = 198
  plan x plan every within-plan product                  9*10/2 = 45

If `obs x plan` alone reproduces B4 while `plan x plan` alone collapses to B2, then
the entire nonlinear gain is the bilinear state-action term -- and the structural
model is a hand-built rank-1 slice of exactly that block. That converts "structure
is cheaper" into "structure writes down the term that matters, and leaves 180
redundant copies of it to the learner".

If instead plan x plan carries the gain, the story is about curvature in the plan,
which is a different mechanism and a different paper.
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


class _Blocks(P.Ridge):
    """Base design split into [obs part, plan part, is_factual] so the blocks can
    be recombined selectively."""

    def _parts(self, tab):
        n, k, m, a = tab.plans.shape
        d = tab.obs.shape[1]
        steps = m if self.plan_steps is None else int(self.plan_steps)
        is_fact = np.zeros((n, k, 1))
        is_fact[:, 0, 0] = 1.0
        obs = np.broadcast_to(tab.obs[:, None, :], (n, k, d))
        plan = tab.plans[:, :, :steps, :].reshape(n, k, steps * a)
        return obs, plan, is_fact


class MainOnly(_Blocks):
    name = "T1 main only (=B2)"

    def _design(self, tab):
        o, p, f = self._parts(tab)
        return np.concatenate([o, p, f], axis=-1)


class ObsXPlan(_Blocks):
    name = "T2 main + obs x plan"

    def _design(self, tab):
        o, p, f = self._parts(tab)
        n, k, do = o.shape
        dp = p.shape[-1]
        cross = (o[:, :, :, None] * p[:, :, None, :]).reshape(n, k, do * dp)
        return np.concatenate([o, p, f, cross], axis=-1)


class PlanXPlan(_Blocks):
    name = "T3 main + plan x plan"

    def _design(self, tab):
        o, p, f = self._parts(tab)
        n, k, dp = p.shape
        ii, jj = np.triu_indices(dp)
        quad = (p[:, :, :, None] * p[:, :, None, :]).reshape(n, k, dp * dp)
        quad = quad[:, :, np.ravel_multi_index((ii, jj), (dp, dp))]
        return np.concatenate([o, p, f, quad], axis=-1)


class ObsXObs(_Blocks):
    name = "T4 main + obs x obs (state curvature only)"

    def _design(self, tab):
        o, p, f = self._parts(tab)
        n, k, do = o.shape
        ii, jj = np.triu_indices(do)
        quad = (o[:, :, :, None] * o[:, :, None, :]).reshape(n, k, do * do)
        quad = quad[:, :, np.ravel_multi_index((ii, jj), (do, do))]
        return np.concatenate([o, p, f, quad], axis=-1)


def main():
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])
    models = (("T1 main only (=B2)", lambda: MainOnly()),
              ("T2 main + obs x plan", lambda: ObsXPlan()),
              ("T3 main + plan x plan", lambda: PlanXPlan()),
              ("T4 main + obs x obs", lambda: ObsXObs()),
              ("B4 full poly2", lambda: P.RidgePoly()),
              ("G1 global direction", lambda: P.GlobalDirection()),
              ("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1)))
    acc, nprm = {}, {}
    for cfg in ("m3",):
        for seed in range(12):
            path = RS.ds_path(cfg, seed)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                ds = json.load(fh)
            ft = None
            for sp in range(5):
                rows = SP.assign(ds["episodes"], seed=sp)
                ids = {"train": [], "val": [], "test": []}
                for r in rows:
                    ids[r["split"]].append(r["episode_id"])
                if not (ids["train"] and ids["test"]):
                    continue
                if ft is None:
                    tr_ft = P.features_from(ds, ids["train"])
                    tr_y = P.truth_matrix(ds, "C", ids["train"])
                    for label, factory in models:
                        mdl = factory().fit(tr_ft, tr_y)
                        nprm[label] = int(len(mdl.w)) - 1
                for label, factory in models:
                    res = P.evaluate_predictor(
                        factory(), ds, convention="C", train_ids=ids["train"],
                        test_ids=ids["test"], spec=spec, eps=eps, n_boot=0)
                    acc.setdefault(label, []).append(
                        float(R.summarize(res["report"], res["name"])["dar"]))
            print("  seed %02d done" % seed, flush=True)

    print("\n" + "=" * 70)
    print("INTERACTION-TERM ABLATION  (m3, 12 seeds x 5 template splits)")
    print("=" * 70)
    print("%-32s %7s %8s %8s" % ("design", "params", "DAR", "sd"))
    for label, _f in models:
        v = acc.get(label, [])
        if not v:
            continue
        print("%-32s %7d %8.4f %8.4f"
              % (label, nprm.get(label, 0), mean(v),
                 stdev(v) if len(v) > 1 else 0.0))

    try:
        base = mean(acc["T1 main only (=B2)"])
        print("\nrecovery of the full quadratic's gain over T1:")
        full = mean(acc["B4 full poly2"]) - base
        for label in ("T2 main + obs x plan", "T3 main + plan x plan",
                      "T4 main + obs x obs"):
            g = mean(acc[label]) - base
            print("  %-28s %+.4f  (%.0f%% of B4's %+.4f)"
                  % (label, g, 100.0 * g / full if abs(full) > 1e-9 else float("nan"),
                     full))
    except KeyError:
        pass

    with open(os.path.join(RS.ROOT, "logs", "ablation_terms.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"params": nprm, "dar": {k: [mean(v), stdev(v), len(v)]
                                           for k, v in acc.items()}},
                  fh, indent=1)
    print("\nwrote logs/ablation_terms.json")


if __name__ == "__main__":
    main()
