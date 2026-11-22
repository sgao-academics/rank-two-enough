"""Diagnose C2: a negative control that scores 0.91 is a bug or a finding.

Three questions, answered numerically:
  1. does fit() actually permute the design rows?
  2. if it does, why do the fitted deviation weights still rank arms correctly?
  3. what does DAR look like when the deviation block is explicitly zeroed?
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
N_LEV = 4


def main() -> int:
    ds = S.load(DS)
    sp = {}
    for ep in ds["episodes"]:
        sp.setdefault(ep["split"], []).append(ep["episode_id"])
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])

    ft = P.features_from(ds, sp["train"])
    y = P.truth_matrix(ds, "C", sp["train"])
    x = P._plan_design(ft, 3)
    print("design:", x.shape, " level block width:", N_LEV)

    rng = np.random.default_rng(0)
    perm = np.argsort(rng.random(y.shape), axis=1)
    xp = np.take_along_axis(x, perm[:, :, None], axis=1)
    print("permuted rows identical to originals?",
          np.allclose(xp, x),
          "| frac of steps where perm is identity: %.3f"
          % float(np.mean(np.all(perm == np.arange(perm.shape[1]), axis=1))))

    for label, pred in (("M1 L3", P.StructuredPlan(level=3)),
                        ("C2", P.StructuredPlanShuffled(level=3))):
        pred.fit(ft, y)
        w = pred.w[1:]
        dev = w[N_LEV:]
        print("%-6s |w|=%6.3f  |w_dev|=%6.3f  dev/lev=%.3f  sign pattern=%s"
              % (label, float(np.linalg.norm(w)), float(np.linalg.norm(dev)),
                 float(np.linalg.norm(dev) / max(np.linalg.norm(w), 1e-12)),
                 np.sign(dev).astype(int)))

    # what DAR does a model with the deviation block zeroed produce?
    m1 = P.StructuredPlan(level=3).fit(ft, y)
    ft_te = P.features_from(ds, sp["test"])
    res = P.evaluate_predictor(P.StructuredPlan(level=3), ds, convention="C",
                               train_ids=sp["train"], test_ids=sp["test"],
                               spec=spec, eps=eps, n_boot=0)
    print("\nM1 DAR=%.4f" % res["report"]["metrics"]["dar"]["value"])

    # zero the deviation coefficients and re-score, using the same fitted object
    class Zeroed(P.StructuredPlan):
        def predict(self, test):
            X = P._plan_design(test, self.level)
            z = (X.reshape(-1, X.shape[-1]) - self.mu) / self.sd
            w = self.w.copy()
            w[1 + N_LEV:] = 0.0
            return (w[0] + z @ w[1:]).reshape(X.shape[0], X.shape[1])

    z = Zeroed(level=3).fit(ft, y)
    r2 = P.evaluate_predictor(z, ds, convention="C", train_ids=sp["train"],
                              test_ids=sp["test"], spec=spec, eps=eps, n_boot=0)
    print("M1 with deviation block zeroed: DAR=%.4f (ties -> must be ~0)"
          % r2["report"]["metrics"]["dar"]["value"])

    # does the level block alone vary across arms? it must not.
    print("level block arm-constant:", bool(np.allclose(x[:, :, :N_LEV],
                                                        x[:, [0], :N_LEV])))
    # how much of the within-step variation is deviation vs level?
    dev_part = x[:, :, N_LEV:]
    print("within-step sd: level=%.2e  deviation=%.2e"
          % (float(np.std(x[:, :, :N_LEV] - x[:, [0], :N_LEV])),
             float(np.std(dev_part - dev_part[:, [0], :]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
