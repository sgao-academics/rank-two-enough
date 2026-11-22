"""Is the ranking task effectively ONE-DIMENSIONAL?

If a single fixed direction in plan-deviation space, or even a plain magnitude
heuristic, already ranks the K arms almost perfectly, then:
  - M1's state-dependent structure is decoration,
  - R1's 0.87 is a property of the metric rather than of one-step information,
  - and the negative control cannot fail, because almost any reasonable direction
    ranks well.
That would invalidate the "multi-step structure beats one-step oracle" claim
regardless of how pretty the DAR table looks, so it is measured BEFORE the suite.

Rankers tested (all fitted on TRAIN only, scored on TEST):
    magnitude     ||Delta a|| ascending -- the perturbation size, nothing else
    cum_aligned   summed aligned push, no state interaction
    G1            one global linear direction over the deviation features
    R1 / M1       the references already in the ladder
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


class GlobalDirection:
    """G1: w . deviation_features, no obs, no level block. A pure ranking direction."""
    name = "G1 global direction (no state)"
    kind = "baseline"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w = None
        self.mu = None
        self.sd = None

    def _x(self, tab):
        X = P._plan_design(tab, 3)[:, :, 4:]          # drop the 4 level columns
        return X

    def fit(self, train, y_train):
        X = self._x(train)
        y = np.asarray(y_train, float)
        # fit on WITHIN-ROW contrasts only: subtract the row mean of both X and y,
        # so the fitted direction describes ranking and cannot absorb the row level
        Xc = X - X.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        flat, tgt = Xc.reshape(-1, Xc.shape[-1]), yc.reshape(-1)
        self.mu = flat.mean(0)
        sd = flat.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        z = (flat - self.mu) / self.sd
        A = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.w = np.linalg.solve(A, z.T @ tgt)
        return self

    def predict(self, test):
        X = self._x(test)
        z = (X.reshape(-1, X.shape[-1]) - self.mu) / self.sd
        return (z @ self.w).reshape(X.shape[0], X.shape[1])


class Magnitude:
    """Rank by ||Delta a|| ascending: the perturbation size, nothing else."""
    name = "G0 perturbation magnitude"
    kind = "baseline"

    def fit(self, train, y_train):
        return self

    def predict(self, test):
        d = test.plans - test.plans[:, [0]]
        # total path length of the deviation over the plan, (n, K)
        return -np.linalg.norm(d[..., :2], axis=-1).sum(axis=2)


class CumAligned:
    """Rank by the summed aligned push, using the object->target frame only."""
    name = "G0b summed aligned push"
    kind = "baseline"

    def fit(self, train, y_train):
        return self

    def predict(self, test):
        obs = test.obs
        u = P._unit(-obs[:, 12:14])
        d = test.plans - test.plans[:, [0]]
        du = d[..., 0] * u[:, None, None, 0] + d[..., 1] * u[:, None, None, 1]
        return du.sum(axis=2)


def main() -> int:
    ds = S.load(DS)
    sp = {}
    for ep in ds["episodes"]:
        sp.setdefault(ep["split"], []).append(ep["episode_id"])
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])

    def score(pred, tc=None, conv="C"):
        r = P.evaluate_predictor(pred, ds, convention=conv, train_ids=sp["train"],
                                 test_ids=sp["test"], spec=spec, eps=eps,
                                 n_boot=0, train_convention=tc)
        m = r["report"]["metrics"]
        return m["dar"]["value"], m["regret_norm"]["value"], m["zero_regret_rate"]["value"]

    rows = [
        ("G0 magnitude (fewer |da| is better)", Magnitude()),
        ("G0b summed aligned push", CumAligned()),
        ("G1 global direction a=1", GlobalDirection(1.0)),
        ("G1 global direction a=0.1", GlobalDirection(0.1)),
        ("B2 ridge (full)", P.Ridge()),
        ("M1 L3 a0.1", P.StructuredPlan(level=3, alpha=0.1)),
        ("R1 one-step oracle", P.Myopic()),
    ]
    print("%-38s %6s %8s %6s" % ("ranker", "DAR", "regret", "zero"))
    for label, pred in rows:
        dar, reg, zero = score(pred)
        print("%-38s %6.4f %8.4f %6.3f" % (label, dar, reg, zero))

    # --- intrinsic dimension of the within-step effect -------------------- #
    ft = P.features_from(ds, sp["train"])
    X = P._plan_design(ft, 3)[:, :, 4:]
    Xc = (X - X.mean(axis=1, keepdims=True)).reshape(-1, X.shape[-1])
    sv = np.linalg.svd(Xc, compute_uv=False)
    ev = sv ** 2 / (sv ** 2).sum()
    print("\nwithin-row deviation covariance: top-3 explained variance = %s"
          % np.round(ev[:3], 3))
    print("  effective rank (participation ratio) = %.2f of %d"
          % (float((ev.sum() ** 2) / (ev ** 2).sum()), X.shape[-1]))

    y = P.truth_matrix(ds, "C", sp["train"])
    yc = (y - y.mean(axis=1, keepdims=True)).reshape(-1)
    c = np.corrcoef(Xc.T, yc)[-1, :-1]
    print("  corr(feature, within-step Delta y), train: %s" % np.round(c, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
