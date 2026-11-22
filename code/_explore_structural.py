"""Scratch probe: can a STRUCTURED plan model beat the one-step oracle?

Not a deliverable. It answers one question before anything goes into predictors.py:
is there a model whose FORM encodes the task's causal factorisation -- plan
deviation -> object displacement -> potential -- and does that form beat

    B1 bet-on-expert   0.63
    B2 ridge           0.44   (linear, no interactions: already below chance)
    R1 one-step        0.87   (perfect one-step information, needs the simulator)

The probe adds ONE mechanism at a time, in the order physics suggests, so that a
gain can be attributed to a mechanism instead of to a lucky feature set:

    V1 frame        deviations resolved in the object->target frame (u, n)
    V2 + abs        |lateral| -- pushing off the line hurts regardless of side
    V3 + cum        SUMMED aligned push -- displacement accumulates down the plan
    V4 + phi        aligned push x current potential -- forward value falls as
                    the object nears the target (overshoot)
    V5 + sq         square of the summed push -- saturation/overshoot curvature
    V6 + gate       contact gate (control: contact is ~100% here, must be a no-op)

Selection is on VAL; TEST is reported for the number that would go in the paper.
Every variant is printed, losers included.

    python _explore_structural.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M      # noqa: E402
import predictors as P   # noqa: E402
import sampler as S      # noqa: E402

DATA = os.path.abspath(os.path.join(HERE, os.pardir, "data"))
DS = os.path.join(DATA, "pilot_rsamp_m3_H6_48ep.json")

I_REL, I_DOT, I_DPO, I_PHI = 12, 16, 17, 18


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


def design(tab: P.FeatureTable, level: int) -> np.ndarray:
    """Ladder of structural mechanisms, level 1..6 = V1..V6."""
    obs = tab.obs
    u = _unit(-obs[:, I_REL:I_REL + 2])                  # object -> target
    nvec = np.stack([-u[:, 1], u[:, 0]], axis=1)
    ux, uy = u[:, None, None, 0], u[:, None, None, 1]
    nx, ny = nvec[:, None, None, 0], nvec[:, None, None, 1]

    d = tab.plans - tab.plans[:, [0], :, :]              # (n, K, m, 3)
    du = d[..., 0] * ux + d[..., 1] * uy                 # aligned  (+ = more push)
    dn = d[..., 0] * nx + d[..., 1] * ny                 # lateral
    nrow, K, m = du.shape

    g = 1.0 / (1.0 + np.exp((obs[:, I_DPO] - 0.020) / 0.010)) if level >= 6 \
        else np.ones(nrow)
    G = g[:, None]

    blocks = [du, dn]
    if level >= 2:
        blocks.append(np.abs(dn))
    if level >= 3:
        blocks.append(np.cumsum(du, axis=2)[:, :, [-1]])
        blocks.append(np.abs(np.cumsum(dn, axis=2))[:, :, [-1]])
    if level >= 4:
        phi = obs[:, I_PHI][:, None, None]
        blocks.append(np.cumsum(du, axis=2)[:, :, [-1]] * phi)
    if level >= 5:
        cs = np.cumsum(du, axis=2)[:, :, [-1]]
        blocks.append(cs ** 2)
    if level == 6:
        # Deviation blocks only: the LEVEL block must not be gated, or the model's
        # absolute scale would depend on contact.
        blocks = [b * G[:, :, None] for b in blocks]

    F = [b.reshape(nrow, K, -1) for b in blocks]
    # level block: row-constant, cannot move a ranking; keeps nMAE on the Y scale
    lev = np.stack([np.ones(nrow), obs[:, I_PHI], obs[:, I_DOT],
                    1.0 / (1.0 + obs[:, I_DOT])], axis=1)
    n_lev = lev.shape[1]
    X = np.concatenate([np.broadcast_to(lev[:, None, :], (nrow, K, n_lev))]
                       + F, axis=-1)
    return X


class StructuredPlan:
    kind = "method"

    def __init__(self, level: int, alpha: float = 1.0, name: str | None = None):
        self.level = int(level)
        self.alpha = float(alpha)
        self.name = name or ("M V%d" % level)
        self.w = None
        self.mu = None
        self.sd = None

    def fit(self, train: P.FeatureTable, y_train: np.ndarray):
        X = design(train, self.level)
        y = np.asarray(y_train, dtype=np.float64)
        flat, tgt = X.reshape(-1, X.shape[-1]), y.reshape(-1)
        self.mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        z = (flat - self.mu) / self.sd
        A = z.T @ z + self.alpha * np.eye(z.shape[1])
        b = z.T @ (tgt - tgt.mean())
        self.w = np.concatenate([[tgt.mean()], np.linalg.solve(A, b)])
        return self

    def predict(self, test: P.FeatureTable) -> np.ndarray:
        X = design(test, self.level)
        z = (X.reshape(-1, X.shape[-1]) - self.mu) / self.sd
        return (self.w[0] + z @ self.w[1:]).reshape(X.shape[0], X.shape[1])


def line(rep: dict, label: str) -> str:
    m = rep["metrics"]

    def v(k, d=float("nan")):
        c = m.get(k)
        x = None if c is None else c.get("value")
        return d if x is None else float(x)
    return ("%-26s DAR=%.4f  rho=%+.4f  regret=%.4f  zero=%.3f  nMAE=%.3f"
            % (label[:26], v("dar"), v("rho"), v("regret_norm"),
               v("zero_regret_rate"), v("nmae")))


def main() -> int:
    ds = S.load(DS)
    print("dataset:", DS)
    print("meta:", {k: ds["meta"][k] for k in
                    ("spec_version", "K", "H", "plan_len", "phi_ref")})
    print("scene:", {k: ds["meta"]["adapter"][k] for k in
                     ("pusher_start_offset", "expert_standoff")})
    sp = {}
    for ep in ds["episodes"]:
        sp.setdefault(ep["split"], []).append(ep["episode_id"])
    print("split:", {k: len(v) for k, v in sp.items()})

    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])
    for tag in ("val", "test"):
        ev = sp[tag]
        print("\n=== %s (%d episodes) ===" % (tag.upper(), len(ev)))
        for pred in (P.FactualWins(), P.Ridge(seed=0), P.RidgeArmShuffled(seed=0),
                     P.Myopic()):
            r = P.evaluate_predictor(pred, ds, convention="C", train_ids=sp["train"],
                                     test_ids=ev, spec=spec, eps=eps, n_boot=200)
            print("  " + line(r["report"], r["name"]))
        for lvl in (1, 2, 3, 4, 5, 6):
            for alpha in (0.1, 1.0, 10.0):
                r = P.evaluate_predictor(StructuredPlan(lvl, alpha),
                                         ds, convention="C",
                                         train_ids=sp["train"], test_ids=ev,
                                         spec=spec, eps=eps, n_boot=200)
                print("  " + line(r["report"], "M V%d a=%g" % (lvl, alpha)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
