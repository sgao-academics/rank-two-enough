"""Probe: does the scripted expert ever actually TOUCH the object?

The standoff in `base_policy` is 0.045 m from the object's centre, the object's
half-size is 0.016-0.024 and the pusher's box half-width is 0.012, so the expert
parks the pusher 0.009-0.017 m SHORT of contact. If that is what happens, then

  - the object never moves along the factual trajectory,
  - mass / friction / half-size never influence the outcome, and
  - the "CausalWorld-style factorisation" claim is empty for this scene,

so this must be measured before any model is built on top of it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import mujoco_adapter as MU   # noqa: E402
import sampler as S           # noqa: E402
from run_pilot import make_factors  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    facs = [make_factors(rng, i) for i in range(4)]
    cfg = S.SamplerConfig(K=5, H=6, plan_len=3, T=6, seed=0)
    ds = S.generate(MU.MuJoCoPush3D(substeps=cfg.control_substeps), facs,
                    cfg=cfg, ids=["c%d" % i for i in range(4)])

    print("phi_ref = %.5f" % ds["meta"]["phi_ref"])
    for ep in ds["episodes"]:
        f = ep["factors"]
        print("\n%s  half=%.3f mass=%.3f fric=%.2f" %
              (ep["episode_id"], f["half_size"], f["mass"], f["friction"]))
        prev = None
        for st in ep["steps"]:
            o = np.asarray(st["obs"])
            d_po = float(o[17])
            d_ot = float(o[16])
            do = 0.0 if prev is None else float(np.linalg.norm(o[0:2] - prev[0:2]))
            print("  t=%2d  d_po=%.4f  gap=%.4f  d_ot=%.4f  phi=%.4f  |dobj|=%.2e"
                  % (st["t"], d_po, d_po - f["half_size"] - 0.012, d_ot,
                     float(o[18]), do))
            prev = o
        # how big is the counterfactual effect the plans actually produce?
        y = np.array([st["y_C"] for st in ep["steps"]])
        print("  y_C[0] (factual): min %.4f max %.4f | spread across arms: "
              "mean %.4f max %.4f"
              % (y[:, 0].min(), y[:, 0].max(),
                 float(np.mean(y.max(axis=1) - y.min(axis=1))),
                 float(np.max(y.max(axis=1) - y.min(axis=1)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
