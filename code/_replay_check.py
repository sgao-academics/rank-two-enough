"""Step A of the order test: can an episode be replayed from its stored factors?

Before asking whether permuting a plan's steps moves the outcome, the replay has to
be shown to reproduce the stored trajectory. If `obs` does not come back to machine
precision at every decision point, any order effect measured afterwards would be
contaminated by a drift that has nothing to do with order.

Compares, for every step t of one episode:
    obs     (22-d) from the dataset vs from the replay
    phi     the potential at the decision point

Exits non-zero if the worst deviation exceeds 1e-9.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import mujoco_adapter as MU          # noqa: E402
import sampler as S                  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data", "suite", "scene2_spec13",
                    "ds_m3_s00_e96.json")


def main() -> int:
    ds = json.load(io.open(DATA, encoding="utf-8"))
    meta = ds["meta"]
    cfg = meta["config"]
    H = int(cfg["H"])
    warmup = int(cfg["warmup"])
    substeps = int(cfg["control_substeps"])

    ep = ds["episodes"][0]
    factors = dict(ep["factors"])
    target_xy = factors["target_xy"]

    ad = MU.MuJoCoPush3D(substeps=substeps)
    ad.set_factors(**factors)
    ad.reset(int(ep["seed"]))

    for _ in range(warmup):
        ad.step(S.base_policy(ad, target_xy))

    worst_obs = 0.0
    worst_phi = 0.0
    print("%-4s %12s %12s %12s" % ("t", "|dobs|max", "phi stored", "phi replay"))
    for st in ep["steps"]:
        t = int(st["t"])
        phi_now = ad.outcome()
        obs = S.observe(ad, phi_now)
        d_obs = float(np.max(np.abs(obs - np.asarray(st["obs"], dtype=np.float64))))
        d_phi = abs(phi_now - float(st["phi_start"]))
        worst_obs = max(worst_obs, d_obs)
        worst_phi = max(worst_phi, d_phi)
        print("%-4d %12.3e %12.8f %12.8f" % (t, d_obs, float(st["phi_start"]), phi_now))
        # advance exactly as run_episode does: the factual plan's first action
        plans = np.asarray(st["plans"], dtype=np.float64)
        ad.step(plans[0][0])

    print("\nworst |dobs| = %.3e   worst |dphi| = %.3e" % (worst_obs, worst_phi))
    ok = worst_obs < 1e-9 and worst_phi < 1e-9
    print("replay reproduces the stored trajectory:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
