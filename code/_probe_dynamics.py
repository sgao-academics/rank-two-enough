"""Probe: is the logged trajectory a usable dynamics dataset?

Before any model is built, three facts must be verified rather than assumed:

  1. consecutive decision points are exactly ONE control step apart;
  2. the commanded setpoint moves by a *linear* function of the action
     (c' = c + s*a), i.e. the action interface is identifiable from logs;
  3. the pusher body tracks that setpoint with a first-order lag p' = p + k(c' - p).

If (1) is false the whole "learn dynamics from logs" plan is void; if (2)/(3) hold,
the two hardest-to-guess parts of the scene become one-parameter fits.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sampler as S  # noqa: E402

DS = os.path.abspath(os.path.join(HERE, os.pardir, "data",
                                  "pilot_rsamp_m3_H6_48ep.json"))


def main() -> int:
    ds = S.load(DS)
    dep = np.array([s["t"] for ep in ds["episodes"] for s in ep["steps"]])
    for ep in ds["episodes"][:3]:
        t = [s["t"] for s in ep["steps"]]
        print("episode %s t = %s" % (ep["episode_id"], t))

    # --- (1) consecutive rows, one control step apart ---------------------- #
    o, p, c, a = [], [], [], []
    gaps = []
    for ep in ds["episodes"]:
        st = ep["steps"]
        for i in range(len(st) - 1):
            ob = np.asarray(st[i]["obs"])
            nb = np.asarray(st[i + 1]["obs"])
            gaps.append(int(st[i + 1]["t"] - st[i]["t"]))
            o.append((ob[0:2], nb[0:2]))
            p.append((ob[6:8], nb[6:8]))
            c.append((ob[19:21], nb[19:21]))
            a.append(np.asarray(st[i]["plans"])[0][0][:2])
    print("\nt gap histogram:", np.unique(gaps, return_counts=True))

    o0 = np.array([x[0] for x in o]); o1 = np.array([x[1] for x in o])
    p0 = np.array([x[0] for x in p]); p1 = np.array([x[1] for x in p])
    c0 = np.array([x[0] for x in c]); c1 = np.array([x[1] for x in c])
    A = np.array(a)

    # --- (2) setpoint update is linear in the action ----------------------- #
    dc = (c1 - c0).reshape(-1)
    aa = A.reshape(-1)
    s_hat = float((dc @ aa) / (aa @ aa))
    resid = dc - s_hat * aa
    print("\nsetpoint:  c1-c0 = %.6f * a   (action_scale in adapter = 0.008)" % s_hat)
    print("           max|resid| = %.3e, p99 = %.3e"
          % (np.abs(resid).max(), np.percentile(np.abs(resid), 99)))

    # --- (3) pusher tracking: p1-p0 = k*(c1-p0)? --------------------------- #
    dp = (p1 - p0).reshape(-1)
    g = (c1 - p0).reshape(-1)
    k_hat = float((dp @ g) / (g @ g))
    r2 = 1.0 - np.sum((dp - k_hat * g) ** 2) / np.sum((dp - dp.mean()) ** 2)
    print("pusher   :  p1-p0 = %.4f * (c1-p0)   R2=%.4f" % (k_hat, r2))
    print("           |dp| median = %.3e m per step" % np.median(np.linalg.norm(p1 - p0, axis=1)))

    # --- object motion: is it driven by penetration? ----------------------- #
    d_po = np.linalg.norm(p0 - o0, axis=1)
    do = np.linalg.norm(o1 - o0, axis=1)
    print("\nobject   : |do| median = %.3e m per step" % np.median(do))
    for lo, hi in ((0.0, 0.030), (0.030, 0.040), (0.040, 0.060), (0.060, 0.100)):
        m = (d_po >= lo) & (d_po < hi)
        if m.sum():
            print("           d_po in [%.3f,%.3f): n=%4d  median|do|=%.3e  mean=%.3e"
                  % (lo, hi, m.sum(), np.median(do[m]), do[m].mean()))
    # direction of object motion vs pusher->object axis
    if do.max() > 0:
        ux = (o0 - p0) / np.maximum(d_po[:, None], 1e-9)
        mo = (o1 - o0)
        nz = np.linalg.norm(mo, axis=1) > 1e-7
        cosang = np.sum(ux[nz] * mo[nz], axis=1) / np.linalg.norm(mo[nz], axis=1)
        print("           cos(do, pusher->object) over moving steps: median %.3f"
              % np.median(cosang))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
