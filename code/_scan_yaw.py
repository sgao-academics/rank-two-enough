# -*- coding: utf-8 -*-
"""Scan: how much yaw can a 3-step plan actually produce?

The previous probe measured 0.03% order-induced spread in Y on the long box. The
reason is a time-scale mismatch, not a modelling failure: torque has to be applied
for long enough to rotate the box against four corner friction patches, and a
3-step plan (0.3 s) at 8 mm/step starts the rotation and then stops.

This scan answers the only question that can redirect the design: over
(action_scale, friction, ramp amplitude), does a 3-step zero-sum lateral ramp and
its permutations produce a yaw difference large enough to matter?

Reported per cell:
    dyaw  = max-min yaw over the 6 permutations of the ramp, in degrees
    dY%   = (max-min Y) / |mean Y| over the same 6 arms, in percent
    touch = fraction of arms that actually contacted the box during the plan

The design target is not "the biggest dyaw" -- it is a cell where dyaw is O(10 deg)
and touch = 1.0, because a big number produced by arms that fly past the box is not
an order effect, it is a miss.
"""
from __future__ import annotations

import io
import sys
from itertools import permutations

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.path.insert(0, ".")

import _probe_order_longbox as PL   # noqa: E402


def measure(scale: float, friction: float, amp: float, m: int = 3,
            warm: int = 2, seed: int = 0) -> dict:
    PL.SCALE = float(scale)
    PL.FRICTION = float(friction)
    s = PL.Sim()
    s.reset(pusher_xy=(-0.070, 0.0))
    for _ in range(warm):
        for _ in range(PL.H):
            s.step(s.expert_action())
    snap = PL.snapshot(s)

    u = s.target - s.pos()
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    n_hat = np.array([-u[1], u[0]])
    p = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
    p = p / float(np.max(np.abs(p)))
    b = np.zeros((m, 3))
    b[:, :2] = (amp * p)[:, None] * n_hat[None, :]

    plan0 = []
    for _ in range(m):
        a = s.expert_action()
        plan0.append(a)
        s.step(a)
    plan0 = np.array(plan0)
    PL.restore(s, snap)

    ys, yaws, touched = [], [], []
    for perm in permutations(range(m)):
        PL.restore(s, snap)
        gap_min = 1e9
        for k in range(m):
            s.step(plan0[k] + b[list(perm)[k]])
            gap_min = min(gap_min, float(np.linalg.norm(s.psh() - s.pos())))
        touched.append(gap_min < (PL.HL + PL.HS + 0.05))
        for _ in range(PL.H - m):
            s.step(s.expert_action())
        ys.append(s.outcome())
        yaws.append(np.degrees(s.yaw()))

    ys = np.array(ys)
    yaws = np.array(yaws)
    yaw_wrap = np.degrees(np.unwrap(np.radians(yaws)))
    return dict(dyaw=float(yaw_wrap.max() - yaw_wrap.min()),
                dY=float((ys.max() - ys.min()) / max(abs(ys.mean()), 1e-12)),
                touch=float(np.mean(touched)), Y=float(ys.mean()),
                yaw_abs=float(np.abs(yaw_wrap - yaw_wrap.mean()).max()))


def main() -> int:
    print("3-step order scan on the long box (6 permutations per cell)")
    print("%-8s %-8s %-6s %10s %9s %7s %9s"
          % ("scale", "friction", "amp", "dyaw[deg]", "dY[%]", "touch", "meanY"))
    for scale in (0.008, 0.015, 0.025, 0.040):
        for friction in (0.90, 0.45, 0.20):
            for amp in (0.6, 1.0):
                try:
                    r = measure(scale, friction, amp)
                except Exception as exc:                     # noqa: BLE001
                    print("%-8.3f %-8.2f %-6.2f  FAILED: %s"
                          % (scale, friction, amp, str(exc)[:40]))
                    continue
                print("%-8.3f %-8.2f %-6.2f %10.3f %9.3f %7.2f %9.4f"
                      % (scale, friction, amp, r["dyaw"], 100 * r["dY"],
                         r["touch"], r["Y"]))
    PL.SCALE, PL.FRICTION = 0.008, 0.90
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
