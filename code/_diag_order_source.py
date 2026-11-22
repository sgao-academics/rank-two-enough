# -*- coding: utf-8 -*-
"""Where does the 14.6% order-induced spread at friction 0.45 actually come from?

The scan found one cell with a large spread and a healthy Y level:
    scale 0.008, friction 0.45, amp 0.6  ->  dY = 14.6%, dyaw = 0.5 deg
A 0.5-degree attitude change cannot move Y by 14%, so the mechanism is positional.
Three candidates, and they are very different scientific claims:

  (a) TIME-IRREVERSIBILITY: with lower friction the object coasts, so a push
      delivered early is retained and a push delivered at the end has no time to
      accumulate. This is a real non-commutativity of the dynamics.
  (b) CONTACT LOSS: the object slides away faster than the pusher, the arm's later
      steps push nothing, and "order" is really "which arm lost contact first".
      Also real, but a different phenomenon -- contact-event prediction, not
      displacement accumulation.
  (c) ARTEFACT: an arm is clipped at the joint limit or the reset differs.

This script runs the 6 permutations with full per-step traces so the difference can
be attributed rather than guessed: for each arm it prints the per-step gap, the
object's travel, and whether the pusher's commanded setpoint was clipped.
"""
from __future__ import annotations

import io
import sys
from itertools import permutations

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.path.insert(0, ".")

import _probe_order_longbox as PL   # noqa: E402

PL.SCALE, PL.FRICTION = 0.008, 0.45
M, AMP, WARM = 3, 0.6, 2


def main() -> int:
    s = PL.Sim()
    s.reset(pusher_xy=(-0.070, 0.0))
    for _ in range(WARM):
        for _ in range(PL.H):
            s.step(s.expert_action())
    snap = PL.snapshot(s)
    print("at plan time: obj=(%+.4f,%+.4f) yaw=%+.3f psh=(%+.4f,%+.4f) Y=%.4f"
          % (*s.pos(), np.degrees(s.yaw()), *s.psh(), s.outcome()))

    u = s.target - s.pos()
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    n_hat = np.array([-u[1], u[0]])
    p = np.arange(M, dtype=np.float64) - (M - 1) / 2.0
    p = p / float(np.max(np.abs(p)))
    b = np.zeros((M, 3))
    b[:, :2] = (AMP * p)[:, None] * n_hat[None, :]
    print("force line u=(%+.3f,%+.3f), lateral n=(%+.3f,%+.3f)" % (*u, *n_hat))

    plan0 = []
    for _ in range(M):
        a = s.expert_action()
        plan0.append(a)
        s.step(a)
    plan0 = np.array(plan0)
    PL.restore(s, snap)
    print("factual plan (arm 0):\n%s" % np.array2string(plan0, precision=3))

    rows = []
    for perm in permutations(range(M)):
        PL.restore(s, snap)
        o_start = s.pos().copy()
        gaps, travels, clipped = [], [], 0
        for k in range(M):
            act = plan0[k] + b[list(perm)[k]]
            ctrl_before = np.asarray(s.data.ctrl).copy()
            s.step(act)
            # clipped = the commanded setpoint ran into the joint limit
            if np.any(np.abs(np.asarray(s.data.ctrl) - s._hi) < 1e-9) or \
               np.any(np.abs(np.asarray(s.data.ctrl) - s._lo) < 1e-9):
                clipped += 1
            gaps.append(float(np.linalg.norm(s.psh() - s.pos())
                              - (PL.HL + PL.PUSHER_HALF)))
            travels.append(float(np.linalg.norm(s.pos() - o_start)))
        o_after_plan = s.pos().copy()
        for _ in range(PL.H - M):
            s.step(s.expert_action())
        rows.append(dict(perm=tuple(int(v) for v in perm), Y=s.outcome(),
                         travel_plan=float(np.linalg.norm(o_after_plan - o_start)),
                         travel_total=float(np.linalg.norm(s.pos() - o_start)),
                         yaw=np.degrees(s.yaw()), gaps=gaps, clipped=clipped,
                         x=float(s.pos()[0])))

    print("\n%-12s %8s %10s %10s %8s %9s %6s"
          % ("order", "Y", "travel(plan)", "travel(all)", "yaw[deg]", "min gap", "clip"))
    for r in rows:
        print("%-12s %8.4f %10.5f %10.5f %8.3f %9.5f %6d"
              % (str(r["perm"]), r["Y"], r["travel_plan"], r["travel_total"],
                 r["yaw"], min(r["gaps"]), r["clipped"]))

    print("\nper-step gap after each planned step (m; negative = penetrating):")
    for r in rows:
        print("  %-12s %s" % (str(r["perm"]),
                              " ".join("%+.4f" % g for g in r["gaps"])))

    Y = np.array([r["Y"] for r in rows])
    tr = np.array([r["travel_total"] for r in rows])
    print("\nspread Y = %.4f (%.2f%%), spread travel = %.5f m"
          % (Y.max() - Y.min(), 100 * (Y.max() - Y.min()) / abs(Y.mean()),
             tr.max() - tr.min()))
    print("corr(Y, travel_total) over 6 arms = %+.4f"
          % float(np.corrcoef(Y, tr)[0, 1]) if len(set(tr)) > 1 else "  (travel identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
