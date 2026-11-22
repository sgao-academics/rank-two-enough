# -*- coding: utf-8 -*-
"""Step-by-step diagnostic: where does the pusher actually go, and does it touch?"""
from __future__ import annotations

import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.path.insert(0, ".")

import _probe_order_longbox as PL   # noqa: E402

s = PL.Sim()
s.reset(pusher_xy=(-0.070, 0.0))
print("init: psh=(%+.4f,%+.4f) obj=(%+.4f,%+.4f) yaw=%+.3f"
      % (*s.psh(), *s.pos(), np.degrees(s.yaw())))
print("ctrl=%s  qpos_psh=%s"
      % (np.array2string(np.asarray(s.data.ctrl), precision=4),
         np.array2string(np.asarray([s.data.qpos[i] for i in s.q_psh]), precision=4)))
print("\n%-3s %-14s %-20s %8s %-20s %9s %5s" %
      ("k", "action", "pusher", "|dpsh|", "object", "|dobj|", "ncon"))
for k in range(18):
    a = s.expert_action()
    p0, o0 = s.psh().copy(), s.pos().copy()
    s.step(a)
    print("%-3d (%+.2f,%+.2f,%.2f) (%+.4f,%+.4f) %8.4f (%+.4f,%+.4f) %9.5f %5d"
          % (k, a[0], a[1], a[2], *s.psh(), float(np.linalg.norm(s.psh() - p0)),
             *s.pos(), float(np.linalg.norm(s.pos() - o0)), s.data.ncon))

print("\ncontacts at the end:")
for i in range(s.data.ncon):
    c = s.data.contact[i]
    print("  geom %d-%d dist %+.5f pos (%+.4f,%+.4f,%+.4f)"
          % (c.geom1, c.geom2, c.dist, c.pos[0], c.pos[1], c.pos[2]))
print("pusher aabb-ish: xpos=%s, half=0.012/0.012/0.035"
      % np.array2string(s.data.xpos[s.b_psh], precision=4))
print("object xpos=%s" % np.array2string(s.data.xpos[s.b_obj], precision=4))
print("gap along x between pusher right face and object left face = %+.5f"
      % float((s.pos()[0] - PL.HL) - (s.psh()[0] + PL.PUSHER_HALF)))
print("\ngeom ids: table=%d pusher=%d object=%d"
      % tuple(mujoco_id for mujoco_id in (
          PL.mujoco.mj_name2id(s.model, PL.mujoco.mjtObj.mjOBJ_GEOM, "table"),
          PL.mujoco.mj_name2id(s.model, PL.mujoco.mjtObj.mjOBJ_GEOM, "pg"),
          PL.mujoco.mj_name2id(s.model, PL.mujoco.mjtObj.mjOBJ_GEOM, "og"))))
print("support(u) = %.4f m, standoff total = %.4f"
      % (s.support(np.array([1.0, 0.0])), s.support(np.array([1.0, 0.0])) + 0.012))
