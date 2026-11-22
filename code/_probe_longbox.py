# -*- coding: utf-8 -*-
"""Probe: does a LONG BOX rotate when pushed off-centre, and does ORDER change the end state?

Why this probe exists
---------------------
The v1.3 scene uses a CUBE. A cube has an isotropic inertia tensor and a geometry
with 90-degree symmetry, so its orientation is both physically uninteresting and
invisible in the outcome: Y = reach(d_obj, d_ee) depends on positions only. Adding
up the pushed distances is then an almost-additive operation, which is why one
scalar (the summed aligned push) explains R^2 = 0.80 of the within-step truth and a
zero-parameter heuristic beats the one-step oracle. The task is order-invariant by
construction, and no model can find sequence information that physics never put
there.

A long box removes that symmetry. Push it off the long axis and the contact force
has a lever arm, so the object turns; turning changes where the NEXT contact
happens; and "which end you touched first" stops being irrelevant.

Three measurements, all in simulation, no dataset and no metrics involved:
  1. torque-free push (aim at the centroid): |dyaw| must be ~0
  2. off-centre push: |dyaw| must grow with the lever arm
  3. same action multiset, two orders: final (x, y, yaw) must differ

If 1 holds and 2-3 do not, the long box buys nothing and the task needs a
different kind of non-commutativity (a gate, a wall, a socket) before anything
downstream is worth rebuilding.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import mujoco

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

# long axis along +x; half-extents in metres
HL, HS, HT = 0.034, 0.014, 0.008          # half-long, half-short, half-thick
MASS, FRICTION = 0.05, 0.90
PUSHER_HALF = 0.012
PUSHER_HH = 0.035
SUBSTEPS = 50                              # one control step = 0.1 s
SCALE = 0.008                              # metres of setpoint per unit action

XML = """<mujoco model="longbox">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="Euler" cone="pyramidal" solver="Newton"
          iterations="100" noslip_iterations="0"/>
  <default>
    <geom condim="3" friction="{fr} 0.005 0.0001" solref="0.004 1"
          solimp="0.95 0.99 0.001"/>
    <position kp="400" kv="40" ctrllimited="true" ctrlrange="-9 9"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" size="0.6 0.6 0.05" pos="0 0 -0.0005"/>
    <body name="pusher" pos="0 0 {pz}">
      <joint name="px" type="slide" axis="1 0 0" range="-0.30 0.30" damping="1.5"/>
      <joint name="py" type="slide" axis="0 1 0" range="-0.30 0.30" damping="1.5"/>
      <joint name="pz" type="slide" axis="0 0 1" range="-0.030 0.050" damping="1.5"/>
      <geom name="pg" type="box" size="0.012 0.012 0.035" mass="0.15"/>
    </body>
    <body name="object" pos="0 0 {oz}">
      <freejoint name="of"/>
      <geom name="og" type="box" size="{hl} {hs} {ht}" mass="{m}"/>
    </body>
    <site name="target" pos="0.12 0 0.0" size="0.012"/>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" ctrlrange="-0.30 0.30"/>
    <position name="ay" joint="py" ctrlrange="-0.30 0.30"/>
    <position name="az" joint="pz" ctrlrange="-0.030 0.050"/>
  </actuator>
</mujoco>
"""


class Sim:
    def __init__(self) -> None:
        xml = XML.format(fr=FRICTION, pz=HT + 0.025, oz=HT, hl=HL, hs=HS, ht=HT,
                         m=MASS)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.b_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.b_psh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
        j_of = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "of")
        self.q_obj = int(self.model.jnt_qposadr[j_of])
        self.q_psh = [int(self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in ("px", "py", "pz")]

    def reset(self, *, obj_xy=(0.0, 0.0), obj_yaw=0.0, pusher_xy=(0.0, 0.0)) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos
        q[self.q_obj:self.q_obj + 3] = [obj_xy[0], obj_xy[1], HT]
        cy, sy = np.cos(obj_yaw / 2.0), np.sin(obj_yaw / 2.0)
        q[self.q_obj + 3:self.q_obj + 7] = [cy, 0.0, 0.0, sy]
        for i, v in enumerate((pusher_xy[0], pusher_xy[1], 0.0)):
            q[self.q_psh[i]] = v
        self.data.ctrl[:] = [pusher_xy[0], pusher_xy[1], 0.0]
        mujoco.mj_forward(self.model, self.data)

    def step(self, action) -> None:
        for i in range(3):
            lo, hi = self.model.jnt_range[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, ("px", "py", "pz")[i])]
            self.data.ctrl[i] = float(np.clip(self.data.ctrl[i]
                                              + SCALE * float(action[i]), lo, hi))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    def push(self, action, n: int = 8) -> None:
        a = np.asarray(action, dtype=np.float64)
        for _ in range(int(n)):
            self.step(a)

    # -- readings --------------------------------------------------------- #
    def pos(self):
        return self.data.xpos[self.b_obj].copy()

    def rpy(self):
        """Euler roll/pitch/yaw of the object, from its quaternion."""
        w, x, y, z = self.data.xquat[self.b_obj]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)

    def gap(self):
        """Signed clearance between the pusher box and the object along the shortest axis."""
        return float(np.linalg.norm(
            self.data.xpos[self.b_obj][:2] - self.data.xpos[self.b_psh][:2]))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def run(seq, *, lever: float, seg: int = 8, n_rep: int = 1):
    """Execute `n_rep` repeats of each action in `seq`, from a fresh reset.

    The pusher starts at the object's -y side, shifted by `lever` along +x, i.e.
    off the long axis. `lever = 0` aims the force line through the centroid
    (torque-free); `lever = 0.025` puts it near the end of the long side.
    """
    s = Sim()
    s.reset(pusher_xy=(lever, -0.030))
    yaw0 = s.rpy()[2]
    for a in seq:
        s.push(a, seg)
    r, p, y = s.rpy()
    return dict(pos=s.pos(), yaw=y, dyaw=wrap(y - yaw0), roll=r, pitch=p)


def main() -> int:
    print("long box: half-extents (%.3f, %.3f, %.3f) m, mass %.2f kg, mu %.2f"
          % (HL, HS, HT, MASS, FRICTION))
    print("contact distance along the SHORT side = HS + %.3f = %.3f m\n"
          % (PUSHER_HALF, HS + PUSHER_HALF))

    # -- 1. torque-free: aim the force line through the centroid ---------- #
    print("1) push along +y with the pusher on the long axis (lever = 0)")
    print("   %-12s %9s %9s %9s %9s" % ("lever", "dx", "dy", "dyaw[deg]", "pitch[deg]"))
    for lever in (0.0, 0.010, 0.020, 0.025):
        r = run([[0.0, 1.0, 0.0]], lever=lever, seg=10)
        print("   %-12.3f %9.4f %9.4f %9.3f %9.3f"
              % (lever, r["pos"][0], r["pos"][1],
                 np.degrees(r["dyaw"]), np.degrees(r["pitch"])))

    # -- 2. does the torque grow with the lever arm? --------------------- #
    print("\n2) same push, longer: 20 segments of +y (2.0 s of contact)")
    for lever in (0.0, 0.020):
        r = run([[0.0, 1.0, 0.0]], lever=lever, seg=20)
        print("   lever %.3f -> dyaw %7.3f deg, dy %7.4f m"
              % (lever, np.degrees(r["dyaw"]), r["pos"][1]))

    # -- 3. same multiset, two orders ------------------------------------ #
    # A: shove the far end first, then push straight.  B: the reverse.
    print("\n3) order matters? same two actions, both orders, lever 0.020")
    A = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    ra = run(A, lever=0.020, seg=10)
    rb = run(list(reversed(A)), lever=0.020, seg=10)
    print("   A (+y then +x): pos=(%+.4f, %+.4f) yaw=%+7.3f deg"
          % (ra["pos"][0], ra["pos"][1], np.degrees(ra["yaw"])))
    print("   B (+x then +y): pos=(%+.4f, %+.4f) yaw=%+7.3f deg"
          % (rb["pos"][0], rb["pos"][1], np.degrees(rb["yaw"])))
    print("   |dpos| = %.4f m   |dyaw| = %.3f deg"
          % (float(np.linalg.norm(ra["pos"][:2] - rb["pos"][:2])),
             abs(np.degrees(wrap(ra["yaw"] - rb["yaw"])))))

    # -- 4. is the rotation monotone in the lever, or does it flip? ------ #
    print("\n4) sign check: push +y with the pusher offset +x vs -x")
    rp = run([[0.0, 1.0, 0.0]], lever=0.022, seg=10)
    rn = run([[0.0, 1.0, 0.0]], lever=-0.022, seg=10)
    print("   lever +0.022 -> dyaw %+7.3f deg" % np.degrees(rp["dyaw"]))
    print("   lever -0.022 -> dyaw %+7.3f deg" % np.degrees(rn["dyaw"]))
    print("   (opposite signs would confirm the lever arm, not solver noise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
