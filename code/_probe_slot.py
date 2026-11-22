# -*- coding: utf-8 -*-
"""Probe: a slot task, where the outcome is event-gated instead of additive.

The problem, reduced to one line
--------------------------------
In a quasi-static pushing task the outcome is a function of TOTAL DISPLACEMENT, and
total displacement is nearly the sum of the commanded actions. Measured on the cube
task: scalar R^2 = 0.80, full feature set 0.84, pairwise objective identical to
within-step ridge (-0.0019 DAR), i.e. no linear representation can exceed the scalar
channel. Adding torque (long box) did not help at m=3 because the rotation needs
longer than the plan to accumulate (dyaw <= 0.5 deg in the usable regime).

What would help is a physical NON-ADDITIVE: an event that truncates displacement. A
slot does exactly that. Push the box into a gap 4 mm wider than the box on each side:
a lateral error of 4 mm stops it dead, and no amount of further pushing recovers the
lost displacement. Order and lateral placement now decide the outcome discontinuously.

This probe measures:
  1. does the scripted expert actually get through (a task nobody can do is useless)
  2. yaw-free lateral sensitivity: how far off-centre can the plan drift before the
     box jams -- in metres, so the required plan resolution is a number, not a wish
  3. for the 6 permutations of a zero-sum lateral ramp: spread in Y, and whether that
     spread is explained by total push (it must NOT be, or nothing was gained)
"""
from __future__ import annotations

import io
import sys
from itertools import permutations

import numpy as np
import mujoco

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

HALF = 0.014          # object half-width; box is 0.028 across
HL2, HT2 = 0.020, 0.008
MASS, FRICTION = 0.05, 0.70
PH, PHH = 0.012, 0.035
SUBSTEPS, SCALE = 50, 0.008
H, M = 6, 3
S_NORM = 0.10
GAP_HALF = 0.018      # half-width of the corridor -> 4 mm clearance per side


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


XML = """<mujoco model="slot">
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
      <joint name="pz2" type="slide" axis="0 0 1" range="-0.030 0.100" damping="1.5"/>
      <geom name="pg" type="box" size="0.012 0.012 0.035" mass="0.15"/>
    </body>
    <body name="object" pos="0 0 {oz}">
      <freejoint name="of"/>
      <geom name="og" type="box" size="{hl} {hs} {ht}" mass="{m}"/>
    </body>
    <geom name="wall_up" type="box" size="0.020 0.100 0.030"
          pos="0.060 {gap} 0.030" rgba="0.5 0.5 0.5 1"/>
    <geom name="wall_dn" type="box" size="0.020 0.100 0.030"
          pos="0.060 {gapn} 0.030" rgba="0.5 0.5 0.5 1"/>
    <site name="target" pos="0.180 0 0.0" size="0.012"/>
  </worldbody>
  <actuator>
    <position name="ax" joint="px" ctrlrange="-0.30 0.30"/>
    <position name="ay" joint="py" ctrlrange="-0.30 0.30"/>
    <position name="az" joint="pz2" ctrlrange="-0.030 0.100"/>
  </actuator>
</mujoco>
"""


class Sim:
    def __init__(self, gap_half: float = GAP_HALF) -> None:
        self.gap_half = float(gap_half)
        xml = XML.format(fr=FRICTION, pz=PHH + 0.0005, oz=HT2, hl=HL2, hs=HALF,
                         ht=HT2, m=MASS,
                         gap=self.gap_half + 0.100, gapn=-(self.gap_half + 0.100))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.b_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.b_psh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
        self.q_obj = int(self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "of")])
        self.q_psh = [int(self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in ("px", "py", "pz2")]
        self.target = np.array([0.18, 0.0])
        self._lo = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
            for n in ("px", "py", "pz2")])
        self._hi = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
            for n in ("px", "py", "pz2")])

    def reset(self, *, obj_xy=(0.0, 0.0), pusher_xy=(-0.075, 0.0), obj_yaw=0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos
        q[self.q_obj:self.q_obj + 3] = [obj_xy[0], obj_xy[1], HT2]
        cy, sy = np.cos(obj_yaw / 2.0), np.sin(obj_yaw / 2.0)
        q[self.q_obj + 3:self.q_obj + 7] = [cy, 0.0, 0.0, sy]
        for i, v in enumerate((pusher_xy[0], pusher_xy[1], 0.0)):
            q[self.q_psh[i]] = v
        self.data.ctrl[:] = [pusher_xy[0], pusher_xy[1], 0.0]
        mujoco.mj_forward(self.model, self.data)

    def step(self, action) -> None:
        a = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:] = np.clip(np.asarray(self.data.ctrl) + SCALE * a[:3],
                                    self._lo, self._hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    def pos(self) -> np.ndarray:
        return self.data.xpos[self.b_obj][:2].copy()

    def psh(self) -> np.ndarray:
        return self.data.xpos[self.b_psh][:2].copy()

    def yaw(self) -> float:
        w, x, y, z = self.data.xquat[self.b_obj]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def outcome(self) -> float:
        d = float(np.linalg.norm(self.pos() - self.target))
        d_ee = float(np.linalg.norm(self.psh() - self.pos()))
        return float((1.0 / (1.0 + d / S_NORM)) * (0.6 + 0.4 / (1.0 + d_ee / S_NORM)))

    def expert_action(self, pad: float = 0.012) -> np.ndarray:
        p_obj, p_psh = self.pos(), self.psh()
        d = self.target - p_obj
        n = float(np.linalg.norm(d))
        u = np.array([1.0, 0.0]) if n < 1e-9 else d / n
        psi = self.yaw() - float(np.arctan2(u[1], u[0]))
        support = HL2 * abs(np.cos(psi)) + HALF * abs(np.sin(psi))
        contact = p_obj - u * (support + pad)
        a = contact - p_psh
        na = float(np.linalg.norm(a))
        if na < 0.006:
            return np.array([u[0], u[1], 0.0])
        return np.array([a[0] / na, a[1] / na, 0.0])


def snapshot(s: Sim):
    return dict(qpos=s.data.qpos.copy(), qvel=s.data.qvel.copy(),
                ctrl=np.asarray(s.data.ctrl).copy(), act=s.data.act.copy())


def restore(s: Sim, snap) -> None:
    s.data.qpos[:] = snap["qpos"]
    s.data.qvel[:] = snap["qvel"]
    s.data.ctrl[:] = snap["ctrl"]
    if s.data.act.size:
        s.data.act[:] = snap["act"]
    mujoco.mj_forward(s.model, s.data)


def run_free(gap_half: float, drift: float = 0.0):
    """Can the expert get the box through a corridor of this half-width?"""
    s = Sim(gap_half)
    s.reset(pusher_xy=(-0.075, drift))
    for _ in range(40):
        s.step(s.expert_action())
    p = s.pos()
    return p, bool(p[0] > 0.10)


def main() -> int:
    print("slot task: box 0.028 wide, corridor half-width varies; target at x=0.18")
    print("\n1) can the scripted expert pass? (40 control steps, no perturbation)")
    print("   %-12s %-10s %9s %9s %8s" % ("gap_half", "clearance", "x", "y", "passed"))
    for gh in (0.014, 0.016, 0.018, 0.020, 0.024):
        p, ok = run_free(gh)
        print("   %-12.3f %-10.4f %9.4f %9.4f %8s"
              % (gh, gh - HALF, p[0], p[1], "YES" if ok else "no"))

    gh = 0.018
    print("\n2) lateral drift tolerance (pusher starts `drift` off centre)")
    print("   %-10s %9s %9s %8s" % ("drift", "x", "y", "passed"))
    for drift in (0.0, 0.002, 0.004, 0.006, 0.008, 0.012):
        p, ok = run_free(gh, drift=drift)
        print("   %-10.3f %9.4f %9.4f %8s" % (drift, p[0], p[1], "YES" if ok else "no"))

    # -- 3. order effect on a zero-sum lateral ramp ----------------------- #
    s = Sim(gh)
    s.reset(pusher_xy=(-0.075, 0.0))
    for _ in range(2):
        for _ in range(H):
            s.step(s.expert_action())
    snap = snapshot(s)
    print("\nat plan time: obj=(%+.4f,%+.4f) psh=(%+.4f,%+.4f) Y=%.4f"
          % (*s.pos(), *s.psh(), s.outcome()))

    u = s.target - s.pos()
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    nh = np.array([-u[1], u[0]])
    p = np.arange(M, dtype=np.float64) - (M - 1) / 2.0
    p = p / float(np.max(np.abs(p)))

    print("\n3) same total push, different order (ramp amplitude scan)")
    print("   %-8s %-14s %9s %9s %9s %8s"
          % ("amp", "order", "Y", "x", "y", "passed"))
    for amp in (0.4, 0.8):
        b = np.zeros((M, 3))
        b[:, :2] = (amp * p)[:, None] * nh[None, :]
        plan0 = []
        for _ in range(M):
            a = s.expert_action()
            plan0.append(a)
            s.step(a)
        plan0 = np.array(plan0)
        restore(s, snap)
        Ys = []
        for perm in permutations(range(M)):
            restore(s, snap)
            for k in range(M):
                s.step(plan0[k] + b[list(perm)[k]])
            for _ in range(H - M):
                s.step(s.expert_action())
            Ys.append(s.outcome())
            print("   %-8.2f %-14s %9.4f %9.4f %9.4f %8s"
                  % (amp, str(tuple(int(v) for v in perm)), Ys[-1],
                     s.pos()[0], s.pos()[1], "YES" if s.pos()[0] > 0.10 else "no"))
        Ys = np.array(Ys)
        print("   amp %.1f -> spread |dY| = %.4f (%.1f%% of mean), distinct outcomes: %d"
              % (amp, Ys.max() - Ys.min(),
                 100 * (Ys.max() - Ys.min()) / max(abs(Ys.mean()), 1e-12),
                 len(np.unique(np.round(Ys, 4)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
