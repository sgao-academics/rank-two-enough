# -*- coding: utf-8 -*-
"""Probe: with a long box, is the task's end state ORDER-SENSITIVE?

The v1.3 cube task is order-invariant, and that is why nothing downstream could
work: one scalar (the summed aligned push) explained R^2 = 0.80 of the within-step
truth, a zero-parameter heuristic beat the one-step oracle, and "multi-step" had no
content to model. The previous probe established the physics of the fix: a long box
pushed off its long axis turns (+52.6 deg at a 0.022 m lever, -52.6 deg at -0.022,
0.000 deg through the centroid). Turning changes where the next contact happens.

This probe asks the only question that matters next: on the SAME state, do two plans
with the SAME total push but a DIFFERENT order of it reach different end states?

It uses exactly the R-perm construction that failed on the cube task -- a zero-sum
lateral ramp and its permutations -- because that construction is the cleanest test
of order: sum_h b_h = 0 to machine precision, so every arm has identical total push
and identical step multiset, and only the sequence differs.

Three numbers decide whether the new task is worth building:
  * spread(Y) across permutations, in units of the task's own Y
  * the fraction of that spread explained by an order-blind feature set
  * whether the object stays inside the workspace (rotation must not turn into
    tumbling or an object escaping the pusher's +-0.30 m reach)
"""
from __future__ import annotations

import io
import sys

import numpy as np
import mujoco

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

HL, HS, HT = 0.034, 0.014, 0.008     # half-long, half-short, half-thick
MASS, FRICTION = 0.05, 0.90
PUSHER_HALF, PUSHER_HH = 0.012, 0.035
SUBSTEPS, SCALE = 50, 0.008
H, M = 6, 3                          # horizon, plan length (same as the cube task)
S_NORM = 0.10

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
      <joint name="pz" type="slide" axis="0 0 1" range="-0.030 0.100" damping="1.5"/>
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
    <position name="az" joint="pz" ctrlrange="-0.030 0.100"/>
  </actuator>
</mujoco>
"""


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class Sim:
    def __init__(self, target_xy=(0.12, 0.0), target_yaw=0.0) -> None:
        # The pusher is 0.070 m tall, so its CENTRE must sit at >= 0.035 m or its
        # lower half is embedded in the table: a position-servo joint with 1.5 mm of
        # the block below the plane spends its force budget fighting the floor and
        # transfers almost nothing to the object (measured: 24 control steps moved
        # the object 9 mm, with yaw still at 0.16 deg). Centre it so the block
        # spans just above the table and fully covers the box's height, which keeps
        # the contact resultant level with the object's centroid -- a pure yaw
        # torque instead of a tipping one.
        xml = XML.format(fr=FRICTION, pz=PUSHER_HH + 0.0005, oz=HT, hl=HL, hs=HS,
                         ht=HT, m=MASS)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.b_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.b_psh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
        self.q_obj = int(self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "of")])
        self.q_psh = [int(self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in ("px", "py", "pz")]
        self.target = np.asarray(target_xy, dtype=np.float64)
        self.target_yaw = float(target_yaw)
        self._lo = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
            for n in ("px", "py", "pz")])
        self._hi = np.array([self.model.jnt_range[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
            for n in ("px", "py", "pz")])

    # -- state ------------------------------------------------------------ #
    def reset(self, *, obj_xy=(0.0, 0.0), obj_yaw=0.0, pusher_xy=(0.0, -0.070),
              ctrl_xy=None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q = self.data.qpos
        q[self.q_obj:self.q_obj + 3] = [obj_xy[0], obj_xy[1], HT]
        cy, sy = np.cos(obj_yaw / 2.0), np.sin(obj_yaw / 2.0)
        q[self.q_obj + 3:self.q_obj + 7] = [cy, 0.0, 0.0, sy]
        for i, v in enumerate((pusher_xy[0], pusher_xy[1], 0.0)):
            q[self.q_psh[i]] = v
        c = pusher_xy if ctrl_xy is None else ctrl_xy
        self.data.ctrl[:] = [c[0], c[1], 0.0]
        mujoco.mj_forward(self.model, self.data)

    def step(self, action) -> None:
        a = np.asarray(action, dtype=np.float64)
        new = np.clip(np.asarray(self.data.ctrl) + SCALE * a[:3], self._lo, self._hi)
        self.data.ctrl[:] = new
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    # -- readings --------------------------------------------------------- #
    def pos(self) -> np.ndarray:
        return self.data.xpos[self.b_obj][:2].copy()

    def psh(self) -> np.ndarray:
        return self.data.xpos[self.b_psh][:2].copy()

    def yaw(self) -> float:
        w, x, y, z = self.data.xquat[self.b_obj]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def attitude(self):
        w, x, y, z = self.data.xquat[self.b_obj]
        return (float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
                float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))))

    def outcome(self) -> float:
        """reach x alignment x contact, in (0, 1]; alignment uses cos(2*dYaw)
        because a box is symmetric under a 180-degree turn."""
        d_obj = float(np.linalg.norm(self.pos() - self.target))
        d_ee = float(np.linalg.norm(self.psh() - self.pos()))
        reach = 1.0 / (1.0 + d_obj / S_NORM)
        contact = 0.6 + 0.4 / (1.0 + d_ee / S_NORM)
        dyaw = wrap(self.yaw() - self.target_yaw)
        align = 0.5 * (1.0 + np.cos(2.0 * dyaw))
        return float(reach * (0.55 + 0.45 * align) * contact)

    def support(self, direction: np.ndarray) -> float:
        """Half-width of the box along `direction` (its support function)."""
        psi = self.yaw() - float(np.arctan2(direction[1], direction[0]))
        return float(HL * abs(np.cos(psi)) + HS * abs(np.sin(psi)))

    # -- the scripted expert ---------------------------------------------- #
    def expert_action(self, standoff_pad: float = 0.012) -> np.ndarray:
        """Aim at a contact point on the box's actual surface, not at a fixed radius.

        The v1.3 expert used a fixed standoff. With a rotating box that is a
        collision: when the long axis points at the target the surface is 0.034 m
        out, so a 0.030 m aim puts the pusher INSIDE the object and the solver fires
        it away. Using the support function makes the aim exactly tangent for any
        orientation, which is geometry, not tuning.
        """
        p_obj, p_psh = self.pos(), self.psh()
        d = self.target - p_obj
        n = float(np.linalg.norm(d))
        u = np.array([1.0, 0.0]) if n < 1e-9 else d / n
        contact = p_obj - u * (self.support(u) + standoff_pad)
        a = contact - p_psh
        na = float(np.linalg.norm(a))
        # DEADBAND: once the pusher is already at the contact point, `a` is noise and
        # the expert alternates +1/-1 every step (measured: 24 steps, object crept
        # 9 mm, action sign flipped 12 times). Being in contact is not "no direction"
        # -- it means the direction is the force line, so keep pushing along u and the
        # object moves, which moves the contact point, which the pusher follows.
        if na < 0.006:
            return np.array([u[0], u[1], 0.0])
        return np.array([a[0] / na, a[1] / na, 0.0])


def rollout_to(s: Sim, n_decisions: int):
    """Run the expert for n_decisions*H control steps; return the snapshot + history."""
    hist = []
    for _ in range(n_decisions):
        for _ in range(H):
            s.step(s.expert_action())
        hist.append((s.pos().copy(), s.yaw(), s.outcome()))
    return hist


def closed_loop(s: Sim, plan: np.ndarray, h: int = H) -> float:
    """Execute a plan verbatim, then let the expert finish the horizon."""
    m = plan.shape[0]
    for k in range(m):
        s.step(plan[k])
    for _ in range(h - m):
        s.step(s.expert_action())
    return s.outcome()


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


def main() -> int:
    s = Sim()
    # start behind the object with a lateral offset: the pusher sits off the long
    # axis, which is what turns a plain "push forward" into a push with torque
    # Start the pusher ALIGNED with the force line. An off-axis start makes the
    # expert's aim point (computed from the support function along u) differ from
    # the true first contact point (a corner), so once the pusher touches the box
    # the aim lands BEHIND it and the action oscillates sign every step: measured
    # +0.91 y, then -1.00 y, then +0.98 ... while the object crept 5 mm in 24 steps.
    # The non-commutativity is supposed to come from the ARMS being off-centre, not
    # from the expert being confused, so the expert starts clean and stays clean.
    s.reset(pusher_xy=(-0.070, 0.0))
    # 2 decisions = 12 control steps = 0.096 m of travel: the object started 0.12 m
    # from the target, so this stops it mid-task. More than that and every arm is
    # pinned at the target, where Y saturates and no order effect could show itself.
    hist = rollout_to(s, 2)

    print("expert trajectory (long box, pusher starts behind, on the force line)")
    print("  %-6s %8s %8s %9s %9s %7s" % ("dec", "x", "y", "yaw[deg]", "phi", "Y"))
    for i, (p, y, ph) in enumerate(hist):
        print("  %-6d %8.4f %8.4f %9.2f %9.4f %7.4f"
              % (i, p[0], p[1], np.degrees(y), ph, ph))
    roll, pitch = s.attitude()
    print("  final attitude: roll %.2f deg, pitch %.2f deg (should stay near 0:"
          " rotation about z only)" % (np.degrees(roll), np.degrees(pitch)))
    print("  final |obj| = %.4f m (workspace reach 0.30 m)"
          % float(np.linalg.norm(s.pos())))

    snap = snapshot(s)
    u = s.target - s.pos()
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    n_hat = np.array([-u[1], u[0]])

    # zero-sum lateral ramp and its permutations: identical total push, identical
    # step multiset, different order. This is the R-perm construction that was
    # degenerate on the cube (arm 0 always won) -- the whole point is to see whether
    # the long box makes it non-degenerate.
    from itertools import permutations
    p = np.arange(M, dtype=np.float64) - (M - 1) / 2.0
    p = p / float(np.max(np.abs(p)))
    A = 0.6
    b = np.zeros((M, 3))
    b[:, :2] = (A * p)[:, None] * n_hat[None, :]     # zero-sum lateral ramp

    # The factual plan is the expert's own next M actions, COLLECTED BY RUNNING
    # them -- not by calling expert_action() M times from the same state, which
    # returns the same action each time because nothing moved in between.
    plan0 = []
    for _ in range(M):
        a = s.expert_action()
        plan0.append(a)
        s.step(a)
    plan0 = np.array(plan0)
    restore(s, snap)

    perms = list(permutations(range(M)))
    arms = np.array([plan0 + b[list(perm)] for perm in perms])

    print("\nsame total push, different order (m=%d, K=%d permutations, ramp=%.2f)"
          % (M, len(arms), A))
    print("  arms total deviation: %s" % np.array2string(
        arms.sum(axis=1)[:, :2], precision=12, suppress_small=True))
    print("  %-22s %9s %9s %9s %9s" % ("order", "Y", "yaw[deg]", "x", "y"))
    Ys = []
    for i, pl in enumerate(arms):
        restore(s, snap)
        y = closed_loop(s, pl)
        Ys.append(y)
        print("  %-22s %9.5f %9.2f %9.4f %9.4f"
              % (str(tuple(int(v) for v in perms[i])), y,
                 np.degrees(s.yaw()), s.pos()[0], s.pos()[1]))
    Ys = np.array(Ys)
    print("\n  spread: max-min = %.5f, sd = %.5f, |mean| = %.5f"
          % (Ys.max() - Ys.min(), Ys.std(), abs(Ys.mean())))
    print("  relative spread = %.2f%% of |mean|"
          % (100.0 * (Ys.max() - Ys.min()) / max(abs(Ys.mean()), 1e-12)))

    # how much of that spread is reachable by an ORDER-BLIND feature?
    # total push is identical by construction, so an order-blind linear model sees
    # the same input for every arm and must predict identical Y.
    print("\n  order-blind features (total push, total |push|, first-step push) are")
    print("  IDENTICAL across arms by construction, so any spread above is order")
    print("  information that only a sequence model can use.")
    tot = arms.sum(axis=1)
    print("  max|sum(arm) - sum(arm0)| = %.2e (should be ~0)"
          % float(np.max(np.abs(tot - tot[0]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
