"""
mujoco_adapter.py -- a REAL physics engine behind the Gate-0 SimAdapter surface.

Why MuJoCo and not CausalWorld
-------------------------------
The proposal named CausalWorld as the main validation platform. CausalWorld is a
wrapper over PyBullet, and PyBullet publishes no cp312 Windows wheel (checked
against PyPI on 2026-11-21: releases 3.2.4-3.2.7 ship cp37-cp311 only), so
`pip install pybullet` falls back to a source build and dies on the missing MSVC
toolchain. CausalWorld additionally pins the gym 0.15-0.17 era. On this machine
(Python 3.12.13) neither is installable.

MuJoCo 3.17.0 does ship a cp312 win_amd64 wheel and is a contact-rich engine with
the exact knobs the spec's fingerprint asks for (solver iterations, integrator,
cone, timestep, noslip). So Gate-0 runs here first, on a scene that reproduces
CausalWorld's *pushing* factorisation: one manipulable object, one distractor, a
3-DoF point pusher, and a target site. The causal factors are the object's mass,
half-size, friction and the target position -- the same factor set CausalWorld
exposes for causal-structure tasks.

This file therefore measures the simulator, it does not measure the method. If a
CausalWorld environment is ever stood up (separate py3.8 env), `causalworld_adapter.py`
implements the same surface and the gate re-runs unchanged.

The one thing to keep in mind
-----------------------------
`outcome()` must never saturate. A clipped distance would flat-line the moment the
pusher wanders off, and Gate-0 would then report "bit-identical replay" for a
simulator that is actually noisy. Every term below is a strictly decaying
function of a distance, and none of them is clipped.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "mujoco_adapter.py requires mujoco. Install with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

from gate0_replay import SimAdapter


# --------------------------------------------------------------------------- #
# scene constants that the scene's MEANING depends on
# --------------------------------------------------------------------------- #
# The pusher must start BEHIND the object, not on top of it. Starting at the
# world origin put it inside the object (object jittered to within +-0.02 of the
# origin, half-size 0.016-0.024, pusher half-width 0.012): the contact solver then
# resolved the overlap with an impulse and flung the object up to 0.46 m away,
# past the pusher's own +-0.30 m joint range, so it could never be reached again.
# Measured before being fixed: |obj - target| at t=0 was 0.4646 m in one episode
# while it stayed at 0.1270 in another -- and in neither did the object ever move
# again, which made mass / friction / half-size causally inert.
PUSHER_START_OFFSET = 0.070     # metres behind the object along -(target - object)

# The expert aims this far from the object's CENTRE. Contact happens at
# half_size + 0.012 (the pusher's box half-width), i.e. 0.028-0.036 m, so an aim
# of 0.045 parked the pusher 0.009-0.017 m short: the scripted expert never
# touched the object and the "pushing" task never pushed anything.
EXPERT_STANDOFF = 0.030
XML_TEMPLATE = """<mujoco model="push3d">
  <compiler angle="radian" autolimits="true"/>

  <option timestep="{timestep}" integrator="{integrator}" cone="{cone}"
          solver="Newton" iterations="{solver_iters}" noslip_iterations="{noslip_iters}"/>

  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>

  <default>
    <geom condim="3" friction="{friction}" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <position kp="{kp}" kv="{kv}" ctrllimited="true" ctrlrange="-9 9"/>
  </default>

  <worldbody>
    <geom name="table" type="plane" size="0.6 0.6 0.05" pos="0 0 -0.0005"
          rgba="0.55 0.55 0.58 1" contype="1" conaffinity="1"/>

    <body name="pusher" pos="0 0 {pusher_z}">
      <joint name="px" type="slide" axis="1 0 0" range="-0.30 0.30" damping="1.5"/>
      <joint name="py" type="slide" axis="0 1 0" range="-0.30 0.30" damping="1.5"/>
      <joint name="pz" type="slide" axis="0 0 1" range="-0.030 0.050" damping="1.5"/>
      <geom name="pusher_geom" type="box" size="0.012 0.012 0.035" mass="0.15"
            rgba="0.20 0.35 0.90 1" friction="{friction}"/>
    </body>

    <body name="object" pos="0 0 {half_s}">
      <freejoint name="obj_free"/>
      <geom name="obj_geom" type="box" size="{half_s} {half_s} {half_s}"
            mass="{mass}" rgba="0.90 0.40 0.15 1"/>
    </body>

    <body name="distractor" pos="{dx} {dy} {half_s}">
      <freejoint name="dis_free"/>
      <geom name="dis_geom" type="box" size="{half_s} {half_s} {half_s}"
            mass="{mass}" rgba="0.55 0.55 0.20 1"/>
    </body>

    <site name="target" pos="{tx} {ty} {half_s}" size="0.012"
          rgba="0.10 0.80 0.30 0.55"/>
  </worldbody>

  <actuator>
    <position name="ax" joint="px" ctrlrange="-0.30 0.30"/>
    <position name="ay" joint="py" ctrlrange="-0.30 0.30"/>
    <position name="az" joint="pz" ctrlrange="-0.030 0.050"/>
  </actuator>
</mujoco>
"""


def build_xml(*, mass: float = 0.05, half_size: float = 0.020,
              friction: float = 0.90, target_xy: Tuple[float, float] = (-0.12, 0.12),
              distractor_xy: Tuple[float, float] = (0.10, -0.10),
              timestep: float = 0.002, integrator: str = "Euler", cone: str = "pyramidal",
              solver_iters: int = 100, noslip_iters: int = 0,
              kp: float = 400.0, kv: float = 40.0) -> str:
    """Deterministic MJCF string for one factorised pushing scene.

    `friction` is the sliding coefficient; torsional/rolling terms are fixed at
    MuJoCo's small defaults because they are not causal factors here.
    """
    fr = "%g 0.005 0.0001" % friction
    return XML_TEMPLATE.format(
        timestep=timestep, integrator=integrator, cone=cone,
        solver_iters=solver_iters, noslip_iters=noslip_iters,
        friction=fr, kp=kp, kv=kv,
        half_s=half_size, mass=mass,
        pusher_z=half_size + 0.025,
        tx=target_xy[0], ty=target_xy[1],
        dx=distractor_xy[0], dy=distractor_xy[1],
    )


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #
class MuJoCoPush3D(SimAdapter):
    """3-DoF point pusher on a tabletop, one object and one distractor.

    Action: 3-vector of *incremental* setpoints in metres (dx, dy, dz), scaled by
    `action_scale`. Incremental rather than absolute so that the N(0,1) action
    scripts used by the gate stay inside the workspace instead of pinning the
    pusher against a joint limit -- a pinned joint is a constant, and constants
    are exactly what a determinism test can not distinguish from determinism.

    Control cadence: `substeps` physics ticks per control step. The default of 50
    makes one control step 0.1 s (10 Hz), a policy-class decision rate. It used to
    be 5, i.e. 100 Hz, which made the H=3 horizon 30 ms -- three servo ticks, over
    which the counterfactual spread was 0.6% of Y's between-step variation. That is
    not a decision, it is a rounding error, and it was measured before being fixed.
    """

    name = "mujoco:push3d"

    def __init__(self, *, mass: float = 0.05, half_size: float = 0.020,
                 friction: float = 0.90, target_xy: Tuple[float, float] = (-0.12, 0.12),
                 distractor_xy: Tuple[float, float] = (0.10, -0.10),
                 timestep: float = 0.002, substeps: int = 50,
                 action_scale: float = 0.008,
                 integrator: str = "Euler", cone: str = "pyramidal",
                 solver_iters: int = 100, noslip_iters: int = 0):
        # The factor set and the scene parameters must stay in lockstep: a factor
        # that exists in the XML but not here would be silently left at its default
        # by set_factors(), i.e. do(.) would be a no-op that still returns numbers.
        self.factors = dict(mass=mass, half_size=half_size, friction=friction,
                            target_xy=tuple(target_xy),
                            distractor_xy=tuple(distractor_xy), timestep=timestep,
                            integrator=integrator, cone=cone,
                            solver_iters=solver_iters, noslip_iters=noslip_iters)
        self.substeps = int(substeps)
        self.action_scale = float(action_scale)
        # read by sampler.base_policy: where the scripted expert aims, relative to
        # the object's centre. Part of the scene's definition, so it is in the
        # fingerprint below.
        self.expert_standoff = float(EXPERT_STANDOFF)
        self.pusher_start_offset = float(PUSHER_START_OFFSET)

        xml = build_xml(mass=mass, half_size=half_size, friction=friction,
                        target_xy=target_xy, distractor_xy=distractor_xy,
                        timestep=timestep,
                        integrator=integrator, cone=cone,
                        solver_iters=solver_iters, noslip_iters=noslip_iters)
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._b_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._b_dis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "distractor")
        self._b_psh = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
        self._s_tgt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self._j_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        self._q_obj = int(self.model.jnt_qposadr[self._j_obj])
        # pusher slide joints: needed by reset() to place the pusher somewhere other
        # than the world origin, and by the actuator setpoint that must match it.
        self._q_psh = [int(self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in ("px", "py", "pz")]

        self._ctrl = np.zeros(3)
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._state_spec = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._state_size = int(mujoco.mj_stateSize(self.model, self._state_spec))
        self._half = float(half_size)

        self.reset(0)

    # -- SimAdapter surface -------------------------------------------------- #
    def reset(self, seed: int) -> None:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        # small deterministic jitter so different seeds are different states,
        # but the same seed is always the same state
        q = self.data.qpos
        q[self._q_obj:self._q_obj + 3] = [rng.uniform(-0.02, 0.02),
                                          rng.uniform(-0.02, 0.02),
                                          self._half]
        q[self._q_obj + 3:self._q_obj + 7] = [1.0, 0.0, 0.0, 0.0]

        # Place the pusher BEHIND the object (opposite the target) and command the
        # setpoint to that same point. Both halves matter: qpos alone would leave the
        # position servo aiming at 0, i.e. the very overlap this is avoiding.
        obj_xy = np.array(q[self._q_obj:self._q_obj + 2], dtype=np.float64)
        tgt = np.asarray(self.factors["target_xy"], dtype=np.float64)[:2]
        u = tgt - obj_xy
        nu = float(np.linalg.norm(u))
        u = np.array([1.0, 0.0]) if nu < 1e-9 else u / nu
        start = obj_xy - u * PUSHER_START_OFFSET
        q[self._q_psh[0]] = start[0]
        q[self._q_psh[1]] = start[1]
        q[self._q_psh[2]] = 0.0
        self._start_xy = start.copy()

        self._ctrl = np.array([start[0], start[1], 0.0], dtype=np.float64)
        self.data.ctrl[:] = self._ctrl
        mujoco.mj_forward(self.model, self.data)

    def save_state(self):
        buf = np.empty(self._state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, buf, self._state_spec)
        # mjSTATE_FULLPHYSICS does NOT include ctrl. Restoring physics without
        # ctrl leaks the last action of the previous run into the next one.
        return (buf.copy(), self._ctrl.copy())

    def load_state(self, state) -> None:
        buf, ctrl = state
        mujoco.mj_setState(self.model, self.data, np.ascontiguousarray(buf), self._state_spec)
        self._ctrl = np.array(ctrl, dtype=np.float64, copy=True)
        self.data.ctrl[:] = self._ctrl
        mujoco.mj_forward(self.model, self.data)

    def step(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.size != 3:
            raise ValueError("expected a 3-vector action, got %r" % (a.shape,))
        nxt = self._ctrl + self.action_scale * a
        nxt = np.clip(nxt, self._ctrl_lo, self._ctrl_hi)
        self._ctrl = nxt
        self.data.ctrl[:] = nxt
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def controller_state(self) -> np.ndarray:
        """Commanded setpoint of the 3 position actuators (metres).

        Part of the observable state: actions are incremental, so "where the pusher
        was told to go" is not recoverable from `poses()` alone (the body lags the
        setpoint through the PD loop).
        """
        return np.array(self._ctrl, dtype=np.float64, copy=True)

    def poses(self) -> np.ndarray:
        """(n_objects, 7) = [x, y, z, qw, qx, qy, qz] for object, distractor, pusher, target."""
        d = self.data
        rows = [
            np.concatenate([d.xpos[self._b_obj], d.xquat[self._b_obj]]),
            np.concatenate([d.xpos[self._b_dis], d.xquat[self._b_dis]]),
            np.concatenate([d.xpos[self._b_psh], d.xquat[self._b_psh]]),
            np.concatenate([d.site_xpos[self._s_tgt], [1.0, 0.0, 0.0, 0.0]]),
        ]
        return np.stack(rows, axis=0).astype(np.float64)

    def outcome(self) -> float:
        """Progress in (0, 1]; strictly decaying in both distances, never clipped.

        y = [1 / (1 + d_obj/S)] * [0.6 + 0.4 / (1 + d_ee/S)]

        The first factor is how close the object is to the target, the second how
        close the pusher is to the object (contact progress). Neither is clipped,
        so no state of the world produces a flat response.
        """
        S = 0.10
        p_obj = self.data.xpos[self._b_obj]
        p_tgt = self.data.site_xpos[self._s_tgt]
        p_psh = self.data.xpos[self._b_psh]
        d_obj = float(np.linalg.norm(p_obj[:2] - p_tgt[:2]))
        d_ee = float(np.linalg.norm(p_psh[:2] - p_obj[:2]))
        reach = 1.0 / (1.0 + d_obj / S)
        contact = 0.6 + 0.4 / (1.0 + d_ee / S)
        return float(reach * contact)

    # -- provenance ---------------------------------------------------------- #
    def fingerprint(self) -> Dict[str, object]:
        return {
            "sim": "mujoco",
            "mujoco_version": mujoco.__version__,
            "scene": "push3d (CausalWorld-style factorisation)",
            "timestep": float(self.model.opt.timestep),
            "substeps_per_control_step": self.substeps,
            "control_dt": float(self.model.opt.timestep) * self.substeps,
            "integrator": {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}.get(
                int(self.model.opt.integrator), int(self.model.opt.integrator)),
            "cone": {0: "pyramidal", 1: "elliptic"}.get(int(self.model.opt.cone),
                                                        int(self.model.opt.cone)),
            "solver": int(self.model.opt.solver),
            "solver_iterations": int(self.model.opt.iterations),
            "noslip_iterations": int(self.model.opt.noslip_iterations),
            "n_joint_dof": int(self.model.nv),
            "n_bodies": int(self.model.nbody),
            # Scene-definition constants. Without these two, two runs can produce
            # different Y from the same factors and nothing in the payload would say
            # why: the pusher's start point and the expert's aim decide whether the
            # object is ever touched at all.
            "pusher_start_offset": self.pusher_start_offset,
            "expert_standoff": self.expert_standoff,
            "causal_factors": {
                "object_mass": self.factors["mass"],
                "object_half_size": self.factors["half_size"],
                "sliding_friction": self.factors["friction"],
                "target_xy": list(self.factors["target_xy"]),
                "distractor_xy": list(self.factors["distractor_xy"]),
            },
        }

    # -- extras used by later stages ---------------------------------------- #
    def set_factors(self, **kw) -> None:
        """Rebuild the model with different causal factors (for do(.) sampling)."""
        raisable = set(kw) - set(self.factors)
        if raisable:
            raise KeyError("unknown factors: %s" % sorted(raisable))
        new = dict(self.factors)
        new.update(kw)
        self.__init__(mass=new["mass"], half_size=new["half_size"],
                      friction=new["friction"], target_xy=new["target_xy"],
                      distractor_xy=new["distractor_xy"],
                      timestep=new["timestep"], integrator=new["integrator"],
                      cone=new["cone"], solver_iters=new["solver_iters"],
                      noslip_iters=new["noslip_iters"],
                      substeps=self.substeps, action_scale=self.action_scale)


FACTORY = {"mujoco:push3d": MuJoCoPush3D}


if __name__ == "__main__":  # tiny smoke run, no gate yet
    a = MuJoCoPush3D()
    print("xml bytes:", len(a.xml))
    print("nv =", a.model.nv, " nbody =", a.model.nbody,
          " ncon =", a.data.ncon)
    a.reset(0)
    print("outcome @ reset: %.6f" % a.outcome())
    for k in range(5):
        a.step(np.array([1.0, 0.0, 0.0]))
    print("outcome after 5 pushes: %.6f" % a.outcome())
    print("poses shape:", a.poses().shape)
