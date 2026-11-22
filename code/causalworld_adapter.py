"""
causalworld_adapter.py -- the proposal's intended main platform, currently BLOCKED.

Status (2026-11-21): NOT EXECUTED, NOT VERIFIED. Every method below is written
against CausalWorld's documented interface but has never been run, because the
dependency cannot be installed on this machine. Do not cite a Gate-0 number for
this adapter until it has actually been produced. `Gate-0 -> mujoco:push3d` is
the only real-engine measurement that exists today.

Why it is blocked
-----------------
1. CausalWorld is a wrapper over PyBullet. PyPI ships no cp312 Windows wheel for
   pybullet (3.2.4-3.2.7 cover cp37-cp311 only), so pip falls back to a source
   build and fails on the missing MSVC 14 build tools.
2. CausalWorld pins the gym 0.15-0.17 era, which does not install on Python 3.12
   without patching. Its asset bundle is a separate multi-hundred-MB download.

What it would take
------------------
    conda create -n cw python=3.8 -y
    conda activate cw
    pip install pybullet==3.2.5 gym==0.17.2
    pip install git+https://github.com/aair-lab/causal_world.git
    python -m causal_world.download_assets      # ~1 GB, S3

Then run, from THIS directory, with the py3.8 interpreter:

    python gate0_replay.py --sim causalworld:push --noise-floor

The adapter only imports numpy and the local SimAdapter, so it will load in a
py3.8 env without dragging numpy 2.x in.

Decision this implies for the paper
-----------------------------------
Either (a) run the main experiments in a py3.8 CausalWorld env and accept an old
numpy/torch there, or (b) keep MuJoCo as the physics engine and state plainly in
Methods that the scene reproduces CausalWorld's factorisation rather than being
CausalWorld itself. Writing "we evaluate on CausalWorld" without having run it
would be a factual misstatement; that is not an option.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from gate0_replay import SimAdapter

BLOCKER = (
    "causalworld_adapter is BLOCKED on this machine: pybullet has no cp312 Windows "
    "wheel (install falls back to a source build and fails for lack of MSVC 14), and "
    "CausalWorld pins the gym 0.15-0.17 era. Use --sim mujoco:push3d for the real-engine "
    "Gate-0 run, or build the py3.8 env described in causalworld_adapter.py."
)

# CausalWorld task names that map onto a causal-structure benchmark. Not exhaustive.
TASKS = ("push", "pick_and_place", "stacking", "creative_async_collecting")


class CausalWorldAdapter(SimAdapter):  # pragma: no cover - cannot run here
    """Thin SimAdapter over a CausalWorld env. UNVERIFIED -- see module docstring."""

    name = "causalworld"

    def __init__(self, task: str = "push", *, seed: int = 0, action_mode: str = "joint_positions"):
        try:
            import causal_world  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(BLOCKER) from exc

        from causal_world.envs.causalworld import CausalWorld  # UNVERIFIED path

        self.task = task
        self.name = "causalworld:%s" % task
        self.env = CausalWorld(task_name=task, enable_visualization=False,
                               seed=seed, action_mode=action_mode)
        self._obs = None
        self._action_dim = int(self.env.action_space.shape[0]) if hasattr(
            self.env, "action_space") else 1

    # -- SimAdapter surface -------------------------------------------------- #
    def reset(self, seed: int) -> None:
        self._obs = self.env.reset()

    def save_state(self):
        """Snapshot: full simulator state dict plus the realised control vector."""
        return (self.env.get_state(), self._last_action())

    def load_state(self, state) -> None:
        snap, action = state
        self.env.set_state(snap)
        self._restore_action(action)

    def step(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float64).ravel()
        if self._action_dim > 1 and a.size != self._action_dim:
            a = np.resize(a, self._action_dim)
        self._obs, _, _, info = self.env.step(a)

    def poses(self) -> np.ndarray:
        """(n_actors, 7) end-effector + object poses from the observation dict.

        CausalWorld returns unordered observation entries; the adapter must sort
        them by actor name so the column order is stable across replays. Filenames
        and keys differ between CausalWorld releases -- verify before trusting.
        """
        obs = self._obs
        if obs is None:
            raise RuntimeError("call reset() before poses()")
        if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], dict):
            obs = obs[0]
        rows = []
        for key in sorted(k for k in obs if "end_effector" in k or "position" in k):
            v = np.asarray(obs[key], dtype=np.float64).ravel()
            rows.append(np.concatenate([v[:3], np.array([1.0, 0.0, 0.0, 0.0])]))
        if not rows:
            raise RuntimeError("no pose-like entries in observation: %r" % list(obs)[:8])
        return np.stack(rows, axis=0)

    def outcome(self) -> float:
        """Progress in (0, 1]; not clipped, for the same reason as the MuJoCo adapter."""
        obs = self._obs
        if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], dict):
            obs = obs[0]
        if not isinstance(obs, dict):
            raise RuntimeError("outcome() needs the dict observation")
        cube = np.asarray(obs.get("achieved_goal"), dtype=np.float64).ravel()[:2]
        goal = np.asarray(obs.get("desired_goal"), dtype=np.float64).ravel()[:2]
        d = float(np.linalg.norm(cube - goal))
        return float(1.0 / (1.0 + d / 0.10))

    def fingerprint(self) -> Dict[str, object]:
        try:
            import causal_world
            ver = getattr(causal_world, "__version__", "unknown")
        except Exception:
            ver = "unknown"
        return {
            "sim": "causal_world",
            "causal_world_version": ver,
            "task": self.task,
            "num_actors": int(getattr(self.env, "num_actors", -1)),
            "status": "UNVERIFIED: adapter never executed on this machine",
        }

    # -- internals ----------------------------------------------------------- #
    def _last_action(self) -> Optional[np.ndarray]:
        return getattr(self, "_last_a", None)

    def _restore_action(self, action) -> None:
        self._last_a = None if action is None else np.array(action, copy=True)


def _factory():
    """Only build the factory if causal_world imports; keeps the CLI honest."""
    out = {}
    try:
        import causal_world  # noqa: F401
    except ModuleNotFoundError:
        return out
    for t in TASKS:
        out["causalworld:%s" % t] = (lambda _t=t: CausalWorldAdapter(_t))
    return out


FACTORY = _factory()
