"""
gate0_replay.py -- Gate-0: is the simulator bit-reproducible under replay?

Why this file exists before any model code
------------------------------------------
Contact-rich physics engines are not bit-reproducible. Change `num_envs`, the
solver iteration count, threading, or an RNG stream and the same action sequence
lands in a different final state. If that drift is not measured first, every
"causal effect" we later report may be solver noise wearing a causal costume.

The gate does three things:
  1. Replay one action script N times from one saved state and measure pose drift.
  2. Classify the simulator: DETERMINISTIC / STABLE_STOCHASTIC / DRIFTING.
  3. Optionally estimate the test-retest noise floor `eps` that the metrics
     module needs (spec 5.1: eps comes from repeated replay, never from taste).

Real adapters (CausalWorld, AI2-THOR) plug in by implementing SimAdapter.
Three mock adapters are included so the gate itself can be verified today,
before any simulator is installed.

Run
---
    python gate0_replay.py --sim mock:det
    python gate0_replay.py --sim mock:noisy --repeats 20
    python gate0_replay.py --sim mock:drift      # exits 1 on purpose
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, os.pardir, "logs")

SPEC_POSE_TOL = 1e-9        # spec: replay_determinism_gate.pose_max_abs_diff_threshold
SPEC_DRIFT_TOL = 1e-3       # spec: replay_determinism_gate.drift_threshold
SPEC_REPEATS = 20           # spec: replay_determinism_gate.repeats


# --------------------------------------------------------------------------- #
# adapter interface
# --------------------------------------------------------------------------- #
class SimAdapter:
    """Minimal surface a simulator must expose for Gate-0.

    Implement these five methods for CausalWorld / AI2-THOR and the rest of the
    gate (drift classification, fingerprinting, noise floor) works unchanged.
    """

    name = "abstract"

    def reset(self, seed: int) -> None:
        raise NotImplementedError

    def save_state(self):
        raise NotImplementedError

    def load_state(self, state) -> None:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def poses(self) -> np.ndarray:
        """(n_objects, d) pose array; d may be 3 (xyz) or 7 (xyz+quat)."""
        raise NotImplementedError

    def outcome(self) -> float:
        """Scalar progress in [0, 1] used as the primary outcome Y_prog."""
        raise NotImplementedError

    def fingerprint(self) -> Dict[str, object]:
        return {}


# --------------------------------------------------------------------------- #
# mock adapters (for verifying the gate itself)
# --------------------------------------------------------------------------- #
@dataclass
class _PoseState:
    pos: np.ndarray
    target: np.ndarray


class DeterministicMock(SimAdapter):
    """Closed-form integrator: same state + same actions => identical poses."""

    name = "mock:det"

    def __init__(self, n_obj: int = 3, gain: float = 0.05):
        self.n_obj = n_obj
        self.gain = gain
        self._s = _PoseState(np.zeros((n_obj, 3)), np.ones(n_obj) * 1.0)
        self._seed = 0

    def reset(self, seed: int) -> None:
        self._seed = seed
        rng = np.random.default_rng(seed)
        self._s = _PoseState(rng.uniform(-0.2, 0.2, (self.n_obj, 3)),
                             rng.uniform(0.6, 1.0, self.n_obj))

    def save_state(self):
        return (self._s.pos.copy(), self._s.target.copy())

    def load_state(self, state) -> None:
        self._s = _PoseState(state[0].copy(), state[1].copy())

    def step(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=float).reshape(-1)
        push = self.gain * np.pad(a, (0, max(0, 3 - a.size)))[:3]
        self._s.pos = self._s.pos + push[None, :] * (1.0 + 0.1 * np.arange(self.n_obj))[:, None]

    def poses(self) -> np.ndarray:
        return self._s.pos.copy()

    def outcome(self) -> float:
        # Smooth and never saturating: 1/(1+d) instead of clip(1-d). A clipped
        # outcome flat-lines at 0 once the agent wanders off, which silently
        # swallows the very noise the gate is supposed to measure.
        dist = np.linalg.norm(self._s.pos - self._s.target[:, None], axis=1).mean()
        return float(1.0 / (1.0 + dist))

    def fingerprint(self) -> Dict[str, object]:
        return {"mock": "deterministic_integrator", "gain": self.gain, "n_obj": self.n_obj}


class NoisyMock(DeterministicMock):
    """Same physics, plus fresh-entropy noise injected at *every step*.

    This is what a non-deterministic solver actually looks like: replaying from
    the same saved state diverges because the integrator itself is noisy, with a
    stable variance and no trend. (Noise at reset only would be reset by
    load_state and the gate would see nothing -- that was a real bug caught by
    the self-test.)
    """

    name = "mock:noisy"

    def __init__(self, sigma: float = 1e-6, **kw):
        super().__init__(**kw)
        self.sigma = sigma
        self._rng = np.random.default_rng()   # OS entropy: differs per run

    def step(self, action: np.ndarray) -> None:
        super().step(action)
        self._s.pos = self._s.pos + self._rng.normal(0.0, self.sigma, self._s.pos.shape)

    def fingerprint(self) -> Dict[str, object]:
        d = super().fingerprint()
        d.update(mock="noisy_integrator", sigma=self.sigma)
        return d


class DriftingMock(NoisyMock):
    """Noise grows with every replay attempt -- the scheduling-dependent failure
    mode (worse after each restart)."""

    name = "mock:drift"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._attempt = 0

    def load_state(self, state) -> None:
        self._attempt += 1
        self.sigma = min(1e-6 * (10.0 ** self._attempt), 1e-1)
        super().load_state(state)

    def fingerprint(self) -> Dict[str, object]:
        d = super().fingerprint()
        d.update(mock="drifting_integrator", attempts=self._attempt)
        return d


MOCKS = {"mock:det": DeterministicMock, "mock:noisy": NoisyMock, "mock:drift": DriftingMock}


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
def env_fingerprint() -> Dict[str, object]:
    """The 12 fields the spec requires to be logged with every result."""
    fp: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "os": platform.system() + " " + platform.release(),
    }
    for mod in ("numpy", "torch", "mujoco", "pybullet", "ai2thor"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "?")
        except Exception:
            fp[mod] = None
    try:
        import torch  # noqa: F401
        fp["cuda_deterministic"] = bool(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        fp["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        fp["cuda_deterministic"] = None
        fp["cuda_available"] = None
    # filled by the caller; Gate-0 has no access to the sim config otherwise
    for k in ("solver_iterations", "num_envs", "threads", "vectorized"):
        fp.setdefault(k, "UNSET")
    return fp


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    sim: str
    classification: str
    passed: bool
    repeats: int
    steps: int
    max_abs_diff: float
    mean_abs_diff: float
    drift_slope: float
    drift_growth: float = 1.0
    per_run_max: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    fingerprint: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gate": "Gate-0_replay_determinism",
            "sim": self.sim,
            "classification": self.classification,
            "passed": self.passed,
            "repeats": self.repeats,
            "steps": self.steps,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "drift_slope": self.drift_slope,
            "drift_growth": self.drift_growth,
            "per_run_max": self.per_run_max,
            "notes": self.notes,
            "fingerprint": self.fingerprint,
            "thresholds": {"pose_tol": SPEC_POSE_TOL, "drift_tol": SPEC_DRIFT_TOL},
        }


def make_action_script(rng: np.random.Generator, steps: int, action_dim: int = 3) -> np.ndarray:
    return rng.normal(0.0, 1.0, (steps, action_dim))


def run_gate(adapter: SimAdapter, *, seed: int = 0, repeats: int = SPEC_REPEATS,
             steps: int = 20) -> GateResult:
    """Replay one script `repeats` times and classify the simulator."""
    rng = np.random.default_rng(seed)
    script = make_action_script(rng, steps)

    adapter.reset(seed)
    state0 = adapter.save_state()

    runs = []
    for _ in range(repeats):
        adapter.load_state(state0)
        for a in script:
            adapter.step(a)
        runs.append(adapter.poses())

    base = runs[0]
    per_run_max = [float(np.max(np.abs(r - base))) for r in runs]
    all_pair = np.stack([np.abs(r - base) for r in runs[1:]], axis=0) if repeats > 1 else np.zeros((1,))
    max_abs = float(np.max(np.abs(all_pair))) if all_pair.size else 0.0
    mean_abs = float(np.mean(np.abs(all_pair))) if all_pair.size else 0.0

    arr = np.asarray(per_run_max, dtype=float)
    idx = np.arange(len(arr), dtype=float)
    slope = float(np.polyfit(idx, arr, 1)[0]) if len(arr) > 2 and np.ptp(arr) > 0 else 0.0

    # Drift is judged on early-vs-late *level*, not on a linear slope: a
    # saturating failure (noise plateaus after N restarts) flattens the slope
    # while the divergence is plainly real.
    #
    # The early window MUST include run 0. Run 0 is compared against itself and
    # is therefore exactly 0; if it is excluded, the early median is already the
    # plateau and a saturating drift reports growth ~= 1 (this was a real false
    # negative: mock:drift scored 0.90 at both ends). A median over a window that
    # holds one guaranteed zero is robust to that zero but lethal to a plateau --
    # stable-stochastic runs sit at a constant c, so median([0,c,c,c,c,c]) = c.
    third = max(1, len(arr) // 3)
    if len(arr) >= 4:
        early = float(np.median(arr[:third]))
        late = float(np.median(arr[-third:]))
    else:  # too few repeats for a windowed median
        early = float(arr[0]) if len(arr) else 0.0
        late = float(arr[-1]) if len(arr) else 0.0
    if early > 0:
        growth = late / early
    else:
        growth = float("inf") if late > SPEC_DRIFT_TOL else 1.0

    notes: List[str] = []
    if max_abs <= SPEC_POSE_TOL:
        cls = "DETERMINISTIC"
        notes.append("bit-identical replay; treat ground truth as a single deterministic rollout")
    elif max_abs > SPEC_DRIFT_TOL and late > SPEC_DRIFT_TOL and growth > 2.0:
        cls = "DRIFTING"
        notes.append("divergence grows with replay count (early %.2e -> late %.2e, x%.1f); "
                     "fix the environment (threads, solver iterations, vectorisation, "
                     "deterministic kernels) before running any experiment"
                     % (early, late, growth))
    else:
        cls = "STABLE_STOCHASTIC"
        notes.append("run-to-run variation without a growing trend (early %.2e -> late %.2e); "
                     "use Monte-Carlo ground truth and estimate eps by test-retest"
                     % (early, late))

    if cls == "DETERMINISTIC":
        notes.append("eps = 1e-6 (spec fallback); no repeated sampling needed")
    elif cls == "STABLE_STOCHASTIC":
        notes.append("eps must come from estimate_noise_floor(); M capped at %d"
                     % 30)

    return GateResult(sim=adapter.name, classification=cls, passed=cls != "DRIFTING",
                      repeats=repeats, steps=steps, max_abs_diff=max_abs,
                      mean_abs_diff=mean_abs, drift_slope=slope, drift_growth=growth,
                      per_run_max=per_run_max, notes=notes,
                      fingerprint={**env_fingerprint(), **adapter.fingerprint()})


# --------------------------------------------------------------------------- #
# noise floor (test-retest)
# --------------------------------------------------------------------------- #
def estimate_noise_floor(adapter: SimAdapter, *, n_units: int = 40, k_arms: int = 5,
                         repeats: int = 2, horizon: int = 3, seed: int = 0,
                         quantile: float = 0.95) -> dict:
    """Test-retest estimate of eps (spec 5.1).

    For each sampled state we run every arm twice and look at how much the
    *effect* Delta = Y^k - Y^0 moves between the two replays. The 95th percentile
    of |Delta_1 - Delta_2| is the noise floor. Effects smaller than eps are
    undecidable and must not enter the direction-agreement denominator.
    """
    rng = np.random.default_rng(seed)
    adapter.reset(seed)
    warm = np.random.default_rng(seed + 1).normal(0.0, 1.0, (8, 3))
    for a in warm:
        adapter.step(a)

    states = []
    for _ in range(n_units):
        for _ in range(3):                      # random walk to sample a state
            adapter.step(rng.normal(0.0, 1.0, 3))
        states.append(adapter.save_state())

    diffs = []
    for st in states:
        arms = rng.normal(0.0, 1.0, (k_arms, 3))
        ys = np.zeros((repeats, k_arms))
        for r in range(repeats):
            for j in range(k_arms):
                adapter.load_state(st)
                adapter.step(arms[j])
                for _ in range(horizon - 1):
                    adapter.step(np.zeros(3))
                ys[r, j] = adapter.outcome()
        d1 = ys[0] - ys[0, 0]
        d2 = ys[-1] - ys[-1, 0]
        diffs.append(np.abs(d1[1:] - d2[1:]))

    d = np.concatenate(diffs) if diffs else np.array([0.0])
    eps = float(np.percentile(d, quantile * 100))
    return {
        "gate": "noise_floor_test_retest",
        "quantile": quantile,
        "eps": eps,
        "n_units": n_units,
        "k_arms": k_arms,
        "repeats": repeats,
        "max_abs_diff_between_replays": float(d.max()) if d.size else 0.0,
        "note": "use 1e-6 instead when Gate-0 classifies the simulator as DETERMINISTIC",
    }


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
SIM_HELP = ("mock:det | mock:noisy | mock:drift | mujoco:push3d | causalworld:<task>")


def make_adapter(name: str) -> SimAdapter:
    """Resolve a --sim name to an adapter, importing real simulators lazily.

    Mocks must keep working on a machine with no physics engine installed, so the
    heavy imports live behind the prefix check rather than at module scope.
    """
    if name in MOCKS:
        return MOCKS[name]()
    prefix = name.split(":", 1)[0] if ":" in name else name
    if prefix == "mujoco":
        try:
            from mujoco_adapter import FACTORY as MU_FACTORY
        except ModuleNotFoundError as exc:
            raise SystemExit("--sim %r needs mujoco: %s" % (name, exc))
        if name not in MU_FACTORY:
            raise SystemExit("unknown mujoco scene %r; available: %s"
                             % (name, sorted(MU_FACTORY)))
        return MU_FACTORY[name]()
    if prefix == "causalworld":
        try:
            from causalworld_adapter import FACTORY as CW_FACTORY, BLOCKER
        except ModuleNotFoundError as exc:
            raise SystemExit("--sim %r needs causal_world: %s" % (name, exc))
        if name not in CW_FACTORY:
            raise SystemExit("unknown causalworld task %r; available: %s\n%s"
                             % (name, sorted(CW_FACTORY), BLOCKER))
        return CW_FACTORY[name]()
    raise SystemExit("unknown --sim %r; available: %s" % (name, SIM_HELP))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Gate-0 replay determinism check")
    ap.add_argument("--sim", default="mock:det", help=SIM_HELP)
    ap.add_argument("--repeats", type=int, default=SPEC_REPEATS)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-floor", action="store_true", help="also estimate eps")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args(argv)

    adapter = make_adapter(args.sim)
    res = run_gate(adapter, seed=args.seed, repeats=args.repeats, steps=args.steps)
    payload = res.to_dict()

    print("\n=== Gate-0 ===")
    print("sim            :", res.sim)
    print("classification :", res.classification, "(PASS)" if res.passed else "(FAIL)")
    print("repeats        :", res.repeats, "steps:", res.steps)
    print("max |dpose|    :", "%.3e" % res.max_abs_diff, " tol:", "%.0e" % SPEC_POSE_TOL)
    print("mean|dpose|    :", "%.3e" % res.mean_abs_diff)
    print("drift slope    :", "%.3e" % res.drift_slope,
          " growth: x%.1f" % res.drift_growth)
    for n in res.notes:
        print("  -", n)

    if args.noise_floor:
        nf = estimate_noise_floor(adapter, seed=args.seed)
        payload["noise_floor"] = nf
        print("\n=== noise floor ===")
        if nf["eps"] < 1e-12:
            print("eps            : ~0  -> use the 1e-6 deterministic fallback"
                  " (quantile %.2f)" % nf["quantile"])
        else:
            print("eps            :", "%.3e" % nf["eps"], " (quantile %.2f)" % nf["quantile"])

    out = args.out or os.path.join(LOG_DIR, "gate0_%s.json" % args.sim.replace(":", "_"))
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("\nwritten:", out)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
