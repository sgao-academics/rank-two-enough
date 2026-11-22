"""
sampler.py -- twin-rollout ground truth for the counterfactual protocol.

What this file produces
-----------------------
For every decision point (episode, t) it produces K ground-truth outcomes under
two conventions, computed only from the simulator:

    y_C[k]  closed loop : do(a_k) at t, then the FROZEN base policy re-plans for H-1 steps
    y_O[k]  open loop   : do(a_k) at t, then the FACTUAL action sequence is replayed
    y_S[k]  one step    : diagnostic only, never a main result (spec: forbidden_as_main)

Both y_C and y_O are emitted because the spec requires the secondary convention to
be reported alongside the primary one: the gap between them is the part of the
decision's effect that a re-planning policy can absorb, and that gap is itself a
result, not noise.

Three things this file deliberately does NOT do
-----------------------------------------------
1. It never scores a candidate with the H-step truth. The candidate set is built
   with a ONE-STEP myopic heuristic. Scoring candidates by their H-step outcome
   would make the candidate set oracle-selected and every DAR number afterwards a
   tautology. (Caught by hand-checking: the first draft did exactly this.)
2. It never trains anything. y is simulator output; a predictor supplies y_pred.
   Blurring the two is how "we fit the simulator" gets written down as "we
   recovered the causal structure".
3. It never clips the potential used for noise detection. `adapter.outcome()` is
   the Gate-0 potential and stays unclipped. Y_prog is a different quantity: as of
   spec v1.1 it is SIGNED and may be negative -- see `progress_from_potential`.
4. It never omits the decision-time observation. A dataset that stores only arms
   and truths can be scored but cannot be *predicted from*; every predictor,
   including the SCM and the VLA, needs what the agent could see before acting.
   `observe()` is that vector and `predictors.FeatureTable` is its only consumer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import metrics as M

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class SamplerConfig:
    K: int = 5                 # spec: candidate_set.K_primary
    H: int = 6                 # spec v1.3: outcome.primary_horizon (was 3)
    plan_len: int = 3          # spec v1.3: decision_unit.plan_len (must be < H)
    T: int = 15                # decision points per episode
    warmup: int = 6            # base-policy steps before the first decision point
    rule: str = "R-samp"       # spec v1.2: candidate_set.rule (was R-top, see below)
    n_cand: int = 8            # pool size scored by the myopic heuristic (R-top only)
    proposal_sigma: float = 0.8   # dispersion of the alternatives, in action units
    min_arm_sep: float = 0.25     # alternatives differ less than this = duplicates
    min_spread: float = 0.125     # ... and their MEAN difference must clear this too
    plan_alt_frac: float = 0.5    # share of the perturbation that alternates sign
    control_substeps: int = 50    # physics substeps per control step
    eps_degen: float = 1e-6    # spec: noise_floor.deterministic_fallback
    seed: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# frozen base policy
# --------------------------------------------------------------------------- #
def base_policy(adapter, target_xy: Sequence[float]) -> np.ndarray:
    """Frozen scripted expert: get behind the object, then push it at the target.

    In the paper this slot is the VLA policy; here it must be deterministic so
    that "closed loop" is well defined before any policy exists. It reads object
    and target positions directly (privileged state) -- acceptable for a scripted
    expert, and it must be stated as such: it is NOT a learned policy and no
    claim about learned-policy behaviour should rest on it.

    Deterministic argmax with a fixed tiebreak, per spec:
      - if the object is already at the target, direction defaults to +x
      - if the pusher is already at the contact point, the action is exactly zero
    """
    poses = adapter.poses()
    obj = poses[0, :2]
    psh = poses[2, :2]
    d = np.asarray(target_xy, dtype=np.float64)[:2] - obj
    n = float(np.linalg.norm(d))
    u = np.array([1.0, 0.0]) if n < 1e-9 else d / n        # fixed tiebreak
    # The expert must aim INSIDE the contact distance, or it never pushes anything.
    # Contact happens at half_size + 0.012 = 0.028-0.036 m; the previous 0.045 left
    # the pusher 0.009-0.017 m short on every step of every episode, so the object
    # never moved and the scene's causal factors (mass, friction, half-size) were
    # inert. The value is supplied by the adapter because it is a property of the
    # scene geometry, and it is recorded in the fingerprint.
    standoff = float(getattr(adapter, "expert_standoff", 0.030))
    contact = obj - u * standoff
    err = contact - psh
    # Gain 5x the per-step displacement, not 2x. At 2x the expert's position error
    # (2-4 cm) divided by 0.016 m is 1.25-2.5, so the action saturates at +/-1 on
    # essentially every step: the expert was a bang-bang controller sitting exactly
    # on a corner of the action box. Perturbing a corner and clipping lands back on
    # the corner, so "K alternatives" collapsed into duplicates of arm 0 (measured
    # 42-76% of components pinned at the limit). At 5x the same error maps to
    # 0.5-0.8 and the policy has room around it.
    scale = max(getattr(adapter, "action_scale", 0.008) * 5.0, 1e-9)
    a_xy = np.clip(err / scale, -1.0, 1.0)
    if float(np.max(np.abs(err))) < 1e-9:                  # fixed zero tiebreak
        a_xy = np.zeros(2)
    return np.array([a_xy[0], a_xy[1], 0.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
def progress_from_potential(phi_start: float, phi_end: float, phi_ref: float) -> float:
    """Y_prog = SIGNED normalised potential GAIN over H steps (spec v1.1).

        Y = (phi_end - phi_start) / phi_ref      # positive = progressed, negative = regressed

    Sign convention matters and I got it backwards once: `adapter.outcome()` is a
    "higher is better" potential, so progress is phi_end - phi_start. Using the
    reversed order turned every real gain into a negative number, which the v1.0
    floor then flattened to 0 -- producing a dataset where 36.7% of steps looked
    degenerate and 95.7% of arms looked better than the scripted expert. Those two
    symptoms were the sign bug wearing the costume of a metric problem.

    v1.0 floored this at zero and the pilot showed why that is wrong: every arm
    worse than the factual arm collapsed to exactly 0, so Delta = Y^k - Y^0 was the
    same number for all of them. "This arm hurts" became indistinguishable from
    "this arm does nothing", DAR degenerated into 5-way accuracy of "which arm is
    best", and Gate-2's chance level fell from 0.50 to 0.20 (measured). A signed Y
    keeps the direction a direction metric needs. See spec v1.1 changelog.

    Note the asymmetry with the Gate-0 potential: `adapter.outcome()` is never
    clipped (clipping it would flat-line the determinism test), while Y is allowed
    to be negative. They are different quantities and must not be conflated.
    """
    if phi_ref <= 0:
        raise ValueError("phi_ref must be positive, got %r" % (phi_ref,))
    return float((float(phi_end) - float(phi_start)) / float(phi_ref))


# --------------------------------------------------------------------------- #
# candidate set
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# decision-time observation
# --------------------------------------------------------------------------- #
OBS_NAMES = (
    "obj_x", "obj_y", "obj_z",
    "dis_x", "dis_y", "dis_z",
    "psh_x", "psh_y", "psh_z",
    "tgt_x", "tgt_y", "tgt_z",
    "rel_obj_tgt_x", "rel_obj_tgt_y",
    "rel_psh_obj_x", "rel_psh_obj_y",
    "d_obj_tgt", "d_psh_obj",
    "phi",
    "ctrl_x", "ctrl_y", "ctrl_z",
)


def observe(adapter, phi_now: float) -> np.ndarray:
    """The observation at decision time, as a flat vector named by OBS_NAMES.

    Built from `poses()` (part of the SimAdapter surface) plus an optional
    `controller_state()` hook. Nothing here is future information -- no arm has
    been rolled out when this is called.

    Deliberately excludes mass, friction and half-size. Those are the causal
    factors the method is supposed to *infer*; feeding them in would turn "our
    model recovered the effect of friction" into "we handed it friction". A
    robot could not sense them either.
    """
    p = np.asarray(adapter.poses(), dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 4 or p.shape[1] < 3:
        raise ValueError("poses() must be (>=4, >=3), got %r" % (p.shape,))
    obj, dis, psh, tgt = p[0, :3], p[1, :3], p[2, :3], p[3, :3]

    hook = getattr(adapter, "controller_state", None)
    ctrl = (np.asarray(hook(), dtype=np.float64) if callable(hook)
            else np.zeros(3, dtype=np.float64))

    vec = np.concatenate([
        obj, dis, psh, tgt,
        obj[:2] - tgt[:2],
        psh[:2] - obj[:2],
        [float(np.linalg.norm(obj[:2] - tgt[:2])),
         float(np.linalg.norm(psh[:2] - obj[:2])),
         float(phi_now)],
        ctrl,
    ])
    if vec.size != len(OBS_NAMES):
        raise ValueError("observation has %d slots but OBS_NAMES has %d"
                         % (vec.size, len(OBS_NAMES)))
    return vec


def _myopic_score(adapter, state, action: np.ndarray, phi_now: float) -> float:
    """One-step potential GAIN (higher is better). Never the H-step outcome.

    Sign slipped here once too (returned phi_now - phi_after), which made R-top
    keep the five WORST candidates: 95.7% of arms then looked worse than the
    scripted expert and DAR became a near-free "predict the factual arm" test.
    The lesson generalises: phi is "higher is better", so every `before - after`
    in this codebase is a bug unless the comment says it is a loss.
    """
    adapter.load_state(state)
    adapter.step(action)
    return float(adapter.outcome()) - float(phi_now)


def factual_plan(adapter, state, target_xy: Sequence[float], H: int
                 ) -> Tuple[np.ndarray, float]:
    """The factual H-step action sequence and its end potential, from `state`.

    One rollout serves three purposes, which is why it is computed once:

      - its first `plan_len` rows are the factual PLAN, i.e. arm 0;
      - all H rows are what open-loop replay must reuse in the tail;
      - the end potential is phi for the factual arm under both conventions.

    The adapter is left back at `state`. Callers that roll candidates afterwards
    must not assume anything about where the simulator sits.
    """
    adapter.load_state(state)
    acts: List[np.ndarray] = []
    for _ in range(int(H)):
        a = np.asarray(base_policy(adapter, target_xy), dtype=np.float64)
        acts.append(a)
        adapter.step(a)
    phi_end = float(adapter.outcome())
    adapter.load_state(state)
    return np.asarray(acts, dtype=np.float64), phi_end


def _order_permutations(fact: np.ndarray, rng: np.random.Generator,
                        cfg: SamplerConfig, n: int,
                        frame: Optional[Tuple[np.ndarray, np.ndarray]] = None
                        ) -> np.ndarray:
    """K-1 plans that differ ONLY in the ORDER of the same deviation multiset.

    Why this rule exists (measured, not preferred)
    ----------------------------------------------
    With the v1.3 perturbation (persistent bias + alternating term), a SINGLE scalar
    -- the summed aligned push sum_h(du_h) -- explains R^2 = 0.80 of the within-step
    truth, and all eleven structural features together explain 0.84. A zero-parameter
    heuristic ranks arms at DAR 0.897, above the one-step oracle's 0.869. So on that
    candidate set the ranking is rank-1: "did this plan push more or less than the
    expert", and no amount of structure has anything else to find.

    This rule removes that channel by construction. Every arm receives a permutation
    of the SAME zero-sum deviation pattern b:

        b_h = g_h - mean_h(g),        so sum_h b_h = 0 exactly
        plan_k = fact + P_k(b),       P_k a distinct permutation

    Consequences, all of them checkable rather than asserted:
      * every arm has the same total push as the factual plan (sum b = 0),
      * every arm has the same multiset of step deviations, hence the same
        magnitude, the same |lateral| sum, the same everything except order,
      * a monotone-in-total-push heuristic cannot discriminate and must land at
        chance, while a model that gets the sequence right can still rank.

    For m = 3 there are exactly 6 permutations, which is exactly K = 5 arms plus the
    factual plan, so no arm has to be dropped. For m > 3 a deterministic subset is
    used; the rule requires m! >= K.
    """
    from itertools import permutations

    m = int(cfg.plan_len)
    if m > fact.shape[0]:
        raise ValueError("plan_len=%d exceeds the factual length %d"
                         % (m, fact.shape[0]))
    perms = [p for p in permutations(range(m))]
    if len(perms) < n + 1:
        raise ValueError("R-perm needs m! >= K (m=%d gives %d permutations, K=%d)"
                         % (m, len(perms), n + 1))
    if frame is None:
        raise ValueError("R-perm needs the task frame (u, n) to place its pattern")
    n_hat = np.asarray(frame[1], dtype=np.float64)[:2]

    # A zero-sum RAMP along the lateral axis: p_h = h - (m-1)/2, normalised so that
    # max|p| = 1. Its permutations are the arms, so every arm has the same total push
    # (sum p = 0), the same multiset of steps, and differs only in WHEN the lateral
    # motion happens.
    #
    # Why lateral and not aligned: the scripted expert sits on the +1 limit of the
    # component pointing at the target (it is a bang-bang controller), so the upward
    # headroom on that axis is zero and any zero-sum pattern there must be identically
    # zero. Alternating projections onto {sum = 0} cap {box} found exactly that and
    # collapsed to repeated steps -- which is how the constraint was discovered. Order
    # in this scene can only live in the lateral channel, and that is a property of
    # the expert, not of the code.
    p = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
    p = p / float(np.max(np.abs(p)))

    a_n = fact[:m, :2] @ n_hat                       # lateral component per step
    # the ramp has a zero entry; divide only where it is non-zero
    pm = np.maximum(np.abs(p), 1e-12)
    A = float(min(cfg.proposal_sigma, np.min((1.0 - np.abs(a_n)) / pm)))
    if not np.isfinite(A) or A < float(cfg.min_arm_sep):
        raise RuntimeError(
            "R-perm has only %.3f of lateral headroom at this decision (needs %.3f): "
            "the pusher is pinned laterally too, so no order-only pattern fits in the "
            "action box. State property, not a tuning issue."
            % (A, cfg.min_arm_sep))

    b = np.zeros((m, 3), dtype=np.float64)
    b[:, :2] = (A * p)[:, None] * n_hat[None, :]

    out: List[np.ndarray] = []
    for q in range(len(perms)):
        cand = fact[:m] + b[list(perms[q])]
        if np.max(np.abs(cand)) > 1.0 + 1e-12:
            continue                      # would clip: multiset no longer equal
        if float(np.abs(cand - fact[:m]).max()) < float(cfg.min_arm_sep):
            continue
        if any(float(np.abs(cand - o).max()) < float(cfg.min_arm_sep) for o in out):
            continue
        out.append(cand)
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(
            "R-perm produced only %d distinct permutations (needed %d) at m=%d"
            % (len(out), n, m))
    return np.stack(out)


def _distinct_plans(fact: np.ndarray, rng: np.random.Generator,
                    cfg: SamplerConfig, n: int) -> np.ndarray:
    """Draw n (m,3) plans that really differ from arm 0 and from each other.

    TWO separation tests, not one, because they catch different failures:

      max|Delta a| >= min_arm_sep   the plans are genuinely different somewhere
      mean|Delta a| >= min_spread   the difference is not one single spike

    A plan that clips back onto the factual plan is not an alternative: it
    duplicates arm 0, drives Delta to exactly zero, and silently turns "K options"
    into fewer than K. Reject-and-resample deterministically, and fail loudly if the
    action box cannot host K distinct plans -- a saturated box is a scene bug.

    The perturbation is a MIX of a persistent bias and a per-step alternating term:

        eps_h = (1-w) * b + w * (-1)^h * q,     b, q ~ N(0, sigma^2)^3

    With only the persistent term, every plan is the same direction scaled over the
    plan, i.e. the single action multiplied by plan_len -- multi-step content stays
    exactly zero while the code looks fixed. The alternating term is what makes
    "left then right" distinguishable from "right then left", and it is the only
    part that can carry information the first action does not already hold.
    """
    m = fact.shape[0]
    min_sep = float(cfg.min_arm_sep)
    min_spread = float(cfg.min_spread)
    w = float(cfg.plan_alt_frac)
    sign = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)[:, None]
    # Normalise so that EACH step's perturbation has marginal std == proposal_sigma
    # regardless of w and of m. Without it, m=1 would draw sqrt((1-w)^2 + w^2) =
    # 0.707 sigma while m=3 drew sigma -- and then "m=1 collapses, m=3 does not"
    # would have a dispersion explanation that had nothing to do with the plan.
    norm = 1.0 / float(np.hypot(1.0 - w, w))

    out: List[np.ndarray] = []
    max_tries = 500 * max(n, 1)
    for _ in range(max_tries):
        if len(out) == n:
            break
        b = rng.normal(0.0, cfg.proposal_sigma, 3)
        q = rng.normal(0.0, cfg.proposal_sigma, 3)
        p = np.clip(fact + norm * ((1.0 - w) * b[None, :] + w * sign * q[None, :]),
                    -1.0, 1.0)
        d = np.abs(p - fact)
        if float(d.max()) < min_sep or float(d.mean()) < min_spread:
            continue
        if any(float(np.abs(p - o).max()) < min_sep
               or float(np.abs(p - o).mean()) < min_spread for o in out):
            continue
        out.append(p)
    if len(out) < n:
        raise RuntimeError(
            "could not draw %d plans with max|da| >= %.2f and mean|da| >= %.3f "
            "(sigma=%.2f); the action box is saturated -- check the expert gain "
            "before lowering sigma" % (n, min_sep, min_spread, cfg.proposal_sigma))
    return np.stack(out)


def build_candidates(adapter, state, fact_actions: np.ndarray, phi_now: float,
                     rng: np.random.Generator, cfg: SamplerConfig
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (plans (K,m,3), myopic (K,)) with plan 0 = the factual plan.

    The myopic score is the ONE-STEP gain of each plan's FIRST action. It is
    computed for every arm, including arm 0. It used to be zeroed under R-samp,
    which turned the myopic reference into "no effect" -- a reference that silently
    stops being one is worse than a slow one.

    Note what "myopic" now means: it is the strongest information available from one
    step, and whether it can rank an m-step plan is exactly the question the plan
    unit was introduced to make testable. It is deliberately NOT the sum of one-step
    gains along the plan, because that telescopes to the plan's own progress and
    would be the truth wearing a reference's clothes.
    """
    m = int(cfg.plan_len)
    if not (1 <= m <= fact_actions.shape[0]):
        raise ValueError("plan_len=%d is outside 1..%d" % (m, fact_actions.shape[0]))
    a_fact = fact_actions[:m]

    if cfg.rule == "R-samp":
        alts = _distinct_plans(a_fact, rng, cfg, cfg.K - 1)
        plans = np.concatenate([a_fact[None, :, :], alts], axis=0)
    elif cfg.rule == "R-perm":
        # order-only candidates: same total push, same multiset, different sequence.
        # Any ranking skill on this set is sequence skill by construction, which is
        # what makes "multi-step" testable at all (see _order_permutations).
        # The task frame comes from the CURRENT poses, not from a stored target: the
        # decision is made at this state, and the lateral axis is what "sideways"
        # means here.
        ps = np.asarray(adapter.poses(), dtype=np.float64)
        rel = (ps[3, :2] - ps[0, :2])            # object -> target
        nrel = float(np.linalg.norm(rel))
        u_hat = np.array([1.0, 0.0]) if nrel < 1e-9 else rel / nrel
        alts = _order_permutations(a_fact, rng, cfg, cfg.K - 1,
                                   frame=(u_hat, np.array([-u_hat[1], u_hat[0]])))
        plans = np.concatenate([a_fact[None, :, :], alts], axis=0)
    elif cfg.rule == "R-top":
        # Selecting candidates by their one-step myopic score would bake the
        # one-step oracle into the data and make "can one-step information rank a
        # plan" unanswerable. Legal only when the plan IS one step.
        if m > 1:
            raise ValueError(
                "R-top is forbidden with plan_len=%d: it selects candidates by a "
                "one-step score, i.e. it would answer the multi-step question by "
                "construction (spec v1.3 decision_unit.rule_forbidden_when_plan)"
                % m)
        cand = _distinct_plans(a_fact, rng, cfg, cfg.n_cand)
        sc = np.array([_myopic_score(adapter, state, c[0], phi_now) for c in cand])
        picked = [int(i) for i in np.argsort(-sc)[: cfg.K - 1]]
        plans = np.concatenate([a_fact[None, :, :], cand[picked]], axis=0)
    else:
        raise ValueError("unknown candidate rule %r" % (cfg.rule,))

    myopic = np.array([_myopic_score(adapter, state, p[0], phi_now) for p in plans])
    return plans, myopic


# --------------------------------------------------------------------------- #
# one decision point
# --------------------------------------------------------------------------- #
def twin_rollout(adapter, state, phi_now: float, plans: np.ndarray,
                 fact_actions: np.ndarray, phi_fact_end: float, *,
                 H: int, target_xy: Sequence[float]) -> Dict[str, np.ndarray]:
    """K plans x 2 conventions, all from the same saved `state`.

    Shape of a decision: the plan's m action steps are executed verbatim under BOTH
    conventions -- the decision is the plan, so replaying it is not a modelling
    choice. The conventions only diverge in the remaining `H - m` steps:

        C  re-plan with the frozen policy            (what a closed-loop agent gets)
        O  replay the factual actions from the plan  (what the same plan does on rails)

    When `plan_len == H` the two are identical by construction, which is why the
    spec requires m < H: otherwise the C/O gap, a reported result, is a constant 0.
    """
    K, m, _ = plans.shape
    if not (1 <= m <= H):
        raise ValueError("plan_len=%d outside 1..%d" % (m, H))
    tail = H - m

    phi_C = np.empty(K)
    phi_O = np.empty(K)
    phi_S = np.empty(K)

    # Arm 0 runs through the same path as every other arm on purpose. It used to be
    # assigned phi_fact_end directly, and then asserted equal to phi_fact_end --
    # a check that could not fail because it compared a value with itself. Rolling
    # it out costs H steps and turns that tautology into the real invariant:
    # executing the factual plan and continuing with the frozen policy must land
    # exactly where the factual rollout landed, or the plan and the policy have
    # silently diverged and every Delta is measured against a different baseline
    # than the one in the report.
    for k in range(K):
        # S: EXACTLY one step -- the plan's first action, nothing else. This column
        # is the one-step truth, and `horizon_collapse` measures "can one-step
        # information rank the decision" by comparing it with C. Leaving it as "all
        # m steps of the plan" (which it became for one revision) quietly changed
        # what that statistic meant, and broke the identity with the one-step
        # oracle -- which is exactly how the drift was caught.
        adapter.load_state(state)
        adapter.step(plans[k][0])
        phi_S[k] = adapter.outcome()

        adapter.load_state(state)
        for h in range(m):
            adapter.step(plans[k][h])
        for h in range(tail):
            adapter.step(base_policy(adapter, target_xy))
        phi_C[k] = adapter.outcome()

        adapter.load_state(state)
        for h in range(m):
            adapter.step(plans[k][h])
        for h in range(tail):
            adapter.step(fact_actions[m + h])
        phi_O[k] = adapter.outcome()

    # Tolerance, measured rather than guessed. Recomputing `base_policy` from a
    # reloaded state gives actions that differ by ~4e-14 (the poses go through a
    # different sequence of floating-point operations even for bit-identical state).
    # In a contact-rich scene that is amplified: the worst observed deviation over a
    # 6-step rollout was 6.2e-7 in potential, i.e. 2.7e-6 in Y units, against Delta
    # magnitudes of ~1e-2. A STRUCTURAL mismatch -- wrong actions or wrong order --
    # lands at 1e-3..1e-1, so 1e-5 separates the two cleanly; 1e-9 separated nothing
    # and only asserted more precision than the scene provides.
    ARM0_TOL = 1e-5
    dev = max(abs(float(phi_C[0]) - float(phi_fact_end)),
              abs(float(phi_O[0]) - float(phi_fact_end)))
    if dev > ARM0_TOL:
        raise AssertionError(
            "arm 0 ended at %.12f (C) / %.12f (O) but the factual rollout ended at "
            "%.12f -- deviation %.2e exceeds the %.0e tolerance, so the factual plan "
            "is not the policy's own action sequence"
            % (phi_C[0], phi_O[0], phi_fact_end, dev, ARM0_TOL))

    return {"phi_start": phi_now, "phi_C": phi_C, "phi_O": phi_O,
            "phi_S": phi_S, "phi_fact_end": phi_fact_end,
            "fact_actions": np.asarray(fact_actions), "arm0_dev": dev}


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def run_episode(adapter, *, episode_id: str, factors: Dict[str, object],
                cfg: SamplerConfig, phi_ref: float, seed: int) -> Dict[str, object]:
    adapter.set_factors(**factors)
    adapter.reset(seed)
    target_xy = factors["target_xy"]

    for _ in range(cfg.warmup):
        adapter.step(base_policy(adapter, target_xy))

    rng = np.random.default_rng(seed * 100003 + 17)
    steps: List[Dict[str, object]] = []
    arm0_dev = 0.0

    for t in range(cfg.T):
        state = adapter.save_state()
        phi_now = adapter.outcome()
        # observe BEFORE anything rolls out: factual_plan and build_candidates both
        # move the simulator, and an observation taken afterwards would describe a
        # possible future rather than the present.
        obs = observe(adapter, phi_now)
        fact_actions, phi_fact_end = factual_plan(adapter, state, target_xy, cfg.H)
        plans, scores = build_candidates(adapter, state, fact_actions, phi_now, rng, cfg)

        tr = twin_rollout(adapter, state, phi_now, plans, fact_actions, phi_fact_end,
                          H=cfg.H, target_xy=target_xy)
        y_C = np.array([progress_from_potential(phi_now, p, phi_ref) for p in tr["phi_C"]])
        y_O = np.array([progress_from_potential(phi_now, p, phi_ref) for p in tr["phi_O"]])
        y_S = np.array([progress_from_potential(phi_now, p, phi_ref) for p in tr["phi_S"]])

        degen_C = bool(np.max(np.abs(y_C - y_C[0])) <= cfg.eps_degen)
        degen_O = bool(np.max(np.abs(y_O - y_O[0])) <= cfg.eps_degen)

        arm0_dev = max(arm0_dev, float(tr["arm0_dev"]))
        steps.append({
            "t": t,
            "phi_start": float(phi_now),
            "obs": obs.tolist(),
            # the decision unit (K, m, 3) and its first-step view. `arms` is derived,
            # not independent: it exists so that "what a one-step agent could see of
            # this decision" is answerable without re-deriving it in three places.
            "plans": plans.tolist(),
            "arms": plans[:, 0, :].tolist(),
            "myopic": scores.tolist(),
            "y_C": y_C.tolist(), "y_O": y_O.tolist(), "y_S": y_S.tolist(),
            "phi_C": tr["phi_C"].tolist(), "phi_O": tr["phi_O"].tolist(),
            "degenerate_C": degen_C, "degenerate_O": degen_O,
        })

        # Advance the TRAJECTORY by exactly one control step, with the expert's own
        # first action -- the action the closed-loop convention would have taken.
        #
        # This line is load-bearing and it was missing for one revision. twin_rollout
        # returns with the simulator sitting at the last candidate's open-loop END,
        # so without advancing here the next decision point starts wherever that
        # rollout happened to stop, not at step t+1 of the expert's trajectory. The
        # 15 "decision points" of an episode then do not form a trajectory at all:
        # they are 15 independent probes from drifting, off-policy states.
        #
        # It was found by measuring the cadence instead of assuming it: consecutive
        # decision points moved the pusher by 0.021 m (median), while one control
        # step can move it at most action_scale*sqrt(2) = 0.0113 m, and the commanded
        # setpoint advanced by 3x action_scale. Everything downstream --
        # horizon_collapse, the DAR ladder, any learned dynamics -- inherits whatever
        # trajectory this line produces, which is why the check lives in run_selftest.
        adapter.load_state(state)
        adapter.step(fact_actions[0])

    return {
        "episode_id": episode_id,
        "factors": {k: (list(v) if isinstance(v, (tuple, list)) else v)
                    for k, v in factors.items()},
        "seed": seed,
        "phi_ref": phi_ref,
        "arm0_dev": arm0_dev,
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def estimate_phi_ref(adapter, factor_list: Sequence[Dict[str, object]],
                     cfg: SamplerConfig, seeds: Sequence[int]) -> float:
    """Reference potential = median potential at the first decision point.

    A per-episode normaliser would make cross-episode Y values incomparable under
    the cluster bootstrap; a single global scalar keeps them on one scale and is
    fixed before any arm is ever rolled out, so it cannot be tuned to the result.
    """
    vals = []
    for fac, sd in zip(factor_list, seeds):
        adapter.set_factors(**fac)
        adapter.reset(sd)
        for _ in range(cfg.warmup):
            adapter.step(base_policy(adapter, fac["target_xy"]))
        vals.append(float(adapter.outcome()))
    ref = float(np.median(vals))
    if ref <= 0:
        raise ValueError("phi_ref came out non-positive (%r); check the adapter" % (ref,))
    return ref


def generate(adapter, factor_list: Sequence[Dict[str, object]], *,
             cfg: SamplerConfig, ids: Sequence[str]) -> Dict[str, object]:
    # The decision cadence is a property of the task, so it is declared once in the
    # config and the adapter must agree. 50 substeps at 2 ms = 10 Hz, a policy-class
    # rate: at 5 substeps the horizon H=3 was 30 ms, i.e. three servo ticks, and the
    # counterfactual spread was 0.6% of Y's between-step variation.
    sub = getattr(adapter, "substeps", None)
    if sub is not None and int(sub) != int(cfg.control_substeps):
        raise ValueError(
            "adapter runs %s substeps/control step but the config declares %d: "
            "build the adapter with substeps=cfg.control_substeps"
            % (sub, cfg.control_substeps))
    # plan_len < H is a structural requirement, not a preference: at plan_len == H
    # the closed- and open-loop conventions execute the same actions and the C/O
    # gap degenerates to a constant zero -- a reported result turning into an
    # artifact of the setting.
    if not (1 <= int(cfg.plan_len) < int(cfg.H)):
        raise ValueError("plan_len=%d must satisfy 1 <= plan_len < H=%d"
                         % (cfg.plan_len, cfg.H))
    if cfg.rule == "R-top" and cfg.plan_len > 1:
        raise ValueError("R-top cannot be used with plan_len=%d (spec v1.3: it "
                         "selects candidates by a one-step score)" % cfg.plan_len)

    seeds = [cfg.seed * 7919 + i for i in range(len(factor_list))]
    phi_ref = estimate_phi_ref(adapter, factor_list, cfg, seeds)
    episodes = [run_episode(adapter, episode_id=ids[i], factors=factor_list[i],
                            cfg=cfg, phi_ref=phi_ref, seed=seeds[i])
                for i in range(len(factor_list))]
    return {
        "meta": {
            "spec_version": M.SPEC_VERSION,
            "K": cfg.K, "H": cfg.H, "T": cfg.T, "rule": cfg.rule,
            "decision_unit": "plan",
            "plan_len": cfg.plan_len,
            "plan_tail": int(cfg.H) - int(cfg.plan_len),
            # Worst observed |arm0 rollout - factual rollout| over the whole dataset.
            # Not a pass/fail flag: it is the measured precision of "the factual plan
            # is the policy's own sequence", so a reader can see how far Delta's
            # baseline can be from the reported one. 0.0 would be a lie.
            "arm0_max_dev": max(ep["arm0_dev"] for ep in episodes),
            "phi_ref": phi_ref,
            "eps_degen": cfg.eps_degen,
            "config": cfg.as_dict(),
            "adapter": adapter.fingerprint(),
            "conventions": {"primary": "C_closed_loop", "secondary": "O_open_loop",
                            "diagnostic_only": "S_one_step"},
            "base_policy": "scripted_frozen_expert (NOT a learned policy)",
        },
        "episodes": episodes,
    }


def records_from(dataset: Dict[str, object], *, convention: str = "C",
                 y_pred: Optional[np.ndarray] = None,
                 episode_ids: Optional[Sequence[str]] = None) -> List[dict]:
    """Flatten a dataset into metrics.py records for one convention.

    `y_pred` is (n_steps, K) in the same row order as the emitted rows; if omitted,
    the factual arm is used as the prediction, which is the "no-effect" baseline and
    must score DAR=0.

    `episode_ids` restricts the output to a subset (a split). The row order is the
    dataset order, filtered -- the same order `predictors.features_from` produces,
    which is what makes it safe to hand a prediction matrix in from the outside.
    """
    key = "y_%s" % convention
    dkey = "degenerate_%s" % convention
    keep = None if episode_ids is None else set(episode_ids)
    rows: List[dict] = []
    i = 0
    for ep_id, st in _iter_steps(dataset, keep):
        k_arms = len(st[key])
        if y_pred is not None and i >= len(y_pred):
            raise ValueError("y_pred has %d rows but the selection has more steps"
                             % len(y_pred))
        pred = y_pred[i] if y_pred is not None else np.array([st[key][0]] * k_arms)
        if len(pred) != k_arms:
            raise ValueError("y_pred row %d has %d arms, record has %d"
                             % (i, len(pred), k_arms))
        for k in range(k_arms):
            rows.append({"episode": ep_id, "t": st["t"], "k": k,
                         "y_true": float(st[key][k]),
                         "y_pred": float(pred[k]),
                         "degenerate": bool(st[dkey])})
        i += 1
    if y_pred is not None and i != len(y_pred):
        raise ValueError("y_pred has %d rows but %d steps were emitted"
                         % (len(y_pred), i))
    return rows


def _iter_steps(dataset: Dict[str, object], keep: Optional[set]):
    """Yield (episode_id, step) in dataset order, filtered by `keep`.

    Features and truths must be built from this one iterator, or a prediction
    matrix can end up zipped against the wrong rows and every number becomes
    fiction that still looks plausible.
    """
    for ep in dataset["episodes"]:                      # type: ignore[index]
        if keep is not None and ep["episode_id"] not in keep:
            continue
        for st in ep["steps"]:                          # type: ignore[index]
            yield ep["episode_id"], st


def observation_matrix(dataset: Dict[str, object],
                       episode_ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """(n_steps, len(OBS_NAMES)) decision-time observations, in dataset order."""
    keep = None if episode_ids is None else set(episode_ids)
    rows = []
    for ep_id, st in _iter_steps(dataset, keep):
        if "obs" not in st:
            raise KeyError("dataset has no 'obs' field: regenerate it, a dataset "
                           "without observations cannot feed any predictor")
        rows.append(np.asarray(st["obs"], dtype=np.float64))
    return np.vstack(rows) if rows else np.zeros((0, len(OBS_NAMES)))


def n_steps_of(dataset: Dict[str, object]) -> int:
    return sum(len(ep["steps"]) for ep in dataset["episodes"])   # type: ignore[index]


def horizon_collapse(dataset: Dict[str, object], convention: str = "C") -> Dict[str, float]:
    """How much of the decision's ranking the FIRST step already decides.

    A task property, not a metric of any method -- and the one number that says
    whether "multi-step reasoning" is even testable on this data. It is the SAME
    statistic for both decision units, which is what makes the v1.2 -> v1.3 change
    a controlled comparison rather than a story:

      single-action unit (v1.2, H=3)    rank rho 0.905 | 73.9% identical | argmax 88.1%
      plan unit (v1.3, m=3, H=6)        measured by run_pilot, reported as plan content

    Why the v1.2 number was so high: the closed-loop convention lets the scripted
    expert re-plan immediately after the perturbation, so it undoes the deviation
    during the remaining H-1 steps. Enlarging candidate dispersion did NOT help
    (0.92-0.94 across sigma 0.8-3.2), so it was structural, not a tuning issue.

    Two consequences that must be stated wherever these numbers appear:
      - the one-step probe is the H=1 oracle (its delta equals the 1-step truth delta
        exactly), so it belongs among the references, not the baselines;
      - a high value means a DAR improvement does NOT license a claim about
        multi-step reasoning, because there is little multi-step content to reason
        about. That is exactly the condition the plan unit exists to remove, and why
        m=1 must be reported alongside m=3 as the ablation it is.
    """
    from scipy.stats import spearmanr

    rho, same = [], 0
    for ep in dataset["episodes"]:                              # type: ignore[index]
        for st in ep["steps"]:                                  # type: ignore[index]
            a = np.asarray(st["y_S"], dtype=np.float64)
            b = np.asarray(st["y_%s" % convention], dtype=np.float64)
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            rho.append(float(spearmanr(a, b).statistic))
            same += int(np.argmax(a) == np.argmax(b))
    if not rho:
        return {"n_steps": 0, "rho_mean": float("nan"),
                "identical_ranking_rate": float("nan"),
                "argmax_agreement": float("nan")}
    arr = np.asarray(rho)
    return {"n_steps": int(arr.size), "rho_mean": float(arr.mean()),
            "identical_ranking_rate": float(np.mean(np.abs(arr - 1.0) < 1e-9)),
            "argmax_agreement": same / float(arr.size)}


def save(dataset: Dict[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=1)
    return os.path.abspath(path)


def load(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
