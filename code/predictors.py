"""
predictors.py -- the ONE interface every method must implement, plus the baseline
ladder that a method has to beat before its numbers mean anything.

Contract
--------
    class SomePredictor:
        name: str
        def fit(self, train: FeatureTable, y_train: np.ndarray) -> "SomePredictor"
        def predict(self, test: FeatureTable) -> np.ndarray    # (n_steps, K)

`y` is on the same scale as `sampler.progress_from_potential`: the SIGNED
normalised potential gain over the horizon. Only within-step differences relative
to arm 0 matter for DAR; the absolute level matters for nMAE and regret, so a
predictor that only ranks arms should still return defensible magnitudes.

Two guarantees enforced by structure, not discipline
----------------------------------------------------
1. **A predictor cannot see y_true.** `FeatureTable` has no field holding it. There
   is nothing to read by accident. The oracle is therefore NOT implemented here --
   it lives in `metrics.ruler_selftest`, where it belongs, as a ruler rather than a
   method. That is a stronger guarantee than "we promise not to use the labels".
2. **A predictor cannot see `degenerate` either.** Degeneracy is defined as
   max|y_C - y_C[0]| <= eps, i.e. from the OUTCOME. Using it as an input is label
   leakage dressed up as a tidy rule ("we skip uninformative steps"). `metrics`
   already excludes those steps from the numerator and denominator, so no predictor
   ever needs it.

What IS available at decision time
----------------------------------
    obs     : geometry + potential + commanded setpoint (sampler.OBS_NAMES)
    plans   : the K candidate PLANS (K, m, 3). Plan 0 is the sequence the scripted
              expert would execute. The decision unit of spec v1.3 -- a plan, not a
              single action, because a single-action unit made the horizon decoration
              (see sampler.horizon_collapse).
    arms    : property = plans[:, :, 0] -- the first step of each plan. Not a second
              field, so "what a one-step agent can see" has exactly one definition.
    myopic  : one-step potential gain of each plan's FIRST action. Legal -- it is
              computed by rolling exactly one step, i.e. it is weaker information
              than the H-step truth it is asked to predict.
    phi_ref : dataset-level normaliser, fixed before any plan was ever rolled out

Deliberately absent: mass, friction, half-size. They are the causal factors the
method is meant to infer; feeding them in would turn "our model recovered the
effect of friction" into "we handed it friction".

The ladder, and why each rung exists (each carries a `kind`)
-----------------------------------------------------------
    B0 no-effect      every plan predicted to change nothing. Any method scoring at
                      or below this has no directional information at all.
    B1 bet-on-expert  plan 0 is predicted to win. The honest "do nothing but what
                      the expert did" reference: strong in a scripted-expert regime,
                      and a trap for methods that quietly learn "copy arm 0".
    B2 ridge          linear surrogate on [obs, plan flattened, is_factual]. Purpose:
                      show what is achievable WITHOUT modelling any interaction -- a
                      linear model cannot express "this push is good only when the
                      object is on this side", which is the gap the structural
                      method is supposed to close. The myopic score is NOT an input:
                      see `Ridge._design`.
    M1 structured plan  kind=method. Linear in features derived from the task's
                      causal factorisation -- plan deviation resolved in the
                      object->target frame, |lateral|, and the SUMMED aligned push --
                      instead of in the raw world-frame action. See `_plan_design`
                      for what each mechanism is and why it is there.
    C1 ridge arm-shuffled   negative control for B2: the action-outcome
                      correspondence is permuted within each step. It MUST land at
                      chance. If it does not, the pipeline manufactures skill out of
                      noise and no method's score on this data can be trusted.
    C2 plan-outcome shuffled  negative control for M1, and the one that matters: the
                      same structural design, but the plan-outcome correspondence is
                      permuted within each step. If M1's skill survives destroying
                      that correspondence, M1's skill was never about the plan.
    R1 one-step oracle      kind=reference, not baseline: it steps the simulator once
                      per candidate. Its DAR is the answer to "can one-step
                      information rank an m-step plan?" -- the whole point of the
                      plan unit, and the number M1 exists to beat.

On the control's design (this was wrong once)
---------------------------------------------
The first version permuted the *labels* within a step and kept the arms in place.
That is a no-op whenever the K outcomes of a step are nearly equal -- and they were:
within-step spread was about 0.5% of Y's between-step variation, so the permuted
target differed from the true target by ~1e-4 while the row level was ~8e-3. The
control model came out with weights identical to the real one to four decimals and
the same DAR (0.7714 vs 0.7714). It "passed" nothing and tested nothing.

Permuting the position block instead -- arms, myopic score and the is_factual flag
moved together, outcomes left in place -- destroys the mapping the model is supposed
to learn while preserving the marginal distribution of Y exactly. That is an
intervention the model cannot absorb.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M      # noqa: E402
import sampler as S      # noqa: E402


# --------------------------------------------------------------------------- #
# the feature table: everything a predictor may legitimately look at
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureTable:
    """Decision-time inputs only. No truths, no degeneracy flags, by construction."""
    obs: np.ndarray            # (n_steps, d_obs)
    plans: np.ndarray          # (n_steps, K, m, 3)  the decision unit (spec v1.3)
    myopic: np.ndarray         # (n_steps, K)  one-step gain of each plan's FIRST action
    phi_ref: float             # dataset-level normaliser, in potential units
    episode_ids: Tuple[str, ...]
    k: int

    def __len__(self) -> int:
        return int(self.plans.shape[0])

    @property
    def arms(self) -> np.ndarray:
        """The first step of each plan, (n_steps, K, 3). A view, not a second field.

        Kept as a property so that "what a one-step agent can see of this decision"
        has exactly one definition. Storing it separately would let the two drift.
        """
        return self.plans[:, :, 0, :]

    def __post_init__(self) -> None:
        if self.plans.ndim != 4:
            raise ValueError("plans must be (n, K, m, 3), got %r"
                             % (self.plans.shape,))
        if self.obs.shape[0] != self.plans.shape[0]:
            raise ValueError("obs rows %d != plans rows %d"
                             % (self.obs.shape[0], self.plans.shape[0]))
        if self.myopic.shape != self.plans.shape[:2]:
            raise ValueError("myopic shape %r != plans[:2] %r"
                             % (self.myopic.shape, self.plans.shape[:2]))
        if not (np.isfinite(self.phi_ref) and self.phi_ref > 0):
            raise ValueError("phi_ref must be finite and positive, got %r"
                             % (self.phi_ref,))
        if not np.isfinite(self.obs).all():
            raise ValueError("obs contains non-finite values")
        if not np.isfinite(self.plans).all():
            raise ValueError("plans contain non-finite values")

    def episodes(self) -> set:
        return set(self.episode_ids)


def features_from(dataset: Dict[str, object],
                  episode_ids: Optional[Sequence[str]] = None) -> FeatureTable:
    """Build the decision-time feature table for a whole dataset or a subset."""
    keep = None if episode_ids is None else set(episode_ids)
    obs, plans, myopic, eps = [], [], [], []
    shape = None
    for ep_id, st in S._iter_steps(dataset, keep):
        for key in ("obs", "plans", "myopic"):
            if key not in st:
                raise KeyError("dataset step is missing %r: regenerate it. A "
                               "dataset without %r cannot feed a predictor."
                               % (key, key))
        p = np.asarray(st["plans"], dtype=np.float64)
        if shape is None:
            shape = p.shape
        elif p.shape != shape:
            raise ValueError("plan shape changed mid-dataset (%r vs %r)"
                             % (p.shape, shape))
        obs.append(np.asarray(st["obs"], dtype=np.float64))
        plans.append(p)
        myopic.append(np.asarray(st["myopic"], dtype=np.float64))
        eps.append(ep_id)
    if not obs:
        raise ValueError("empty episode selection")

    meta = dataset["meta"]                                  # type: ignore[index]
    return FeatureTable(obs=np.vstack(obs), plans=np.stack(plans),
                        myopic=np.vstack(myopic),
                        phi_ref=float(meta["phi_ref"]),     # type: ignore[index]
                        episode_ids=tuple(eps), k=int(shape[0]))


def truth_matrix(dataset: Dict[str, object], convention: str,
                 episode_ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """(n_steps, K) truths in the SAME row order as `features_from`.

    Only `fit_predict` may call this, and only for the training split. It is kept
    public so tests can assert the row alignment, not so methods can call it.
    """
    key = "y_%s" % convention
    keep = None if episode_ids is None else set(episode_ids)
    rows = [np.asarray(st[key], dtype=np.float64)
            for _, st in S._iter_steps(dataset, keep)]
    if not rows:
        raise ValueError("empty episode selection")
    return np.stack(rows)


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
class Predictor(Protocol):
    name: str

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "Predictor": ...

    def predict(self, test: FeatureTable) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class NoEffect:
    """B0: every arm is predicted to change nothing. Delta is identically zero."""
    name = "B0 no-effect"
    kind = "baseline"

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "NoEffect":
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        return np.zeros((len(test), test.k), dtype=np.float64)


class FactualWins:
    """B1: arm 0 (the expert's action) is predicted to beat every alternative.

    The gap is set from the training split, so nMAE stays on the Y scale and this
    baseline is comparable with the others rather than being arbitrarily large.
    A method that cannot beat "always bet on the expert" has not learned anything
    about the counterfactual.
    """
    name = "B1 bet-on-expert"
    kind = "baseline"

    def __init__(self) -> None:
        self.gap = 1.0

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "FactualWins":
        d = y_train - y_train[:, [0]]
        vals = np.abs(d[:, 1:])
        vals = vals[np.isfinite(vals)]
        med = float(np.median(vals)) if vals.size else 1.0
        self.gap = med if (np.isfinite(med) and med > 0) else 1.0
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        out = np.zeros((len(test), test.k), dtype=np.float64)
        out[:, 0] = self.gap
        return out


class Myopic:
    """R1: the ONE-STEP ORACLE. Not a baseline -- it needs simulator access.

    `myopic` is produced by actually stepping the simulator once from the saved
    state and reading the potential. No agent has that faculty: it is a one-step
    model-based probe with a perfect model. So this belongs among the REFERENCES
    (like the full oracle), not among the baselines a method must beat. Listing it
    as a baseline invited the reading "a trivial heuristic already wins", which is
    the wrong conclusion -- the trivial heuristic was consulting the oracle.

    Its delta is exactly the one-step truth delta, not approximately:

        myopic_k - myopic_0 = phi_ref * (y_S_k - y_S_0)

    which `run_selftest.py` section G asserts. That identity is why this class is
    also the sharpest available evidence about the task: it measures how much of the
    H-step ranking the first step already decides (see sampler.horizon_collapse).
    """
    name = "R1 one-step oracle (model-based)"
    kind = "reference"

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "Myopic":
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        return test.myopic / float(test.phi_ref)


class HeuristicAlignedSum:
    """G0b: rank by the SUMMED aligned push. Zero fitted parameters.

    This exists because it was measured to matter: on the v1.3 candidate set it
    scores DAR 0.897, above the one-step oracle (0.869) and within 0.04 of the
    fifteen-parameter structural model. Any table that omits it invites the reviewer
    to compute it themselves and conclude the model is decoration -- and on that
    candidate set they would be right.

    It is a baseline precisely because it is the trivial explanation of the result.
    """

    name = "G0b summed aligned push (0 param)"
    kind = "baseline"

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "HeuristicAlignedSum":
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        u = _unit(-test.obs[:, _I_REL:_I_REL + 2])
        d = test.plans - test.plans[:, [0], :, :]
        du = d[..., 0] * u[:, None, None, 0] + d[..., 1] * u[:, None, None, 1]
        return du.sum(axis=2) / float(test.phi_ref)


class HeuristicMagnitude:
    """G0: rank by total path length of the deviation. Zero parameters.

    The silliest possible ranker. It is here to make "the task is trivial" a
    measurable statement rather than an opinion.
    """

    name = "G0 perturbation magnitude (0 param)"
    kind = "baseline"

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "HeuristicMagnitude":
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        d = test.plans - test.plans[:, [0], :, :]
        return -np.linalg.norm(d[..., :2], axis=-1).sum(axis=2) / float(test.phi_ref)


class GlobalDirection:
    """G1: one fixed linear direction over the SAME deviation features as M1.

    This class was written as "M1 without state interaction", and that description is
    WRONG. Measured: with the deviation blocks centred, the level block (row-constant)
    is exactly orthogonal to them, so the ridge solution for the deviation
    coefficients is the same with or without the level block. The within-row
    contrasts of M1's prediction therefore equal G1's to 1e-14, and the two rank
    identically at every alpha (rank-identical rate 1.000 at alpha 0.1/1/10). M1 and
    G1 are the SAME model, not two models.

    What that means for the paper, stated plainly: M1's content is the FEATURE MAP
    (project the plan deviation onto the object->target frame, take |lateral|, sum
    the aligned push) and a single fitted direction over it. It is a linear model on
    hand-designed state-dependent features, not a learned structural model, and any
    claim of the form "structure beats a global direction" is vacuous by algebra.

    It is kept because it is the cheapest statement of that algebra, and
    `run_selftest` asserts the identity: if someone ever adds a state interaction
    that actually changes the ranking, the assertion fires and the claim has to be
    revisited rather than assumed.
    """

    name = "G1 global direction (no state, 11 param)"
    kind = "baseline"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.w: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.n_lev = 4

    def _x(self, tab: FeatureTable) -> np.ndarray:
        return _plan_design(tab, 3)[:, :, self.n_lev:]

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "GlobalDirection":
        x = self._x(train)
        y = np.asarray(y_train, dtype=np.float64)
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        flat, tgt = xc.reshape(-1, xc.shape[-1]), yc.reshape(-1)
        self.mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        z = (flat - self.mu) / self.sd
        a = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.w = np.linalg.solve(a, z.T @ tgt)
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("GlobalDirection.predict called before fit")
        x = self._x(test)
        z = (x.reshape(-1, x.shape[-1]) - self.mu) / self.sd
        return (z @ self.w).reshape(x.shape[0], x.shape[1])


class Ridge:
    """B3: linear surrogate on [obs, arm, myopic, is_factual].

    Closed-form ridge, no dependency beyond numpy, because a baseline that needs a
    framework is a baseline nobody re-runs.

    Note what this model structurally CANNOT do: with no interaction terms it can
    only rank arms by a fixed linear functional of the action and the one-step
    score. "Push left is good only when the object sits right of the target" is not
    expressible. That limitation is the point -- it marks the gap the structural
    method must close, and if a fancy method does not clear this bar, the
    interactions it learned are not paying rent.
    """

    name = "B2 ridge surrogate"
    kind = "baseline"

    def __init__(self, alpha: float = 1.0, destroy_arm_association: bool = False,
                 seed: int = 0, plan_steps: Optional[int] = None) -> None:
        self.alpha = float(alpha)
        self.destroy_arm_association = bool(destroy_arm_association)
        self.seed = int(seed)
        # B3: truncate the plan to its first `plan_steps` steps. The plan-content
        # ablation for the LINEAR rung: if B3 matches B2, the m-step plan adds
        # nothing a linear model can use, and any claim about multi-step structure
        # would have to come from somewhere else.
        self.plan_steps = plan_steps
        self.w: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    # -- design matrix ---------------------------------------------------- #
    def _design(self, tab: FeatureTable) -> np.ndarray:
        """[obs, plan flattened (m*3), is_factual] -- myopic score EXCLUDED.

        Including the myopic score turns any learner into a wrapper around the
        one-step oracle: DAR is a step function of the sign of that one weight, so a
        model that happens to load it positively scores 1.000 and one that loads it
        negatively scores 0.000 -- nothing in between. The negative control could
        therefore only ever return 0 or 1, never chance, and "the method beats the
        baselines" would have been untestable.

        The myopic score is a BASELINE'S OUTPUT, not an observation. Feeding a
        baseline's output into the model being compared against it is the same
        mistake as selecting the candidate set with that baseline.

        The whole plan is flattened into the design. That is deliberate: this is the
        linear rung of the ladder, and it should be given every input a linear model
        could use. Flattening does not give it interaction terms, so "push left is
        good only when the object is right of the target" is still inexpressible --
        which is the gap the structural method is supposed to close.
        """
        n, k, m, a_dim = tab.plans.shape
        d = tab.obs.shape[1]
        steps = m if self.plan_steps is None else int(self.plan_steps)
        if not (1 <= steps <= m):
            raise ValueError("plan_steps=%r outside 1..%d" % (self.plan_steps, m))
        is_fact = np.zeros((n, k, 1), dtype=np.float64)
        is_fact[:, 0, 0] = 1.0
        obs = np.broadcast_to(tab.obs[:, None, :], (n, k, d))
        plan = tab.plans[:, :, :steps, :].reshape(n, k, steps * a_dim)
        return np.concatenate([obs, plan, is_fact], axis=-1)

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "Ridge":
        x = self._design(train)                       # (n, k, f)
        y = np.array(y_train, dtype=np.float64, copy=True)
        if self.destroy_arm_association:
            # Move the POSITION BLOCK (arms, myopic, is_factual) of each training
            # step, leaving the outcomes where they are. The model then sees real
            # outcomes attached to the wrong actions: the arm -> effect mapping is
            # gone, the marginal distribution of Y is untouched, and any skill left
            # over is skill the pipeline invented.
            #
            # Permuting the labels instead was a no-op: when the K outcomes of a step
            # differ by ~5e-4 while their level varies by ~1e-2, a within-row
            # permutation barely moves the regression target (measured: identical
            # weights to 4 decimals, identical DAR).
            rng = np.random.default_rng(self.seed)
            perm = np.argsort(rng.random(y.shape), axis=1)
            steps = train.plans.shape[2] if self.plan_steps is None else self.plan_steps
            w = steps * train.plans.shape[3] + 1                  # plan + is_factual
            block = x[:, :, -w:]
            x = np.concatenate(
                [x[:, :, :-w],
                 np.take_along_axis(block, perm[:, :, None], axis=1)], axis=-1)

        flat = x.reshape(-1, x.shape[-1])
        tgt = y.reshape(-1)
        self.mean = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self.scale = np.where(sd > 1e-12, sd, 1.0)
        z = (flat - self.mean) / self.scale

        a = z.T @ z + self.alpha * np.eye(z.shape[1])
        b = z.T @ (tgt - tgt.mean())
        self.w = np.linalg.solve(a, b)
        self.w = np.concatenate([[tgt.mean()], self.w])   # [intercept, weights]
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Ridge.predict called before fit")
        x = self._design(test)
        z = (x.reshape(-1, x.shape[-1]) - self.mean) / self.scale
        pred = self.w[0] + z @ self.w[1:]
        return pred.reshape(x.shape[0], x.shape[1])


class RidgeArmShuffled(Ridge):
    """C1: the negative control. Must land at chance, or nothing else counts."""
    name = "C1 ridge arm-shuffled (control)"
    kind = "control"

    def __init__(self, alpha: float = 1.0, seed: int = 0) -> None:
        super().__init__(alpha=alpha, destroy_arm_association=True, seed=seed)


# --------------------------------------------------------------------------- #
# M1: the structural model the baselines exist to be beaten by
# --------------------------------------------------------------------------- #
# Observation indices used below (sampler.OBS_NAMES): 12-13 = obj-tgt (xy),
# 16 = |obj-tgt|, 17 = |pusher-obj|, 18 = potential.
_I_REL, _I_DOT, _I_DPO, _I_PHI = 12, 16, 17, 18


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


def _plan_design(tab: FeatureTable, level: int = 3) -> np.ndarray:
    """Structural design matrix (n, K, f) -- one mechanism per level.

    The claim behind this file is not "more features win". It is that the effect of
    a plan factorises along the task's causal chain:

        plan deviation -> pusher displacement -> object displacement -> potential

    and that writing the input in the coordinates OF THAT CHAIN is what a flat
    [obs, plan] regression cannot do. Each level adds exactly one link, so the
    ablation can attribute a gain to a mechanism:

      1 frame   deviation resolved as aligned (u = object->target) and lateral (n).
                "push harder" and "push sideways" are the same two numbers in any
                frame, but only in this frame is the aligned one the one that moves
                the object toward the goal.
      2 + abs   |lateral|. Pushing off the line hurts whichever side it is on, and a
                signed term cannot say that.
      3 + cum   the SUMMED aligned deviation. Displacement accumulates down the
                plan; the sum over steps is the quantity the object actually feels,
                whereas per-step terms can only express independent nudges.
      4 + phi   summed aligned push x current potential. Forward value falls as the
                object nears the target (overshoot), so the same push is worth less
                at high phi. This is the interaction B2 structurally cannot write.
      5 + sq    square of the summed push: curvature/saturation of that value.
      6 + gate  soft contact gate. Kept as a registered variant because it was a
                plausible mechanism and it must be shown to be a no-op when contact
                is continuous (measured: 100% of steps here) rather than dropped
                silently.

    Deliberately excluded: `myopic`. It is R1's output, and feeding a baseline's
    output into the model that is compared against it is the same error as selecting
    candidates with it (see `Ridge._design`).
    """
    obs = tab.obs
    u = _unit(-obs[:, _I_REL:_I_REL + 2])                      # object -> target
    nvec = np.stack([-u[:, 1], u[:, 0]], axis=1)
    ux, uy = u[:, None, None, 0], u[:, None, None, 1]
    nx, ny = nvec[:, None, None, 0], nvec[:, None, None, 1]

    d = tab.plans - tab.plans[:, [0], :, :]                    # (n, K, m, 3)
    du = d[..., 0] * ux + d[..., 1] * uy                       # aligned
    dn = d[..., 0] * nx + d[..., 1] * ny                       # lateral
    nrow, k, _m = du.shape

    if level == 0:
        # The plan-content ablation, and the single most important control here:
        # the SAME frame, the SAME mechanisms, but only the plan's FIRST step. If
        # this matches level 3, then everything the model knows comes from the first
        # action and "multi-step plan" is a label, not a capability.
        blocks = [du[:, :, 0:1], dn[:, :, 0:1], np.abs(dn[:, :, 0:1])]
    else:
        blocks = [du, dn]
        if level >= 2:
            blocks.append(np.abs(dn))
        if level >= 3:
            cum_u = np.cumsum(du, axis=2)[:, :, [-1]]
            blocks.append(cum_u)
            blocks.append(np.abs(np.cumsum(dn, axis=2))[:, :, [-1]])
        if level >= 4:
            blocks.append(cum_u * obs[:, _I_PHI][:, None, None])
        if level >= 5:
            blocks.append(cum_u ** 2)
    if level == 6:
        # Deviation blocks only. Gating the LEVEL block would make the model's
        # absolute scale depend on contact, which is not a property of the task.
        g = 1.0 / (1.0 + np.exp((obs[:, _I_DPO] - 0.020) / 0.010))
        blocks = [b * g[:, None, None] for b in blocks]

    # CENTRE every deviation block across arms. Two reasons, both measured:
    #
    # 1. Meaning. A deviation block should say how the arms DIFFER; the row level is
    #    the level block's job. Uncentred, the block also carries a row-level
    #    component that duplicates (badly) what the level block already provides.
    # 2. The negative control. With uncentred blocks, C2 -- fitted on SCRAMBLED
    #    plan-outcome pairs -- still reached DAR 0.89 (10 seeds: 0.891 +- 0.015),
    #    although its weight direction was uncorrelated with M1's (cos +0.07). The
    #    row-mean of the deviation vector survives any within-row permutation, and
    #    that rank-1 channel alone ranks arms well. A control that cannot fail is not
    #    a control; centring removes the channel at its source.
    dev = [b.reshape(nrow, k, -1) for b in blocks]
    dev = [b - b.mean(axis=1, keepdims=True) for b in dev]

    # Level block: identical across arms, so it cannot move a ranking. It exists so
    # that nMAE/regret stay on the Y scale instead of being fitted to differences.
    lev = np.stack([np.ones(nrow), obs[:, _I_PHI], obs[:, _I_DOT],
                    1.0 / (1.0 + obs[:, _I_DOT])], axis=1)
    n_lev = lev.shape[1]
    return np.concatenate([np.broadcast_to(lev[:, None, :], (nrow, k, n_lev))]
                          + dev, axis=-1)


class StructuredPlan:
    """M1: linear in structurally-derived features; the structure IS the hypothesis.

    Why this is a *method* and not a bigger baseline: every feature is a statement
    about the causal chain (aligned push moves the object toward the goal, lateral
    push does not, displacement accumulates, forward value saturates), and the
    ablation in `run_ablation.py` shows which statements pay. A generic expansion of
    [obs, plan] -- B2 with polynomials -- would be a baseline; this is a model of the
    mechanism, with four fitted coefficients per mechanism.

    Arm 0 is exactly zero in every deviation block by construction, so the model is
    "everything is a deviation from the expert's own plan": it cannot predict an
    absolute action effect, only the difference from the factual plan. That is also
    why the level block must carry the absolute scale.
    """

    kind = "method"
    name = "M1 structured plan (frame+abs+cum)"

    def __init__(self, level: int = 3, alpha: float = 0.1,
                 destroy_plan_association: bool = False, seed: int = 0) -> None:
        self.level = int(level)
        self.alpha = float(alpha)
        self.destroy_plan_association = bool(destroy_plan_association)
        self.seed = int(seed)
        self.w: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "StructuredPlan":
        x = _plan_design(train, self.level)
        y = np.array(y_train, dtype=np.float64, copy=True)
        if self.destroy_plan_association:
            # Same intervention as C1, applied to the structural design: permute the
            # ARMS within each step, leaving outcomes in place. Every step keeps its
            # own K row vectors and its own K outcomes (marginals untouched), and the
            # only thing destroyed is which plan produced which outcome -- which is
            # precisely the relation M1 claims to model.
            rng = np.random.default_rng(self.seed)
            perm = np.argsort(rng.random(y.shape), axis=1)
            x = np.take_along_axis(x, perm[:, :, None], axis=1)

        flat, tgt = x.reshape(-1, x.shape[-1]), y.reshape(-1)
        self.mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        z = (flat - self.mu) / self.sd
        a = z.T @ z + self.alpha * np.eye(z.shape[1])
        b = z.T @ (tgt - tgt.mean())
        self.w = np.concatenate([[tgt.mean()], np.linalg.solve(a, b)])
        return self

    def predict(self, test: FeatureTable) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("%s.predict called before fit" % type(self).__name__)
        x = _plan_design(test, self.level)
        z = (x.reshape(-1, x.shape[-1]) - self.mu) / self.sd
        return (self.w[0] + z @ self.w[1:]).reshape(x.shape[0], x.shape[1])


class StructuredPlanShuffled(StructuredPlan):
    """C2: negative control for M1. Must land at chance, or M1's skill is not the plan."""
    kind = "control"
    name = "C2 plan-outcome shuffled (control)"

    def __init__(self, level: int = 3, alpha: float = 0.1, seed: int = 0) -> None:
        super().__init__(level=level, alpha=alpha,
                         destroy_plan_association=True, seed=seed)


class RidgePoly(Ridge):
    """B4: the same linear learner on a degree-2 expansion of [obs, plan].

    The control that decides whether "structure" is doing the work at all. A
    quadratic expansion can express EVERY action x state interaction -- including
    the ones M1 hard-codes (aligned push, cumulative displacement, saturation) -- if
    it can find them among ~560 terms with 7k rows. If B4 matches M1, then M1's
    advantage is generic nonlinearity and its story about the causal chain is
    decoration. If B4 does NOT reach M1, the gap is the factorisation itself.
    """

    name = "B4 ridge poly2 (generic interactions)"
    kind = "baseline"

    def _design(self, tab: FeatureTable) -> np.ndarray:
        base = super()._design(tab)                        # (n, K, f)
        n, k, f = base.shape
        x = base.reshape(-1, f)
        ii, jj = np.triu_indices(f)
        quad = x[:, ii] * x[:, jj]                         # all i<=j products
        return np.concatenate([x, quad], axis=-1).reshape(n, k, -1)


class LearnedOneStepLinear(Ridge):
    """L1: the linear model restricted to the FIRST action, trained on one-step truth.

    R1 needs a simulator; this does not. Fitted on logged one-step outcomes y_S (the
    gain of the first action alone) and then asked to rank the whole m-step plan.
    Together with R1 it separates "the oracle has a perfect model" from "the model
    sees the plan": R1 - L1 is what the simulator is worth, M1 - L2 is what
    composing over the horizon is worth.
    """

    name = "L1 learned one-step (linear)"
    kind = "baseline"

    def __init__(self, alpha: float = 1.0, seed: int = 0) -> None:
        super().__init__(alpha=alpha, seed=seed, plan_steps=1)


class LearnedOneStepFrame(StructuredPlan):
    """L2: M1's structural form, trained on the ONE-STEP outcome instead of the H-step one."""

    name = "L2 learned one-step (frame)"
    kind = "baseline"

    def __init__(self, alpha: float = 0.1, seed: int = 0) -> None:
        super().__init__(level=0, alpha=alpha, seed=seed)


def make_ladder(seed: int = 0) -> Dict[str, Predictor]:
    """Everything that gets scored, keyed by name, each carrying a `kind`.

    Three kinds, and the distinction is not cosmetic:

      baseline   observation-only. These are what a method must beat.
      reference  needs something no agent has -- here, the simulator itself or the
                 truth. Reported in the same table so the reader can see the gap,
                 never used to claim a method "failed".
      control    a deliberately broken learner. It must land at chance; if it does
                 not, the pipeline invents skill out of noise and nothing else on
                 this table means anything.

    Putting `Myopic` (a one-step probe with a perfect model) in the baseline block
    was a category error: it made "a trivial heuristic already wins" look like a
    finding about the task when it was a finding about the inventory.
    """
    ps: Sequence[Predictor] = (
        NoEffect(),
        FactualWins(),
        Ridge(seed=seed),
        StructuredPlan(seed=seed),
        RidgeArmShuffled(seed=seed),
        StructuredPlanShuffled(seed=seed),
        Myopic(),
    )
    return {p.name: p for p in ps}


# --------------------------------------------------------------------------- #
# the only sanctioned way to score a learned predictor
# --------------------------------------------------------------------------- #
def fit_predict(pred: Predictor, dataset: Dict[str, object], *,
                convention: str, train_ids: Sequence[str],
                test_ids: Sequence[str],
                train_convention: Optional[str] = None
                ) -> Tuple[M.RolloutTable, dict]:
    """Fit on `train_ids`, predict on `test_ids`, return the TEST table.

    Disjointness is asserted here because it is the single mistake that would turn
    every learned number into fiction while leaving all the machinery working. The
    predictor never receives a truth matrix for the test split -- it receives a
    `FeatureTable`, which has no truth field at all.

    `train_convention` exists for the learned one-step baselines (L1/L2): they must
    be supervised on y_S and scored on y_C, and having them fake that inside `fit`
    would be worse than a parameter. It defaults to `convention`, so every existing
    call is unchanged; a value different from `convention` is recorded in `info` so
    no payload can be misread as "trained and tested on the same outcome".
    """
    tr = tuple(dict.fromkeys(train_ids))
    te = tuple(dict.fromkeys(test_ids))
    if not tr or not te:
        raise ValueError("both splits must be non-empty")
    overlap = set(tr) & set(te)
    if overlap:
        raise ValueError("train/test overlap on %d episode(s): %s"
                         % (len(overlap), sorted(overlap)[:4]))

    tc = convention if train_convention is None else train_convention
    if tc not in ("C", "O", "S"):
        raise ValueError("train_convention must be C, O or S, got %r" % (tc,))

    ft_train = features_from(dataset, tr)
    y_train = truth_matrix(dataset, tc, tr)
    if y_train.shape != ft_train.arms.shape[:2]:
        raise ValueError("truth matrix %r does not align with features %r"
                         % (y_train.shape, ft_train.arms.shape[:2]))

    ft_test = features_from(dataset, te)
    pred.fit(ft_train, y_train)
    y_pred = np.asarray(pred.predict(ft_test), dtype=np.float64)
    if y_pred.shape != ft_test.arms.shape[:2]:
        raise ValueError("%s returned %r, expected %r"
                         % (pred.name, y_pred.shape, ft_test.arms.shape[:2]))
    if not np.isfinite(y_pred).all():
        raise ValueError("%s returned non-finite predictions" % pred.name)

    rows = S.records_from(dataset, convention=convention, y_pred=y_pred,
                          episode_ids=te)
    tab = M.build_table(rows, k_arms=ft_test.k)
    info = {"n_train_episodes": len(tr), "n_test_episodes": len(te),
            "n_train_steps": int(y_train.shape[0]), "n_test_steps": len(ft_test),
            "train_convention": tc, "eval_convention": convention}
    return tab, info


def evaluate_predictor(pred: Predictor, dataset: Dict[str, object], *,
                       convention: str, train_ids: Sequence[str],
                       test_ids: Sequence[str], spec: Optional[dict] = None,
                       eps: Optional[float] = None,
                       n_boot: Optional[int] = None,
                       train_convention: Optional[str] = None) -> dict:
    """fit_predict + metrics.evaluate, with the fit metadata carried along."""
    spec = spec or M.load_spec()
    tab, info = fit_predict(pred, dataset, convention=convention,
                            train_ids=train_ids, test_ids=test_ids,
                            train_convention=train_convention)
    rep = M.evaluate(tab, spec, eps=eps, n_boot=n_boot)
    return {"name": pred.name, "table": repr(tab), "report": rep, "info": info}
