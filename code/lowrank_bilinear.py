"""Low-rank bilinear interaction model: the state x action block at rank r.

The motivation is measured on this suite, not assumed. Three facts from
the task diagnostics stored with the results:

  * Every bit of the nonlinear gain that a full degree-2 expansion buys over the
    linear skeleton lives in the 22 x 9 state x action block: +0.3595 DAR
    against +0.1066 for plan x plan and -0.0005 for state x state.
  * The singular spectrum of that fitted 22 x 9 block is not flat but it is not
    rank one either: the leading direction holds 0.502 +- 0.075 of the energy
    against 0.385 +- 0.078 under a permutation null, with a participation ratio
    of 2.92 out of a possible 9.
  * 230 explicit terms buy 0.9216 while 560 buy 0.9335; the last 330 terms are
    worth +0.009.

A 198-entry block that behaves like rank three is over-parameterised by
construction. This model keeps the block bilinear but constrains it to r rank-one
slices:

    y = b0 + s.bs + a.ba + isf.bf + sum_k (s . u_k) (a . v_k)

Fitted by alternating least squares, so that every half-step is a closed-form
ridge solve: no optimiser state, no learning rate, no seed dependence beyond the
initialisation of one factor matrix. Parameters: 32 + 31 r, counting b0, bs,
ba, bf and the two factor matrices. At r = 3 that is 125, against 230 for the
explicit block and 560 for the full quadratic.

The model is a strict generalisation of the hand-built frame: `GlobalDirection`
and `StructuredPlan` hard-code one direction of exactly this block and fit its
magnitudes. At r = 1 with u_1 fixed to the frame's own direction the two
coincide; letting u_1 be estimated is the point of the exercise.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from predictors import FeatureTable, Ridge

__all__ = ["LowRankBilinear", "StateActionParts", "split_parts"]


class StateActionParts:
    """The decision unit split into the three blocks the model treats separately.

    `main` is [intercept, obs, plan, is_factual] -- the same 32 columns `Ridge`
    builds, so the skeleton is literally the linear baseline's design and any
    difference in score is attributable to the interaction block.
    """

    __slots__ = ("main", "state", "action")

    def __init__(self, main: np.ndarray, state: np.ndarray,
                 action: np.ndarray) -> None:
        self.main = main          # (n, K, 33) incl. intercept column
        self.state = state        # (n, K, d_obs)
        self.action = action      # (n, K, m*3)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.main.shape[0]), int(self.main.shape[1])


def split_parts(tab: FeatureTable) -> StateActionParts:
    """Build the shared linear skeleton and the two interaction operands.

    Uses `Ridge._design` verbatim rather than re-deriving it: if the skeleton here
    drifted from the linear baseline's, the comparison between the two would be
    measuring the drift.
    """
    base = Ridge()._design(tab)                    # (n, K, d_obs + m*3 + 1) = 32
    n, k, f = base.shape
    one = np.ones((n, k, 1), dtype=np.float64)
    # The state is shared by the K arms of one decision; the interaction operand
    # must be broadcast to (n, K, d) or the (s.u_k)(a.v_k) product pairs rows from
    # different decisions. Ridge._design does this same broadcast internally.
    obs = np.ascontiguousarray(
        np.broadcast_to(tab.obs[:, None, :], (n, k, tab.obs.shape[1])))
    plan = tab.plans.reshape(n, k, -1)
    # intercept, then Ridge's 32 columns VERBATIM (obs, plan, is_factual). Dropping
    # is_factual here would have made the skeleton differ from the linear baseline's
    # in exactly the column that tells a model which arm was taken.
    main = np.concatenate([one, base], axis=-1)              # 33 incl. intercept
    return StateActionParts(main=main, state=obs, action=plan)


class LowRankBilinear:
    """Rank-r bilinear interaction on (state, action), fitted by ALS.

    Alternating least squares rather than gradient descent is deliberate. Every
    half-step is `(Z'Z + alpha I)^-1 Z'y` on a design whose interaction columns
    are the OTHER factor times one operand, which means: deterministic given the
    initialisation, no schedule to tune, no training curve to read, and the same
    closed-form estimator the linear rung uses. A method whose selling point is
    "fewer free parameters" cannot afford to add an optimiser's worth of them.
    """

    kind = "model"

    def __init__(self, rank: int = 3, alpha: float = 1.0, n_iter: int = 8,
                 seed: int = 0, init: str = "random",
                 plan_quad: bool = False, tol: float = 1e-9) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.init = str(init)
        self.plan_quad = bool(plan_quad)
        self.tol = float(tol)
        self.name = "LRB r%d a%g%s" % (self.rank, self.alpha,
                                       "+pq" if self.plan_quad else "")

        self.u: Optional[np.ndarray] = None        # (r, d_obs)
        self.v: Optional[np.ndarray] = None        # (r, d_action)
        self.coef: Optional[np.ndarray] = None     # skeleton incl. intercept
        self.state_mu: Optional[np.ndarray] = None
        self.state_sd: Optional[np.ndarray] = None
        self.action_mu: Optional[np.ndarray] = None
        self.action_sd: Optional[np.ndarray] = None
        self.main_mu: Optional[np.ndarray] = None
        self.main_sd: Optional[np.ndarray] = None
        self.history: list = []                    # per-iteration training MSE

    # -- standardisation --------------------------------------------------- #
    def _standardise(self, parts: StateActionParts,
                     fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        s = parts.state.reshape(-1, parts.state.shape[-1])
        a = parts.action.reshape(-1, parts.action.shape[-1])
        if fit:
            self.state_mu = s.mean(axis=0)
            sd = s.std(axis=0)
            self.state_sd = np.where(sd > 1e-12, sd, 1.0)
            self.action_mu = a.mean(axis=0)
            sd = a.std(axis=0)
            self.action_sd = np.where(sd > 1e-12, sd, 1.0)
        s = (s - self.state_mu) / self.state_sd
        a = (a - self.action_mu) / self.action_sd
        return s, a

    # -- the linear skeleton, standardised --------------------------------- #
    def _skeleton(self, parts: StateActionParts, fit: bool = False) -> np.ndarray:
        """[intercept, obs, plan, is_factual, (plan x plan)] with columns scaled.

        Standardising the skeleton matters as much as standardising the two
        interaction operands: ridge penalises coefficients on the scale they are
        estimated in, so leaving 22 position columns and 9 action columns on
        their raw units (metres vs action-box units) makes one alpha mean two
        different things at once. The intercept column is exempted.
        """
        main = parts.main.reshape(-1, parts.main.shape[-1])
        if self.plan_quad:
            a_raw = parts.action.reshape(-1, parts.action.shape[-1])
            ii, jj = np.triu_indices(a_raw.shape[1])
            main = np.concatenate([main, a_raw[:, ii] * a_raw[:, jj]], axis=1)
        if fit:
            self.main_mu = main.mean(axis=0)
            sd = main.std(axis=0)
            self.main_sd = np.where(sd > 1e-12, sd, 1.0)
            self.main_mu[0] = 0.0
            self.main_sd[0] = 1.0            # the constant stays a constant
        return (main - self.main_mu) / self.main_sd

    # -- one ridge half-step ----------------------------------------------- #
    def _interaction_columns(self, s: np.ndarray, a: np.ndarray,
                             factors: np.ndarray, other: str) -> np.ndarray:
        """Columns (s.u_k)(a) for k=1..r, or (a.v_k)(s) for the mirror step."""
        n = s.shape[0]
        r = factors.shape[0]
        cols = np.empty((n, r * (s.shape[1] if other == "action" else a.shape[1])),
                        dtype=np.float64)
        j = 0
        for k in range(r):
            if other == "action":          # estimating u_k: weight = (a . v_k)
                w = a @ factors[k]
                block = w[:, None] * s
            else:                          # estimating v_k: weight = (s . u_k)
                w = s @ factors[k]
                block = w[:, None] * a
            wd = block.shape[1]
            cols[:, j:j + wd] = block
            j += wd
        return cols

    @staticmethod
    def _ridge(design: np.ndarray, target: np.ndarray,
               alpha: float) -> np.ndarray:
        """Closed-form ridge; column 0 is the constant and is left unpenalised.

        The target is centred first so that column 0 absorbs exactly the mean,
        which is what makes leaving it out of the penalty honest rather than a
        back door that lets the model buy a level shift for free.
        """
        y = target - target.mean()
        gram = design.T @ design
        pen = alpha * np.eye(design.shape[1])
        pen[0, 0] = 0.0                    # never shrink the intercept
        return np.linalg.solve(gram + pen, design.T @ y)

    # -- fit ---------------------------------------------------------------- #
    def fit(self, train: FeatureTable, y_train: np.ndarray) -> "LowRankBilinear":
        parts = split_parts(train)
        s, a = self._standardise(parts, fit=True)
        main = self._skeleton(parts, fit=True)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)

        rng = np.random.default_rng(self.seed)
        d_s, d_a = s.shape[1], a.shape[1]
        if self.init == "random":
            self.v = rng.normal(0.0, 1.0 / np.sqrt(d_s), size=(self.rank, d_a))
            self.u = rng.normal(0.0, 1.0 / np.sqrt(d_a), size=(self.rank, d_s))
        elif self.init == "svd":
            # Spectral start: fit the unconstrained block, take its leading r
            # singular triplets, then let the RANK-CONSTRAINED objective refine
            # them. Only the start point is borrowed -- the fitted model keeps
            # 32 + 31 r parameters and its solution minimises the low-rank
            # residual, not the Frobenius distance to the full block. A random
            # start lands in local minima (measured: r=1 scored 0.769 against a
            # 0.920 frame) because the product (s.u)(a.v) has a flat direction:
            # u -> cu, v -> v/c leaves every prediction unchanged.
            inter = (s[:, :, None] * a[:, None, :]).reshape(
                -1, d_s * d_a)
            design = np.concatenate([main, inter], axis=1)
            coef = self._ridge(design, y, self.alpha)
            W = coef[main.shape[1]:].reshape(d_s, d_a)
            uu, sv, vt = np.linalg.svd(W, full_matrices=False)
            r = min(self.rank, len(sv))
            self.u = (uu[:, :r] * sv[:r]).T
            self.v = vt[:r]
        else:
            raise ValueError("unknown init %r" % (self.init,))

        self.history = []
        prev = np.inf
        for _it in range(self.n_iter):
            # -- u given v (and the whole linear skeleton) --
            cols = self._interaction_columns(s, a, self.v, other="action")
            design = np.concatenate([main, cols], axis=1)
            coef = self._ridge(design, y, self.alpha)
            skel = coef[:main.shape[1]]
            self.u = coef[main.shape[1]:].reshape(self.rank, d_s)

            # -- v given u --
            cols = self._interaction_columns(s, a, self.u, other="state")
            design = np.concatenate([main, cols], axis=1)
            coef = self._ridge(design, y, self.alpha)
            skel = coef[:main.shape[1]]
            self.v = coef[main.shape[1]:].reshape(self.rank, d_a)
            self.coef = skel

            pred = self._raw_predict(main, s, a)
            mse = float(np.mean((pred - y) ** 2))
            self.history.append(mse)
            # inf - mse is inf, and tol * inf is inf, so a naive guard stops the
            # loop after ONE iteration while looking like a converged run. That
            # bug made every earlier "ALS" result a single alternating sweep.
            if np.isfinite(prev) and abs(prev - mse) <= self.tol * max(1.0, abs(prev)):
                break
            prev = mse
        return self

    # -- predict ------------------------------------------------------------ #
    def _raw_predict(self, main: np.ndarray, s: np.ndarray,
                     a: np.ndarray) -> np.ndarray:
        out = main @ self.coef
        for k in range(self.rank):
            out = out + (s @ self.u[k]) * (a @ self.v[k])
        return out

    def predict(self, test: FeatureTable) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("LowRankBilinear.predict called before fit")
        parts = split_parts(test)
        s, a = self._standardise(parts, fit=False)
        main = self._skeleton(parts, fit=False)
        pred = self._raw_predict(main, s, a)
        return pred.reshape(parts.main.shape[0], parts.main.shape[1])

    # -- accounting --------------------------------------------------------- #
    @property
    def n_params(self) -> int:
        """Fitted slopes, counted the way the rest of the ladder counts them.

        `Ridge.w` is [intercept, coefs] and the price table reports len(w) - 1, so
        the intercept is excluded here too. Including it would inflate every model
        in the comparison by one and make LRB look costlier than it is at small r,
        which is exactly where the parameter claim is supposed to bite.
        """
        return int(self.coef.shape[0]) - 1 + self.rank * (self.u.shape[1]
                                                          + self.v.shape[1])
