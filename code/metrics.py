"""
metrics.py -- the single source of truth for every number that goes into the paper.

Contract (see README.md, section "The frozen measurement contract"):
  * Every reported metric MUST come from this module. In-place metric
    implementations inside experiment scripts are forbidden: two divergent
    definitions of "direction agreement" is the fastest way to lose a paper.
  * All metric parameters (K, H, eps policy, bootstrap size, thresholds) are read
    from the frozen spec JSON. Nothing here is hard-coded except numerical
    tolerances that the spec itself fixes.

Frozen spec: "1.0" (2026-11-21).

Typical use
-----------
    import metrics
    spec  = metrics.load_spec()
    table = metrics.build_table(records, k_arms=spec["candidate_set"]["K_primary"])
    report = metrics.evaluate(table, spec)                 # point + 95% CI
    assert metrics.ruler_selftest(table, spec)["passed"]   # Gate-2, always run

`records` is a list of dicts, one per rollout:
    {"episode": "cw_pick_0007", "t": 12, "k": 0,
     "y_true": 0.31, "y_pred": 0.29, "degenerate": False}
with k == 0 being the factual arm (mandatory, exactly once per (episode, t)).

Dependency: scipy (installed; see PINNED INTERPRETER below). Rank statistics
delegate to scipy.stats -- see the note above spearman_per_step for why the
hand-rolled version had to go.

PINNED INTERPRETER (2026-11-21, verified by probe):
    D:/Miniconda3/envs/torch-cuda128/python.exe
    python 3.12.13 | numpy 2.4.3 | scipy 1.17.1 | pandas 2.3.3
    torch 2.11.0+cu128 (CUDA available on RTX 5060 Laptop) | pymupdf 1.27.2.3
Do NOT run this module with D:/python.exe: that is a bare CPython 3.13.5 whose
prefix is D:/ and whose site-packages has no scipy.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import kendalltau, rankdata
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise ModuleNotFoundError(
        "metrics.py requires scipy. Install the pinned dependencies with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

SPEC_VERSION = "1.3"
_SPEC_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.pardir, "spec", "counterfactual_contract_v1.3.json")


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
def load_spec(path: Optional[str] = None) -> dict:
    """Load the frozen spec and refuse to run on a version mismatch."""
    path = path or os.environ.get("SCM_SPEC_PATH") or _SPEC_DEFAULT
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    got = spec.get("spec_version")
    if got != SPEC_VERSION:
        raise RuntimeError(
            "spec_version mismatch: metrics.py implements %r, spec file says %r (%s). "
            "Update the code or the spec -- never silently mix the two."
            % (SPEC_VERSION, got, path)
        )
    return spec


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
class RolloutTable:
    """Dense (n_steps, K) view of a rollout record set.

    Reshaping into a dense matrix once, up front, is what makes the bootstrap
    cheap enough to run 10,000 times.
    """

    __slots__ = ("y_true", "y_pred", "degen", "episode_of_step", "episodes", "k")

    def __init__(self, y_true, y_pred, degen, episode_of_step, episodes, k):
        self.y_true = y_true
        self.y_pred = y_pred
        self.degen = degen
        self.episode_of_step = episode_of_step
        self.episodes = episodes
        self.k = k

    def __repr__(self):  # pragma: no cover - cosmetic
        return "RolloutTable(steps=%d, K=%d, episodes=%d, degenerate=%.1f%%)" % (
            self.y_true.shape[0], self.k, len(self.episodes),
            100.0 * float(self.degen.mean()) if len(self.degen) else 0.0)


def build_table(records: Iterable[dict], k_arms: Optional[int] = None) -> RolloutTable:
    """Pivot records into a dense table and validate the structure.

    Hard failures here are intentional: a malformed record set must never reach
    a metric, because the resulting number would look plausible and be wrong.
    """
    recs = list(records)
    if not recs:
        raise ValueError("empty record set")

    eps_guess: Dict[Tuple[str, int], List[int]] = {}
    for r in recs:
        key = (str(r["episode"]), int(r["t"]))
        eps_guess.setdefault(key, []).append(int(r["k"]))

    if k_arms is None:
        k_arms = max(len(v) for v in eps_guess.values())

    # validate completeness
    for key, ks in eps_guess.items():
        if sorted(ks) != list(range(k_arms)):
            raise ValueError(
                "step %r has arms %r, expected 0..%d exactly once"
                % (key, sorted(ks), k_arms - 1))

    steps = sorted(eps_guess.keys())
    step_index = {s: i for i, s in enumerate(steps)}
    n_steps = len(steps)

    y_true = np.full((n_steps, k_arms), np.nan, dtype=np.float64)
    y_pred = np.full((n_steps, k_arms), np.nan, dtype=np.float64)
    degen = np.zeros(n_steps, dtype=bool)

    for r in recs:
        i = step_index[(str(r["episode"]), int(r["t"]))]
        j = int(r["k"])
        y_true[i, j] = float(r["y_true"])
        y_pred[i, j] = float(r["y_pred"])
        if r.get("degenerate"):
            degen[i] = True

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("missing (episode, t, k) cells after pivot")

    episodes = sorted({e for e, _ in steps})
    ep_pos = {e: i for i, e in enumerate(episodes)}
    episode_of_step = np.array([ep_pos[e] for e, _ in steps], dtype=np.int64)

    return RolloutTable(y_true, y_pred, degen, episode_of_step, episodes, k_arms)


def steps_of_episodes(table: RolloutTable) -> List[np.ndarray]:
    """Precomputed step indices for each episode, for the cluster bootstrap."""
    return [np.flatnonzero(table.episode_of_step == e) for e in range(len(table.episodes))]


# --------------------------------------------------------------------------- #
# rank helpers
# --------------------------------------------------------------------------- #
def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Average ranks along axis=1, 1-based, tie-aware (scipy.rankdata)."""
    return rankdata(a, axis=1)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.full(a.shape[0], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def spearman_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Spearman rho per step over the non-factual arms; NaN where undefined.

    Verified equivalent to the previous hand-rolled version (max |diff| = 0.0
    on continuous, discrete-with-ties, rounded, and constant-row inputs), so
    switching to scipy moved no published number.
    """
    return _pearson_rows(_rank_rows(y_true), _rank_rows(y_pred))


def kendall_per_step(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Kendall tau-b per step, tie-corrected; NaN where undefined.

    This replaces a hand-rolled tau-b that was WRONG whenever ties existed: its
    tie counts `n1`/`n2` were accumulated across all steps as scalars and then
    used as a single shared denominator, so any step with a tied outcome could
    return |tau| > 1 (observed |diff| = 4.09 vs scipy on 5-level discrete
    outcomes). Continuous outcomes hid the bug, which is exactly why it would
    have survived into the paper: task success and clipped returns are discrete.
    """
    n_steps = y_true.shape[0]
    out = np.full(n_steps, np.nan)
    for i in range(n_steps):
        a = y_true[i]
        b = y_pred[i]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        out[i] = kendalltau(a, b, variant="b").statistic
    return out


# --------------------------------------------------------------------------- #
# core metric computation
# --------------------------------------------------------------------------- #
_CORE_KEYS = (
    "dar", "mae", "nmae", "rho", "tau", "regret", "regret_p90", "regret_norm",
    "zero_regret_rate", "undecidable_rate", "nontrivial_step_rate",
    "n_steps_ranked", "n_steps", "n_units_usable",
)


def _core(y_t: np.ndarray, y_p: np.ndarray, degen: np.ndarray, eps: float,
          zero_regret_tol: float, fast: bool = False) -> Dict[str, float]:
    """All scalar metrics on one (possibly resampled) table.

    Always returns every key in _CORE_KEYS (NaN where undefined). Partial dicts
    are a trap: the bootstrap loop indexes these keys 10,000 times, and one
    early return with a missing field crashes the whole run.
    """
    k = y_t.shape[1]
    d_t = y_t - y_t[:, [0]]
    d_p = y_p - y_p[:, [0]]

    nf = slice(1, k)                      # non-factual arms
    d_t_nf, d_p_nf = d_t[:, nf], d_p[:, nf]
    valid_step = ~degen

    decided = np.abs(d_t_nf) > eps        # noise-floor mask, per element
    usable = decided & valid_step[:, None]

    out: Dict[str, float] = {key: float("nan") for key in _CORE_KEYS}
    n_usable = int(usable.sum())
    n_denom = int((valid_step[:, None]).sum()) * max(k - 1, 1)

    out["n_steps"] = float(y_t.shape[0])
    out["n_units_usable"] = float(n_usable)
    out["undecidable_rate"] = float(1.0 - n_usable / n_denom) if n_denom else float("nan")

    if n_usable == 0:
        return out

    hit = (np.sign(d_p_nf) == np.sign(d_t_nf))[usable]
    out["dar"] = float(hit.mean())

    err = np.abs(d_p_nf - d_t_nf)[usable]
    out["mae"] = float(err.mean())
    scale = float(np.median(np.abs(d_t_nf)[usable]))
    out["nmae"] = float(err.mean() / scale) if scale > 0 else float("nan")

    # rank agreement: only over steps where every non-factual arm is decidable
    all_decided = decided.all(axis=1) & valid_step
    if all_decided.any():
        rho = spearman_per_step(d_t_nf[all_decided], d_p_nf[all_decided])
        out["rho"] = float(np.nanmean(rho)) if np.isfinite(rho).any() else float("nan")
        # Kendall tau is SECONDARY and is reported as a point estimate only. It is
        # skipped inside the bootstrap because it is the only metric that loops over
        # steps calling scipy (tau-b with tie correction), i.e. ~100 us x n_steps per
        # resample: 2000 resamples x 11 predictors turned one evaluation into ten
        # minutes of pure Python. The number it produces is unchanged -- `fast`
        # affects only whether the resampled draws exist, so its CI is None and the
        # payload says so rather than inventing one.
        if fast:
            out["tau"] = float("nan")
        else:
            tau = kendall_per_step(d_t_nf[all_decided], d_p_nf[all_decided])
            out["tau"] = float(np.nanmean(tau)) if np.isfinite(tau).any() else float("nan")
        out["n_steps_ranked"] = float(all_decided.sum())
    else:
        out["rho"] = float("nan")
        out["tau"] = float("nan")
        out["n_steps_ranked"] = 0.0

    # regret over non-degenerate steps
    if valid_step.any():
        yt, yp = y_t[valid_step], y_p[valid_step]
        a_hat = np.argmax(yp, axis=1)
        a_star = np.argmax(yt, axis=1)
        r = yt[np.arange(len(yt)), a_star] - yt[np.arange(len(yt)), a_hat]
        gap = yt[np.arange(len(yt)), a_star] - yt[:, 0]
        denom = np.where(np.abs(gap) > 1e-12, gap, np.nan)
        r_norm = np.where(np.isfinite(denom), r / denom, np.nan)
        out["regret"] = float(np.mean(r))
        out["regret_p90"] = float(np.percentile(r, 90))
        out["regret_norm"] = float(np.nanmean(r_norm)) if np.isfinite(r_norm).any() else float("nan")
        out["zero_regret_rate"] = float(np.mean(r <= zero_regret_tol))
        nontrivial = a_star != 0
        out["nontrivial_step_rate"] = float(np.mean(nontrivial))
    else:
        out.update(regret=float("nan"), regret_p90=float("nan"),
                   regret_norm=float("nan"), zero_regret_rate=float("nan"),
                   nontrivial_step_rate=float("nan"))
    return out


# --------------------------------------------------------------------------- #
# public: evaluate
# --------------------------------------------------------------------------- #
def evaluate(table: RolloutTable, spec: Optional[dict] = None, *,
             eps: Optional[float] = None, n_boot: Optional[int] = None,
             seed: int = 0) -> dict:
    """Point estimates + episode-level bootstrap CI for every primary metric."""
    spec = spec or load_spec()

    if eps is None:
        eps = float(spec["noise_floor"].get("deterministic_fallback", 1e-6))
        src = "spec.noise_floor.deterministic_fallback"
    else:
        src = "caller"

    if n_boot is None:
        n_boot = int(spec["estimation"]["bootstrap_n"])

    zero_tol = float(spec["noise_floor"].get("deterministic_fallback", 1e-6))

    point = _core(table.y_true, table.y_pred, table.degen, eps, zero_tol)
    # inclusive-of-degenerate variants (spec: must be reported alongside)
    point_incl = _core(table.y_true, table.y_pred,
                       np.zeros_like(table.degen), eps, zero_tol)

    rng = np.random.default_rng(seed)
    per_ep = steps_of_episodes(table)
    n_ep = len(per_ep)
    keys = ["dar", "mae", "nmae", "rho", "tau", "regret", "regret_norm",
            "zero_regret_rate", "undecidable_rate", "nontrivial_step_rate"]
    draws = {k: np.empty(n_boot) for k in keys}

    for b in range(n_boot):
        pick = rng.integers(0, n_ep, size=n_ep)
        idx = np.concatenate([per_ep[i] for i in pick])
        m = _core(table.y_true[idx], table.y_pred[idx],
                  table.degen[idx], eps, zero_tol, fast=True)
        for k in keys:
            draws[k][b] = m.get(k, float("nan"))

    report = {
        "spec_version": spec["spec_version"],
        "eps": eps,
        "eps_source": src,
        "n_episodes": n_ep,
        "n_steps": int(table.y_true.shape[0]),
        "K": table.k,
        "degenerate_rate": float(table.degen.mean()) if len(table.degen) else 0.0,
        "n_boot": n_boot,
        "cluster_unit": "episode",
        "metrics": {},
        "metrics_including_degenerate": {},
    }

    def _pack(d: Dict[str, float], keyset: Sequence[str]) -> Dict[str, dict]:
        out = {}
        for k in keyset:
            v = d.get(k)
            if v is None or not np.isfinite(v):
                out[k] = {"value": None}
                continue
            col = draws[k]
            col = col[np.isfinite(col)]
            ci = ([float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]
                  if col.size >= 100 else None)
            out[k] = {"value": float(v), "ci95": ci}
        return out

    report["metrics"] = _pack(point, keys)
    report["metrics"]["regret_p90"] = {"value": point["regret_p90"]}
    report["metrics_including_degenerate"] = _pack(point_incl, ["dar", "nmae", "regret_norm"])
    return report


# --------------------------------------------------------------------------- #
# Gate-2: the ruler must prove itself before anyone trusts its readings
# --------------------------------------------------------------------------- #
def oracle_predictions(table: RolloutTable) -> np.ndarray:
    return table.y_true.copy()


def random_predictions(table: RolloutTable, seed: int = 0) -> np.ndarray:
    """Shuffle the arm order of the TRUTH within each step.

    Deliberately permutes `y_true`, not `y_pred`. Permuting the prediction is a
    no-op whenever the prediction carries no information -- e.g. the "no effect"
    baseline whose every arm equals the factual one -- so a chance test built that
    way silently measures the baseline instead of chance, and reports DAR=0.000
    where chance should be ~0.5 (found on 2026-11-21, pilot #1).
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(table.y_true)
    for i in range(table.y_true.shape[0]):
        out[i] = table.y_true[i][rng.permutation(table.y_true.shape[1])]
    return out


def _p_positive(table: RolloutTable, eps: float) -> float:
    """Share of usable (step, arm) cells whose effect is positive.

    Reported, never used as a pass/fail threshold: it tells you whether the table
    can support a direction metric at all. A table where p is 0 or 1 has no
    direction to predict, and DAR numbers from it mean nothing.
    """
    d_t_nf = (table.y_true - table.y_true[:, [0]])[:, 1:]
    usable = (np.abs(d_t_nf) > eps) & (~table.degen)[:, None]
    return float((d_t_nf[usable] > 0).mean()) if usable.any() else float("nan")


def ruler_selftest(table: RolloutTable, spec: Optional[dict] = None,
                   seed: int = 0, chance_permutations: int = 200) -> dict:
    """Gate-2. A predictor with perfect knowledge must score perfect; a predictor
    with no information must score chance. If chance scores 0.8, the metric is
    broken -- fix the metric before running any experiment."""
    spec = spec or load_spec()
    k = table.k
    tol = float(spec["noise_floor"].get("deterministic_fallback", 1e-6))

    tt = RolloutTable(table.y_true, oracle_predictions(table), table.degen,
                      table.episode_of_step, table.episodes, k)
    o = _core(tt.y_true, tt.y_pred, tt.degen, tol, tol)

    rnd = random_predictions(table, seed=seed)
    tr = RolloutTable(table.y_true, rnd, table.degen,
                      table.episode_of_step, table.episodes, k)
    r = _core(tr.y_true, tr.y_pred, tr.degen, tol, tol)

    oracle_ok = (abs(o["dar"] - 1.0) < 1e-9
                 and abs(o["nmae"]) < 1e-9
                 and abs(o["regret_norm"]) < 1e-9
                 and abs(o["zero_regret_rate"] - 1.0) < 1e-9)

    # Chance level is MEASURED, not derived. Each additional permutation of the
    # truth is another no-information predictor, so the distribution of DAR over
    # permutations is the chance distribution for this table.
    #
    # An algebraic shortcut is tempting and wrong: P(sign match) is NOT
    # p^2 + (1-p)^2, because every comparison in a row shares perm[0] as its
    # baseline, so the comparisons are not independent. That formula returned 0.917
    # where permutation returned 0.493 on real pilot data (2026-11-21). Measure it.
    dars, rhos = [], []
    rng = np.random.default_rng(seed + 991)
    for _ in range(chance_permutations):
        perm = np.empty_like(table.y_true)
        for i in range(table.y_true.shape[0]):
            perm[i] = table.y_true[i][rng.permutation(k)]
        tp = RolloutTable(table.y_true, perm, table.degen,
                          table.episode_of_step, table.episodes, k)
        mp = _core(tp.y_true, tp.y_pred, tp.degen, tol, tol)
        dars.append(mp["dar"])
        rhos.append(mp["rho"])
    chance_mean = float(np.nanmean(dars))
    chance_sd = float(np.nanstd(dars))
    chance_rho = float(np.nanmean(rhos)) if np.isfinite(rhos).any() else float("nan")

    # A no-information predictor must not score high. The band is deliberately wide
    # (0.25-0.75): with a scripted expert the sign balance is skewed, so the exact
    # chance level moves; what must never happen is chance scoring 0.8.
    chance_ok = (0.25 <= chance_mean <= 0.75) and abs(chance_rho) <= 0.15 \
        and abs(r["regret_norm"]) > 1e-6

    return {
        "passed": bool(oracle_ok and chance_ok),
        "oracle": {"dar": o["dar"], "nmae": o["nmae"],
                   "regret_norm": o["regret_norm"],
                   "zero_regret_rate": o["zero_regret_rate"], "ok": bool(oracle_ok)},
        "chance": {"dar": r["dar"], "dar_perm_mean": chance_mean,
                   "dar_perm_sd": chance_sd, "rho": chance_rho,
                   "p_positive": _p_positive(table, tol),
                   "regret_norm": r["regret_norm"],
                   "permutations": int(chance_permutations),
                   "accepted_band": [0.25, 0.75], "ok": bool(chance_ok),
                   "permuted": "y_true"},
    }


# --------------------------------------------------------------------------- #
# diagnosis (H3) and recovery
# --------------------------------------------------------------------------- #
def diagnosis_metrics(pred_ranked: Sequence[Sequence[Tuple[str, str]]],
                      true_sets: Sequence[Sequence[Tuple[str, str]]]) -> dict:
    """Hit@1 and MRR for `variable = (object_id, attribute)` predictions.

    `pred_ranked[i]` is the ranked list of variables blamed for failure i;
    `true_sets[i]` is the set of variables that, when reset, restore success
    (built by per-variable I-s intervention -- independent of any model).
    """
    if len(pred_ranked) != len(true_sets):
        raise ValueError("prediction/truth length mismatch")
    hits, rrs, n_no_truth = 0, [], 0
    for pred, truth in zip(pred_ranked, true_sets):
        tset = set(truth)
        if not tset:
            n_no_truth += 1
            continue
        if pred and pred[0] in tset:
            hits += 1
        rr = 0.0
        for rank, var in enumerate(pred, start=1):
            if var in tset:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    n = len(pred_ranked) - n_no_truth
    return {
        "hit_at_1": hits / n if n else float("nan"),
        "mrr": float(np.mean(rrs)) if rrs else float("nan"),
        "n_evaluated": n,
        "n_without_truth": n_no_truth,
    }


def recovery_rates(ours: Sequence[bool], random_retry: Sequence[bool],
                   oracle: Sequence[bool], *, n_boot: int = 10000, seed: int = 0) -> dict:
    """Recovery must be reported against BOTH the random-retry and the oracle
    baseline. Beat 'doing nothing' is not evidence."""
    o = np.asarray(ours, dtype=float)
    r = np.asarray(random_retry, dtype=float)
    g = np.asarray(oracle, dtype=float)
    if not (len(o) == len(r) == len(g)):
        raise ValueError("recovery arrays must share length")
    diff = o - r

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(diff)
    for b in range(n_boot):
        i = rng.integers(0, n, size=n)
        boot[b] = diff[i].mean()
    return {
        "ours": float(o.mean()),
        "random_retry": float(r.mean()),
        "oracle": float(g.mean()),
        "delta_vs_random": float(diff.mean()),
        "delta_vs_random_ci95": [float(np.percentile(boot, 2.5)),
                                 float(np.percentile(boot, 97.5))],
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Gate-1 support: intervention locality
# --------------------------------------------------------------------------- #
def intervention_locality(natural_std: float, delta_after_intervention: np.ndarray,
                          neighbors: np.ndarray, *, tol_frac: float = 0.01) -> dict:
    """Gate-1. Intervening on edge e must move its neighbours and nobody else.

    `delta_after_intervention` is |prediction change| per node; `neighbors` is a
    boolean mask over the same node axis.
    """
    d = np.asarray(delta_after_intervention, dtype=float)
    nb = np.asarray(neighbors, dtype=bool)
    tol = tol_frac * natural_std
    non_nb = d[~nb]
    nb_d = d[nb]
    return {
        "max_non_neighbor_change": float(non_nb.max()) if non_nb.size else 0.0,
        "tol": float(tol),
        "neighbor_mean_change": float(nb_d.mean()) if nb_d.size else 0.0,
        "passed": bool((non_nb.size == 0 or non_nb.max() < tol)
                       and (nb_d.size > 0 and nb_d.mean() > tol)),
    }


if __name__ == "__main__":  # pragma: no cover
    print("metrics.py -- spec", SPEC_VERSION)
    s = load_spec()
    print("loaded spec:", s["spec_version"], "frozen at", s["frozen_at"])
    print("primary metrics:", s["metrics"]["primary"])
    print("bootstrap:", s["estimation"])
