# -*- coding: utf-8 -*-
"""Task diagnostics: is this ranking problem one-dimensional?

Why this module exists
----------------------
A plan-ranking benchmark can look perfectly healthy -- K candidates per decision,
two execution conventions, a plausible spread of outcomes -- while its ranking is a
function of ONE scalar. When that happens, no method can demonstrate multi-step
reasoning, because there is none in the data, and a zero-parameter heuristic is the
correct model. This was measured, not suspected: on the v1.3 cube task a single
scalar (the summed aligned push) reached out-of-sample within-step R^2 = 0.799, a
0-parameter ranker scored DAR 0.897 against the one-step oracle's 0.869, and the
within-step deviation covariance had effective rank 3.45 of 11.

Trying candidate tasks and hoping is how a project spends weeks. This module is the
cheap pre-flight instead: four numbers, computed from a dataset alone, that say
whether a task can support the claim before any method is trained on it.

The four numbers
----------------
1. scalar_r2      out-of-sample within-step R^2 of Delta-y on sum_h(aligned push).
                  The single best one-dimensional explanation. Above ~0.7, the task
                  is effectively rank 1 and a heuristic is the right model.
2. full_r2        the same R^2 from the full structural design (frame projection,
                  |lateral|, accumulated aligned push). The gap to scalar_r2 is all
                  the multi-dimensional content there is.
3. eff_rank       effective rank (sum of eigenvalues)^2 / sum of squares of the
                  within-step deviation covariance. "How many independent directions
                  can an arm move the outcome in."
4. order_pairs    among arm pairs whose total aligned push is equal to within 2% of
                  the step's range, the fraction whose TRUTH differs by more than 10%
                  of the median |Delta y|. If this is near 0, the task never poses an
                  order question, whatever the plan length says.

Reading the verdict: a task is "one-dimensional" when scalar_r2 > 0.7 AND
eff_rank < 0.5 * n_deviation_features AND order_pairs < 0.3. All three must agree,
because each alone has a benign explanation (a task can have a dominant scalar and
still contain order information in the residual).

Usage
-----
    python task_diagnostics.py --dataset <path> [--conv C] [--json out.json]

Nothing here reads truth from the test set to FIT anything: the scalar coefficient
and the structural model are fitted on train and scored on test, as in the pilot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import predictors as P   # noqa: E402
import sampler as S      # noqa: E402

# Observation indices, looked up rather than hard-coded so that adding an observation
# cannot silently shift the diagnosis to the wrong quantity.
_I_REL = S.OBS_NAMES.index("rel_obj_tgt_x")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


def aligned_push(tab: P.FeatureTable) -> np.ndarray:
    """(n, K, m) aligned deviation of each plan from the factual plan.

    Aligned = resolved along the object->target direction, which is the component
    that can move the object toward the goal. Any order-blind scalar worth
    considering is a function of this block summed over the plan.
    """
    u = _unit(-tab.obs[:, _I_REL:_I_REL + 2])
    d = tab.plans - tab.plans[:, [0], :, :]
    du = d[..., 0] * u[:, None, None, 0] + d[..., 1] * u[:, None, None, 1]
    return du


def deviation_design(tab: P.FeatureTable, level: int = 3) -> np.ndarray:
    """The structural DEVIATION block only (level block dropped: it is row-constant
    and cannot move a ranking, so including it would inflate the effective rank)."""
    x = P._plan_design(tab, level)
    n_lev = 4                     # ones, phi, d_obj_tgt, 1/(1+d_obj_tgt)
    return x[:, :, n_lev:]


def _r2(y: np.ndarray, yhat: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        y, yhat = y[mask], yhat[mask]
    den = float((y ** 2).sum())
    if den <= 0:
        return float("nan")
    return 1.0 - float(((y - yhat) ** 2).sum()) / den


def effective_rank(a: np.ndarray) -> Dict[str, float]:
    """Effective rank of the within-step deviation covariance.

    `a` is (n, K, f): for each decision, K feature vectors. Centring across arms
    removes the row level, then the eigenvalues of the pooled covariance describe how
    many independent directions the arms span.
    """
    xc = a - a.mean(axis=1, keepdims=True)
    flat = xc.reshape(-1, xc.shape[-1])
    cov = np.cov(flat, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    tot = float(lam.sum())
    if tot <= 0:
        return dict(eff_rank=0.0, n_features=int(xc.shape[-1]), top_share=0.0)
    return dict(eff_rank=float(tot ** 2 / float((lam ** 2).sum())),
                n_features=int(xc.shape[-1]),
                top_share=float(lam.max() / tot))


def diagnose(dataset: Dict[str, object], conv: str = "C",
             splits: Optional[Dict[str, list]] = None) -> Dict[str, object]:
    eps = list(dataset["episodes"])                      # type: ignore[index]
    if splits is None:
        splits = {}
        for ep in eps:
            splits.setdefault(ep.get("split", "train"), []).append(ep["episode_id"])
    if "test" not in splits or "train" not in splits:
        raise ValueError("dataset has no train/test split; run split.assign first")

    ft_tr = P.features_from(dataset, splits["train"])
    ft_te = P.features_from(dataset, splits["test"])
    y_tr = P.truth_matrix(dataset, conv, splits["train"])
    y_te = P.truth_matrix(dataset, conv, splits["test"])

    # -- degenerate steps must be excluded, or the numbers lie -------------- #
    # A decision whose arms all land at (nearly) the same Y carries no ranking
    # information, but it does contribute rows where y_c ~ 0 and the denominator of
    # R^2 is dominated by noise. Longer horizons make this worse: once the object
    # reaches the target, every remaining decision is saturated. Without this filter,
    # "the ranking becomes less one-dimensional as the horizon grows" is just
    # "there are more saturated steps" -- which is an artefact, not a finding.
    # Threshold: a step is degenerate when its arm spread is below 25% of the median
    # spread, i.e. the arms are not separable on the Y scale of this dataset.
    spread_tr = np.ptp(y_tr, axis=1)
    spread_te = np.ptp(y_te, axis=1)
    thr = 0.25 * float(np.median(spread_tr)) if spread_tr.size else 0.0
    keep_tr = spread_tr > thr
    keep_te = spread_te > thr
    degen_frac = float(1.0 - keep_te.mean()) if keep_te.size else 0.0

    # -- 1. one scalar, fitted on train, scored on test --------------------- #
    s_tr = aligned_push(ft_tr).sum(axis=2)
    s_te = aligned_push(ft_te).sum(axis=2)
    y_tr_k = y_tr[keep_tr] if keep_tr.any() else y_tr
    y_te_k = y_te[keep_te] if keep_te.any() else y_te
    s_tr_k = s_tr[keep_tr] if keep_tr.any() else s_tr
    s_te_k = s_te[keep_te] if keep_te.any() else s_te
    sc_tr = (s_tr_k - s_tr_k.mean(axis=1, keepdims=True)).reshape(-1)
    yc_tr = (y_tr_k - y_tr_k.mean(axis=1, keepdims=True)).reshape(-1)
    beta = float((sc_tr @ yc_tr) / max(float(sc_tr @ sc_tr), 1e-30))
    yc_te = (y_te_k - y_te_k.mean(axis=1, keepdims=True)).reshape(-1)
    sc_te = (s_te_k - s_te_k.mean(axis=1, keepdims=True)).reshape(-1)
    r2_scalar = _r2(yc_te, beta * sc_te)

    # -- 2. the full structural deviation block, same protocol -------------- #
    # Fitted on ALL train steps (more data, and the model has no access to the
    # degenerate flag at test time), scored on the non-degenerate test steps only.
    m1 = P.StructuredPlan(level=3, alpha=0.1).fit(ft_tr, y_tr)
    pr_te = m1.predict(ft_te)
    pr_te_k = pr_te[keep_te] if keep_te.any() else pr_te
    r2_full = _r2(yc_te, (pr_te_k - pr_te_k.mean(axis=1, keepdims=True)).reshape(-1))

    # -- 3. effective rank of the deviation design -------------------------- #
    dev_te = deviation_design(ft_te, 3)
    if keep_te.any():
        dev_te = dev_te[keep_te]
    rank = effective_rank(dev_te)

    # -- 4. do equal-total-push arms ever disagree? -------------------------- #
    k = s_te.shape[1]
    yc_te_2d = y_te - y_te.mean(axis=1, keepdims=True)
    scale = float(np.median(np.abs(yc_te_2d))) or 1.0
    close = 0
    differing = 0
    gaps = []
    for i in range(s_te.shape[0]):
        if not (keep_te[i] if keep_te.size else True):
            continue                      # saturated step: no ranking question here
        rng = float(np.ptp(s_te[i])) or 1e-12
        for a in range(1, k):
            for b in range(a + 1, k):
                if abs(s_te[i, a] - s_te[i, b]) <= 0.02 * rng:
                    close += 1
                    d = abs(yc_te_2d[i, a] - yc_te_2d[i, b])
                    gaps.append(d)
                    differing += int(d > 0.10 * scale)
    order_frac = float(differing) / max(close, 1)

    verdict_1d = (r2_scalar > 0.70
                  and rank["eff_rank"] < 0.5 * rank["n_features"]
                  and order_frac < 0.30)

    return {
        "conv": conv,
        "n_steps": {"train": int(s_tr.shape[0]), "test": int(s_te.shape[0])},
        "degenerate_frac_test": degen_frac,
        "n_steps_scored_test": int(keep_te.sum()) if keep_te.size else int(s_te.shape[0]),
        "K": int(k), "plan_len": int(ft_te.plans.shape[2]),
        "scalar_r2": r2_scalar,
        "scalar_beta": beta,
        "full_r2": r2_full,
        "full_minus_scalar": r2_full - r2_scalar,
        "eff_rank": rank["eff_rank"],
        "n_dev_features": rank["n_features"],
        "top_eig_share": rank["top_share"],
        "order_pairs_n": int(close),
        "order_pairs_frac": order_frac,
        "order_gap_median": float(np.median(gaps)) if gaps else None,
        "verdict_one_dimensional": bool(verdict_1d),
    }


def format_report(d: Dict[str, object], name: str = "") -> str:
    lines = []
    if name:
        lines.append("dataset: %s" % name)
    lines.append("  convention            %s" % d["conv"])
    lines.append("  steps train/test      %d / %d   (K=%d, m=%d)"
                 % (d["n_steps"]["train"], d["n_steps"]["test"], d["K"], d["plan_len"]))
    lines.append("  degenerate (test)     %.1f%%   (scored on %d steps; saturated steps"
                 " carry no ranking question)"
                 % (100.0 * d["degenerate_frac_test"], d["n_steps_scored_test"]))
    lines.append("  1. scalar R^2         %.4f   (beta=%.4f on sum(aligned push))"
                 % (d["scalar_r2"], d["scalar_beta"]))
    lines.append("  2. full design R^2    %.4f   (gain over scalar: %+.4f)"
                 % (d["full_r2"], d["full_minus_scalar"]))
    lines.append("  3. effective rank     %.2f / %d   (top eigenvalue share %.2f)"
                 % (d["eff_rank"], d["n_dev_features"], d["top_eig_share"]))
    lines.append("  4. equal-push pairs   %d, of which disagree >10%%: %.1f%%"
                 % (d["order_pairs_n"], 100.0 * d["order_pairs_frac"]))
    lines.append("  VERDICT               %s"
                 % ("ONE-DIMENSIONAL (a heuristic is the right model)"
                    if d["verdict_one_dimensional"]
                    else "multi-dimensional (order information is available)"))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--conv", default="C")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ds = S.load(a.dataset)
    out = diagnose(ds, conv=a.conv)
    print(format_report(out, os.path.basename(a.dataset)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
