# -*- coding: utf-8 -*-
"""Does modelling the CONTRAST beat modelling the LEVEL?

What the diagnosis said
-----------------------
On the v1.3 task, 84.6% of arm pairs with near-equal total push disagree in truth by
more than 10% of a typical |Delta y|: the order information IS in the data. Yet the
11 structural features gain only +0.037 R^2 over a single scalar, and a 0-parameter
heuristic matches the structural model (DAR 0.897 vs 0.928 vs oracle 0.869).

That combination -- information present, but no feature-level gain -- has one
reading that is testable: the information lives in how arms DIFFER, and a model that
regresses each arm's absolute outcome must first estimate a level that is dominated
by a large arm-independent component. Ranking, though, only ever needs differences.

So: fit the two models on IDENTICAL features and compare them as rankers.

  LEVEL    within-step ridge on the deviation block, i.e. what M1 does
  CONTRAST linear pairwise logistic (RankNet/Bradley-Terry with an L2 penalty),
           trained on sign(y_a - y_b) for every within-step pair

If CONTRAST wins, the fix is the objective, not more features -- and that is a
falsifiable, cheap claim rather than a new scene.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

import metrics as M       # noqa: E402
import predictors as P    # noqa: E402
import sampler as S       # noqa: E402
import split as SP        # noqa: E402
import task_diagnostics as TD   # noqa: E402

DS = os.path.join(HERE, os.pardir, "data", "pilot_rsamp_m3_H6_48ep.json")


def design(tab: P.FeatureTable, level: int = 3) -> np.ndarray:
    """The structural deviation block, already centred across arms by _plan_design."""
    return P._plan_design(tab, level)[:, :, 4:]


class PairwiseLogit:
    kind = "method"
    name = "P1 pairwise logistic (same features)"

    def __init__(self, lam: float = 1e-2, steps: int = 600, lr: float = 0.05,
                 seed: int = 0) -> None:
        self.lam, self.steps, self.lr, self.seed = lam, steps, lr, seed
        self.w: np.ndarray | None = None
        self.mu = self.sd = None

    def fit(self, tab: P.FeatureTable, y_train: np.ndarray):
        x = design(tab)
        y = np.asarray(y_train, dtype=np.float64)
        n, k, f = x.shape
        flat = x.reshape(-1, f)
        self.mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        z = ((x - self.mu) / self.sd).astype(np.float32)

        ia, ib, sg = [], [], []
        for i in range(k):
            for j in range(i + 1, k):
                d = y[:, i] - y[:, j]
                keep = np.abs(d) > 1e-12          # ties carry no preference
                if not keep.any():
                    continue
                ia.append(i)
                ib.append(j)
                sg.append(np.sign(d))
        ia, ib = np.array(ia), np.array(ib)
        s = np.concatenate([sg[p] for p in range(len(ia))]) if ia.size else np.zeros(0)
        # rows: pair index -> step index
        step_idx = np.concatenate([np.arange(n) for _ in range(len(ia))]) if ia.size else None
        pair_idx = np.concatenate([np.full(n, p) for p in range(len(ia))]) if ia.size else None

        torch.manual_seed(self.seed)
        zt = torch.from_numpy(z)
        w = torch.zeros(f, requires_grad=True)
        opt = torch.optim.Adam([w], lr=self.lr)
        a_idx = torch.from_numpy(ia[pair_idx])
        b_idx = torch.from_numpy(ib[pair_idx])
        st = torch.from_numpy(step_idx)
        sv = torch.from_numpy(s).float()
        for _ in range(self.steps):
            d = zt[st, a_idx] - zt[st, b_idx]
            loss = -torch.nn.functional.logsigmoid((d @ w) * sv).mean() \
                + self.lam * (w ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.w = w.detach().numpy().astype(np.float64)
        return self

    def predict(self, tab: P.FeatureTable) -> np.ndarray:
        x = design(tab)
        z = (x - self.mu) / self.sd
        return z @ self.w


class LevelRidge:
    kind = "baseline"
    name = "L0 within-step ridge (same features)"

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.w = self.mu = self.sd = None

    def fit(self, tab, y_train):
        x = design(tab)
        y = np.asarray(y_train, dtype=np.float64)
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        flat = xc.reshape(-1, xc.shape[-1])
        self.mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        z = (flat - self.mu) / self.sd
        a = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.w = np.linalg.solve(a, z.T @ yc.reshape(-1))
        return self

    def predict(self, tab):
        x = design(tab)
        z = ((x - self.mu) / self.sd).reshape(-1, x.shape[-1])
        return (z @ self.w).reshape(x.shape[0], x.shape[1])


def main() -> int:
    ds = S.load(DS)
    rows = SP.assign(ds["episodes"], seed=0)
    ids: dict[str, list] = {"train": [], "val": [], "test": []}
    for r in rows:
        ids[r["split"]].append(r["episode_id"])
    print("split: train/val/test = %d/%d/%d episodes"
          % (len(ids["train"]), len(ids["val"]), len(ids["test"])))

    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])

    diag = TD.diagnose(ds, conv="C", splits=ids)
    print("\n" + TD.format_report(diag, "pilot"))

    cands = {
        "R1 one-step oracle": P.Myopic(),
        "G0b aligned sum (0p)": P.HeuristicAlignedSum(),
        "M1 structured (level)": P.StructuredPlan(level=3, alpha=0.1),
        "L0 ridge (same feats)": LevelRidge(0.1),
    }
    for lam in (1e-3, 1e-2, 1e-1):
        cands["P1 pairwise lam=%g" % lam] = PairwiseLogit(lam=lam)

    print("\n%-26s %8s %9s %8s" % ("ranker", "DAR", "regret", "nMAE"))
    out = {}
    for lab, pred in cands.items():
        r = P.evaluate_predictor(pred, ds, convention="C", train_ids=ids["train"],
                                 test_ids=ids["test"], spec=spec, eps=eps, n_boot=0)
        m = r["report"]["metrics"]
        g = lambda k: m[k]["value"] if m[k]["value"] is not None else float("nan")
        out[lab] = (g("dar"), g("regret_norm"), g("nmae"))
        print("%-26s %8.4f %9.4f %8.4f" % (lab, g("dar"), g("regret_norm"), g("nmae")))

    base = out["M1 structured (level)"][0]
    best_p = max(v[0] for k, v in out.items() if k.startswith("P1"))
    print("\nbest pairwise - M1(level) = %+.4f DAR" % (best_p - base))
    print("best pairwise - G0b      = %+.4f DAR"
          % (best_p - out["G0b aligned sum (0p)"][0]))
    print("best pairwise - R1       = %+.4f DAR"
          % (best_p - out["R1 one-step oracle"][0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
