"""Upper bound for the rank-r hypothesis, before any optimisation.

If the 22 x 9 block fitted by T2 loses little when truncated to rank r, then the
low-rank parameterisation is the right form and only the optimisation is at
fault. If truncation collapses the score, the block genuinely needs its 198
degrees of freedom and the story changes.
"""
import os

import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import numpy as np                # noqa: E402
import metrics as M               # noqa: E402
import predictors as P            # noqa: E402
import report as R                # noqa: E402
import run_suite as RS            # noqa: E402
import split as SP                # noqa: E402
from lowrank_bilinear import split_parts        # noqa: E402

spec = M.load_spec()
eps = 1e-6


def flat(tab):
    p = split_parts(tab)
    return (p.main.reshape(-1, p.main.shape[-1]),
            p.state.reshape(-1, p.state.shape[-1]),
            p.action.reshape(-1, p.action.shape[-1]))


class TruncatedW(P.Ridge):
    """T2's explicit block, then SVD-truncated to rank r. An upper bound, not a model."""

    kind = "reference"

    def __init__(self, rank=3, alpha=1.0) -> None:
        super().__init__(alpha=alpha)
        self.rank = int(rank)
        self.name = "T2trunc r%d" % self.rank
        self.b = None
        self.W = None
        self.svd = None

    def fit(self, train, y_train):
        main, s, a = flat(train)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        inter = (s[:, :, None] * a[:, None, :]).reshape(-1, s.shape[1] * a.shape[1])
        design = np.concatenate([main, inter], axis=1)
        mu, sd = design.mean(axis=0), design.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        z = (design - mu) / sd
        z[:, 0] = 1.0                                  # keep the constant a constant
        A = z.T @ z + self.alpha * np.eye(z.shape[1])
        A[0, 0] -= self.alpha
        w = np.linalg.solve(A, z.T @ (y - y.mean()))
        self.mu, self.sd = mu, sd
        self.b = w[:main.shape[1]]
        Wz = w[main.shape[1]:].reshape(s.shape[1], a.shape[1])
        # undo the standardisation on the interaction block: column j of the flat
        # block is s_i * a_j with scale sd[i*d_a + j]
        scale = self.sd[main.shape[1]:].reshape(s.shape[1], a.shape[1])
        self.W = Wz / scale
        u, sv, vt = np.linalg.svd(self.W)
        self.svd = sv
        r = min(self.rank, len(sv))
        self.Ur = u[:, :r] * sv[:r]
        self.Vr = vt[:r].T
        return self

    def predict(self, test):
        main, s, a = flat(test)
        out = main @ self.b + np.einsum("ni,ij,nj->n", s, self.W, a)
        if self.rank < len(self.svd):
            out = main @ self.b + np.einsum("ni,ij,nj->n", s,
                                            self.Ur @ self.Vr.T, a)
        return out.reshape(main.shape[0] // split_k(test), split_k(test))


def split_k(tab):
    return int(tab.plans.shape[1])


for cfg, seed, sp in (("m3", 0, 0), ("m3", 1, 0)):
    ds = json.load(open(RS.ds_path(cfg, seed), encoding="utf-8"))
    rows = SP.assign(ds["episodes"], seed=sp)
    ids = {"train": [], "val": [], "test": []}
    for row in rows:
        ids[row["split"]].append(row["episode_id"])
    print("=== %s seed=%d split=%d ===" % (cfg, seed, sp))
    for label, fac in (("B4", lambda: P.RidgePoly(alpha=1.0)),
                       ("T2", None)):
        if fac is None:
            continue
        res = P.evaluate_predictor(fac(), ds, convention="C",
                                   train_ids=ids["train"],
                                   test_ids=ids["test"], spec=spec, eps=eps,
                                   n_boot=0)
        print("  B4  DAR=%.4f" % float(R.summarize(res["report"], "B4")["dar"]))
    for r in (1, 2, 3, 4, 5, 6, 8, 9):
        res = P.evaluate_predictor(TruncatedW(rank=r), ds, convention="C",
                                   train_ids=ids["train"],
                                   test_ids=ids["test"], spec=spec, eps=eps,
                                   n_boot=0)
        dar = float(R.summarize(res["report"], "x")["dar"])
        print("  T2 truncated to r=%d  DAR=%.4f" % (r, dar))
    # the spectrum itself
    tab_tr = P.features_from(ds, ids["train"])
    ytr = P.truth_matrix(ds, "C", ids["train"])
    tw = TruncatedW(rank=9).fit(tab_tr, ytr)
    sv = tw.svd
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    print("  singular values", np.round(sv, 4))
    print("  cumulative energy", np.round(energy, 4))
