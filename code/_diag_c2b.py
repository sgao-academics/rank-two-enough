"""Is C2's 0.91 a fluke of one permutation seed, or structural?

If a within-step permutation of the design rows leaves the fitted direction aligned
with the truth, then either (a) the permutation is not doing what I think, or (b)
the design has an arm-symmetric channel that survives permutation. Both matter:
a negative control that cannot fail is not a control.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import sampler as S          # noqa: E402

DS = os.path.abspath(os.path.join(HERE, os.pardir, "data",
                                  "pilot_rsamp_m3_H6_48ep.json"))
N_LEV = 4


def main() -> int:
    ds = S.load(DS)
    sp = {}
    for ep in ds["episodes"]:
        sp.setdefault(ep["split"], []).append(ep["episode_id"])
    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])

    ft_tr = P.features_from(ds, sp["train"])
    y_tr = P.truth_matrix(ds, "C", sp["train"])

    m1 = P.StructuredPlan(level=3).fit(ft_tr, y_tr)
    w_ref = m1.w[1:][N_LEV:]
    print("M1 dev weights:", np.round(w_ref, 3))

    base = P.evaluate_predictor(P.StructuredPlan(level=3), ds, convention="C",
                                train_ids=sp["train"], test_ids=sp["test"],
                                spec=spec, eps=eps, n_boot=0)
    print("M1  DAR(test)=%.4f" % base["report"]["metrics"]["dar"]["value"])

    for name, kind in (("C2 within-step", "perm_arms"),
                       ("C2 across-rows ", "perm_rows")):
        dars, coss = [], []
        for s in range(10):
            class Shuf(P.StructuredPlan):
                def fit(self, train, y):
                    x = P._plan_design(train, self.level)
                    yy = np.array(y, dtype=np.float64, copy=True)
                    rng = np.random.default_rng(1000 + s)
                    if kind == "perm_arms":
                        perm = np.argsort(rng.random(yy.shape), axis=1)
                        x = np.take_along_axis(x, perm[:, :, None], axis=1)
                    else:
                        n = x.shape[0]
                        x = x[rng.permutation(n)]
                        yy = yy[rng.permutation(n)]
                    flat, tgt = x.reshape(-1, x.shape[-1]), yy.reshape(-1)
                    self.mu = flat.mean(0)
                    sd = flat.std(0)
                    self.sd = np.where(sd > 1e-12, sd, 1.0)
                    z = (flat - self.mu) / self.sd
                    a = z.T @ z + self.alpha * np.eye(z.shape[1])
                    b = z.T @ (tgt - tgt.mean())
                    self.w = np.concatenate([[tgt.mean()], np.linalg.solve(a, b)])
                    return self
            p = Shuf(level=3, alpha=0.1)
            r = P.evaluate_predictor(p, ds, convention="C", train_ids=sp["train"],
                                     test_ids=sp["test"], spec=spec, eps=eps,
                                     n_boot=0)
            dars.append(r["report"]["metrics"]["dar"]["value"])
            wd = p.w[1:][N_LEV:]
            coss.append(float(wd @ w_ref / (np.linalg.norm(wd) *
                                            np.linalg.norm(w_ref) + 1e-12)))
        print("%s DAR mean=%.4f sd=%.4f range=[%.3f,%.3f] | cos(w, w_M1) mean=%+.3f"
              % (name, np.mean(dars), np.std(dars), min(dars), max(dars),
                 np.mean(coss)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
