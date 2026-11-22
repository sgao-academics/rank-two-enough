"""Which block, exactly, buys B4's last point over the low-rank state x action term?

The ablation in section 6.2 added ONE block at a time to the linear skeleton and
found state x state worthless on its own. B4 is not one block at a time: it has
all three at once, and interactions between blocks are exactly what a one-at-a-
-time ablation cannot see. This asks the joint question, because it decides
whether a low-rank state x action plus an explicit plan x plan block can close
the gap to 560 parameters.

    T1  main                                       32
    T2  main + s x a                              230
    T3  main + s x a + a x a                      275
    T4  main + s x a + s x s                      483
    B4  main + s x a + s x s + a x a              528 (+32 main = 560)
"""

import glob
import io
import json
import os
import re
import sys
from statistics import mean, stdev

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import numpy as np                # noqa: E402
import metrics as M               # noqa: E402
import predictors as P            # noqa: E402
import report as R                # noqa: E402
import run_suite as RS            # noqa: E402
import split as SP                # noqa: E402
from lowrank_bilinear import split_parts       # noqa: E402

spec = M.load_spec()
eps = float(spec["noise_floor"]["deterministic_fallback"])


class Combo(P.Ridge):
    kind = "baseline"

    def __init__(self, sxa=True, sxs=False, axa=False, alpha=1.0):
        super().__init__(alpha=alpha)
        self.sxa, self.sxs, self.axa = bool(sxa), bool(sxs), bool(axa)
        tag = []
        if sxa:
            tag.append("sxa")
        if sxs:
            tag.append("sxs")
        if axa:
            tag.append("axa")
        self.name = "+".join(tag) if tag else "main"
        self.n_cols = None

    def _design(self, tab):
        p = split_parts(tab)
        n, k = p.main.shape[0], p.main.shape[1]
        s = p.state.reshape(-1, p.state.shape[-1])
        a = p.action.reshape(-1, p.action.shape[-1])
        blocks = [p.main.reshape(-1, p.main.shape[-1])]
        ds, da = s.shape[1], a.shape[1]
        if self.sxa:
            blocks.append((s[:, :, None] * a[:, None, :]).reshape(-1, ds * da))
        if self.sxs:
            ii, jj = np.triu_indices(ds)
            blocks.append(s[:, ii] * s[:, jj])
        if self.axa:
            ii, jj = np.triu_indices(da)
            blocks.append(a[:, ii] * a[:, jj])
        x = np.concatenate(blocks, axis=1)
        self.n_cols = x.shape[1] - 1                 # minus the intercept
        return x.reshape(n, k, -1)


COMBOS = [("main", dict(sxa=False)),
          ("sxa", dict()),
          ("sxa+axa", dict(axa=True)),
          ("sxa+sxs", dict(sxs=True)),
          ("sxa+sxs+axa", dict(sxs=True, axa=True))]

acc = {}
for seed in range(12):
    path = RS.ds_path("m3", seed)
    if not os.path.exists(path):
        continue
    ds = json.load(open(path, encoding="utf-8"))
    for sp in range(5):
        rows = SP.assign(ds["episodes"], seed=sp)
        ids = {"train": [], "val": [], "test": []}
        for row in rows:
            ids[row["split"]].append(row["episode_id"])
        for label, kw in COMBOS:
            m = Combo(**kw)
            res = P.evaluate_predictor(m, ds, convention="C",
                                       train_ids=ids["train"],
                                       test_ids=ids["test"], spec=spec,
                                       eps=eps, n_boot=0)
            acc.setdefault(label, []).append(
                float(R.summarize(res["report"], label)["dar"]))
            acc.setdefault(label + "|cols", []).append(m.n_cols)
    print("  [s%02d]" % seed, flush=True)

print("\n%-14s %8s %8s %8s" % ("design", "cols", "DAR", "sd"))
for label, _kw in COMBOS:
    v = acc[label]
    print("%-14s %8d %8.4f %8.4f"
          % (label, int(acc[label + "|cols"][0]), mean(v), stdev(v)))

out = {label: [mean(acc[label]), stdev(acc[label]),
               int(acc[label + "|cols"][0])] for label, _kw in COMBOS}
with open(os.path.join(RS.ROOT, "logs", "block_combo.json"), "w",
          encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote logs/block_combo.json")
