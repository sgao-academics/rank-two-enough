# -*- coding: utf-8 -*-
"""Decompose the structural gain by mechanism, per horizon.

Same protocol as task_diagnostics.diagnose: fit on train steps, score on
non-degenerate test steps, both centred across arms. Reads the seven tail-sweep
datasets; nothing is re-simulated.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)

import predictors as P          # noqa: E402
import sampler as S             # noqa: E402
import split as SP              # noqa: E402
import task_diagnostics as TD   # noqa: E402

LOGS = os.path.join(ROOT, "logs")


def r2(y2d, yhat2d, mask):
    y, yhat = y2d[mask].reshape(-1), yhat2d[mask].reshape(-1)
    den = float((y ** 2).sum())
    if den <= 0:
        return float("nan")
    return 1.0 - float(((y - yhat) ** 2).sum()) / den


def one(H):
    ds = S.load(os.path.join(LOGS, "tail_scan", "ds_H%02d_m3.json" % H))
    rows = SP.assign(ds["episodes"], seed=0)
    ids = {"train": [], "val": [], "test": []}
    for r in rows:
        ids[r["split"]].append(r["episode_id"])

    ft_tr = P.features_from(ds, ids["train"])
    ft_te = P.features_from(ds, ids["test"])
    y_tr = P.truth_matrix(ds, "C", ids["train"])
    y_te = P.truth_matrix(ds, "C", ids["test"])

    thr = 0.25 * float(np.median(np.ptp(y_tr, axis=1)))
    keep = np.ptp(y_te, axis=1) > thr

    s_tr = TD.aligned_push(ft_tr).sum(axis=2)
    s_te = TD.aligned_push(ft_te).sum(axis=2)
    sc = (s_tr - s_tr.mean(axis=1, keepdims=True)).reshape(-1)
    yc = (y_tr - y_tr.mean(axis=1, keepdims=True)).reshape(-1)
    beta = float((sc @ yc) / max(float(sc @ sc), 1e-30))
    out = {"H": H,
           "scalar": r2(y_te - y_te.mean(axis=1, keepdims=True),
                        beta * (s_te - s_te.mean(axis=1, keepdims=True)), keep)}

    for lev in (1, 2, 3, 4, 5):
        mdl = P.StructuredPlan(level=lev, alpha=0.1).fit(ft_tr, y_tr)
        pr = mdl.predict(ft_te)
        out["L%d" % lev] = r2(y_te - y_te.mean(axis=1, keepdims=True),
                              pr - pr.mean(axis=1, keepdims=True), keep)

    out["d_frame"] = out["L1"] - out["scalar"]
    out["d_abs"] = out["L2"] - out["L1"]
    out["d_cum"] = out["L3"] - out["L2"]
    out["d_phi"] = out["L4"] - out["L3"]
    return out


rows = [one(H) for H in range(4, 11)]
hdr = ("  H  tail%| scalar     L1     L2     L3     L4     L5  |"
       " dFrame   dAbs    dCum    dPhi")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print("%3d %5.1f|%7.4f%7.4f%7.4f%7.4f%7.4f%7.4f |%7.4f %7.4f %7.4f %7.4f"
          % (r["H"], 100.0 * (r["H"] - 3) / r["H"], r["scalar"], r["L1"], r["L2"],
             r["L3"], r["L4"], r["L5"], r["d_frame"], r["d_abs"], r["d_cum"], r["d_phi"]))

tail = np.array([100.0 * (r["H"] - 3) / r["H"] for r in rows])
print()
for k in ("d_frame", "d_abs", "d_cum", "d_phi"):
    v = np.array([r[k] for r in rows])
    print("corr(tail, %-7s) = %+.4f   range %+.4f .. %+.4f"
          % (k, float(np.corrcoef(tail, v)[0, 1]), v.min(), v.max()))
