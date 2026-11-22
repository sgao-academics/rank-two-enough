# -*- coding: utf-8 -*-
"""Statistics for the seven-point tail relation.

Reports, for each candidate parameterisation: in-sample R^2, leave-one-out R^2,
slope and intercept with bootstrap intervals, and the through-origin fit. Also
reports the identity between the odds and the tail-steps forms.

Reads logs/tail_scan/tail_scan.csv; rewrites nothing.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
PATH = os.path.join(ROOT, "logs", "tail_scan", "tail_scan.csv")

rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
H = np.array([float(r["H"]) for r in rows])
m = 3.0
tail = H - m
t = tail / H
odds = tail / m
gain = np.array([float(r["gain"]) for r in rows])

print("H      tail  t      odds   gain")
for i in range(len(H)):
    print("%4.0f  %4.0f  %.3f  %.3f  %.5f" % (H[i], tail[i], t[i], odds[i], gain[i]))


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return coef, 1.0 - ss_res / ss_tot


def loo(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    err = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        A = np.stack([np.ones(keep.sum()), x[keep]], axis=1)
        coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        err.append(coef[0] + coef[1] * x[i] - y[i])
    err = np.array(err)
    return 1.0 - float((err ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def boot(x, y, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    sl, ic = [], []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        A = np.stack([np.ones(len(idx)), x[idx]], axis=1)
        c, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
        sl.append(c[1])
        ic.append(c[0])
    return np.percentile(sl, [2.5, 97.5]), np.percentile(ic, [2.5, 97.5])


forms = [("odds (H-m)/m", odds), ("tail steps H-m", tail), ("tail share t", t),
         ("log odds", np.log(odds)), ("H", H)]

print()
print("%-18s %8s %8s %9s %9s %24s %24s"
      % ("form", "R2", "LOO R2", "slope", "intercept", "slope 95% CI", "intercept 95% CI"))
for name, x in forms:
    coef, r2 = fit(x, gain)
    sci, ici = boot(x, gain)
    print("%-18s %8.4f %8.4f %9.5f %9.5f  [%+.5f, %+.5f]  [%+.5f, %+.5f]"
          % (name, r2, loo(x, gain), coef[1], coef[0], sci[0], sci[1], ici[0], ici[1]))

k = float((odds @ gain) / (odds @ odds))
yhat = k * odds
ss_res = float(((gain - yhat) ** 2).sum())
print()
print("through-origin, odds form: slope %.5f" % k)
print("  uncentred R2  = %.4f" % (1 - ss_res / float((gain ** 2).sum())))
print("  centred  R2  = %.4f" % (1 - ss_res / float(((gain - gain.mean()) ** 2).sum())))
print("  residuals:", np.round(gain - yhat, 5).tolist())

c_odds, _ = fit(odds, gain)
c_tail, _ = fit(tail, gain)
print()
print("slope ratio odds/tail-steps = %.6f  (equals m = %.0f)" % (c_odds[1] / c_tail[1], m))
print("LOO identical: %.10f vs %.10f" % (loo(odds, gain), loo(tail, gain)))
