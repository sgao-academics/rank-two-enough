# -*- coding: utf-8 -*-
"""Effective sample size for the rank comparison.

The 180 evaluation cells are twelve seeds crossed with five template partitions.
The partitions reuse the same sixteen templates, so the seed is the inferential
unit. Reports the paired difference at cell and seed level, a cluster bootstrap
over seeds, and a variance decomposition. Reads logs/lrb_vs_bilinear.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
d = json.load(open(os.path.join(ROOT, "logs", "lrb_vs_bilinear.json"), encoding="utf-8"))
pc = d["per_cell"]
A = "LRB r2 (139)"
BASE = "sxa+sxs+axa (528)"

print("seeds=%d splits=%d cfgs=%d cells=%d"
      % (len(d["seeds"]), len(d["splits"]), len(d["configs"]), len(pc)))

diff = [r[A] - r[BASE] for r in pc]


def stats(x):
    x = np.asarray(x, float)
    n = len(x)
    return x.mean(), x.std(ddof=1), x.std(ddof=1) / np.sqrt(n), \
        x.mean() / (x.std(ddof=1) / np.sqrt(n)), n


m, s, se, t, n = stats(diff)
print()
print("CELL level (n=%d, cells treated as independent)" % n)
print("  mean %+.5f  sd %.5f  se %.5f  t %.2f" % (m, s, se, t))

seed_vals = {}
for r, df in zip(pc, diff):
    seed_vals.setdefault(r["seed"], []).append(df)
vals = np.array([float(np.mean(seed_vals[k])) for k in sorted(seed_vals)])
m2, s2, se2, t2, n2 = stats(vals)
print()
print("SEED level (n=%d, the independent unit)" % n2)
print("  mean %+.5f  sd %.5f  se %.5f  t %.2f (df %d)" % (m2, s2, se2, t2, n2 - 1))
print("  seed means:", np.round(vals, 5).tolist())
print("  SE inflation from cell level: x%.2f" % (se2 / se))

rng = np.random.default_rng(11)
boot = np.array([float(vals[rng.integers(0, n2, n2)].mean()) for _ in range(50000)])
print()
print("cluster bootstrap: 95%% CI [%+.5f, %+.5f]"
      % tuple(np.percentile(boot, [2.5, 97.5])))

print()
print("per-horizon seed-level means:")
for cfg in d["configs"]:
    v = np.array([float(np.mean([df for r, df in zip(pc, diff)
                                 if r["cfg"] == cfg and r["seed"] == sd]))
                  for sd in d["seeds"]])
    print("  %-4s mean %+.5f  sd %.5f  se %.5f  t %.2f"
          % (cfg, v.mean(), v.std(ddof=1), v.std(ddof=1) / np.sqrt(len(v)),
             v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))))

grand = float(np.mean(diff))
ss_b, ss_w = 0.0, 0.0
for sd in d["seeds"]:
    rows = [df for r, df in zip(pc, diff) if r["seed"] == sd]
    sm = float(np.mean(rows))
    ss_b += len(rows) * (sm - grand) ** 2
    ss_w += float(((np.array(rows) - sm) ** 2).sum())
print()
print("variance decomposition: between-seed per df %.3g, within-seed per df %.3g, "
      "ratio %.3f" % (ss_b / 11, ss_w / 168, (ss_w / 168) / (ss_b / 11)))
