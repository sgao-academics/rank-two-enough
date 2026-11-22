# -*- coding: utf-8 -*-
"""Does the ranking's dimensionality depend on the DECISION UNIT rather than the scene?

The suite already contains six configurations that differ only in the decision unit:
    m1    plan_len 1, H 6     -> the decision is one action
    m3    plan_len 3, H 6     -> primary
    H4    plan_len 3, H 4     -> the plan is 3 of 4 steps (less tail)
    H8    plan_len 3, H 8     -> the plan is 3 of 8 steps (more tail)
    sig04/sig16  plan_len 3, sigma 0.4/1.6  -> how far arms are spread

If the ranking's non-degeneracy is a property of the SCENE, these six numbers should
look alike. If it is a property of the DECISION UNIT -- the resolution at which arms
differ, and how much of the outcome the tail is allowed to absorb -- they should move
with plan_len and with the tail fraction. That distinction decides what the paper can
claim is wrong: the task, or the unit we chose to test it with.

Everything here is computed from datasets that already exist on disk; no simulation.
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

import sampler as S              # noqa: E402
import split as SP               # noqa: E402
import task_diagnostics as TD    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir))
DATA = os.path.join(ROOT, "data", "suite", "scene2_spec13")

CFGS = [("m1", 6, 1), ("H4", 4, 3), ("m3", 6, 3), ("H8", 8, 3),
        ("sig04", 6, 3), ("sig16", 6, 3)]


def main() -> int:
    print("%-7s %4s %4s %7s %8s %10s %10s %10s %10s %9s"
          % ("cfg", "H", "m", "tail%", "degen%", "scalarR2", "fullR2", "gain",
             "effRank", "order%"))
    rows = []
    for name, H, m in CFGS:
        path = os.path.join(DATA, "ds_%s_s00_e96.json" % name)
        if not os.path.exists(path):
            print("%-7s  (missing %s)" % (name, os.path.basename(path)))
            continue
        ds = S.load(path)
        rows_s = SP.assign(ds["episodes"], seed=0)
        ids: dict[str, list] = {"train": [], "val": [], "test": []}
        for r in rows_s:
            ids[r["split"]].append(r["episode_id"])
        d = TD.diagnose(ds, conv="C", splits=ids)
        tail = 100.0 * (H - m) / H
        print("%-7s %4d %4d %7.1f %8.1f %10.4f %10.4f %+10.4f %6.2f/%-2d %9.1f"
              % (name, H, m, tail, 100.0 * d["degenerate_frac_test"],
                 d["scalar_r2"], d["full_r2"],
                 d["full_minus_scalar"], d["eff_rank"], d["n_dev_features"],
                 100.0 * d["order_pairs_frac"]))
        rows.append((name, H, m, tail, d))

    print("\n--- what moves with the decision unit ---")
    if len(rows) > 1:
        for name, H, m, tail, d in rows:
            print("  %-7s plan_len=%d tail=%4.1f%%  equal-push pairs=%3d  "
                  "disagree=%5.1f%%  gain over scalar=%+.4f"
                  % (name, m, tail, d["order_pairs_n"],
                     100 * d["order_pairs_frac"], d["full_minus_scalar"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
