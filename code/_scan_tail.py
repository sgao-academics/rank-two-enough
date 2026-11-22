# -*- coding: utf-8 -*-
"""Sweep the decision unit's TAIL FRACTION and measure how one-dimensional the ranking is.

Established so far on the six suite configurations (no simulation, existing data):

    tail%   scalar R^2   gain from the full feature set
    25.0    0.8730       +0.0207        (H=4, m=3)
    50.0    0.8172       +0.0331        (H=6, m=3)
    62.5    0.7472       +0.0520        (H=8, m=3)

Degenerate (saturated) steps were 0.0-0.4%, so that monotone trend is not an artefact
of longer horizons spending more decisions at the target. If it holds over the whole
range, it is a design law rather than a coincidence:

    the plan-to-outcome map becomes LESS linear as the closed-loop tail grows,
    because correction is saturating and time-lagged, so "how much push" stops being
    a sufficient statistic and the multi-dimensional content of the plan starts to pay.

This script fills in H = 4..10 at m = 3, T = 15, one dataset seed (the quantity is a
property of the DATA, so seeds are not the inferential unit here -- the sweep is).
Each dataset is cached by H, so the run is resumable and nothing is recomputed.

Run detached; it writes a CSV and a log. ~2-4 min per new H.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUT = os.path.join(ROOT, "logs", "tail_scan")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "progress.log")


def say(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    import mujoco_adapter as MU       # noqa: E402
    import sampler as S               # noqa: E402
    import split as SP                # noqa: E402
    import task_diagnostics as TD     # noqa: E402
    from run_pilot import make_factors  # noqa: E402

    say("tail scan start: m=3, T=15, K=5, sigma=0.8, 96 episodes, dataset seed 0")
    rows = []
    for H in range(4, 11):
        path = os.path.join(OUT, "ds_H%02d_m3.json" % H)
        if os.path.exists(path):
            ds = S.load(path)
            say("H=%2d  cached  %s" % (H, os.path.basename(path)))
        else:
            rng = np.random.default_rng(0)
            factors = [make_factors(rng, i) for i in range(96)]
            ids = ["s00_%03d" % i for i in range(96)]
            cfg = S.SamplerConfig(K=5, H=H, plan_len=3, T=15, seed=0,
                                  proposal_sigma=0.8, control_substeps=50)
            adapter = MU.MuJoCoPush3D(substeps=cfg.control_substeps)
            ds = S.generate(adapter, factors, cfg=cfg, ids=ids)
            ds["meta"]["cache"] = {"scene_tag": "tail_scan", "H": H,
                                   "plan_len": 3, "episodes": 96}
            S.save(ds, path)
            say("H=%2d  GEN     %s" % (H, os.path.basename(path)))

        rows_s = SP.assign(ds["episodes"], seed=0)
        ids: dict[str, list] = {"train": [], "val": [], "test": []}
        for r in rows_s:
            ids[r["split"]].append(r["episode_id"])
        d = TD.diagnose(ds, conv="C", splits=ids)
        tail = 100.0 * (H - 3) / H
        rows.append(dict(H=H, m=3, tail_pct=tail,
                         degenerate=100.0 * d["degenerate_frac_test"],
                         scalar_r2=d["scalar_r2"], full_r2=d["full_r2"],
                         gain=d["full_minus_scalar"], eff_rank=d["eff_rank"],
                         order_pct=100.0 * d["order_pairs_frac"],
                         n_order_pairs=d["order_pairs_n"]))
        say("H=%2d tail=%5.1f%%  degen=%4.1f%%  scalarR2=%.4f  fullR2=%.4f  "
            "gain=%+.4f  order=%5.1f%% (%d pairs)"
            % (H, tail, rows[-1]["degenerate"], d["scalar_r2"], d["full_r2"],
               d["full_minus_scalar"], rows[-1]["order_pct"], d["order_pairs_n"]))

    csv = os.path.join(OUT, "tail_scan.csv")
    cols = list(rows[0].keys())
    with open(csv, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("%.6g" % r[c] if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    say("wrote %s" % csv)

    # correlation of the two headline quantities with the tail fraction
    tail = np.array([r["tail_pct"] for r in rows])
    sc = np.array([r["scalar_r2"] for r in rows])
    gn = np.array([r["gain"] for r in rows])
    say("corr(tail, scalarR2) = %+.4f   corr(tail, gain) = %+.4f"
        % (float(np.corrcoef(tail, sc)[0, 1]), float(np.corrcoef(tail, gn)[0, 1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
