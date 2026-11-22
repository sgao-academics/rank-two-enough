# -*- coding: utf-8 -*-
"""Sweep the PLAN LENGTH at a fixed horizon.

The tail sweep of `_scan_tail.py` holds m at three and varies H, which makes the
odds (H-m)/m an affine transform of the number of tail steps H-m: the two cannot be
told apart. Fixing H and varying m separates them, at the price of a second
collinearity, because with H fixed the odds are H/m - 1, an affine transform of
1/m. Both limitations are reported rather than assumed away.

Design: H = 6, T = 15, K = 5, sigma = 0.8, 96 episodes, dataset seed 0, plan length
m in {1, 2, 4}. The m = 3 cell is the cached H06 dataset of the tail sweep; it is
read from disk rather than regenerated, so the two sweeps share one point.

Each configuration is cached by (H, m), so the run is resumable. Run detached; it
writes a CSV and a log.
"""
from __future__ import annotations

import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

# the stored-results directory is `logs` in the package and `05_Logs` beside it in
# the working tree; pick whichever exists so that one script serves both
LOGS_DIR = None
for _cand in ("logs", "05_Logs"):
    if os.path.isdir(os.path.join(ROOT, _cand)):
        LOGS_DIR = os.path.join(ROOT, _cand)
        break
if LOGS_DIR is None:
    raise SystemExit("no stored-results directory found beside %s" % HERE)

OUT = os.path.join(LOGS_DIR, "mdim_scan")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "progress.log")

H_FIXED = 6
T = 15
M_LIST = (1, 2, 4)
N_EP = 96


def say(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def dataset_path(m: int) -> str:
    return os.path.join(OUT, "ds_H%02d_m%d.json" % (H_FIXED, m))


def main() -> int:
    import mujoco_adapter as MU       # noqa: E402
    import sampler as S               # noqa: E402
    import split as SP                # noqa: E402
    import task_diagnostics as TD     # noqa: E402
    from run_pilot import make_factors  # noqa: E402

    say("plan-length sweep at fixed H=%d: m in %s, T=%d, K=5, sigma=0.8, %d episodes"
        % (H_FIXED, list(M_LIST), T, N_EP))

    rows = []
    for m in M_LIST:
        path = dataset_path(m)
        if os.path.exists(path):
            ds = S.load(path)
            say("m=%d  cached  %s" % (m, os.path.basename(path)))
        else:
            rng = np.random.default_rng(0)
            factors = [make_factors(rng, i) for i in range(N_EP)]
            ids = ["s00_%03d" % i for i in range(N_EP)]
            cfg = S.SamplerConfig(K=5, H=H_FIXED, plan_len=m, T=T, seed=0,
                                  proposal_sigma=0.8, control_substeps=50)
            adapter = MU.MuJoCoPush3D(substeps=cfg.control_substeps)
            ds = S.generate(adapter, factors, cfg=cfg, ids=ids)
            ds["meta"]["cache"] = {"scene_tag": "mdim_scan", "H": H_FIXED,
                                   "plan_len": m, "episodes": N_EP}
            S.save(ds, path)
            say("m=%d  GEN     %s" % (m, os.path.basename(path)))

        rows_s = SP.assign(ds["episodes"], seed=0)
        ids_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
        for r in rows_s:
            ids_by_split[r["split"]].append(r["episode_id"])
        d = TD.diagnose(ds, conv="C", splits=ids_by_split)

        tail = float(H_FIXED - m)
        odds = tail / float(m)
        rows.append(dict(H=H_FIXED, m=m, tail_steps=tail,
                         tail_pct=100.0 * tail / H_FIXED, odds=odds,
                         degenerate=d["degenerate_frac_test"],
                         scalar_r2=d["scalar_r2"], full_r2=d["full_r2"],
                         gain=d["full_minus_scalar"], eff_rank=d["eff_rank"]))
        say("m=%d  tail=%.2f steps (%.1f%%)  odds=%.3f  degen=%4.1f%%  scalarR2=%.4f  "
            "fullR2=%.4f  gain=%+.4f"
            % (m, tail, rows[-1]["tail_pct"], odds, rows[-1]["degenerate"],
               d["scalar_r2"], d["full_r2"], d["full_minus_scalar"]))

    # ---- the m = 3 anchor, read from the tail sweep rather than regenerated ---- #
    anchor_path = os.path.join(LOGS_DIR, "tail_scan", "tail_scan.csv")
    if os.path.exists(anchor_path):
        with io.open(anchor_path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("H"):
                    continue
                if int(r["H"]) == H_FIXED and int(r["m"]) == 3:
                    rows.append(dict(H=H_FIXED, m=3, tail_steps=float(H_FIXED - 3),
                                     tail_pct=100.0 * (H_FIXED - 3) / H_FIXED,
                                     odds=(H_FIXED - 3) / 3.0, degenerate=float(r["degenerate"]),
                                     scalar_r2=float(r["scalar_r2"]),
                                     full_r2=float(r["full_r2"]),
                                     gain=float(r["gain"]), eff_rank=float(r["eff_rank"])))
        rows.sort(key=lambda d: d["m"])
        say("anchor m=3 taken from %s" % os.path.basename(anchor_path))

    csv_out = os.path.join(OUT, "mdim_scan.csv")
    cols = list(rows[0].keys())
    with open(csv_out, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("%.6g" % r[c] if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    say("wrote %s (%d rows)" % (csv_out, len(rows)))

    odds_v = np.array([r["odds"] for r in rows])
    tail_v = np.array([r["tail_steps"] for r in rows])
    gain_v = np.array([r["gain"] for r in rows])
    if len(rows) >= 3:
        say("corr(odds, tail steps) over this grid = %+.4f"
            % float(np.corrcoef(odds_v, tail_v)[0, 1]))
        for name, x in (("odds (H-m)/m", odds_v), ("tail steps H-m", tail_v),
                        ("1/m", 1.0 / np.array([r["m"] for r in rows], float))):
            A = np.stack([np.ones_like(x), x], axis=1)
            c, *_ = np.linalg.lstsq(A, gain_v, rcond=None)
            r2 = 1 - float(((gain_v - A @ c) ** 2).sum()) / \
                float(((gain_v - gain_v.mean()) ** 2).sum())
            say("gain ~ %-14s slope %+.5f  R2 %.4f" % (name, c[1], r2))
        say("gains: " + ", ".join("%.4f" % r["gain"] for r in
                                 sorted(rows, key=lambda d: d["m"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
