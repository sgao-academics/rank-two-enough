"""Does a 560-parameter quadratic need more data than a 11-parameter frame?

The suite already refits M1 and B2 at 10/25/50/100% of the training split, but the
model that actually beats M1 -- B4, the generic quadratic -- was never put on that
curve. So the one remaining positive claim for structure ("it buys sample
efficiency") was untested: the comparison it had was against B2, which scores 0.557
against a ZERO-parameter heuristic's 0.891 and is therefore not a competitive
baseline at all.

This script refits G0b / G1 / M1 / B4 at four training sizes on the datasets that
already exist. No simulation is run; the cost is a closed-form solve per cell.

    python _curve_lowdata.py --configs m3 H4 H8
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import report as R           # noqa: E402
import split as SP           # noqa: E402
import run_suite as RS       # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

FRACS = (0.10, 0.25, 0.50, 1.00)
MODELS = (("G0b summed aligned push", lambda: P.HeuristicAlignedSum()),
          ("G1 global direction", lambda: P.GlobalDirection()),
          ("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1)),
          ("B4 ridge poly2", lambda: P.RidgePoly()))


def sub_train(ids, frac):
    """Same rule as run_suite._sub_train: the FIRST n sorted ids.

    Note the consequence, which matters for interpretation: at 10% the model sees
    roughly one template's worth of episodes, so this curve measures "one template
    of data" more than "10% of the data". It is still the stored protocol and is
    reported as such rather than silently changed.
    """
    ordered = sorted(ids)
    n = max(1, int(round(frac * len(ordered))))
    return ordered[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["m3", "H4", "H8"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
    ap.add_argument("--splits", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--convention", default="C")
    ap.add_argument("--json", default=os.path.join(
        RS.ROOT, "logs", "route_b_lowdata.json"))
    args = ap.parse_args()

    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])
    acc = {}
    n_cells = 0

    for cfg in args.configs:
        for seed in args.seeds:
            path = RS.ds_path(cfg, seed)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                ds = json.load(fh)
            for sp in args.splits:
                rows = SP.assign(ds["episodes"], seed=sp)
                ids = {"train": [], "val": [], "test": []}
                for r in rows:
                    ids[r["split"]].append(r["episode_id"])
                if not (ids["train"] and ids["test"]):
                    continue
                for frac in FRACS:
                    tr = sub_train(ids["train"], frac)
                    for label, factory in MODELS:
                        try:
                            res = P.evaluate_predictor(
                                factory(), ds, convention=args.convention,
                                train_ids=tr, test_ids=ids["test"], spec=spec,
                                eps=eps, n_boot=0)
                            # DAR lives in report.summarize's row, not in the report
                            # dict itself: the report keeps `metrics` keyed by name.
                            dar = float(R.summarize(res["report"],
                                                    res["name"])["dar"])
                        except Exception as exc:
                            print("  ! %s s%02d sp%d %.2f %s: %s"
                                  % (cfg, seed, sp, frac, label, exc))
                            continue
                        acc.setdefault((cfg, frac, label), []).append(dar)
                        n_cells += 1
                print("  [%s s%02d sp%d] done" % (cfg, seed, sp), flush=True)

    print("\n" + "=" * 74)
    print("DAR vs training size   (convention %s, unit = dataset seed x split)"
          % args.convention)
    print("=" * 74)

    out = {}
    for cfg in args.configs:
        keys = [k for k in acc if k[0] == cfg]
        if not keys:
            continue
        print("\n[%s]  n per cell ~ %d" % (cfg, len(acc[keys[0]])))
        print("  %-22s %8s %8s %8s %8s %10s"
              % ("model", "10%", "25%", "50%", "100%", "slope"))
        for label, _f in MODELS:
            vals = []
            for frac in FRACS:
                v = acc.get((cfg, frac, label))
                vals.append(mean(v) if v else float("nan"))
                out.setdefault(cfg, {}).setdefault(label, {})["%.2f" % frac] = \
                    (mean(v), stdev(v) if v and len(v) > 1 else 0.0, len(v) if v else 0)
            slope = (vals[3] - vals[0]) if vals[0] == vals[0] and vals[3] == vals[3] \
                else float("nan")
            print("  %-22s %8.4f %8.4f %8.4f %8.4f %+10.4f"
                  % (label, vals[0], vals[1], vals[2], vals[3], slope))

        b4 = [acc.get((cfg, f, "B4 ridge poly2")) for f in FRACS]
        m1 = [acc.get((cfg, f, "M1 L3 a0.1")) for f in FRACS]
        if all(b4) and all(m1):
            print("  B4 - M1: " + "  ".join(
                "%+.4f" % (mean(b) - mean(m)) for b, m in zip(b4, m1)))

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"cells": n_cells, "convention": args.convention,
                   "fracs": list(FRACS), "result": out}, fh, indent=1)
    print("\nwrote %s  (%d cells)" % (args.json, n_cells))


if __name__ == "__main__":
    main()
