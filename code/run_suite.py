"""run_suite.py -- the full experiment matrix, cached and resumable.

What this runs, and why each axis exists
----------------------------------------
Axis 1  dataset seed            0..11   (12 independent scenes/seeds -- not 3)
Axis 2  split seed              0..4    (5 template partitions -- re-splits the same
                                         data, so split variance is estimated
                                         without regenerating anything)
Axis 3  configuration           m3 (primary), m1 (plan ablation), sigma 0.4/1.6
                                (dispersion), H=4/H=8 (horizon)
Axis 4  predictor               B0 B1 B2 B3 B4 (generic interactions) M1 {level
                                0..6} x alpha {0.1,1,10} C1 C2 R1 L1 L2
Axis 5  convention              C (primary) and O (secondary)
Axis 6  training size           M1/B2 refit on 10/25/50/100% of the training split
Axis 7  supervision transfer    M1 trained on C, evaluated on O

Every cell is one file. A file that exists with a matching schema is skipped, so an
interrupted run resumes where it stopped and no run ever has to be repeated from
zero. Nothing here is tuned to a result: the grids are fixed in code below, and the
aggregator (`run_aggregate.py`) applies the pre-declared selection rule (choose on
VAL, report on TEST).

Cache keys include the SCENE TAG. That is deliberate and it is the only defence
against the failure mode that has already cost this project a day: a scene change
(the pusher's start point, the expert's standoff) silently invalidated every number
while the JSON files happily sat there looking valid.

    python run_suite.py                         # everything, resuming
    python run_suite.py --configs m3 --seeds 0 1
    python run_suite.py --budget-minutes 30     # stop cleanly, resume later
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import report as R           # noqa: E402
import sampler as S          # noqa: E402
import split as SP           # noqa: E402

# Bump SCENE_TAG whenever anything that changes the data changes: the scene, the
# expert, the sampler, the spec. Old caches then stay on disk (harmless) instead of
# being silently reused.
SCENE_TAG = "scene2_spec13"
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
OUT = os.path.join(ROOT, "logs", "suite", SCENE_TAG)
DATA = os.path.join(ROOT, "data", "suite", SCENE_TAG)
PROGRESS = os.path.join(OUT, "progress.log")

EPISODES = 96          # per dataset seed (24 test episodes under a 70/15/15 split)
T_STEPS = 15
DATASET_SEEDS = tuple(range(12))
SPLIT_SEEDS = tuple(range(5))
# v3: the eval payload now also stores VAL rows. Without them "choose on val, report
# on test" is a sentence rather than a procedure: the aggregate had nothing to
# choose from and would have silently reported a pre-registered hyperparameter while
# claiming a selection protocol. v2 was the stratified template split. The schema
# string is the cache key for both decisions -- bumping it recomputes every
# evaluation without deleting anything.
EVAL_SCHEMA = "suite-evals/3"

CONFIGS = {
    # name      H  plan_len  sigma   note
    "m3":      dict(H=6, plan_len=3, sigma=0.8),   # spec v1.3 primary
    "m1":      dict(H=6, plan_len=1, sigma=0.8),   # plan-content ablation
    "sig04":   dict(H=6, plan_len=3, sigma=0.4),   # dispersion sensitivity
    "sig16":   dict(H=6, plan_len=3, sigma=1.6),
    "H4":      dict(H=4, plan_len=3, sigma=0.8),   # shorter horizon (tail 1)
    "H8":      dict(H=8, plan_len=3, sigma=0.8),   # longer horizon (tail 5)
    # Plan-length ladder at a fixed tail share. `m1` above varies the plan length
    # at H=6, which moves two things at once: how often a decision is made, and
    # the tail share (H-m)/H -- 5/6 against the primary configuration's 3/6. The
    # ladder holds the share at 1/2 by setting H=2m, so the only axis that moves
    # is m: (1,2), (2,4), (4,8), with (3,6) already on disk as `m3`.
    "pl1":     dict(H=2, plan_len=1, sigma=0.8),   # ladder, tail share 1/2
    "pl2":     dict(H=4, plan_len=2, sigma=0.8),
    "pl4":     dict(H=8, plan_len=4, sigma=0.8),
}
K = 5
HEADLINES = ("B1 bet-on-expert", "B2 ridge surrogate", "M1 L3 a0.1",
             "C1 ridge arm-shuffled", "C2 plan-shuffled",
             "R1 one-step oracle", "L1 learned one-step (linear)",
             "L2 learned one-step (frame)",
             # the trivial rankers MUST carry a full bootstrap CI: they are the
             # lines that decide whether the structural claim survives at all
             "G0 perturbation magnitude", "G0b summed aligned push",
             "G1 global direction")
HEAD_BOOT = 2000       # within-run percentile CI. The spec asks for 10000 and the
                       # FINAL headline table will be produced at 10000 for a subset
                       # of seeds; at 10000 x 11 predictors x 5 splits x 72 datasets
                       # the sweep is ~24 h, and the inferential unit here is not a
                       # single run's bootstrap but the 60 (dataset seed, split seed)
                       # pairs, whose distribution is reported in run_aggregate.
GRID_BOOT = 0          # the ablation grid gets point estimates; inference is
                       # across seeds/splits in run_aggregate (a better unit anyway)


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def _dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)          # atomic: a crash never leaves half a JSON


def _load(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _say(msg: str) -> None:
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def ds_path(cfg: str, seed: int) -> str:
    return os.path.join(DATA, "ds_%s_s%02d_e%d.json" % (cfg, seed, EPISODES))


def ev_path(cfg: str, seed: int, sp: int) -> str:
    return os.path.join(OUT, "ev_%s_s%02d_sp%d.json" % (cfg, seed, sp))


# --------------------------------------------------------------------------- #
# task 1: dataset
# --------------------------------------------------------------------------- #
def ensure_dataset(cfg_name: str, seed: int) -> tuple:
    """Generate (or reuse) one dataset. Returns (path, seconds, was_generated)."""
    p = ds_path(cfg_name, seed)
    c = CONFIGS[cfg_name]
    if os.path.exists(p):
        ds = _load(p)
        m = ds.get("meta", {})
        ok = (m.get("decision_unit") == "plan"
              and int(m.get("H", -1)) == c["H"]
              and int(m.get("plan_len", -1)) == c["plan_len"]
              and abs(float(m.get("config", {}).get("proposal_sigma", -1)) - c["sigma"]) < 1e-12
              and int(m.get("K", -1)) == K)
        if ok:
            return p, 0.0, False
        _say("CACHE MISMATCH %s: regenerating" % os.path.basename(p))

    from run_pilot import make_factors
    import mujoco_adapter as MU

    t0 = time.time()
    rng = np.random.default_rng(seed)
    factors = [make_factors(rng, i) for i in range(EPISODES)]
    ids = ["s%02d_%03d" % (seed, i) for i in range(EPISODES)]
    scfg = S.SamplerConfig(K=K, H=c["H"], plan_len=c["plan_len"], T=T_STEPS,
                           seed=seed, proposal_sigma=c["sigma"],
                           control_substeps=50)
    adapter = MU.MuJoCoPush3D(substeps=scfg.control_substeps)
    ds = S.generate(adapter, factors, cfg=scfg, ids=ids)
    ds["meta"]["cache"] = {"scene_tag": SCENE_TAG, "episodes": EPISODES,
                           "dataset_seed": seed, "config_name": cfg_name}
    # a task property that belongs to the DATA, not to any split: computed once
    ds["meta"]["plan_content"] = S.horizon_collapse(ds, "C")
    _dump(ds, p)
    return p, time.time() - t0, True


# --------------------------------------------------------------------------- #
# task 2: evaluations on one (dataset, split)
# --------------------------------------------------------------------------- #
def predictor_set(seed: int) -> dict:
    """The full inventory, rebuilt per call because predictors are stateful."""
    out = {
        "B0 no-effect": (P.NoEffect(), None),
        "B1 bet-on-expert": (P.FactualWins(), None),
        "B2 ridge surrogate": (P.Ridge(seed=seed), None),
        "B3 ridge first-step only": (P.Ridge(seed=seed, plan_steps=1), None),
        "B4 ridge poly2": (P.RidgePoly(seed=seed), None),
        "C1 ridge arm-shuffled": (P.RidgeArmShuffled(seed=seed), None),
        "C2 plan-shuffled": (P.StructuredPlanShuffled(level=3, alpha=0.1, seed=seed),
                             None),
        "R1 one-step oracle": (P.Myopic(), None),
        # The three trivial rankers. Their presence is the single most important
        # defence in this suite: on the v1.3 candidate set the summed aligned push
        # alone scored DAR 0.897 against the one-step oracle's 0.869, so without
        # them a table showing "M1 > R1" would read as a result when it is not.
        "G0 perturbation magnitude": (P.HeuristicMagnitude(), None),
        "G0b summed aligned push": (P.HeuristicAlignedSum(), None),
        "G1 global direction": (P.GlobalDirection(), None),
        # R1 with a LEARNED model instead of the simulator: isolates the value of
        # simulator access from the value of seeing the plan.
        "L1 learned one-step (linear)": (P.LearnedOneStepLinear(seed=seed), "S"),
        "L2 learned one-step (frame)": (P.LearnedOneStepFrame(seed=seed), "S"),
    }
    for lvl in (0, 1, 2, 3, 4, 5, 6):
        for a in (0.1, 1.0, 10.0):
            out["M1 L%d a%g" % (lvl, a)] = (
                P.StructuredPlan(level=lvl, alpha=a, seed=seed), None)
    return out


def _sub_train(ids, frac: float) -> list:
    ordered = sorted(ids)
    n = max(1, int(round(frac * len(ordered))))
    return ordered[:n]


def run_evals(cfg_name: str, seed: int, split_seed: int) -> dict:
    p = ds_path(cfg_name, seed)
    ds = _load(p)
    rows = SP.assign(ds["episodes"], seed=split_seed)
    leak = SP.check_no_leak(rows)
    ids = {"train": [], "val": [], "test": []}
    for r in rows:
        ids[r["split"]].append(r["episode_id"])
    # val is required too: the method's hyperparameters are chosen on val, so a split
    # without one cannot support the protocol even though its test rows exist.
    if not ids["train"] or not ids["val"] or not ids["test"]:
        return {"schema": EVAL_SCHEMA, "config": cfg_name, "dataset_seed": seed,
                "split_seed": split_seed, "skipped": "empty split",
                "n": {k: len(v) for k, v in ids.items()}, "leak_ok": bool(leak["ok"])}

    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])
    out = {"schema": EVAL_SCHEMA, "config": cfg_name, "dataset_seed": seed,
           "split_seed": split_seed, "leak_ok": bool(leak["ok"]),
           "n": {k: len(v) for k, v in ids.items()},
           "phi_ref": ds["meta"]["phi_ref"],
           "n_episodes": len(rows), "scene_tag": SCENE_TAG,
           "rows": {}, "extras": {}}

    out["val_rows"] = {}
    for conv in ("C", "O"):
        out["rows"][conv] = {}
        for label, (pred, tc) in predictor_set(seed).items():
            head = (cfg_name == "m3" and conv == "C" and label in HEADLINES)
            res = P.evaluate_predictor(
                pred, ds, convention=conv, train_ids=ids["train"],
                test_ids=ids["test"], spec=spec, eps=eps,
                n_boot=HEAD_BOOT if head else GRID_BOOT, train_convention=tc)
            row = R.summarize(res["report"], res["name"])
            row.update(res["info"])
            row["kind"] = getattr(pred, "kind", "baseline")
            out["rows"][conv][label] = row

        # VAL rows: the selection set. Point estimates only (no bootstrap) because
        # selection needs an ordering, not an interval; the interval belongs to the
        # test number it produces. Same predictors, same fit, different held-out set.
        if conv == "C":
            out["val_rows"] = {}
            for label, (pred, tc) in predictor_set(seed).items():
                res = P.evaluate_predictor(
                    pred, ds, convention=conv, train_ids=ids["train"],
                    test_ids=ids["val"], spec=spec, eps=eps, n_boot=0,
                    train_convention=tc)
                row = R.summarize(res["report"], res["name"])
                row.update(res["info"])
                row["kind"] = getattr(pred, "kind", "baseline")
                out["val_rows"][label] = row

        # -- extra 1: supervision transfer (fit on C, score on O) ------------- #
        if conv == "O":
            res = P.evaluate_predictor(
                P.StructuredPlan(level=3, alpha=0.1, seed=seed), ds, convention="O",
                train_ids=ids["train"], test_ids=ids["test"], spec=spec, eps=eps,
                n_boot=0, train_convention="C")
            row = R.summarize(res["report"], "M1 L3 a0.1 (trained on C)")
            row.update(res["info"])
            out["extras"]["transfer_C_to_O"] = row

        # -- extra 2: training-size curve ------------------------------------- #
        if conv == "C":
            for frac in (0.10, 0.25, 0.50):
                sub = _sub_train(ids["train"], frac)
                for label, factory in (
                        ("M1 L3 a0.1", lambda: P.StructuredPlan(level=3, alpha=0.1,
                                                                seed=seed)),
                        ("B2 ridge surrogate", lambda: P.Ridge(seed=seed))):
                    res = P.evaluate_predictor(
                        factory(), ds, convention="C", train_ids=sub,
                        test_ids=ids["test"], spec=spec, eps=eps, n_boot=0)
                    row = R.summarize(res["report"],
                                      "%s @%d%% train" % (label, int(100 * frac)))
                    row.update(res["info"])
                    out["extras"]["curve_%.2f_%s" % (frac, label)] = row
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="full experiment matrix, resumable")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DATASET_SEEDS))
    ap.add_argument("--split-seeds", nargs="*", type=int, default=list(SPLIT_SEEDS))
    ap.add_argument("--budget-minutes", type=float, default=0.0,
                    help="0 = no limit; otherwise stop cleanly between tasks")
    ap.add_argument("--force", action="store_true", help="ignore existing caches")
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    _say("suite start: %d configs x %d seeds x %d splits | episodes=%d T=%d K=%d"
         % (len(args.configs), len(args.seeds), len(args.split_seeds),
            EPISODES, T_STEPS, K))

    t_start = time.time()
    n_done = n_gen = n_skip = n_err = 0
    for cfg in args.configs:
        if cfg not in CONFIGS:
            _say("unknown config %r" % cfg)
            return 2
        for seed in args.seeds:
            if args.budget_minutes and \
                    (time.time() - t_start) / 60.0 > args.budget_minutes:
                _say("budget reached after %d evaluations, %d datasets"
                     % (n_done, n_gen))
                return 0
            try:
                p, secs, gen = ensure_dataset(cfg, seed)
                n_gen += int(gen)
                _say("dataset %-5s s%02d %s %.1fs %s"
                     % (cfg, seed, "GEN " if gen else "cached", secs,
                        os.path.basename(p)))
            except Exception:                                   # noqa: BLE001
                n_err += 1
                _say("DATASET ERROR %s s%02d\n%s" % (cfg, seed, traceback.format_exc()))
                continue

            for sp in args.split_seeds:
                ep = ev_path(cfg, seed, sp)
                if os.path.exists(ep) and not args.force:
                    try:
                        if _load(ep).get("schema") == EVAL_SCHEMA:
                            n_skip += 1
                            continue
                    except Exception:                           # noqa: BLE001
                        pass
                try:
                    t0 = time.time()
                    obj = run_evals(cfg, seed, sp)
                    _dump(obj, ep)
                    n_done += 1
                    _say("evals   %-5s s%02d sp%d %.1fs n=%s"
                         % (cfg, seed, sp, time.time() - t0,
                            obj.get("n", "?")))
                except Exception:                               # noqa: BLE001
                    n_err += 1
                    _say("EVAL ERROR %s s%02d sp%d\n%s"
                         % (cfg, seed, sp, traceback.format_exc()))
    _say("suite done: %d evaluations written, %d skipped (cached), %d datasets "
         "generated, %d errors, %.1f min"
         % (n_done, n_skip, n_gen, n_err, (time.time() - t_start) / 60.0))
    return 0 if n_err == 0 else 1


if __name__ == "__main__":      # required: this module must never run on import
    raise SystemExit(main())
