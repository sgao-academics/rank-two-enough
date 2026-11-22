"""
run_pilot.py -- generate a pilot counterfactual dataset and validate the ladder.

This still does NOT evaluate a method; there is no method yet. What it does is
place every number that a future method will be compared against, so that when the
method arrives the comparison already has a floor, a ceiling and a control:

    B0 no-effect      -> DAR must be 0        (no directional information)
    B1 bet-on-expert  -> the "copy the scripted expert" reference
    B2 myopic         -> one-step information only
    B3 ridge          -> linear surrogate, no interactions
    C1 ridge-shuffled -> MUST land at chance; if not, the pipeline invents skill
    REF oracle        -> DAR 1, regret 0      (ceiling, not a method)
    REF chance        -> permutation mean     (measured, never analytic)

All predictors are fit on the training split and scored on the evaluation split
(test by default). Nothing in `predictors.py` can see a label at predict time, so
the split discipline is enforced by the type system rather than by care.

    python run_pilot.py --episodes 48 --T 15
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as M          # noqa: E402
import predictors as P       # noqa: E402
import report as R           # noqa: E402
import sampler as S          # noqa: E402
import split as SP           # noqa: E402

LOGS = os.path.abspath(os.path.join(HERE, os.pardir, "logs"))
DATA = os.path.abspath(os.path.join(HERE, os.pardir, "data"))

TARGET_QUADRANTS = [(0.13, 0.13), (-0.13, 0.13), (0.13, -0.11), (-0.13, -0.11)]
DISTRACTOR_QUADRANTS = [(0.10, -0.11), (-0.10, -0.11), (0.10, 0.12), (-0.10, 0.12)]


def make_factors(rng: np.random.Generator, i: int) -> dict:
    tq = TARGET_QUADRANTS[i % len(TARGET_QUADRANTS)]
    dq = DISTRACTOR_QUADRANTS[(i // len(TARGET_QUADRANTS)) % len(DISTRACTOR_QUADRANTS)]
    jitter = lambda s: float(rng.uniform(-s, s))  # noqa: E731
    return {
        "mass": float(rng.uniform(0.030, 0.080)),
        "half_size": float(rng.uniform(0.016, 0.024)),
        "friction": float(rng.uniform(0.60, 1.10)),
        "target_xy": (tq[0] + jitter(0.01), tq[1] + jitter(0.01)),
        "distractor_xy": (dq[0] + jitter(0.01), dq[1] + jitter(0.01)),
    }


def ids_by_split(dataset: dict) -> dict:
    out = {"train": [], "val": [], "test": []}
    for ep in dataset["episodes"]:
        out[ep["split"]].append(ep["episode_id"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pilot counterfactual dataset + baseline ladder")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--plan-len", type=int, default=3,
                    help="decision unit length in control steps (must be < H). "
                         "--plan-len 1 is the v1.2 ablation and must show the collapse.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", default="R-samp", choices=["R-samp", "R-top", "R-perm"])
    ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.8, help="candidate dispersion")
    ap.add_argument("--substeps", type=int, default=50,
                    help="physics substeps per control step (50 = 10 Hz at 2 ms)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--eval-split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    spec = M.load_spec()
    if args.K != spec["candidate_set"]["K_primary"] or args.H != spec["outcome"]["primary_horizon"]:
        raise SystemExit("K/H must match the frozen spec (K=%d, H=%d)"
                         % (spec["candidate_set"]["K_primary"], spec["outcome"]["primary_horizon"]))
    if not (1 <= args.plan_len < args.H):
        raise SystemExit("plan_len must satisfy 1 <= plan_len < H=%d (got %d): at "
                         "plan_len == H the C/O gap is a constant zero"
                         % (args.H, args.plan_len))
    if args.plan_len != spec["decision_unit"]["plan_len"]:
        print("!! plan_len=%d differs from the frozen spec (%d): this run is an "
              "ABLATION, not the primary configuration"
              % (args.plan_len, spec["decision_unit"]["plan_len"]))

    import mujoco_adapter as MU

    cfg = S.SamplerConfig(K=args.K, H=args.H, plan_len=args.plan_len, T=args.T,
                          rule=args.rule, n_cand=args.n_cand, seed=args.seed,
                          proposal_sigma=args.sigma,
                          min_spread=float(spec["decision_unit"]["min_spread"]),
                          plan_alt_frac=float(spec["decision_unit"]["alt_frac"]),
                          control_substeps=args.substeps)
    adapter = MU.MuJoCoPush3D(substeps=cfg.control_substeps)

    rng = np.random.default_rng(args.seed)
    factors = [make_factors(rng, i) for i in range(args.episodes)]
    ids = ["pilot_%03d" % i for i in range(args.episodes)]

    print("=" * 78)
    print("pilot: %d episodes x %d decision points, K=%d H=%d rule=%s"
          % (args.episodes, args.T, args.K, args.H, args.rule))
    print("decision unit: %d-step plan, tail %d steps (C re-plans, O replays)"
          % (cfg.plan_len, cfg.H - cfg.plan_len))
    print("adapter:", adapter.name, "| mujoco", adapter.fingerprint()["mujoco_version"])
    print("=" * 78)

    dataset = S.generate(adapter, factors, cfg=cfg, ids=ids)
    dataset["episodes"] = SP.assign(dataset["episodes"], seed=args.split_seed)
    leak = SP.check_no_leak(dataset["episodes"])
    dataset["meta"]["split"] = {"seed": args.split_seed, "on": "template", "report": leak}

    split_ids = ids_by_split(dataset)
    train_ids, eval_ids = split_ids["train"], split_ids[args.eval_split]

    print("phi_ref = %.6f   obs_dim = %d   degenerate(eps=%.0e)"
          % (dataset["meta"]["phi_ref"], len(S.OBS_NAMES), cfg.eps_degen))
    print("arm0 replay  : max |arm0 - factual| = %.2e in potential (%.2e in Y) "
          "over %d decisions"
          % (dataset["meta"]["arm0_max_dev"],
             dataset["meta"]["arm0_max_dev"] / dataset["meta"]["phi_ref"],
             S.n_steps_of(dataset)))
    print("split  :", leak["n_episodes"], "| templates t/v/te =",
          leak["n_train_templates"], leak["n_val_templates"], leak["n_test_templates"])
    print("leak   :", "OK" if leak["ok"] else "FAIL %r" % leak)
    print("fit on %d train episodes, score on %d %s episodes"
          % (len(train_ids), len(eval_ids), args.eval_split))
    if not train_ids or not eval_ids:
        raise SystemExit("empty split: %r" % {k: len(v) for k, v in split_ids.items()})

    # Task property, not a method metric: how much of the decision's ranking the
    # first step already decides. If this is high, a DAR gain does NOT license a
    # claim about multi-step reasoning. Compare it across --plan-len 1 vs 3; that
    # comparison, not any single DAR, is the evidence that the plan unit has content.
    hz = S.horizon_collapse(dataset, "C")
    print("plan content : 1-step vs %d-step rank rho=%.3f | identical ranking %.1f%% | "
          "same argmax %.1f%%  (n=%d steps)"
          % (cfg.H, hz["rho_mean"], 100 * hz["identical_ranking_rate"],
             100 * hz["argmax_agreement"], hz["n_steps"]))

    eps = float(spec["noise_floor"]["deterministic_fallback"])   # Gate-0 says DETERMINISTIC
    n_boot = int(spec["estimation"]["bootstrap_n"])
    per_conv = {}

    for conv in ("C", "O"):
        # -- truth-only reference table for the evaluation split -------------- #
        tab_eval = M.build_table(
            S.records_from(dataset, convention=conv, episode_ids=eval_ids),
            k_arms=args.K)
        ruler = M.ruler_selftest(tab_eval, spec)

        oracle_rows = S.records_from(
            dataset, convention=conv, episode_ids=eval_ids,
            y_pred=M.oracle_predictions(tab_eval))
        rep_oracle = M.evaluate(M.build_table(oracle_rows, k_arms=args.K),
                                spec, eps=eps, n_boot=200)

        # -- the ladder, all through the one interface ------------------------ #
        est = []
        for name, pred in P.make_ladder(seed=args.seed).items():
            res = P.evaluate_predictor(pred, dataset, convention=conv,
                                       train_ids=train_ids, test_ids=eval_ids,
                                       spec=spec, eps=eps, n_boot=n_boot)
            row = R.summarize(res["report"], res["name"])
            row.update(res["info"])
            row["kind"] = getattr(pred, "kind", "baseline")
            est.append(row)

        per_conv[conv] = {
            "table": repr(tab_eval),
            "ruler": ruler,
            "oracle": R.summarize(rep_oracle, "REF oracle (ceiling, not a method)"),
            "estimates": est,
        }

        # Four kinds, kept apart because mixing them is how "a trivial heuristic
        # already wins" gets written down as a result: methods here, observation-only
        # baselines they must beat next, deliberate breakages that must land at
        # chance after that, and simulator/truth references last.
        methods = [e for e in est if e["kind"] == "method"]
        baselines = [e for e in est if e["kind"] == "baseline"]
        controls = [e for e in est if e["kind"] == "control"]
        refs = [e for e in est if e["kind"] == "reference"]

        print("\n--- convention %s (%s) ---" % (
            conv, "closed loop [primary]" if conv == "C" else "open loop [secondary]"))
        print("  ", per_conv[conv]["table"])
        print("   method:")
        print(R.table(methods))
        print("   baselines (observation only -- what a method must beat):")
        print(R.table(baselines))
        print("   negative controls (must sit at chance):")
        print(R.table(controls))
        o = per_conv[conv]["oracle"]
        print("   references (need the simulator or the truth -- not methods):")
        for r in refs:
            print("   %-30s DAR=%6.4f  nMAE=%7.4f  regret=%7.4f  rho=%+.4f"
                  % (r["label"][:30], r["dar"], r["nmae"], r["regret_norm"], r["rho"]))
        print("   %-30s DAR=%6.4f  nMAE=%7.4f  regret=%7.4f"
              % (o["label"][:30], o["dar"], o["nmae"], o["regret_norm"]))
        print("   %-30s DAR=%.4f +- %.4f over %d permutations (p+=%.3f)"
              % ("REF chance (permuted truth)",
                 ruler["chance"]["dar_perm_mean"], ruler["chance"]["dar_perm_sd"],
                 ruler["chance"]["permutations"], ruler["chance"]["p_positive"]))

    # ------------------------------------------------------------------ #
    # verdict
    # ------------------------------------------------------------------ #
    def find(conv, name):
        for e in per_conv[conv]["estimates"]:
            if e["label"].startswith(name):
                return e
        raise KeyError("no estimate named %r in convention %s" % (name, conv))

    def num(conv, name, field):
        v = find(conv, name).get(field)
        return float("nan") if v is None else float(v)

    chance_mean = float(per_conv["C"]["ruler"]["chance"]["dar_perm_mean"])
    p_pos = float(per_conv["C"]["ruler"]["chance"]["p_positive"])
    ctl_dar = num("C", "C1", "dar")
    ctl2_dar = num("C", "C2", "dar")
    m1_dar, m1_reg = num("C", "M1", "dar"), num("C", "M1", "regret_norm")
    r1_dar, r1_reg = num("C", "R1", "dar"), num("C", "R1", "regret_norm")
    b1_dar = num("C", "B1", "dar")

    checks = {
        "oracle DAR == 1 on eval split": abs(per_conv["C"]["oracle"]["dar"] - 1.0) < 1e-9,
        "oracle regret == 0 on eval split":
            abs(per_conv["C"]["oracle"]["regret_norm"]) < 1e-9,
        "no-effect DAR == 0": abs(num("C", "B0", "dar")) < 1e-12,
        "chance is measured, not assumed":
            0.25 <= chance_mean <= 0.75 and abs(per_conv["C"]["ruler"]["chance"]["rho"]) <= 0.15,
        # the negative controls: a model fit on destroyed correspondence must not score
        "negative control sits at chance": abs(ctl_dar - chance_mean) <= 0.15,
        "method's negative control sits at chance": abs(ctl2_dar - chance_mean) <= 0.15,
        "signs are mixed (a direction metric needs both)": 0.05 < p_pos < 0.95,
        "split has no leak": bool(leak["ok"]),
        "train and eval splits are disjoint": not (set(train_ids) & set(eval_ids)),
        "both conventions generated":
            per_conv["C"]["oracle"]["n_steps"] > 0 and per_conv["O"]["oracle"]["n_steps"] > 0,
        "degenerate ratio below no-signal threshold":
            per_conv["C"]["oracle"]["degenerate_rate"]
            < float(spec["degenerate"]["no_signal_threshold"]),
        "Gate-2 ruler passed": all(bool(per_conv[c]["ruler"]["passed"]) for c in ("C", "O")),
        # Needed condition, not a sufficiency claim: if the first step's ordering is
        # identical to the plan's on EVERY step, the plan unit carries no multi-step
        # content and nothing downstream can demonstrate any. Vacuous at plan_len=1,
        # where the two are the same ordering by construction, so skip it there.
        "plan unit: 1-step ordering is not identical on every step":
            cfg.plan_len == 1 or hz["identical_ranking_rate"] < 1.0,
        # Delta is measured relative to arm 0, so "arm 0 is the policy's own
        # sequence" has to hold to well below the Delta scale. Measured, not assumed:
        # 6.2e-7 worst case from a 4e-14 action difference amplified through contact.
        "arm 0 replays to the factual trajectory within tolerance":
            float(dataset["meta"]["arm0_max_dev"]) <= 1e-5,
    }

    # The one-step oracle must BE the H=1 truth, not merely correlated with it. This
    # is what makes "R1" a reference rather than a baseline, and it is checkable
    # exactly rather than rhetorically.
    ft_eval = P.features_from(dataset, eval_ids)
    y_s = P.truth_matrix(dataset, "S", eval_ids)
    mp = ft_eval.myopic / ft_eval.phi_ref
    gap = float(np.max(np.abs((mp - mp[:, [0]]) - (y_s - y_s[:, [0]]))))
    # Two independent replays of one step, so the floor is the Gate-0 tolerance.
    checks["one-step oracle delta == 1-step truth delta"] = gap < 1e-9
    print("   identity check   : max |R1 delta - y_S delta| = %.2e  (Gate-0 tol 1e-9)" % gap)

    # The comparison the project exists to make. Printed, deliberately NOT turned
    # into a pass/fail check: a method that loses here is a scientific result about
    # the method, not a broken pipeline, and the pipeline's verdict must not depend
    # on which way it goes.
    print("\n=== method vs the lines it must clear (convention C) ===")
    print("  B1 bet-on-expert   DAR=%.4f         (floor: do nothing but copy the expert)" % b1_dar)
    print("  R1 one-step oracle DAR=%.4f regret=%.4f (target: perfect one-step model)"
          % (r1_dar, r1_reg))
    print("  M1 structured plan DAR=%.4f regret=%.4f" % (m1_dar, m1_reg))
    print("  -> M1 - R1 = %+.4f DAR, %+.4f regret" % (m1_dar - r1_dar, m1_reg - r1_reg))
    if cfg.plan_len == 1:
        print("  !! plan_len=1: M1 and R1 see the SAME decision (one step) -- any gap "
              "here is model skill, not multi-step content")

    print("\n=== pipeline checks ===")
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))

    payload = R.pack_run(
        label="pilot %dep T%d m%d H%d %s" % (args.episodes, args.T, cfg.plan_len,
                                             cfg.H, args.rule),
        spec=spec, adapter=adapter, dataset=dataset,
        split={"seed": args.split_seed, "on": "template", "report": leak,
               "n_train": len(train_ids), "n_val": len(split_ids["val"]),
               "n_eval": len(eval_ids)},
        estimates=[dict(conv=c, **e) for c in ("C", "O") for e in per_conv[c]["estimates"]],
        checks=checks, config=cfg.as_dict(), eval_split=args.eval_split,
        extra={"per_convention": {c: {"table": per_conv[c]["table"],
                                      "ruler": R.summarize_ruler(per_conv[c]["ruler"]),
                                      "oracle": per_conv[c]["oracle"]}
                                  for c in ("C", "O")},
               "task_properties": {"plan_content_C": hz,
                                   "horizon_collapse_C": hz,
                                   "one_step_oracle_identity_max_abs": gap,
                                   "decision_unit": "plan",
                                   "plan_len": cfg.plan_len,
                                   "plan_tail": cfg.H - cfg.plan_len}})

    # The filename MUST carry every setting that changes the numbers. It did not:
    # the m=1 ablation wrote to the same path as the m=3 run and silently overwrote
    # it, so the primary configuration's payload was gone while its log remained --
    # the exact shape of "a number nobody can trace". plan_len and H go in the name.
    tag = "%s_m%d_H%d" % (args.rule.lower().replace("-", ""), cfg.plan_len, cfg.H)
    out = args.out or os.path.join(
        LOGS, "pilot_%s_%dep_%s.json" % (tag, args.episodes, args.eval_split))
    print("\nwritten:", R.dump(payload, out))
    dpath = S.save(dataset, os.path.join(
        DATA, "pilot_%s_%dep.json" % (tag, args.episodes)))
    print("dataset:", dpath)

    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
