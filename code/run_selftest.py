"""
run_selftest.py -- acceptance test for metrics.py and gate0_replay.py.

Nothing downstream may trust these two modules until this script passes. It
checks the ruler itself (Gate-2), the estimator's basic identities, and the
Gate-0 classifier on three mock simulators with known ground truth.

    python run_selftest.py          # exit 0 == all checks passed
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate0_replay as g0          # noqa: E402
import metrics as M                # noqa: E402

RESULTS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print("  [%s] %-46s %s" % ("PASS" if cond else "FAIL", name, detail))


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def make_records(n_ep=12, steps_per_ep=30, k=5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for e in range(n_ep):
        ep = "ep_%02d" % e
        for t in range(steps_per_ep):
            base = rng.uniform(0.2, 0.8)
            effects = rng.uniform(-0.3, 0.3, k)      # arm 0 keeps the factual level
            effects[0] = 0.0
            y_true = np.clip(base + effects, 0.0, 1.0)
            y_pred = np.clip(y_true + rng.normal(0.0, noise, k), 0.0, 1.0) if noise else y_true.copy()
            for j in range(k):
                recs.append({"episode": ep, "t": t, "k": j,
                             "y_true": float(y_true[j]),
                             "y_pred": float(y_pred[j]),
                             "degenerate": False})
    return recs


def records_with(table_pred: np.ndarray, table_true: np.ndarray, degen=None):
    n_steps, k = table_true.shape
    recs = []
    for i in range(n_steps):
        ep = "ep_%02d" % (i // 30)
        t = i % 30
        for j in range(k):
            recs.append({"episode": ep, "t": t, "k": j,
                         "y_true": float(table_true[i, j]),
                         "y_pred": float(table_pred[i, j]),
                         "degenerate": bool(degen[i]) if degen is not None else False})
    return recs


# --------------------------------------------------------------------------- #
# A. Gate-2: the ruler proves itself
# --------------------------------------------------------------------------- #
def part_a(spec):
    print("\nA. Gate-2 ruler self-test (spec gate 2)")
    tab = M.build_table(make_records(noise=0.0), k_arms=spec["candidate_set"]["K_primary"])
    res = M.ruler_selftest(tab, spec)
    check("oracle -> DAR = 1.0", abs(res["oracle"]["dar"] - 1.0) < 1e-9,
          "dar=%.6f" % res["oracle"]["dar"])
    check("oracle -> nMAE = 0", abs(res["oracle"]["nmae"]) < 1e-9,
          "nmae=%.2e" % res["oracle"]["nmae"])
    check("oracle -> regret_norm = 0", abs(res["oracle"]["regret_norm"]) < 1e-9,
          "regret=%.2e" % res["oracle"]["regret_norm"])
    check("chance -> DAR in [0.40, 0.60]", 0.40 <= res["chance"]["dar"] <= 0.60,
          "dar=%.4f" % res["chance"]["dar"])
    check("chance -> |rho| <= 0.10", abs(res["chance"]["rho"]) <= 0.10,
          "rho=%.4f" % res["chance"]["rho"])
    check("chance -> regret_norm > 0", res["chance"]["regret_norm"] > 1e-6,
          "regret=%.4f" % res["chance"]["regret_norm"])
    check("Gate-2 overall", res["passed"])


# --------------------------------------------------------------------------- #
# B. estimator identities
# --------------------------------------------------------------------------- #
def part_b(spec, tab_base):
    print("\nB. estimator identities")
    yt, k = tab_base.y_true, tab_base.k
    eps = 1e-6

    off = M.build_table(records_with(yt + 0.1, yt), k_arms=k)
    r = M._core(off.y_true, off.y_pred, off.degen, eps, eps)
    check("constant offset cancels in Delta", abs(r["dar"] - 1.0) < 1e-9 and abs(r["mae"]) < 1e-9,
          "dar=%.4f mae=%.2e" % (r["dar"], r["mae"]))

    dbl = M.build_table(records_with(2.0 * yt, yt), k_arms=k)
    r = M._core(dbl.y_true, dbl.y_pred, dbl.degen, eps, eps)
    check("doubling -> DAR 1, nMAE ~ 1", abs(r["dar"] - 1.0) < 1e-9 and 0.7 < r["nmae"] < 1.3,
          "dar=%.4f nmae=%.3f" % (r["dar"], r["nmae"]))

    flat = np.repeat(yt[:, [0]], k, axis=1)
    nf = M.build_table(records_with(flat, yt), k_arms=k)
    r = M._core(nf.y_true, nf.y_pred, nf.degen, eps, eps)
    check("'no effect' prediction -> DAR = 0", abs(r["dar"]) < 1e-12, "dar=%.4f" % r["dar"])

    r = M._core(nf.y_true, nf.y_pred, nf.degen, 10.0, eps)
    check("eps above all effects -> undecidable 1.0", abs(r["undecidable_rate"] - 1.0) < 1e-12
          and np.isnan(r["dar"]), "undecidable=%.2f dar=nan" % r["undecidable_rate"])

    degen_all = np.ones(yt.shape[0], dtype=bool)
    rep = M.evaluate(M.build_table(records_with(yt, yt, degen_all), k_arms=k),
                     spec, eps=eps, n_boot=500)
    check("all-degenerate -> primary DAR is null",
          rep["metrics"]["dar"]["value"] is None,
          "value=%r" % rep["metrics"]["dar"]["value"])
    check("all-degenerate -> inclusive variant still computed",
          rep["metrics_including_degenerate"]["dar"]["value"] is not None,
          "value=%.3f" % (rep["metrics_including_degenerate"]["dar"]["value"] or 0))

    rep = M.evaluate(tab_base, spec, eps=eps, n_boot=500)
    ok_vec = True
    for key in ("dar", "nmae", "regret_norm"):
        cell = rep["metrics"][key]
        if cell.get("ci95") and not (cell["ci95"][0] - 1e-12 <= cell["value"] <= cell["ci95"][1] + 1e-12):
            ok_vec = False
    check("point estimate inside its own 95% CI", ok_vec)
    check("report carries spec_version + cluster unit",
          rep["spec_version"] == M.SPEC_VERSION and rep["cluster_unit"] == "episode",
          "N=%d episodes, %d steps" % (rep["n_episodes"], rep["n_steps"]))


# --------------------------------------------------------------------------- #
# C. diagnosis / recovery / locality
# --------------------------------------------------------------------------- #
def part_c():
    print("\nC. diagnosis, recovery, intervention locality")
    pred = [[("cup", "mass"), ("cup", "friction")],
            [("bowl", "position"), ("cup", "mass")],
            [("cup", "mass")]]
    truth = [[("cup", "mass")],
             [("cup", "mass")],
             [("cup", "friction")]]
    d = M.diagnosis_metrics(pred, truth)
    # case 1: top-1 correct; case 2: correct variable ranked 2nd; case 3: missed
    check("Hit@1 = 1/3", abs(d["hit_at_1"] - 1 / 3) < 1e-9, "hit@1=%.4f" % d["hit_at_1"])
    check("MRR = (1 + 1/2 + 0)/3 = 0.5", abs(d["mrr"] - 0.5) < 1e-9, "mrr=%.4f" % d["mrr"])

    rng = np.random.default_rng(0)
    ours = rng.random(200) < 0.55
    rand = rng.random(200) < 0.35
    orac = rng.random(200) < 0.90
    rec = M.recovery_rates(ours, rand, orac, n_boot=2000)
    check("recovery reports all three arms",
          {"ours", "random_retry", "oracle"} <= set(rec),
          "ours=%.2f rand=%.2f oracle=%.2f" % (rec["ours"], rec["random_retry"], rec["oracle"]))
    check("delta_vs_random CI excludes 0",
          rec["delta_vs_random_ci95"][0] > 0, "CI=[%.3f, %.3f]" % tuple(rec["delta_vs_random_ci95"]))

    good = M.intervention_locality(natural_std=1.0,
                                   delta_after_intervention=np.array([0.9, 0.004, 0.002]),
                                   neighbors=np.array([True, False, False]))
    bad = M.intervention_locality(natural_std=1.0,
                                  delta_after_intervention=np.array([0.9, 0.5, 0.002]),
                                  neighbors=np.array([True, False, False]))
    check("Gate-1 accepts local intervention", good["passed"],
          "max_non_nb=%.4f tol=%.3f" % (good["max_non_neighbor_change"], good["tol"]))
    check("Gate-1 rejects leaking intervention", not bad["passed"],
          "max_non_nb=%.4f" % bad["max_non_neighbor_change"])


# --------------------------------------------------------------------------- #
# C2. rank statistics, including the tie case that broke the hand-rolled tau-b
# --------------------------------------------------------------------------- #
def part_c2():
    print("\nC2. rank statistics (scipy-backed)")
    import scipy
    from scipy.stats import rankdata
    check("scipy importable", scipy.__version__ >= "1.10", "scipy=%s" % scipy.__version__)

    r = rankdata(np.array([[1.0, 1.0, 2.0, 3.0]]), axis=1)[0]
    check("average ranks on ties -> [1.5,1.5,3,4]",
          np.allclose(r, [1.5, 1.5, 3.0, 4.0]), "ranks=%s" % np.round(r, 3).tolist())

    # identical input with ties: tau-b must be exactly 1, not merely near 1
    a = np.array([[1.0, 1.0, 2.0, 3.0, 3.0]])
    tau = M.kendall_per_step(a, a)[0]
    check("tau-b on tied identical rows = 1.0", abs(tau - 1.0) < 1e-12, "tau=%.6f" % tau)

    # exact reversal: tau-b must be -1
    b = np.array([[3.0, 3.0, 2.0, 1.0, 1.0]])
    tau_rev = M.kendall_per_step(a, b)[0]
    check("tau-b on tied reversed rows = -1.0", abs(tau_rev + 1.0) < 1e-12, "tau=%.6f" % tau_rev)

    # the regression that motivated the switch: |tau| must never leave [-1, 1]
    rng = np.random.default_rng(7)
    tied_true = rng.integers(1, 6, size=(200, 4)).astype(float)
    tied_pred = rng.integers(1, 6, size=(200, 4)).astype(float)
    tau_tied = M.kendall_per_step(tied_true, tied_pred)
    tau_tied = tau_tied[np.isfinite(tau_tied)]
    check("tau-b on 200 tied steps stays in [-1,1]",
          bool(np.all(np.abs(tau_tied) <= 1.0 + 1e-12)),
          "max|tau|=%.4f (hand-rolled version reached |tau|>4)" % np.abs(tau_tied).max())

    rho_tied = M.spearman_per_step(tied_true, tied_pred)
    rho_tied = rho_tied[np.isfinite(rho_tied)]
    check("rho on 200 tied steps stays in [-1,1]",
          bool(np.all(np.abs(rho_tied) <= 1.0 + 1e-12)), "max|rho|=%.4f" % np.abs(rho_tied).max())


# --------------------------------------------------------------------------- #
# D. Gate-0 on mocks with known behaviour
# --------------------------------------------------------------------------- #
def part_d():
    print("\nD. Gate-0 replay determinism (mock simulators)")
    cases = [("mock:det", g0.DeterministicMock, "DETERMINISTIC", True),
             ("mock:noisy", g0.NoisyMock, "STABLE_STOCHASTIC", True),
             ("mock:drift", g0.DriftingMock, "DRIFTING", False)]
    for name, cls, expect, expect_pass in cases:
        res = g0.run_gate(cls(), seed=0, repeats=g0.SPEC_REPEATS, steps=15)
        check("%s -> %s" % (name, expect), res.classification == expect and res.passed == expect_pass,
              "got %s max=%.2e slope=%.2e" % (res.classification, res.max_abs_diff, res.drift_slope))

    nf = g0.estimate_noise_floor(g0.DeterministicMock(), n_units=12, seed=0)
    check("noise floor on deterministic sim is ~0", nf["eps"] < 1e-9, "eps=%.2e" % nf["eps"])
    nf2 = g0.estimate_noise_floor(g0.NoisyMock(sigma=1e-3), n_units=12, seed=0)
    check("noise floor on noisy sim is > 0", nf2["eps"] > 0, "eps=%.2e" % nf2["eps"])


# --------------------------------------------------------------------------- #
# E. real engine (skipped cleanly when the engine is absent)
# --------------------------------------------------------------------------- #
def part_e():
    print("\nE. real engine adapter (skipped if mujoco is absent)")
    try:
        import mujoco  # noqa: F401
    except ModuleNotFoundError:
        check("mujoco present", True, "SKIPPED - mocks only on this machine")
        return

    import mujoco_adapter as MU

    a = MU.MuJoCoPush3D()
    p = a.poses()
    check("adapter builds and returns (4,7) poses", p.shape == (4, 7), "shape=%s" % (p.shape,))
    y0 = a.outcome()
    check("outcome in (0,1] at reset", 0.0 < y0 <= 1.0, "y=%.4f" % y0)

    # non-saturation probe: this is the bug that would silently turn a noisy sim
    # into a "deterministic" one. outcome must STRICTLY decrease as the target
    # moves away, at every scale we operate at -- including the far end.
    ys = []
    for d in (0.05, 0.10, 0.20, 0.40):
        a.set_factors(target_xy=(-d, d))
        a.reset(0)
        ys.append(a.outcome())
    strictly_dec = all(ys[i] > ys[i + 1] for i in range(len(ys) - 1))
    check("outcome strictly decreases with distance (no plateau)",
          strictly_dec, "->".join("%.4f" % y for y in ys))

    # two independent model instances, same script => bit-identical poses
    rng = np.random.default_rng(3)
    script = rng.normal(0.0, 1.0, (10, 3))
    a1, a2 = MU.MuJoCoPush3D(), MU.MuJoCoPush3D()
    a1.reset(0)
    a2.reset(0)
    for act in script:
        a1.step(act)
        a2.step(act)
    dmax = float(np.max(np.abs(a1.poses() - a2.poses())))
    check("two instances agree bit-for-bit", dmax == 0.0, "max|diff|=%.2e" % dmax)

    res = g0.run_gate(MU.MuJoCoPush3D(), seed=0, repeats=5, steps=8)
    check("real engine classified DETERMINISTIC",
          res.classification == "DETERMINISTIC" and res.passed,
          "got %s max=%.2e" % (res.classification, res.max_abs_diff))

    # the blocked adapter must fail with a message, not silently misbehave
    try:
        import causalworld_adapter as CW
        try:
            CW.CausalWorldAdapter("push")
            cw_ok = False
        except RuntimeError as exc:
            cw_ok = "BLOCKED" in str(exc)
        except ModuleNotFoundError:
            cw_ok = True
        check("causalworld adapter refuses loudly when blocked", cw_ok,
              "factory entries=%d" % len(CW.FACTORY))
    except Exception as exc:  # noqa: BLE001
        check("causalworld adapter importable", False, type(exc).__name__)


# --------------------------------------------------------------------------- #
# F. data pipeline: the generator must be reproducible and structurally sound
# --------------------------------------------------------------------------- #
def part_f():
    print("\nF. data pipeline (sampler + split)")
    try:
        import mujoco  # noqa: F401
        import mujoco_adapter as MU
        import sampler as SA
        import split as SP
    except ModuleNotFoundError as exc:
        check("sampler dependencies present", True, "SKIPPED - %s" % exc)
        return

    from run_pilot import make_factors

    rng = np.random.default_rng(11)
    factor_list = [make_factors(rng, i) for i in range(2)]
    # v1.3: the decision unit is a 3-step plan over an H=6 horizon. plan_len == H is
    # rejected outright (the C/O gap would be a constant zero), so that rejection is
    # itself a check rather than a comment.
    cfg = SA.SamplerConfig(K=5, H=6, plan_len=3, T=4, n_cand=6, seed=3)
    ids = ["ft_000", "ft_001"]

    for bad, why in ((dict(H=3, plan_len=3), "plan_len == H"),
                     (dict(H=6, plan_len=7), "plan_len > H"),
                     (dict(H=6, plan_len=3, rule="R-top"), "R-top on a plan unit")):
        try:
            SA.generate(MU.MuJoCoPush3D(substeps=50), factor_list[:1],
                        cfg=SA.SamplerConfig(**bad), ids=["x"])
            refused = False
        except ValueError:
            refused = True
        check("generate refuses %s" % why, refused)

    d1 = SA.generate(MU.MuJoCoPush3D(), factor_list, cfg=cfg, ids=ids)
    d2 = SA.generate(MU.MuJoCoPush3D(), factor_list, cfg=cfg, ids=ids)
    same = all(a["y_C"] == b["y_C"] and a["y_O"] == b["y_O"]
               for ep1, ep2 in zip(d1["episodes"], d2["episodes"])
               for a, b in zip(ep1["steps"], ep2["steps"]))
    check("dataset generation is bit-reproducible", same,
          "rebuilt twice from the same factors")

    # structural invariants of the two conventions
    all_steps = [st for ep in d1["episodes"] for st in ep["steps"]]
    # Arm 0 under C and under O is the SAME action sequence, so the two can only
    # differ by replay noise. 1e-12 was a guess and it failed: recomputing
    # `base_policy` from a reloaded state yields actions that differ by ~4e-14, and
    # in contact that amplifies. Measure the gap, assert well above it, and report
    # the number -- a structural mismatch would be >=1e-3, four orders away.
    arm0_gap = max(abs(st["y_C"][0] - st["y_O"][0]) for st in all_steps)
    check("arm 0 identical under C and O (it is the factual arm)", arm0_gap <= 1e-9,
          "max|diff|=%.2e in Y over %d steps (replay floor, not 0)" % (arm0_gap, len(all_steps)))

    conv_differ = sum(1 for st in all_steps
                      if max(abs(np.array(st["y_C"][1:]) - np.array(st["y_O"][1:]))) > 1e-9)
    check("C and O disagree on counterfactual arms", conv_differ > 0,
          "%d/%d steps differ" % (conv_differ, len(all_steps)))

    # -- the plan unit is really a plan ------------------------------------ #
    shapes = {tuple(np.asarray(st["plans"]).shape) for st in all_steps}
    check("every step stores plans (K, m, 3) with m = plan_len",
          shapes == {(cfg.K, cfg.plan_len, 3)}, "shapes=%s" % sorted(shapes))

    arm_view = all(np.array_equal(np.asarray(st["arms"]),
                                  np.asarray(st["plans"])[:, 0, :]) for st in all_steps)
    check("arms is exactly plans[:, 0]", arm_view,
          "the one-step view must not be a second, drift-prone field")

    # The perturbation MUST NOT be a constant offset across the plan. A purely
    # persistent bias makes every plan the same direction scaled over m, i.e. the
    # single action times m -- multi-step content stays zero while the code looks
    # fixed. With eps_h = (1-w) b + w (-1)^h q the lag-1 correlation of the deviation
    # is ((1-w)^2 - w^2) / ((1-w)^2 + w^2), which is 0 at w = 0.5 and +1 at w = 0.
    w = float(cfg.plan_alt_frac)
    pred = ((1 - w) ** 2 - w ** 2) / ((1 - w) ** 2 + w ** 2)
    dev = np.stack([np.asarray(st["plans"])[1:] - np.asarray(st["plans"])[0][None]
                    for st in all_steps])                       # (N, K-1, m, 3)
    flat = dev.transpose(0, 1, 3, 2).reshape(-1, cfg.plan_len)   # rows = (N, K-1, 3)
    lag = float(np.mean([np.corrcoef(flat[:, h], flat[:, h + 1])[0, 1]
                         for h in range(cfg.plan_len - 1)]))
    check("plan perturbation is not a constant offset (lag-1 corr far from +1)",
          abs(lag - pred) < 0.5,
          "measured lag-1 corr=%.3f, predicted %.3f at w=%.1f (w=0 would be +1)"
          % (lag, pred, w))

    # -- the decision points must form a TRAJECTORY, not 15 probes ---------- #
    # Nothing asserted this, and for one revision it was false: run_episode never
    # advanced the simulator, so each decision point began wherever the previous
    # twin_rollout abandoned it (the last candidate's open-loop end). The data still
    # looked completely normal -- K plans, two conventions, plausible Y spread -- it
    # just was not an episode. Two independent bound checks pin the cadence down:
    #
    #   (a) the commanded setpoint advances by exactly action_scale * first action;
    #   (b) the pusher body cannot move more than action_scale*sqrt(2) in one step
    #       (before contact; being pushed only adds, and that is measured by (a)).
    scale = float(getattr(MU.MuJoCoPush3D(substeps=cfg.control_substeps),
                          "action_scale", 0.0))
    dctrl, dpush, moving = [], [], []
    for ep in d1["episodes"]:
        st = ep["steps"]
        for i in range(len(st) - 1):
            o, nx = np.asarray(st[i]["obs"]), np.asarray(st[i + 1]["obs"])
            a0 = np.asarray(st[i]["plans"])[0][0]
            dctrl.append((nx[19:21] - o[19:21]) - scale * a0[:2])
            dpush.append(float(np.linalg.norm(nx[6:8] - o[6:8])))
            moving.append(float(np.max(np.abs(nx[0:2] - o[0:2]))))
    dctrl = np.abs(np.array(dctrl))
    dpush = np.array(dpush)
    # clipping at the ctrl limit legitimately breaks (a); count those separately
    # rather than loosening the tolerance for everyone.
    pinned = int((dctrl.max(axis=1) > 1e-9).sum())
    check("decision points advance by exactly one control step (setpoint identity)",
          pinned == 0,
          "%d/%d transitions deviate from c'=c+%.4f*a, max|dev|=%.2e "
          "(deviation means the trajectory is not 1 step per decision)"
          % (pinned, len(dctrl), scale, dctrl.max()))
    bound = scale * np.sqrt(2.0) * 1.05
    check("pusher moves at most one step per decision point",
          float(dpush.max()) <= bound,
          "max|dpush|=%.4f m vs one-step bound %.4f m (median %.4f)"
          % (dpush.max(), bound, float(np.median(dpush))))
    # The object only moves while in contact, so a median over all steps is ~0 by
    # design (measured 5e-12). What must hold is that contact actually happens: if
    # the scene never pushes the object, every Y is a pure pusher-distance effect and
    # "counterfactual planning" would be about servo lag.
    frac_pushed = float(np.mean(np.array(moving) > 1e-6))
    check("the object is actually pushed on a non-trivial share of steps",
          frac_pushed >= 0.10,
          "%.1f%% of transitions move the object >1e-6 m" % (100 * frac_pushed))

    # Sign convention unit test. Asserting "both signs appear in the dataset" on a
    # two-episode sample is not an invariant -- it failed for exactly that reason --
    # but the convention itself is the thing that was wrong twice, so test it directly.
    up = SA.progress_from_potential(phi_start=0.20, phi_end=0.30, phi_ref=0.30)
    down = SA.progress_from_potential(phi_start=0.30, phi_end=0.20, phi_ref=0.30)
    check("potential rise -> positive Y, fall -> negative Y",
          abs(up - 1.0 / 3.0) < 1e-12 and abs(down + 1.0 / 3.0) < 1e-12,
          "up=%.4f down=%.4f" % (up, down))
    nf = np.concatenate([np.array(st["y_C"][1:]) for st in all_steps])
    check("Y magnitudes are on the phi_ref scale (not raw phi)",
          float(np.max(np.abs(nf))) < 1.0,
          "max|Y|=%.4f on %d arms" % (float(np.max(np.abs(nf))), nf.size))

    # records <-> table round trip
    rows = SA.records_from(d1, convention="C")
    tab = M.build_table(rows, k_arms=cfg.K)
    check("records round-trip into a dense table",
          tab.y_true.shape == (len(all_steps), cfg.K) and len(rows) == len(all_steps) * cfg.K,
          "rows=%d table=%s" % (len(rows), tab.y_true.shape))

    # split must not leak, at template granularity
    assigned = SP.assign(d1["episodes"] + SP.assign(d1["episodes"], seed=3), seed=3)
    rep = SP.check_no_leak(assigned)
    check("split report has no template/instance leak",
          not rep["template_leak"] and not rep["instance_leak"],
          "templates t/v/te=%d/%d/%d" % (rep["n_train_templates"],
                                         rep["n_val_templates"], rep["n_test_templates"]))


# --------------------------------------------------------------------------- #
def part_g():
    print("\nG. predictor interface (leak-proof by construction) + report schema")
    try:
        import mujoco  # noqa: F401
        import mujoco_adapter as MU
        import predictors as PR
        import report as RP
        import sampler as SA
    except ModuleNotFoundError as exc:
        check("predictor dependencies present", True, "SKIPPED - %s" % exc)
        return

    from run_pilot import make_factors

    spec = M.load_spec()
    eps = float(spec["noise_floor"]["deterministic_fallback"])

    # -- structural guarantee: nothing to read a label from ------------------ #
    fields = set(PR.FeatureTable.__dataclass_fields__)
    check("FeatureTable holds no truth/degeneracy field",
          not ({"y_true", "y", "degen", "degenerate", "y_C", "y_S"} & fields),
          "fields=%s" % sorted(fields))

    rng = np.random.default_rng(5)
    facs = [make_factors(rng, i) for i in range(6)]
    cfg = SA.SamplerConfig(K=5, H=6, plan_len=3, T=3, n_cand=6, seed=1)
    ds = SA.generate(MU.MuJoCoPush3D(), facs, cfg=cfg,
                     ids=["g%d" % i for i in range(6)])
    tr, te = ["g0", "g1", "g2", "g3"], ["g4", "g5"]

    ft = PR.features_from(ds, tr)
    check("features_from returns (n_steps, K, m, 3) plans and full-width obs",
          ft.plans.shape == (len(tr) * cfg.T, cfg.K, cfg.plan_len, 3)
          and ft.obs.shape == (len(tr) * cfg.T, len(SA.OBS_NAMES)),
          "plans=%s obs=%s" % (ft.plans.shape, ft.obs.shape))
    check("arms is a derived view of plans[:, :, 0]",
          np.array_equal(ft.arms, ft.plans[:, :, 0, :]),
          "arms=%s" % (ft.arms.shape,))

    # -- split discipline is enforced by the code path, not by the caller --- #
    try:
        PR.fit_predict(PR.Ridge(), ds, convention="C", train_ids=tr,
                       test_ids=["g3", "g4"])
        overlap_refused = False
    except ValueError:
        overlap_refused = True
    check("fit_predict refuses overlapping train/test", overlap_refused)

    # -- the learned model cannot see the myopic column -------------------- #
    # -- M1 and G1 are the SAME ranker (algebra, not opinion) -------------- #
    # After centring, the level block is row-constant and therefore orthogonal to
    # every deviation block, so the ridge coefficients on the deviations do not depend
    # on whether the level block is present. If this ever stops holding, M1 has gained
    # a real state interaction (or the centring was lost), and every sentence built on
    # "structure vs global direction" has to be rewritten -- which is why it is an
    # assertion rather than a note.
    y_tr = PR.truth_matrix(ds, "C", tr)
    m1 = PR.StructuredPlan(level=3, alpha=1.0).fit(ft, y_tr)
    g1 = PR.GlobalDirection(alpha=1.0).fit(ft, y_tr)
    pm = m1.predict(ft)
    pg = g1.predict(ft)
    dgap = float(np.max(np.abs((pm - pm[:, [0]]) - (pg - pg[:, [0]]))))
    check("M1 ranking == single global direction on its own features", dgap < 1e-9,
          "max|Delta pred diff|=%.2e (level block is orthogonal by construction)" % dgap)

    scrambled = PR.FeatureTable(obs=ft.obs, plans=ft.plans,
                                myopic=ft.myopic[::-1].copy(),
                                phi_ref=ft.phi_ref,
                                episode_ids=ft.episode_ids, k=ft.k)
    y_tr = PR.truth_matrix(ds, "C", tr)
    r_a = PR.Ridge(seed=0).fit(ft, y_tr)
    r_b = PR.Ridge(seed=0).fit(scrambled, y_tr)
    gap = float(np.max(np.abs(r_a.predict(ft) - r_b.predict(ft))))
    check("Ridge predictions invariant to the myopic column", gap == 0.0,
          "max|diff|=%.2e (a positive weight on it would make DAR 0 or 1, never chance)"
          % gap)

    # -- the reference is the H=1 truth, exactly --------------------------- #
    ft_all = PR.features_from(ds, [e for e in ("g0", "g1", "g2", "g3", "g4", "g5")])
    y_s = PR.truth_matrix(ds, "S", [e for e in ("g0", "g1", "g2", "g3", "g4", "g5")])
    mp = ft_all.myopic / ft_all.phi_ref
    ident = float(np.max(np.abs((mp - mp[:, [0]]) - (y_s - y_s[:, [0]]))))
    # Two INDEPENDENT replays of the same one step, so the achievable agreement is
    # the Gate-0 replay tolerance (1e-9), not bit-exactness. Asserting 1e-12 here was
    # asserting more precision than the simulator provides.
    check("one-step oracle delta == 1-step truth delta (to replay tolerance)",
          ident < 1e-9, "max|diff|=%.2e, Gate-0 tol=1e-9" % ident)

    # -- the control must not grossly succeed ------------------------------ #
    tab, _ = PR.fit_predict(PR.RidgeArmShuffled(seed=0), ds, convention="C",
                            train_ids=tr, test_ids=te)
    rep = M.evaluate(tab, spec, eps=eps, n_boot=200)
    cm = M.ruler_selftest(tab, spec, chance_permutations=100)["chance"]["dar_perm_mean"]
    ctl = rep["metrics"]["dar"]["value"]
    check("arm-shuffled control stays near chance (gross-failure check)",
          abs(ctl - cm) <= 0.25, "control=%.3f chance=%.3f" % (ctl, cm))

    # -- task property is computable and bounded --------------------------- #
    hz = SA.horizon_collapse(ds, "C")
    check("horizon_collapse returns a bounded rank statistic",
          hz["n_steps"] > 0 and -1.0 <= hz["rho_mean"] <= 1.0,
          "rho=%.3f identical=%.2f" % (hz["rho_mean"], hz["identical_ranking_rate"]))

    # -- report payload carries provenance for all four questions ---------- #
    env = RP.environment(spec)
    need = {"schema_version", "spec_version", "python", "numpy", "scipy"}
    check("environment() records schema+spec+stack", need.issubset(set(env)),
          "keys=%s" % sorted(env))
    payload = RP.pack_run(label="selftest", spec=spec, adapter=None, dataset=ds,
                          split={"on": "template"}, estimates=[], checks={"x": True},
                          config=cfg.as_dict(), eval_split="test")
    ok = (payload["dataset"]["phi_ref"] > 0 and payload["env"]["spec_version"] == spec["spec_version"]
          and payload["all_checks_passed"])
    check("pack_run carries phi_ref + spec_version and folds the verdict", ok,
          "phi_ref=%.6f" % payload["dataset"]["phi_ref"])


# --------------------------------------------------------------------------- #
def main() -> int:
    import scipy
    print("=" * 78)
    print("counterfactual-plan-ranking self-test -- metrics.py (spec %s) + gate0_replay.py" % M.SPEC_VERSION)
    print("=" * 78)
    # Environment banner: a PASS from the wrong interpreter is not evidence.
    print("interpreter :", sys.executable)
    print("python      :", sys.version.split()[0])
    print("numpy       :", np.__version__)
    print("scipy       :", scipy.__version__)
    spec = M.load_spec()
    tab = M.build_table(make_records(noise=0.0), k_arms=spec["candidate_set"]["K_primary"])
    print("base table:", tab)

    part_a(spec)
    part_b(spec, tab)
    part_c()
    part_c2()
    part_d()
    part_e()
    part_f()
    part_g()

    n = len(RESULTS)
    bad = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 78)
    print("TOTAL %d checks, %d failed" % (n, len(bad)))
    for name, _, detail in bad:
        print("  FAILED:", name, detail)
    print("VERDICT:", "PASS" if not bad else "FAIL")
    print("=" * 78)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
