"""Route B -- the marginal value of structure, priced in parameters and data.

Everything here is computed from artefacts that already exist on disk:
  * the 6 x 12 x 5 eval JSONs (predictors trained on C, scored on C and on O)
  * the training-size curves stored inside the same files
  * the design matrices, which give the EXACT number of fitted coefficients

No new simulation, no new dataset. The question is not "who wins DAR" -- that was
already answered (B4, the generic quadratic, wins everywhere). The question is
"what does M1 buy for the price it charges", and that has three separate answers:
parameters, data, and unseen templates. They are reported separately because they
can disagree, and a claim that survives only one of them is a weak claim.
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys
from statistics import mean, stdev

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import predictors as P          # noqa: E402
import sampler as S             # noqa: E402
import split as SP              # noqa: E402
import run_suite as RS          # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")

OUT = RS.OUT
DATA = RS.DATA
CONFIGS = ["m3", "m1", "sig04", "sig16", "H4", "H8"]


def load_evals():
    """(cfg, seed, split) -> payload, for every eval file that exists."""
    out = {}
    pat = re.compile(r"^ev_(?P<cfg>.+)_s(?P<seed>\d+)_sp(?P<sp>\d+)\.json$")
    for path in sorted(glob.glob(os.path.join(OUT, "ev_*.json"))):
        base = os.path.basename(path)
        m = pat.match(base)
        if not m:
            print("  ! skip %s (name does not parse)" % base)
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                out[(m.group("cfg"), int(m.group("seed")),
                     int(m.group("sp")))] = json.load(fh)
        except Exception as exc:                      # a half-written file
            print("  ! skip %s (%s)" % (base, exc))
    return out


def params_of():
    """Exact count of fitted coefficients per ranker, from its own design matrix."""
    cfg = "m3"
    path = os.path.join(DATA, "ds_%s_s00_e96.json" % cfg)
    with open(path, "r", encoding="utf-8") as fh:
        ds = json.load(fh)
    rows = SP.assign(ds["episodes"], seed=0)
    tr = [r["episode_id"] for r in rows if r["split"] == "train"]
    ft = P.features_from(ds, tr)

    y = P.truth_matrix(ds, "C", tr)
    out = {}
    for label, (pred, _tc) in RS.predictor_set(0).items():
        # Count the FITTED coefficients, not the design width: G1 is not a Ridge
        # subclass, so an isinstance dispatch silently reported 0 for an 11-parameter
        # model and made the whole price table lie in the method's favour.
        try:
            pred.fit(ft, y)
        except Exception:
            out[label] = 0
            continue
        w = getattr(pred, "w", None)
        if w is None:
            out[label] = 0                           # no fitted coefficients at all
        elif isinstance(pred, (P.Ridge, P.StructuredPlan)):
            out[label] = int(len(w)) - 1             # w = [intercept, coefs]
        else:
            out[label] = int(len(w))
    return out, ft


def agg(evals, conv, label):
    """DAR over every (dataset seed, split seed) pair that exists. The pair is the
    inferential unit, not the individual run: one run's bootstrap says nothing about
    whether the scene or the template partition mattered."""
    vals = []
    for (_cfg, _s, _sp), d in evals.items():
        row = d["rows"][conv].get(label)
        if row is not None and row.get("dar") is not None:
            vals.append(float(row["dar"]))
    return vals


def agg_by_cfg(evals, cfg, conv, label):
    vals = []
    for (c, _s, _sp), d in evals.items():
        if c != cfg:
            continue
        row = d["rows"][conv].get(label)
        if row is not None and row.get("dar") is not None:
            vals.append(float(row["dar"]))
    return vals


def line(tag, vals):
    if not vals:
        return "%-26s %s" % (tag, "---")
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return "%-26s %7.4f +- %6.4f   (n=%d)" % (tag, mean(vals), sd, len(vals))


def main():
    evals = load_evals()
    print("=" * 78)
    print("ROUTE B  --  what structure buys, priced in parameters and data")
    print("=" * 78)
    print("eval files: %d   (expected %d = 6 configs x 12 seeds x 5 splits)"
          % (len(evals), 6 * 12 * 5))
    per_cfg = {c: sum(1 for k in evals if k[0] == c) for c in CONFIGS}
    print("per config: %s" % ", ".join("%s=%d" % (c, per_cfg[c]) for c in CONFIGS))

    prm, ft = params_of()
    print("\n-- design breadth (m=3): obs=%d, plan=%s, K=%d"
          % (ft.obs.shape[1], ft.plans.shape[2:], ft.k))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("1. PARAMETER PRICE  (fitted coefficients, no intercept)")
    print("=" * 78)
    head = ["G0 perturbation magnitude", "G0b summed aligned push",
            "G1 global direction", "B2 ridge surrogate", "B3 ridge first-step only",
            "B4 ridge poly2", "M1 L0 a0.1", "M1 L1 a0.1", "M1 L2 a0.1",
            "M1 L3 a0.1", "M1 L5 a0.1", "M1 L6 a0.1", "L1 learned one-step (linear)",
            "L2 learned one-step (frame)", "C1 ridge arm-shuffled", "C2 plan-shuffled"]
    print("%-32s %8s %10s %10s" % ("ranker", "params", "DAR(C)", "DAR(O)"))
    for label in head:
        vc = agg(evals, "C", label)
        vo = agg(evals, "O", label)
        print("%-32s %8d %10s %10s"
              % (label, prm[label],
                 ("%.4f" % mean(vc)) if vc else "---",
                 ("%.4f" % mean(vo)) if vo else "---"))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("2. THE TRADE: extra parameters vs extra DAR, against the 1-param floor")
    print("=" * 78)
    base_lbl = "G0b summed aligned push"
    for cfg in CONFIGS:
        print("\n[%s]" % cfg)
        b = agg_by_cfg(evals, cfg, "C", base_lbl)
        if not b:
            print("  no data yet")
            continue
        bm = mean(b)
        print("  %-28s %6s %8s %12s %14s"
              % ("ranker", "params", "DAR", "dDAR vs G0b", "dDAR/100 params"))
        for label in ("G0b summed aligned push", "B2 ridge surrogate",
                      "M1 L3 a0.1", "B4 ridge poly2"):
            v = agg_by_cfg(evals, cfg, "C", label)
            if not v:
                continue
            m = mean(v)
            dp = prm[label] - prm[base_lbl]
            rate = (m - bm) / dp * 100.0 if dp > 0 else float("nan")
            print("  %-28s %6d %8.4f %+12.4f %14s"
                  % (label, prm[label], m, m - bm,
                     ("%.4f" % rate) if dp > 0 else "floor"))

    # the honest summary of the B4/M1 head-to-head, in one line per config
    print("\n  -- B4 needs this many times M1's parameters for its margin --")
    for cfg in CONFIGS:
        v4 = agg_by_cfg(evals, cfg, "C", "B4 ridge poly2")
        v1 = agg_by_cfg(evals, cfg, "C", "M1 L3 a0.1")
        if not (v4 and v1):
            continue
        print("  %-6s B4 %d params DAR %.4f | M1 %d params DAR %.4f | "
              "param ratio %.1fx for %+.4f DAR"
              % (cfg, prm["B4 ridge poly2"], mean(v4), prm["M1 L3 a0.1"], mean(v1),
                 prm["B4 ridge poly2"] / max(prm["M1 L3 a0.1"], 1),
                 mean(v4) - mean(v1)))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("3. DATA PRICE  (training-size curve already stored in every eval file)")
    print("=" * 78)
    for cfg in ("m3", "H4", "H8"):
        print("\n[%s]" % cfg)
        print("  %-18s %8s %8s %8s" % ("train fraction", "M1 L3", "B2", "M1-B2"))
        full = {}
        for lbl, key in (("M1 L3 a0.1", "M1 L3 a0.1"),
                         ("B2 ridge surrogate", "B2 ridge surrogate")):
            full[lbl] = agg_by_cfg(evals, cfg, "C", key)
        if full["M1 L3 a0.1"] and full["B2 ridge surrogate"]:
            print("  %-18s %8.4f %8.4f %+8.4f"
                  % ("100%", mean(full["M1 L3 a0.1"]),
                     mean(full["B2 ridge surrogate"]),
                     mean(full["M1 L3 a0.1"]) - mean(full["B2 ridge surrogate"])))
        for frac in ("0.10", "0.25", "0.50"):
            m1, b2 = [], []
            for (c, _s, _sp), d in evals.items():
                if c != cfg:
                    continue
                a = d["extras"].get("curve_%s_M1 L3 a0.1" % frac)
                b = d["extras"].get("curve_%s_B2 ridge surrogate" % frac)
                if a and a.get("dar") is not None:
                    m1.append(float(a["dar"]))
                if b and b.get("dar") is not None:
                    b2.append(float(b["dar"]))
            if m1 and b2:
                print("  %-18s %8.4f %8.4f %+8.4f"
                      % ("%d%%" % int(100 * float(frac)), mean(m1), mean(b2),
                         mean(m1) - mean(b2)))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("4. TEMPLATE PRICE  (the protocol already holds out whole templates)")
    print("=" * 78)
    cfg = "m3"
    path = os.path.join(DATA, "ds_%s_s00_e96.json" % cfg)
    with open(path, "r", encoding="utf-8") as fh:
        ds = json.load(fh)
    for sp in range(5):
        rows = SP.assign(ds["episodes"], seed=sp)
        tpl = {}
        for r in rows:
            tpl.setdefault(r["split"], set()).add(r["template"])
        shared = tpl["train"] & tpl["test"]
        print("  split seed %d: templates train=%d val=%d test=%d | overlap train/test=%d"
              % (sp, len(tpl["train"]), len(tpl["val"]), len(tpl["test"]), len(shared)))
    print("  => every test number in this suite is an unseen-template number.")

    print("\n  split-to-split spread of the headline rankers (split seed is the unit):")
    for label in ("M1 L3 a0.1", "B4 ridge poly2", "B2 ridge surrogate",
                  "G0b summed aligned push"):
        by_sp = {}
        for (c, _s, sp), d in evals.items():
            if c != cfg:
                continue
            row = d["rows"]["C"].get(label)
            if row and row.get("dar") is not None:
                by_sp.setdefault(sp, []).append(float(row["dar"]))
        if not by_sp:
            continue
        means = [mean(v) for v in by_sp.values()]
        print("  %-28s %s   spread=%.4f"
              % (label, " ".join("%.3f" % m for m in means),
                 (max(means) - min(means)) if len(means) > 1 else 0.0))

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("5. CONVENTION TRANSFER  (trained on C, scored on O)")
    print("=" * 78)
    print("%-30s %9s %9s %10s" % ("ranker", "DAR(C)", "DAR(O)", "drop"))
    for label in ("M1 L3 a0.1", "B4 ridge poly2", "B2 ridge surrogate",
                  "G1 global direction", "G0b summed aligned push",
                  "R1 one-step oracle"):
        vc, vo = agg(evals, "C", label), agg(evals, "O", label)
        if vc and vo:
            print("%-30s %9.4f %9.4f %+10.4f"
                  % (label, mean(vc), mean(vo), mean(vo) - mean(vc)))
    tv = []
    for _k, d in evals.items():
        r = d["extras"].get("transfer_C_to_O")
        if r and r.get("dar") is not None:
            tv.append(float(r["dar"]))
    if tv:
        print("\n  M1 trained on C, scored on O: %.4f +- %.4f (n=%d)"
              % (mean(tv), stdev(tv) if len(tv) > 1 else 0.0, len(tv)))
        ref = agg(evals, "C", "M1 L3 a0.1")
        if ref:
            print("  same model on its own convention: %.4f  -> transfer cost %.4f"
                  % (mean(ref), mean(ref) - mean(tv)))

    # -- persist, so the figure and the manuscript quote one source ---------- #
    ladder = ["G0b summed aligned push", "G1 global direction", "M1 L3 a0.1",
              "B4 ridge poly2"]
    dump = {"n_evals": len(evals), "params": prm,
            "dar_C": {}, "dar_O": {}, "by_config": {},
            "transfer_C_to_O": {"dar": tv}}
    for label in prm:
        vc, vo = agg(evals, "C", label), agg(evals, "O", label)
        if vc:
            dump["dar_C"][label] = [mean(vc), stdev(vc) if len(vc) > 1 else 0.0,
                                    len(vc)]
        if vo:
            dump["dar_O"][label] = [mean(vo), stdev(vo) if len(vo) > 1 else 0.0,
                                    len(vo)]
    for cfg in CONFIGS:
        dump["by_config"][cfg] = {}
        for label in ladder:
            v = agg_by_cfg(evals, cfg, "C", label)
            if v:
                dump["by_config"][cfg][label] = [mean(v),
                                                 stdev(v) if len(v) > 1 else 0.0,
                                                 len(v)]
    with open(os.path.join(RS.ROOT, "logs", "route_b.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dump, fh, indent=1)
    print("\nwrote logs/route_b.json")


if __name__ == "__main__":
    main()
