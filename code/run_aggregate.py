"""run_aggregate.py -- turn the cached suite cells into the tables a reviewer reads.

Reads whatever exists (so it can be run while the suite is still going), and writes
both a JSON and a markdown table set. Three things it does that a per-run table
cannot:

  1. Pairs, not points. Every number is summarised over (dataset seed, split seed)
     pairs, so the spread is the spread a reader would get by re-running the study,
     not the bootstrap spread of one lucky split.
  2. Paired comparisons against the lines that matter: R1 (one-step oracle),
     G0b (zero-parameter aligned-push heuristic) and G1 (global direction). A
     method that does not beat G0b has not modelled anything.
  3. Val-selected hyperparameters, reported on test. The selection rule is fixed in
     advance (best val DAR over the M1 grid); the aggregate reports both the
     selected-on-val number and the pre-registered default, plus how often each
     level was chosen -- a selection rule that keeps picking the same level is
     evidence it is not doing work.

    python run_aggregate.py                 # all configs present
    python run_aggregate.py --configs m3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Single source of truth for the payload schema: importing it is the only way to
# avoid the exact bug this line replaced -- the aggregator hard-coded "/2" while the
# runner had moved to "/3", and it then produced an EMPTY table without an error.
from run_suite import EVAL_SCHEMA, SCENE_TAG        # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, os.pardir, "logs", "suite", SCENE_TAG))
METRICS = ("dar", "regret_norm", "zero_regret_rate", "rho", "nmae")
REF_LINES = ("R1 one-step oracle", "G0b summed aligned push",
             "G1 global direction", "B1 bet-on-expert")


def _load_all(configs) -> dict:
    rows = defaultdict(list)          # (config, conv) -> list of eval dicts
    for path in sorted(glob.glob(os.path.join(OUT, "ev_*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        if obj.get("schema") != EVAL_SCHEMA or "rows" not in obj:
            continue
        if obj.get("skipped"):
            continue
        if configs and obj["config"] not in configs:
            continue
        for conv in obj["rows"]:
            rows[(obj["config"], conv)].append(obj)
    return rows


def _summ(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1))
            if v.size > 1 else 0.0, "median": float(np.median(v))}


def _paired(a, b) -> dict:
    """Paired difference a - b over the same (seed, split) pairs."""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return {"n": len(pairs), "mean_diff": None, "sd": None, "win_rate": None,
                "p_ttest": None}
    d = np.array([x - y for x, y in pairs], float)
    out = {"n": int(d.size), "mean_diff": float(d.mean()),
           "sd": float(d.std(ddof=1)), "win_rate": float((d > 0).mean()),
           "p_ttest": None}
    try:
        from scipy.stats import ttest_rel
        pa = np.array([x for x, _ in pairs], float)
        pb = np.array([y for _, y in pairs], float)
        out["p_ttest"] = float(ttest_rel(pa, pb).pvalue)
    except Exception:                                            # noqa: BLE001
        pass
    return out


def _table(configs) -> tuple:
    rows = _load_all(configs)
    out = {"scene_tag": SCENE_TAG, "configs": {}, "pairs": {}}
    lines = []

    for (cfg, conv), evs in sorted(rows.items()):
        labels = set()
        for e in evs:
            labels |= set(e["rows"][conv])
        labels = sorted(labels)
        # build label -> list of values aligned by (seed, split)
        keys = [(e["dataset_seed"], e["split_seed"]) for e in evs]
        val = {lab: [] for lab in labels}
        for e in evs:
            for lab in labels:
                cell = e["rows"][conv].get(lab) or {}
                val[lab].append(cell.get("dar"))
        block = {"n_pairs": len(evs), "metrics": {}}
        for lab in labels:
            block["metrics"][lab] = {
                "dar": _summ(val[lab]),
                "regret_norm": _summ([(e["rows"][conv].get(lab) or {}).get("regret_norm")
                                      for e in evs]),
                "zero_regret_rate": _summ([(e["rows"][conv].get(lab) or {}).get(
                    "zero_regret_rate") for e in evs]),
                "rho": _summ([(e["rows"][conv].get(lab) or {}).get("rho") for e in evs]),
                "nmae": _summ([(e["rows"][conv].get(lab) or {}).get("nmae") for e in evs]),
            }
        out["configs"]["%s/%s" % (cfg, conv)] = block

        for ref in REF_LINES:
            if ref in val:
                out["pairs"]["%s/%s/M1_vs_%s" % (cfg, conv, ref)] = _paired(
                    val["M1 L3 a0.1"], val[ref])

        if conv == "C":
            lines.append("\n### %s (convention C, %d seed x split pairs)\n" % (cfg, len(evs)))
            lines.append("| ranker | DAR (mean±sd) | regret_norm | zero-regret | nMAE |")
            lines.append("|:--|--:|--:|--:|--:|")
            order = [lab for lab in labels if lab.startswith(("M1", "G", "R1", "B",
                                                             "C1", "C2", "L1", "L2"))]
            order.sort(key=lambda s: -(block["metrics"][s]["dar"]["mean"] or -1))
            for lab in order:
                m = block["metrics"][lab]
                d = m["dar"]
                if d["mean"] is None:
                    continue
                lines.append("| %s | %.4f ± %.4f | %s | %s | %s |"
                             % (lab, d["mean"], d["sd"],
                                _f(m["regret_norm"]["mean"], "%.4f"),
                                _f(m["zero_regret_rate"]["mean"], "%.3f"),
                                _f(m["nmae"]["mean"], "%.3f")))
    return out, lines


def _f(v, fmt):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt % v


def _val_selection(rows) -> dict:
    """Choose (level, alpha) on VAL, report TEST. One fixed rule, applied per pair.

    Rule, declared before looking: among the M1 grid entries present in the payload,
    pick the one with the highest VAL DAR for that (dataset seed, split seed); record
    its TEST DAR. Also record the pre-registered default (level 3, alpha 0.1) so the
    two can be compared -- if selection never helps, saying so is the honest outcome.
    """
    grid = [lab for lab in ("M1 L%d a%g" % (l, a)
                            for l in range(7) for a in (0.1, 1.0, 10.0))]
    picked = defaultdict(int)
    sel, default, val_sel, chosen = [], [], [], []
    for (cfg, conv), evs in rows.items():
        if conv != "C":
            continue
        for e in evs:
            vr = e.get("val_rows") or {}
            tr = e["rows"]["C"]
            if not vr:
                continue
            cand = [(vr[l]["dar"], l) for l in grid
                    if l in vr and vr[l].get("dar") is not None]
            if not cand:
                continue
            best_val, best_lab = max(cand)
            picked[best_lab] += 1
            chosen.append((cfg, e["dataset_seed"], e["split_seed"], best_lab))
            sel.append(tr[best_lab]["dar"])
            val_sel.append(best_val)
            default.append(tr["M1 L3 a0.1"]["dar"])
    return {"picked": dict(picked), "n": len(sel),
            "test_dar_selected": _summ(sel), "val_dar_selected": _summ(val_sel),
            "test_dar_default": _summ(default),
            "paired_selected_minus_default": _paired(sel, default)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[])
    args = ap.parse_args(argv)
    obj, lines = _table(args.configs)
    rows = _load_all(args.configs)
    if not rows:
        # An empty table is not a result. Fail loudly instead of writing one: this is
        # how the schema mismatch above stayed invisible for one run.
        raise SystemExit(
            "no usable evaluations under %s (expected schema %s). "
            "Is the suite still on its first cells, or was the schema bumped?"
            % (OUT, EVAL_SCHEMA))
    obj["val_selection"] = _val_selection(rows)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "aggregate.json"), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    md = "# Suite aggregate (%s)\n" % SCENE_TAG
    md += "\n".join(lines)
    vs = obj["val_selection"]
    md += "\n\n## Hyperparameters chosen on VAL, reported on TEST\n\n"
    if vs["n"]:
        md += ("rule: highest VAL DAR over the M1 grid, applied per (seed, split); "
               "n=%d pairs\n\n" % vs["n"])
        md += "| quantity | mean | sd |\n|:--|--:|--:|\n"
        md += ("| test DAR, selected on val | %s | %s |\n"
               % (_f(vs["test_dar_selected"]["mean"], "%.4f"),
                  _f(vs["test_dar_selected"]["sd"], "%.4f")))
        md += ("| test DAR, pre-registered L3 a0.1 | %s | %s |\n"
               % (_f(vs["test_dar_default"]["mean"], "%.4f"),
                  _f(vs["test_dar_default"]["sd"], "%.4f")))
        p = vs["paired_selected_minus_default"]
        md += ("| paired diff (selected - default) | %s | %s |\n"
               % (_f(p["mean_diff"], "%+.4f"), _f(p["sd"], "%.4f")))
        md += "\n  picked: %s\n" % json.dumps(vs["picked"], sort_keys=True)

    md += "\n\n## Paired differences vs the reference lines\n\n"
    md += "| comparison | mean diff | sd | win rate | p (paired t) |\n"
    md += "|:--|--:|--:|--:|--:|\n"
    for k, v in obj["pairs"].items():
        if v["mean_diff"] is None:
            continue
        md += "| %s | %+.4f | %.4f | %.2f | %s |\n" % (
            k, v["mean_diff"], v["sd"], v["win_rate"], _f(v["p_ttest"], "%.2e"))
    with open(os.path.join(OUT, "aggregate.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md[:4000])
    print("\nwritten:", os.path.join(OUT, "aggregate.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
