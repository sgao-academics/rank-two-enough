"""One table, one source: per-config DAR for every rung, plus LRB by rank.

Reads the 360 stored evaluations for the reference models and the fresh
logs/lrb_final.json for the low-rank family, so the numbers in the manuscript
and the numbers in the figure come from the same place.
"""

import glob
import io
import json
import os
import re
import sys
from statistics import mean, stdev

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import run_suite as RS       # noqa: E402

SUITE = os.path.join(RS.ROOT, "logs", "suite", "scene2_spec13")
PAT = re.compile(r"^ev_(?P<cfg>.+)_s(?P<seed>\d+)_sp(?P<sp>\d+)\.json$")
LABELS = ("G0b summed aligned push", "G1 global direction", "M1 L3 a0.1",
          "B4 ridge poly2")

per = {}
for path in sorted(glob.glob(os.path.join(SUITE, "ev_*.json"))):
    m = PAT.match(os.path.basename(path))
    if not m:
        continue
    cfg = m.group("cfg")
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    for lab in LABELS:
        row = d["rows"]["C"].get(lab)
        if row and row.get("dar") is not None:
            per.setdefault((cfg, lab), []).append(float(row["dar"]))

lrb = json.load(open(os.path.join(RS.ROOT, "logs", "lrb_final.json"),
                     encoding="utf-8"))
CONFIGS = ("m3", "H4", "H8")

print("%-24s %5s %8s %8s %8s %8s" % ("model", "prm", "m3", "H4", "H8", "mean"))
print("-" * 66)
table = {}
for lab in LABELS:
    vals, prm = [], {"G0b summed aligned push": 0, "G1 global direction": 11,
                     "M1 L3 a0.1": 15}.get(lab, 560)
    for cfg in CONFIGS:
        v = per.get((cfg, lab), [])
        vals.append(mean(v) if v else float("nan"))
    allv = [x for cfg in CONFIGS for x in per.get((cfg, lab), [])]
    print("%-24s %5d %8.4f %8.4f %8.4f %8.4f"
          % (lab, prm, vals[0], vals[1], vals[2], mean(allv)))
    table[lab] = {"params": prm, "by_config": vals, "mean": mean(allv),
                  "sd": stdev(allv)}

print("-" * 66)
for r in range(1, 10):
    vals = []
    for cfg in CONFIGS:
        key = "%s|r%d" % (cfg, r)
        vals.append(lrb["rank_curve"][key][0] if key in lrb["rank_curve"]
                    else float("nan"))
    allv = [lrb["rank_curve"]["%s|r%d" % (cfg, r)][0] for cfg in CONFIGS
            if "%s|r%d" % (cfg, r) in lrb["rank_curve"]]
    print("%-24s %5d %8.4f %8.4f %8.4f %8.4f"
          % ("LRB r%d" % r, 32 + 31 * r, vals[0], vals[1], vals[2], mean(allv)))
    table["LRB r%d" % r] = {"params": 32 + 31 * r, "by_config": vals,
                            "mean": mean(allv)}

print("\nrank chosen on validation: %s"
      % ", ".join("r%s=%d" % (k, v) for k, v in
                  sorted(lrb["chosen"].items(), key=lambda kv: int(kv[0]))))
print("cells: %d" % lrb["cells"])

with open(os.path.join(RS.ROOT, "logs", "lrb_table.json"), "w",
          encoding="utf-8") as fh:
    json.dump({"table": table, "chosen": lrb["chosen"]}, fh, indent=1)
print("wrote logs/lrb_table.json")
