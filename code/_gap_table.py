# -*- coding: utf-8 -*-
"""Arm separation per configuration, on one rule.

`r=1` DAR rises with plan length in the ladder, and the obvious explanation is
that the arms separate more when a plan spends more steps, so even a one-direction
score recovers part of the gap.  This prints the median within-decision spread of
the five arm outcomes (max - min) plus the degenerate share, on exactly the cells
the rank curves were fitted on.
"""
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, "data", "suite", "scene2_spec13")
CFG = {"pl1": (1, 2), "pl2": (2, 4), "m3": (3, 6), "pl4": (4, 8),
       "H4": (3, 4), "H8": (3, 8), "m1": (1, 6)}


def main():
    print("%-5s %2s %2s %6s %9s %9s %8s" % ("cfg", "m", "H", "t", "gap_med", "gap_p10", "deg%"))
    for cfg, (m, H) in CFG.items():
        gaps, dec, tot = [], 0, 0
        for sd in range(12):
            p = os.path.join(D, "ds_%s_s%02d_e96.json" % (cfg, sd))
            if not os.path.exists(p):
                continue
            ds = json.load(open(p, encoding="utf-8"))
            for ep in ds["episodes"]:
                for st in ep["steps"]:
                    y = np.array(st["y_C"], float)
                    gaps.append(y.max() - y.min())
                    tot += 1
                    if st.get("degenerate_C") or st.get("degenerate_O"):
                        dec += 1
        print("%-5s %2d %2d %6.3f %9.4f %9.4f %8.2f"
              % (cfg, m, H, (H - m) / H, np.median(gaps),
                 np.percentile(gaps, 10), 100.0 * dec / max(tot, 1)))


if __name__ == "__main__":
    main()
