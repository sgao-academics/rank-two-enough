# -*- coding: utf-8 -*-
"""Generate the tail-matched plan-length ladder datasets.

`ensure_dataset` is the suite's own generator, so this driver adds nothing to the
data format: it loops seeds and calls it, and prints the wall clock per dataset so
that a silently slow configuration is visible in the log.

Windows note: the MuJoCo sampler does not spawn children of its own, but the guard
below is kept because this file is launched through a process pool elsewhere.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_suite as RS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for cfg in a.configs:
        if cfg not in RS.CONFIGS:
            print("unknown config %r" % cfg, flush=True)
            return 2
        c = RS.CONFIGS[cfg]
        print("== %s  H=%d plan_len=%d  tail share %.3f"
              % (cfg, c["H"], c["plan_len"], (c["H"] - c["plan_len"]) / c["H"]),
              flush=True)
        t_cfg = time.time()
        for sd in a.seeds:
            p = RS.ds_path(cfg, sd)
            if os.path.exists(p) and not a.force:
                print("   s%02d exists  %.2f MB" % (sd, os.path.getsize(p) / 1048576.0),
                      flush=True)
                continue
            t0 = time.time()
            path, sec, gen = RS.ensure_dataset(cfg, sd)
            print("   s%02d %s  %6.1fs  %.2f MB"
                  % (sd, "built" if gen else "cached", sec,
                     os.path.getsize(path) / 1048576.0), flush=True)
        print("-- %s done in %.1fs" % (cfg, time.time() - t_cfg), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
