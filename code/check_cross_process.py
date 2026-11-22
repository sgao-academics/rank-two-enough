"""
check_cross_process.py -- does the simulator replay identically across PROCESSES?

Gate-0 replays one script N times inside a single process. That only proves the
state restore works. The sampling pipeline will fan out across processes, and
that is where thread counts, BLAS, import order and allocator behaviour actually
differ. This script is the assumption the pipeline silently makes, checked.

    python check_cross_process.py                    # mujoco:push3d, 8x12
    python check_cross_process.py --sim mock:noisy

Exit 0 == the two independent runs produced identical numbers and fingerprints.
Exit 1 == they differ; treat the sampling pipeline as non-reproducible until fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.abspath(os.path.join(HERE, os.pardir, "logs"))


def run_once(py: str, sim: str, repeats: int, steps: int, out: str) -> dict:
    r = subprocess.run([py, "gate0_replay.py", "--sim", sim,
                        "--repeats", str(repeats), "--steps", str(steps),
                        "--out", out],
                       cwd=HERE, capture_output=True, text=True, timeout=3600)
    if r.returncode not in (0, 1):        # 1 == DRIFTING, still a valid measurement
        sys.stderr.write(r.stderr[-1500:])
        raise SystemExit("gate0_replay failed with rc=%d" % r.returncode)
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="cross-process replay determinism")
    ap.add_argument("--sim", default="mujoco:push3d")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args(argv)

    os.makedirs(LOGS, exist_ok=True)
    py = sys.executable
    a = run_once(py, args.sim, args.repeats, args.steps,
                 os.path.join(LOGS, "xproc_A.json"))
    b = run_once(py, args.sim, args.repeats, args.steps,
                 os.path.join(LOGS, "xproc_B.json"))

    same_series = a["per_run_max"] == b["per_run_max"]
    same_cls = a["classification"] == b["classification"]
    fp_diff = {k: (a["fingerprint"][k], b["fingerprint"].get(k))
               for k in a["fingerprint"] if a["fingerprint"][k] != b["fingerprint"].get(k)}

    print("sim                    :", args.sim)
    print("process A max|dpose|   : %.6e   class %s" % (a["max_abs_diff"], a["classification"]))
    print("process B max|dpose|   : %.6e   class %s" % (b["max_abs_diff"], b["classification"]))
    print("per_run_max identical  :", same_series)
    print("classification equal   :", same_cls)
    print("fingerprint differences:", fp_diff if fp_diff else "none")

    ok = same_series and same_cls and not fp_diff
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
