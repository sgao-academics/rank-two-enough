"""Launch one run_suite.py process per configuration, detached and windowless.

Why shards: the sweep is ~72 datasets x 5 splits x ~35 predictors. One process needs
about a day; six processes on a 16-core box need a few hours and are independent
because every cell is its own cached file. BLAS is pinned to one thread per process
so the six do not fight over the same cores.

Windows discipline (V24): CREATE_NO_WINDOW, and both this launcher and the target
script carry an `if __name__ == "__main__"` guard.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGDIR = os.path.abspath(os.path.join(HERE, os.pardir, "logs"))

CONFIGS = ["m3", "m1", "sig04", "sig16", "H4", "H8"]


def main() -> int:
    env_master = dict(os.environ)
    env_master.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    pids = []
    for cfg in CONFIGS:
        log = os.path.join(LOGDIR, "suite_shard_%s.log" % cfg)
        out = open(log, "a", encoding="utf-8")
        cmd = [PY, os.path.join(HERE, "run_suite.py"), "--configs", cfg]
        p = subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT,
                             creationflags=0x08000000, close_fds=True,
                             env=env_master)
        pids.append((cfg, p.pid))
        print("shard %-6s pid=%d -> %s" % (cfg, p.pid, os.path.basename(log)))
    print("total shards:", len(pids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
