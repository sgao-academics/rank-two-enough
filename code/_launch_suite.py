"""Launch run_suite.py detached, with no console window and a log file.

Windows discipline (learned the hard way):
  * CREATE_NO_WINDOW (0x08000000), NOT DETACHED_PROCESS (0x8): detached means "no
    console", so any child that wants one creates a visible window -- that is how an
    overnight job once produced an unbounded cascade of popups.
  * The target script has an `if __name__ == "__main__"` guard, so it can never
    re-run on import.
  * The launcher itself is guarded too, for the same reason.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG = os.path.join(HERE, os.pardir, "logs", "suite_stdout.log")


def main() -> int:
    os.makedirs(os.path.dirname(os.path.abspath(LOG)), exist_ok=True)
    out = open(os.path.abspath(LOG), "a", encoding="utf-8")
    cmd = [PY, os.path.join(HERE, "run_suite.py")] + sys.argv[1:]
    flags = 0x08000000                      # CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT,
                            creationflags=flags, close_fds=True)
    print("launched pid=%d" % proc.pid)
    print("log:", os.path.abspath(LOG))
    print("progress:", os.path.abspath(os.path.join(
        HERE, os.pardir, "logs", "suite", "scene2_spec13", "progress.log")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
