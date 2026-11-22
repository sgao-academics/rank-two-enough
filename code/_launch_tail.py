import os
# -*- coding: utf-8 -*-
"""Detach the tail sweep so it survives this session.

CREATE_NO_WINDOW (0x08000000), not DETACHED_PROCESS: the latter gives the child no
console at all, so any grandchild that wants one creates a visible window. This job
starts no children, but the same flag is used everywhere in this project so that the
rule has no exceptions to remember.
"""
import subprocess
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(CODE), "logs", "tail_stdout.log")
PY = sys.executable

fh = open(LOG, "w", encoding="utf-8", errors="ignore")
p = subprocess.Popen([PY, "_scan_tail.py"], cwd=CODE, stdout=fh,
                     stderr=subprocess.STDOUT,
                     creationflags=0x08000000)
print("tail scan pid=%d log=%s" % (p.pid, LOG))
