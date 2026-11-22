"""Launch the low-data curve in its own process group, hidden console."""
import os
import subprocess
import sys

PY = sys.executable
_CODE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_CODE)
SCRIPT = os.path.join(_CODE, "_curve_lowdata.py")
LOG = os.path.join(_ROOT, "logs", "lowdata_full.log")

p = subprocess.Popen(
    [PY, SCRIPT, "--configs", "m3", "H4", "H8"],
    stdout=open(LOG, "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    cwd=_CODE,
    creationflags=0x08000000,          # CREATE_NO_WINDOW
)
print("pid", p.pid)
