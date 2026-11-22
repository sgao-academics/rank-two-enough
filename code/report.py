"""
report.py -- one schema for every scored result this project emits.

Why a separate file
-------------------
`run_pilot.py` used to assemble the JSON inline. That works exactly once. The
moment a second script wants to write results, the two payloads drift: one carries
the environment fingerprint and the other does not, one records `phi_ref` and the
other leaves it to be re-derived. Then two numbers differ by 0.01 and there is no
way to tell whether the method changed or the simulator did.

Every payload written here answers four questions:

    which spec version?   -> spec_version (+ the change-control table in spec)
    which environment?    -> fingerprint (MuJoCo version, solver, integrator, cone)
    which normaliser?     -> phi_ref  (Y is meaningless without it)
    which split?          -> train/val/test episode ids and the leak report

A number missing any of those four is not a result, it is an anecdote.
"""

from __future__ import annotations

import json
import os
import platform
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

import metrics as M


SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def environment(spec: Optional[dict] = None, adapter=None) -> dict:
    """Everything needed to reproduce the exact numbers in a payload."""
    import scipy
    spec = spec or M.load_spec()
    env = {
        "schema_version": SCHEMA_VERSION,
        "spec_version": spec["spec_version"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
    if adapter is not None:
        env["adapter"] = getattr(adapter, "name", type(adapter).__name__)
        try:
            env["fingerprint"] = adapter.fingerprint()
        except AttributeError:
            env["fingerprint"] = None
    return env


# --------------------------------------------------------------------------- #
# per-predictor summary
# --------------------------------------------------------------------------- #
_METRIC_FIELDS = ("dar", "nmae", "rho", "tau", "regret", "regret_norm",
                  "zero_regret_rate", "undecidable_rate", "nontrivial_step_rate")


def summarize(rep: dict, label: str) -> dict:
    """Flatten one `metrics.evaluate` report into a row a table can hold."""
    m = rep["metrics"]
    out = {"label": label,
           "n_episodes": rep["n_episodes"], "n_steps": rep["n_steps"],
           "degenerate_rate": rep["degenerate_rate"],
           "eps": rep["eps"], "eps_source": rep["eps_source"],
           "n_boot": rep["n_boot"]}
    for f in _METRIC_FIELDS:
        cell = m.get(f)
        if cell is None:
            out[f] = None
            out[f + "_ci95"] = None
            continue
        v = cell.get("value")
        out[f] = None if v is None else float(v)
        out[f + "_ci95"] = cell.get("ci95")
    p90 = (m.get("regret_p90") or {}).get("value")
    out["regret_p90"] = None if p90 is None else float(p90)
    return out


def summarize_ruler(ruler: dict) -> dict:
    """The Gate-2 self-test, reduced to the fields a reader will actually check."""
    return {
        "passed": bool(ruler["passed"]),
        "oracle_dar": ruler["oracle"]["dar"],
        "oracle_nmae": ruler["oracle"]["nmae"],
        "oracle_regret_norm": ruler["oracle"]["regret_norm"],
        "chance_dar_perm_mean": ruler["chance"]["dar_perm_mean"],
        "chance_dar_perm_sd": ruler["chance"]["dar_perm_sd"],
        "chance_rho": ruler["chance"]["rho"],
        "permutations": ruler["chance"]["permutations"],
        "p_positive": ruler["chance"]["p_positive"],
    }


# --------------------------------------------------------------------------- #
# payload
# --------------------------------------------------------------------------- #
def pack_run(*, label: str, spec: dict, adapter, dataset: dict, split: dict,
             estimates: Iterable[dict], checks: Dict[str, bool],
             config: Optional[dict] = None,
             eval_split: str = "test",
             extra: Optional[dict] = None) -> dict:
    """Assemble one payload. `estimates` are the output of `summarize`."""
    meta = dataset.get("meta", {})
    payload = {
        "run_label": label,
        "env": environment(spec, adapter),
        "dataset": {
            "phi_ref": meta.get("phi_ref"),
            "K": meta.get("K"), "H": meta.get("H"), "T": meta.get("T"),
            "rule": meta.get("rule"),
            # v1.3: a plan unit is not comparable with a single-action unit even at
            # the same K and H, so the payload has to say which one produced it.
            "decision_unit": meta.get("decision_unit"),
            "plan_len": meta.get("plan_len"),
            "plan_tail": meta.get("plan_tail"),
            # Measured worst |arm0 rollout - factual rollout| in potential units.
            # Delta is taken relative to arm 0, so this is how far the reported
            # baseline can be from the one actually used. 0.0 would be a lie.
            "arm0_max_dev": meta.get("arm0_max_dev"),
            "n_episodes": len(dataset.get("episodes", [])),
            "n_steps": int(sum(len(ep["steps"]) for ep in dataset.get("episodes", []))),
            "conventions": meta.get("conventions"),
            "base_policy": meta.get("base_policy"),
        },
        "split": split,
        "eval_split": eval_split,
        "config": config or {},
        "estimates": list(estimates),
        "checks": {str(k): bool(v) for k, v in checks.items()},
        "all_checks_passed": bool(all(bool(v) for v in checks.values())),
    }
    if extra:
        payload["extra"] = extra
    return payload


def dump(payload: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    return os.path.abspath(path)


def read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# console view
# --------------------------------------------------------------------------- #
def table(estimates: Sequence[dict]) -> str:
    """Fixed-width text table; DAR first because it is the primary metric."""
    head = "%-30s %6s %8s %8s %8s %8s" % ("predictor", "DAR", "nMAE", "regret", "rho", "deg%")
    lines = [head, "-" * len(head)]
    for e in estimates:
        def f(key, fmt):
            v = e.get(key)
            return "   -   " if v is None else fmt % v
        lines.append("%-30s %6s %8s %8s %8s %8s" % (
            e["label"][:30], f("dar", "%6.4f"), f("nmae", "%8.4f"),
            f("regret_norm", "%8.4f"), f("rho", "%+8.4f"),
            "  -  " if e.get("degenerate_rate") is None else "%5.1f" % (100 * e["degenerate_rate"])))
    return "\n".join(lines)
