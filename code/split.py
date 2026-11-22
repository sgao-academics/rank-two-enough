"""
split.py -- deterministic train/val/test split with leak assertions.

Spec (1.0, split): unit = scene_instance_and_task_template, ratio 70/15/15, and
three assertions that must hold: no test scene instance in train, no test task
template in train, unseen object-combination rate > 0.

Why a hash and not random.choice
--------------------------------
A shuffled split depends on the order rows happen to be generated in, so adding
one episode silently reshuffles everything and old numbers stop being comparable.
Hashing the *identity* of the unit is order-independent: the same episode lands in
the same split forever, whatever else is generated.

The assertions are checks on this module's own output, not documentation. A split
that leaks is worse than no split at all, because the resulting test number looks
better and nobody can tell.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

SPLITS: Tuple[str, str, str] = ("train", "val", "test")
DEFAULT_RATIO: Tuple[float, float, float] = (0.70, 0.15, 0.15)


def _u01(key: str, seed: int) -> float:
    """Hash a key to a uniform in [0,1). Independent of insertion order."""
    h = hashlib.sha256(("%d|%s" % (seed, key)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(1 << 32)


def split_of(key: str, seed: int = 0,
             ratio: Sequence[float] = DEFAULT_RATIO) -> str:
    u = _u01(key, seed)
    lo = float(ratio[0])
    mid = lo + float(ratio[1])
    if u < lo:
        return "train"
    if u < mid:
        return "val"
    return "test"


def template_key(factors: Mapping[str, object]) -> str:
    """The task template: which quadrant the target and the distractor occupy.

    Quadrant rather than exact coordinates on purpose -- that is what makes a
    template *reusable* (the same relational pattern with different numbers), and
    only reusable templates can be "unseen but in-distribution" later.
    """
    tx, ty = factors["target_xy"]                      # type: ignore[assignment]
    dx, dy = factors.get("distractor_xy", (0.0, 0.0))  # type: ignore[assignment]
    return "t%s%s:d%s%s" % ("+" if tx >= 0 else "-", "+" if ty >= 0 else "-",
                            "+" if dx >= 0 else "-", "+" if dy >= 0 else "-")


def instance_key(factors: Mapping[str, object]) -> str:
    """The scene instance: the exact numeric factor draw."""
    tx, ty = factors["target_xy"]                      # type: ignore[assignment]
    return "%s|m=%.4f|s=%.4f|f=%.3f|t=%.4f,%.4f" % (
        template_key(factors), float(factors["mass"]),
        float(factors["half_size"]), float(factors["friction"]), tx, ty)


def assign(episodes: Iterable[Mapping[str, object]], *, seed: int = 0,
           split_on: str = "template",
           ratio: Sequence[float] = DEFAULT_RATIO) -> List[Dict[str, object]]:
    """Attach a `split` field to each episode dict (copy, does not mutate).

    split_on="template" assigns whole templates to one split -- the strict reading
    of the spec, and the only one that supports an "unseen template" claim.
    split_on="instance" assigns single scenes, which would let near-duplicate
    instances of one template straddle train and test. Do not use it for the
    headline numbers.

    TEMPLATE ASSIGNMENT IS STRATIFIED BY RANK, not per-key thresholded. With 16
    templates and a 70/15/15 ratio, independent per-template draws gave splits like
    train=66, val=0, test=30 (measured), i.e. a split with no validation set at all
    and half the data in test. Ranking the templates by their hash and cutting the
    sorted list at the ratio keeps the two properties that matter -- order
    independence (the rank of a hash does not depend on what else was generated) and
    determinism -- while guaranteeing each split its intended share and making split
    seeds comparable to one another.
    """
    eps = list(episodes)
    out: List[Dict[str, object]] = []

    if split_on != "template":
        for ep in eps:
            fac = ep["factors"]                       # type: ignore[index]
            key = instance_key(fac)
            row = dict(ep)
            row["split"] = split_of(key, seed=seed, ratio=ratio)
            row["split_key"] = key
            row["template"] = template_key(fac)
            out.append(row)
        return out

    tpl = sorted({template_key(ep["factors"]) for ep in eps})   # type: ignore[index]
    ranked = sorted(tpl, key=lambda t: _u01(t, seed))
    n = len(ranked)
    n_tr = int(round(float(ratio[0]) * n))
    n_va = int(round(float(ratio[1]) * n))
    # never hand out an empty split: with n=16 the rounding above is fine, but a
    # 4-template task would otherwise produce val=0 and silently disable the
    # "select on val" protocol.
    n_tr = max(1, min(n_tr, n - 2))
    n_va = max(1, min(n_va, n - n_tr - 1))
    part = {}
    for i, t in enumerate(ranked):
        part[t] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    for ep in eps:
        fac = ep["factors"]                           # type: ignore[index]
        key = template_key(fac)
        row = dict(ep)
        row["split"] = part[key]
        row["split_key"] = key
        row["template"] = key
        out.append(row)
    return out


def check_no_leak(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Verify the three assertions. Returns a report; `ok=False` means stop."""
    def keys(field: str, split: str) -> set:
        return {str(r[field]) for r in rows if r["split"] == split}

    tr_tpl, te_tpl, va_tpl = (keys("template", s) for s in SPLITS)
    tr_ins, te_ins = keys("split_key", "train"), keys("split_key", "test")

    template_leak = sorted(tr_tpl & te_tpl)
    # instance keys are prefixed by template, so a template leak implies an
    # instance leak; check them separately anyway so the report is explicable
    instance_leak = sorted(tr_ins & te_ins)

    n_train_tpl = len(tr_tpl)
    # "unseen object-combination rate > 0": with template-level splitting, EVERY
    # test episode is on an unseen template, so the rate is 1.0 by construction --
    # provided the test split is non-empty.
    n_test = sum(1 for r in rows if r["split"] == "test")
    unseen_rate = 1.0 if n_test and template_leak == [] else 0.0

    ok = (not template_leak) and (not instance_leak) and unseen_rate > 0.0 and n_train_tpl > 0
    return {
        "ok": bool(ok),
        "n_train_templates": n_train_tpl,
        "n_val_templates": len(va_tpl),
        "n_test_templates": len(te_tpl),
        "template_leak": template_leak,
        "instance_leak": instance_leak,
        "unseen_object_combination_rate": unseen_rate,
        "n_episodes": {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS},
        "counts": len(rows),
    }
