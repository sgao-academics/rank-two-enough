"""What ARE the two directions? If the first is the hand-built frame's own axis
and the second is a coupling the frame never encodes, then "rank 2" is a
mechanism, not a fitting artefact.

Reads the fitted factors back into the named observation slots and into the
(step, axis) layout of the plan, averaged over 12 dataset seeds so that one
dataset's accident is not mistaken for structure.
"""
import os

import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import numpy as np                # noqa: E402
import metrics as M               # noqa: E402
import predictors as P            # noqa: E402
import run_suite as RS            # noqa: E402
import split as SP                # noqa: E402
from lowrank_bilinear import LowRankBilinear   # noqa: E402
from sampler import OBS_NAMES      # noqa: E402

spec = M.load_spec()

U, V, signs = [], [], []
per_seed = []
for seed in range(12):
    path = RS.ds_path("m3", seed)
    try:
        ds = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        continue
    rows = SP.assign(ds["episodes"], seed=0)
    ids = {"train": [], "val": [], "test": []}
    for r in rows:
        ids[r["split"]].append(r["episode_id"])
    tab = P.features_from(ds, ids["train"])
    y = P.truth_matrix(ds, "C", ids["train"])
    m = LowRankBilinear(rank=2, alpha=1.0, n_iter=15, seed=0, init="svd",
                        plan_quad=True)
    m.fit(tab, y)
    # fix the sign so that averaging is meaningful: flip each factor pair so its
    # largest-magnitude entry is positive
    u, v = m.u.copy(), m.v.copy()
    for k in range(2):
        j = int(np.argmax(np.abs(u[k])))
        if u[k, j] < 0:
            u[k] *= -1
            v[k] *= -1
    # Per-seed invariants. The averaged factors are NOT quotable for geometry:
    # two near-equal singular values let SVD rotate the pair seed to seed, so the
    # mean of the directions is a blur even though the subspace is stable. These
    # three numbers are intra-seed and therefore rotation-free.
    step1, step2 = v[0, :3], v[0, 3:6]
    per_seed.append(dict(
        seed=seed,
        step_corr=float(np.corrcoef(step1, step2)[0, 1]),
        cos_u=abs(float(u[0] @ u[1]) / (np.linalg.norm(u[0]) * np.linalg.norm(u[1]))),
        cos_v=abs(float(v[0] @ v[1]) / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))),
        z_share=float((v[0, 2] ** 2 + v[0, 5] ** 2 + v[0, 8] ** 2)
                      / np.sum(v[0] ** 2)),
        v1_xy=v[0].tolist(), u_raw=u.tolist(), v_raw=v.tolist()))
    U.append(u)
    V.append(v)

n_fit = len(V)
U = np.mean(U, axis=0)
V = np.mean(V, axis=0)
print("fitted on %d dataset seeds (m3, training split, alpha=1, svd start)\n"
      % n_fit)

for k in range(2):
    print("=" * 74)
    print("direction %d" % (k + 1))
    print("=" * 74)
    order = np.argsort(-np.abs(U[k]))[:8]
    print("  observation weights (top 8 of %d):" % U[k].size)
    for j in order:
        print("    %-14s %+0.4f" % (OBS_NAMES[j], U[k, j]))
    print("  plan weights, as (step, axis):")
    for st in range(3):
        print("    step %d   %s" % (st, " ".join("%+8.4f" % x
                                                 for x in V[k, st * 3:(st + 1) * 3])))
    # alignment with the direction the hand-built frame uses: the unit vector
    # along (obj - target), which lives in slots 12-13
    dirvec = np.zeros(U.shape[1])
    dirvec[12] = 1.0
    dirvec[13] = 1.0
    dirvec /= np.linalg.norm(dirvec)
    print("  |cos| with the frame's object->target axis: %.3f"
          % abs(float(U[k] @ dirvec)))
    print()

print("=" * 74)
print("orthogonality of the two fitted directions")
print("=" * 74)
print("  |cos(u1,u2)| = %.3f" % abs(float(U[0] @ U[1])))
print("  |cos(v1,v2)| = %.3f" % abs(float(V[0] @ V[1])))
print("  step-to-step correlation inside v1: %.3f, v2: %.3f"
      % (float(np.corrcoef(V[0, :3], V[0, 3:6])[0, 1]),
         float(np.corrcoef(V[1, :3], V[1, 3:6])[0, 1])))

import os                                    # noqa: E402
print("\n" + "=" * 74)
print("per-seed invariants (rotation-free)")
print("=" * 74)
print("  %-5s %10s %8s %8s %9s" % ("seed", "step corr", "cos u", "cos v", "z share"))
for r in per_seed:
    print("  %-5d %10.3f %8.3f %8.3f %9.3f"
          % (r["seed"], r["step_corr"], r["cos_u"], r["cos_v"], r["z_share"]))
med = {k: float(np.median([r[k] for r in per_seed]))
       for k in ("step_corr", "cos_u", "cos_v", "z_share")}
print("  median: %s" % ", ".join("%s=%.3f" % (k, v) for k, v in med.items()))

out = {"u": U.tolist(), "v": V.tolist(), "obs_names": list(OBS_NAMES),
       "n_seeds": int(n_fit), "per_seed": per_seed, "median": med}
with open(os.path.join(RS.ROOT, "logs", "lrb_spectrum.json"), "w",
          encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("\nwrote logs/lrb_spectrum.json")
