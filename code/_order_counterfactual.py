"""The interventional version of the order invariance: permute a plan's steps.

Section 7.2 of the manuscript reads the invariances off fitted factors, which makes
them statements about the task *as the training data present it*. The interventional
version is one edit to the suite, and this script is that edit:

    hold a plan's multiset of actions fixed, execute its m steps in a different
    order, let the frozen policy finish the horizon, and ask whether the outcome
    moves.

Every permutation of the same m actions has the same cumulative displacement to
machine precision, so an order-blind task returns the same outcome for all 3! = 6
orderings and any spread above the noise floor is order information.

Ground truth is deterministic (`logs/gate0_mujoco_push3d.json`: max replay drift
1.5e-12, eps = 1e-6), so the threshold is a measured one rather than a convention.

Two levels are reported, because they can disagree:

  * the outcome spread within one plan's six orderings, in Y units;
  * whether any ordering flips the ranking of the K arms -- the level at which DAR
    would actually move.

Usage
    python _order_counterfactual.py --smoke
    python _order_counterfactual.py                       # 12 seeds x 8 episodes
    python _order_counterfactual.py --seeds 0 1 --episodes 4
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from itertools import permutations
from statistics import mean

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)

import mujoco_adapter as MU          # noqa: E402
import sampler as S                  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_GLOB = os.path.join(_ROOT, "data", "suite", "scene2_spec13",
                         "ds_%s_s%02d_e96.json")
OUT = os.path.join(_ROOT, "logs", "order_counterfactual%s.json")
EPS_FLOOR = 1e-6


def step_checked(ad, a):
    """Step once, reporting whether the setpoint hit a joint limit (which would
    silently break the 'same cumulative displacement' premise of the test)."""
    nxt = ad._ctrl + ad.action_scale * np.asarray(a, dtype=np.float64).ravel()
    hit = bool(np.any(nxt < ad._ctrl_lo - 1e-15) or np.any(nxt > ad._ctrl_hi + 1e-15))
    ad.step(a)
    return hit


def run(args) -> int:
    t0 = time.time()
    perms = None          # set from the dataset's own plan length
    m_seen = None

    rows = []            # one row per (seed, episode, t, arm)
    clip_hits = 0
    n_steps = 0

    for seed in args.seeds:
        path = DATA_GLOB % (args.config, seed)
        if not os.path.exists(path):
            print("  ! missing %s" % os.path.basename(path))
            continue
        ds = json.load(io.open(path, encoding="utf-8"))
        cfg = ds["meta"]["config"]
        H, warmup = int(cfg["H"]), int(cfg["warmup"])
        m = int(cfg["plan_len"])
        # The permutation set comes from the data, not from a constant: at m=4
        # there are 24 orderings and the per-plan spread is an extremum over a
        # larger sample, so max-min is not comparable across m. The comparable
        # statistic is the within-plan standard deviation, reported below next to
        # the spread the manuscript quotes at m=3.
        if perms is None:
            m_seen = m
            perms = list(permutations(range(m)))
            print("  m=%d  H=%d  %d orderings per plan  %d seeds x %d episodes"
                  % (m, H, len(perms), len(args.seeds), args.episodes), flush=True)
        elif m != m_seen:
            raise SystemExit("mixed plan lengths in one run: m=%d and m=%d" % (m_seen, m))
        substeps = int(cfg["control_substeps"])

        for ep in ds["episodes"][: args.episodes]:
            factors = dict(ep["factors"])
            target_xy = factors["target_xy"]
            ad = MU.MuJoCoPush3D(substeps=substeps)
            ad.set_factors(**factors)
            ad.reset(int(ep["seed"]))
            for _ in range(warmup):
                step_checked(ad, S.base_policy(ad, target_xy))

            for st in ep["steps"]:
                t = int(st["t"])
                if t >= args.steps:
                    continue
                state = ad.save_state()
                phi_now = ad.outcome()
                plans = np.asarray(st["plans"], dtype=np.float64)     # (K, m, 3)
                y_store = np.asarray(st["y_C"], dtype=np.float64)

                # -- the six orderings of each arm's own plan -- #
                phi_by_arm = np.empty((plans.shape[0], len(perms)))
                for k in range(plans.shape[0]):
                    for j, p in enumerate(perms):
                        ad.load_state(state)
                        for h in p:
                            clip_hits += int(step_checked(ad, plans[k][h]))
                            n_steps += 1
                        for _ in range(H - m):
                            clip_hits += int(step_checked(ad, S.base_policy(ad, target_xy)))
                            n_steps += 1
                        phi_by_arm[k, j] = ad.outcome()

                spread = phi_by_arm.max(axis=1) - phi_by_arm.min(axis=1)
                # ordering 0 is the original sequence; check it still reproduces
                # the stored phi_C so that a spread cannot be a replay artifact
                replay_dev = float(np.max(np.abs(phi_by_arm[:, 0]
                                                 - np.asarray(st["phi_C"], dtype=np.float64))))

                # -- does any ordering flip the ranking of the arms? -- #
                # baseline: every arm in its original order (convention C)
                base_order = phi_by_arm[:, 0]
                flips = 0
                trials = 0
                for k in range(plans.shape[0]):
                    for j in range(1, len(perms)):
                        trial = base_order.copy()
                        trial[k] = phi_by_arm[k, j]
                        trials += 1
                        flips += int(int(np.argmax(trial)) != int(np.argmax(base_order)))

                sd_by_arm = phi_by_arm.std(axis=1, ddof=0)

                rows.append({
                    "seed": seed, "episode": ep["episode_id"], "t": t,
                    "spread": spread.tolist(),
                    "sd": sd_by_arm.tolist(),
                    "spread_max": float(spread.max()),
                    "phi_by_arm": phi_by_arm.tolist(),
                    "y_store": y_store.tolist(),
                    "replay_dev": replay_dev,
                    "flips": flips, "trials": trials,
                    # the decision signal this spread has to be judged against
                    "arm_gap": float(base_order.max() - base_order.min()),
                })

                # advance exactly as run_episode does
                ad.load_state(state)
                step_checked(ad, plans[0][0])

            print("  [s%02d %s] %d rows, %.0fs" % (seed, ep["episode_id"],
                                                   len(rows), time.time() - t0), flush=True)

    if not rows:
        print("no decision points were visited -- check the data glob and --steps")
        return 2

    allspread = np.array([r["spread_max"] for r in rows])
    allflat = np.array([s for r in rows for s in r["spread"]])
    allsd = np.array([s for r in rows for s in r["sd"]])
    gaps = np.array([r["arm_gap"] for r in rows])
    devs = np.array([r["replay_dev"] for r in rows])
    flips = sum(r["flips"] for r in rows)
    trials = sum(r["trials"] for r in rows)

    print("\n" + "=" * 78)
    print("ORDER PERMUTATION TEST  (m=%d, all %d orderings, frozen-policy tail)"
          % (m_seen, len(perms)))
    print("=" * 78)
    print("decision points %d   rows %d   rollouts %d"
          % (len(rows), len(allflat), n_steps))
    print("noise floor %.1e   replay dev max %.2e" % (EPS_FLOOR, float(devs.max())))
    print("\nper-plan outcome spread over its %d orderings (Y units):" % len(perms))
    print("  median %.3e   mean %.3e   95%% %.3e   max %.3e"
          % (float(np.median(allflat)), float(allflat.mean()),
             float(np.percentile(allflat, 95)), float(allflat.max())))
    print("  fraction above the 1e-6 floor: %.3f" % float((allflat > EPS_FLOOR).mean()))
    print("\nwithin-plan standard deviation over the orderings (comparable across m):")
    print("  median %.3e   mean %.3e   95%% %.3e"
          % (float(np.median(allsd)), float(allsd.mean()),
             float(np.percentile(allsd, 95))))
    print("\nper-decision-point worst-arm spread: median %.3e  max %.3e"
          % (float(np.median(allspread)), float(allspread.max())))
    print("\nthe decision signal this is judged against:")
    print("  gap between best and worst arm at the same decision point:")
    print("  median %.3e   mean %.3e" % (float(np.median(gaps)), float(gaps.mean())))
    print("  ratio median spread / median gap = %.4f"
          % (float(np.median(allflat)) / max(float(np.median(gaps)), 1e-30)))
    print("\nranking flips from reordering one arm: %d of %d trials (%.3f%%)"
          % (flips, trials, 100.0 * flips / max(trials, 1)))
    print("\njoint-limit clipping during the test: %d of %d steps (%.4f%%)"
          % (clip_hits, n_steps, 100.0 * clip_hits / max(n_steps, 1)))

    out = {
        "settings": {"config": args.config, "m": m_seen, "perms": len(perms),
                     "seeds": args.seeds, "episodes": args.episodes,
                     "steps": args.steps},
        "n_decision_points": len(rows),
        "n_rollouts_steps": n_steps,
        "noise_floor": EPS_FLOOR,
        "replay_dev_max": float(devs.max()),
        "spread": {"median": float(np.median(allflat)), "mean": float(allflat.mean()),
                   "p95": float(np.percentile(allflat, 95)),
                   "max": float(allflat.max()),
                   "frac_above_floor": float((allflat > EPS_FLOOR).mean())},
        "sd": {"median": float(np.median(allsd)), "mean": float(allsd.mean()),
               "p95": float(np.percentile(allsd, 95))},
        "arm_gap": {"median": float(np.median(gaps)), "mean": float(gaps.mean())},
        "flips": {"count": int(flips), "trials": int(trials),
                  "rate": float(flips / max(trials, 1))},
        "clip": {"hits": int(clip_hits), "steps": int(n_steps),
                 "rate": float(clip_hits / max(n_steps, 1))},
        "rows": rows,
    }
    if args.smoke:
        print("\n(smoke run: result file not written)")
        return 0

    # m=3 keeps the file name the manuscript cites; other configurations get a
    # suffix so that the two runs cannot overwrite each other.
    out_path = OUT % ("" if args.config == "m3" else "_" + args.config)
    with io.open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="m3",
                    help="suite configuration to draw datasets from")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.episodes, a.steps = [0], 1, 3
    raise SystemExit(run(a))
