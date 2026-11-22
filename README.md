# Replication package — *Rank Two Is Enough*

Code, datasets and stored results for

> **Rank Two Is Enough: Measuring and Matching the Interaction Structure of
> Counterfactual Plan Ranking**
> Shuaidong Gao, Chongqing Institute of Foreign Studies
> ORCID 0009-0004-5641-3581

The manuscript and the figure files are **not** part of this repository; see
[What is deliberately absent](#what-is-deliberately-absent). Everything else that
a published number depends on is here: the simulator and sampler that generate
the task, the analysis scripts, the datasets the models were fitted on, and the
result files the tables were read from.

---

## What the package contains

```
code/        59 Python modules: simulator, sampler, metrics, analysis scripts
spec/        the frozen measurement contract (v1.3), English
data/        9 suite configurations x 12 scene seeds + 4 pilot datasets, gzipped
logs/        stored results: evaluation payloads, tail sweep, rank curves,
             order counterfactual, determinism gates (gzipped where large)
tools/       unpack.py
MANIFEST.md  which script writes which stored number
```

Sizes: the clone is about **136 MB**; unpacked the payload is about **412 MB**.

## Quick start

```bash
git clone https://github.com/sgao-academics/rank-two-enough.git
cd rank-two-enough

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tools/unpack.py                 # 484 files, about 412 MB written
python tools/unpack.py pl4             # or only the files you need
python code/_gap_table.py              # arm separation per configuration
```

Two worked examples, both of which print a number that appears in the paper:

```bash
# Table 5, first row: the rank curve at plan length one
python code/_lrb_final.py --configs pl1 --ranks 1 2 3 --out lrb_pl1_rebuilt.json

# Section 4.5: permute each plan's steps, hold its multiset fixed
python code/_order_counterfactual.py --config m3
```

`_lrb_final.py --out` writes into `logs/`; keep the `_rebuilt` suffix so the
stored file the paper quotes is never overwritten.

## Verified from this package

These were run from a clean copy of this tree, against the shipped datasets:

| command | result | paper |
|:--|:--|:--|
| `_lrb_final.py --configs pl1 --ranks 1 2 3` | 0.772201 / 0.941295 / 0.941573 | Table 5, row m=1; byte-identical to `logs/lrb_pl1.json` |
| `_replay_check.py` | worst abs. deviation 1.5e-12 (obs), 2.2e-16 (phi) | Section 3: the simulator reproduces stored trajectories |
| `_gap_table.py` | 0.0307 / 0.0476 / 0.0763 / 0.0946 at m=1..4 | Section 4.4: arm separation along the plan-length ladder |
| `_order_counterfactual.py --config pl2 --smoke` | 120 control steps, 0 joint-limit clips | Section 4.5 test path |

## The task in one paragraph

A planar pusher moves a box toward a target on a table. At each decision point the
state is shared by five candidate **plans**, each a sequence of `plan_len` action
increments drawn as a persistent bias plus a per-step alternating term; a frozen
base policy executes the chosen plan for its `plan_len` steps and finishes the
horizon. The outcome is the normalised signed potential gain over `H` steps. The
metric, DAR, is the share of decision points at which the model ranks the arm whose
plan produced the larger gain first. Two conventions are reported: **C** executes
the model's chosen plan (closed loop), **O** replays the factual plan for the tail
(open loop).

## Configurations

Nine configurations, each with twelve scene seeds, 96 episodes per dataset, 16
scene templates split 11 / 2 / 3 (fit / select / score) and five template
partitions, so one cell of a table is one (seed, partition) pair and a table row
is 60 cells.

| name | m | H | tail share t=(H-m)/H | perturbation scale | role |
|:--|--:|--:|--:|--:|:--|
| `m3` | 3 | 6 | 0.500 | 0.8 | primary configuration |
| `m1` | 1 | 6 | 0.833 | 0.8 | plan-length ablation |
| `H4` | 3 | 4 | 0.250 | 0.8 | shorter horizon |
| `H8` | 3 | 8 | 0.625 | 0.8 | longer horizon |
| `sig04` | 3 | 6 | 0.500 | 0.4 | dispersion sensitivity |
| `sig16` | 3 | 6 | 0.500 | 1.6 | dispersion sensitivity |
| `pl1` | 1 | 2 | 0.500 | 0.8 | plan-length ladder, t held fixed |
| `pl2` | 2 | 4 | 0.500 | 0.8 | plan-length ladder |
| `pl4` | 4 | 8 | 0.500 | 0.8 | plan-length ladder |

The ladder holds `H = 2m` so that the tail share is constant and the only axis that
moves is how many steps a decision is cut into. `H4` and `H8` move the tail share at
fixed plan length; the tail law of Table 1 is a seven-point sweep of the same kind
(`tail_scan/`, `H = 4..10` at `m = 3`).

## The frozen measurement contract

`spec/counterfactual_contract_v1.3.json` is the machine-readable contract every
reported number is computed under. `metrics.py` reads it; nothing in the analysis
code hard-codes a metric parameter. The fields that matter most:

* **outcome** — signed, normalised potential gain, no zero floor (a floor collapses
  inferior arms together and degrades DAR to a five-way classification);
* **degenerate decision** — the five arms end within `eps` of each other, or all
  outcomes equal the factual arm's; such cells are excluded and the exclusion rate
  is reported;
* **noise floor** — `1e-06` for a deterministic simulator whose replay drift is
  `1.5e-12`; a difference below the floor is not a difference;
* **chance level** — computed from the data by permuting `y_true` 200 times, not
  from the analytic `p^2+(1-p)^2`, which ignores the within-row correlation and
  measures 0.917 where permutation measures 0.493;
* **gates** — a ruler self-test (`ruler_selftest`) runs before any experiment; its
  criterion is that an uninformative predictor must not score above 0.75.

`code/run_selftest.py` exercises the contract and the adapter.

## Determinism and seeds

The physics is deterministic: given a saved state, a rollout reproduces the stored
trajectory to `1.5e-12` in observation space (`logs/gate0_mujoco_push3d.json`,
reproduced by `_replay_check.py`). Every dataset is generated by `run_suite.py`
under `numpy.random.default_rng(seed)` with `seed` in `0..11`; splits are
deterministic functions of the template index and the partition id. There is no
sampling anywhere in the analysis scripts, so the rank curves, the order test and
the block decomposition are exactly reproducible; the only variability in the
paper is across seeds and partitions, which is what the reported spreads measure.

## Cost of a full rebuild

Measured on the machine that produced the paper (AMD Ryzen 9, 16 cores; no GPU is
used by any analysis script):

| step | cost |
|:--|:--|
| generate one dataset (`96` episodes) | 22 s at m=1, 35 s at m=2, 62 s at m=4 |
| the 36 ladder datasets | about 14 min in parallel |
| a rank curve over 60 cells | 80 s (3 ranks) to about 13 min (9 ranks at m=4) |
| the order counterfactual, 1440 decision points | about 2 min at m=3 (6 orderings), about 10 min at m=4 (24 orderings) |
| replay determinism check | under 1 min |

## What is deliberately absent

* **The manuscript.** No `.tex`, no `.pdf`.
* **Figures and figure scripts.** Every figure in the paper is a redraw of numbers
  that are stored in `logs/`; the plotting code is not distributed because it
  depends on a journal style package. The numbers themselves are all here.
* **Internal notes.** The planning notes, the review reports and the Chinese
  working documents are not part of the scientific record and are not published.

## Citing

If you use this package, cite the paper. A `CITATION.cff` is included for
reference managers.

## License

Code: MIT. Datasets, contract and stored results: CC BY 4.0. See `LICENSE.md`.
