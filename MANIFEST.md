# Manifest — which file backs which number

Every number in the paper is read from a stored file in `logs/`. This table maps
the paper's elements to the script that produces the file and to the file itself,
so that a reader can check a specific claim without rebuilding the whole suite.

Paths are relative to the repository root. Files ending in `.gz` must be unpacked
first (`python tools/unpack.py <substring>`). Commands assume `tools/unpack.py`
has been run and that the working directory is the repository root.

## Tables

| table | content | produces | stored evidence |
|:--|:--|:--|:--|
| **Table 1** | tail-share law: seven horizons, gain and scalar `R^2` linear in `t/(1-t)` | `code/_scan_tail.py` | `logs/tail_scan/ds_H04..H10_m3.json`, `logs/tail_scan/tail_scan.csv` |
| **Table 2** | reference points on the full matrix, 360 evaluations, six configurations | `code/run_suite.py`, then `code/run_aggregate.py` | `logs/suite/scene2_spec13/ev_*.json`, `logs/suite/scene2_spec13/aggregate.json`, `logs/route_b.json` |
| **Table 3** | block decomposition of the degree-2 expansion | `code/_ablation_terms.py`, `code/_block_combo.py` | `logs/ablation_terms.json`, `logs/block_combo.json` |
| **Table 4** | rank curve on three horizons + the explicit 230/275 rungs | `code/_lrb_final.py --configs m3 H4 H8` | `logs/lrb_final.json` (summary: `logs/lrb_table.json`) |
| **Table 5** | plan-length ladder at a fixed tail share | `code/_lrb_final.py --configs pl1 pl2 pl4 --ranks …` | `logs/lrb_pl1.json`, `logs/lrb_pl2.json`, `logs/lrb_pl4.json` |
| **Table 6** | price in parameters and in data | `code/_curve_lrb.py`, `code/_curve_lowdata.py` | `logs/lrb_lowdata.json`, `logs/route_b_lowdata.json` |

Rank ceilings differ per configuration because the unconstrained block is
`d_obs x 3m`: `pl1` admits ranks 1–3, `pl2` 1–6, `pl4` 1–9. Passing a larger rank
raises inside the fit rather than silently truncating.

## Text claims

| section | claim | produces | stored evidence |
|:--|:--|:--|:--|
| 3 | the simulator is deterministic; replay drift 1.5e-12, noise floor 1e-6 | `code/gate0_replay.py`, `code/_replay_check.py` | `logs/gate0_mujoco_push3d.json` |
| 3 | metric definitions, chance level by permutation, gate results | `code/run_selftest.py`, `code/metrics.py` | `logs/selftest_*.log`, `spec/counterfactual_contract_v1.3.json` |
| 3 | cross-process reproducibility of the split and the fits | `code/check_cross_process.py` | `logs/xproc_A.json`, `logs/xproc_B.json` |
| 4.2 | the zero-parameter score occupies 91% of the metric's range | `code/_route_b.py` | `logs/route_b.json` |
| 4.3 | state curvature moves DAR by -0.0005; plan curvature supplies 29% | `code/_ablation_terms.py` | `logs/ablation_terms.json` |
| 4.4 | the 198-term block and its 62-term rank-two fit agree to 5e-6 | `code/_lrb_vs_bilinear.py` | `logs/lrb_vs_bilinear.json` |
| 4.4 | arm separation along the ladder (0.031 → 0.095) | `code/_gap_table.py` | the datasets themselves |
| 4.5 | the top singular share is 0.502 ± 0.075; plan weights identical across steps (0.9993) | `code/_lrb_spectrum.py` | `logs/lrb_spectrum.json` |
| 4.5 | six orderings per plan move the outcome by 7.5e-4, 0.76% of substitutions flip the leader | `code/_order_counterfactual.py --config m3` | `logs/order_counterfactual.json` |
| 4.5 | at plan length four: sd 5.0e-4, 5.5% of the gap, 0.78% flips, 0.22% joint-limit clips | `code/_order_counterfactual.py --config pl4` | `logs/order_counterfactual_pl4.json` |
| 4.6 | rank-one participation ratio against an arm-shuffled null | `code/_rank1.py`, `code/_rank1_null.py` | `logs/rank1.json`, `logs/rank1_null.json` |
| 4.6 | the fitted rank-two block transfers from convention C to O better than the 560-term expansion | `code/_lrb_transfer.py` | `logs/lrb_transfer.json` |
| 4.6 | penalty sensitivity: tripling the shrinkage on the same 139 coefficients costs 0.0032 | `code/_lrb_sweep.py` | `logs/lrb_sweep.json` |

## Datasets

| directory | content |
|:--|:--|
| `data/suite/scene2_spec13/` | 108 datasets: 9 configurations x 12 seeds x 96 episodes |
| `data/pilot_*.json.gz` | the four pilot datasets, kept for the record; no published number depends on them |

Each dataset carries its own provenance in `meta`: the configuration it was
generated under (`H`, `plan_len`, `proposal_sigma`), the scene tag, the episode
count and the dataset seed. `ensure_dataset()` in `code/run_suite.py` refuses to
reuse a file whose `meta` does not match the requested configuration, so a stale
cache cannot be mistaken for the right dataset.

## Regenerating from scratch

```bash
python tools/unpack.py
python code/run_suite.py --configs m3 --seeds 0 1        # datasets + evaluations
python code/run_aggregate.py                             # aggregate over splits
python code/_lrb_final.py --configs m3 H4 H8 --out lrb_final.json
python code/_order_counterfactual.py --config m3
```

The commands in this file are the ones that were used; the ones that were re-run
from a clean copy of this tree are listed in the README's verification table.
