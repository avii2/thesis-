# HAI 21.03 Dataset Deep Study

This report studies the local HAI 21.03 files used for Cluster 1. It does not modify raw data and does not use held-out test files for preprocessing, threshold tuning, balancing, clustering, or ratio selection.

## Target Label Decision

- Recommended target: `attack` as one binary attack-vs-normal label.
- Auxiliary attack columns present: `attack_P1`, `attack_P2`, `attack_P3`.
- `attack == OR(attack_P1, attack_P2, attack_P3)`: False.
- Mismatch rows between `attack` and auxiliary OR: 179.
- Rows where more than one auxiliary attack column is active at the same time: 1,091.
- Attack rows with no auxiliary attack flag: 179.
- Normal rows with any auxiliary attack flag: 0.

Conclusion: use single-label binary intrusion detection with `attack`. `attack_P1`, `attack_P2`, and `attack_P3` are auxiliary labels/metadata and must be excluded from model inputs. They should not be used as model features. They should only become targets if the thesis scope changes from binary detection to multi-output process localization.

## Raw File Audit

| file | rows | attack=0 | attack=1 | attack rate | read from | columns | expected counts match |
|---|---:|---:|---:|---:|---|---:|---|
| `train1.csv` | 216,001 | 216,001 | 0 | 0% | .csv | 84 | True |
| `train2.csv` | 226,801 | 226,801 | 0 | 0% | .csv | 84 | True |
| `train3.csv` | 478,801 | 478,801 | 0 | 0% | .csv | 84 | True |
| `test1.csv` | 43,201 | 42,572 | 629 | 1.456% | .csv | 84 | True |
| `test2.csv` | 118,801 | 115,352 | 3,449 | 2.903% | .csv | 84 | True |
| `test3.csv` | 108,001 | 106,466 | 1,535 | 1.421% | .csv | 84 | True |
| `test4.csv` | 39,601 | 38,444 | 1,157 | 2.922% | .csv | 84 | True |
| `test5.csv` | 92,401 | 90,224 | 2,177 | 2.356% | .csv | 84 | True |

## Schema

- Common column count: 84
- Telemetry feature count after dropping leakage labels/time: 79
- Leakage columns to drop from model input: `attack`, `attack_P1`, `attack_P2`, `attack_P3`, `time`
- Non-numeric common columns excluded from telemetry: none

Process feature counts:
- `P1`: 38 features
- `P2`: 22 features
- `P3`: 7 features
- `P4`: 12 features

Feature type counts from observed values:
- `binary_or_boolean`: 11 features
- `continuous_numeric`: 56 features
- `low_cardinality_numeric`: 12 features

## Split-Level Imbalance

| split | source files | rows | attack=0 | attack=1 | attack rate |
|---|---|---:|---:|---:|---:|
| `train_source` | `train1.csv`, `train2.csv`, `train3.csv`, `test1.csv`, `test2.csv` | 1,083,605 | 1,079,527 | 4,078 | 0.3763% |
| `validation_test3` | `test3.csv` | 108,001 | 106,466 | 1,535 | 1.421% |
| `heldout_test4_test5` | `test4.csv`, `test5.csv` | 132,002 | 128,668 | 3,334 | 2.526% |
| `all_rows` | `train1.csv`, `train2.csv`, `train3.csv`, `test1.csv`, `test2.csv`, `test3.csv`, `test4.csv`, `test5.csv` | 1,323,608 | 1,314,661 | 8,947 | 0.676% |

Important: `train1.csv`, `train2.csv`, and `train3.csv` are fully normal. Under the leakage-safe split, positive training rows/windows come only from `test1.csv` and `test2.csv`.

## Window-Level Imbalance

Window rule: length=32, stride=8, label=positive if any row inside the window has `attack=1`.

| file | windows | positive windows | negative windows | positive window rate | max attack rows/window |
|---|---:|---:|---:|---:|---:|
| `train1.csv` | 26,997 | 0 | 26,997 | 0% | 0 |
| `train2.csv` | 28,347 | 0 | 28,347 | 0% | 0 |
| `train3.csv` | 59,847 | 0 | 59,847 | 0% | 0 |
| `test1.csv` | 5,397 | 98 | 5,299 | 1.816% | 32 |
| `test2.csv` | 14,847 | 507 | 14,340 | 3.415% | 32 |
| `test3.csv` | 13,497 | 224 | 13,273 | 1.66% | 32 |
| `test4.csv` | 4,947 | 165 | 4,782 | 3.335% | 32 |
| `test5.csv` | 11,547 | 316 | 11,231 | 2.737% | 32 |

Training-source windows before balancing: 605 positive and 134,830 negative.
Validation windows: 224 positive and 13,273 negative.
Held-out test windows: 481 positive and 16,013 negative.

## Feature Quality Summary

- Features with missing or non-finite values: 0
- Globally constant features: 19 (`P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P2_ASD`, `P2_AutoGO`, `P2_MSD`, `P2_ManualGO`, `P2_RTR`, `P2_TripEx`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P3_LH`, `P3_LL`)
- Constant within training-source files: 19 (`P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P2_ASD`, `P2_AutoGO`, `P2_MSD`, `P2_ManualGO`, `P2_RTR`, `P2_TripEx`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P3_LH`, `P3_LL`)
- Highly correlated feature pairs with |r| >= 0.98: 19

Top univariate features by directionless ROC-AUC against the binary `attack` label. This is diagnostic only, not a feature-selection decision by itself:

| feature | process | AUC direct | AUC directionless | best AP | SMD attack-vs-normal |
|---|---|---:|---:|---:|---:|
| `P1_PCV02Z` | `P1` | 0.754049 | 0.754049 | 0.0858291 | 0.355387 |
| `P2_VYT03` | `P2` | 0.381057 | 0.618943 | 0.0538273 | -0.275233 |
| `P2_VXT03` | `P2` | 0.602698 | 0.602698 | 0.0459533 | 0.0907988 |
| `P1_FCV03D` | `P1` | 0.400789 | 0.599211 | 0.103257 | -0.302935 |
| `P1_FCV03Z` | `P1` | 0.401342 | 0.598658 | 0.102604 | -0.30644 |
| `P2_VYT02` | `P2` | 0.416528 | 0.583472 | 0.0541761 | -0.209224 |
| `P1_FT03Z` | `P1` | 0.422095 | 0.577905 | 0.0384418 | -0.245681 |
| `P1_FT03` | `P1` | 0.422497 | 0.577503 | 0.0393945 | -0.239288 |
| `P1_B3005` | `P1` | 0.423528 | 0.576472 | 0.00882355 | -0.22329 |
| `P1_LCV01D` | `P1` | 0.570816 | 0.570816 | 0.278851 | 0.517843 |

Largest validation mean shifts compared with training-source rows, in training standard-deviation units. Features constant in the training source are excluded from this ranking:

| feature | validation shift | train mean | validation mean |
|---|---:|---:|---:|
| `P2_VYT03` | -1.70838 | 6.19632 | 5.48313 |
| `P2_VYT02` | -1.12472 | 3.94371 | 3.36206 |
| `P1_B2004` | -0.749634 | 0.0945965 | 0.0820142 |
| `P1_B4002` | -0.670949 | 32.3478 | 31.9503 |
| `P1_B4022` | -0.564111 | 36.0913 | 35.7055 |
| `P2_SIT02` | -0.395745 | 783.69 | 778.865 |
| `P2_SIT01` | -0.395641 | 783.69 | 778.865 |
| `P1_FCV01Z` | -0.386992 | 55.7695 | 39.0473 |
| `P1_FCV01D` | -0.386377 | 55.9463 | 39.2597 |
| `P2_24Vdc` | -0.38461 | 28.0286 | 28.0273 |

## Attack Episodes

| file | label | episodes | total rows | min rows | median rows | max rows |
|---|---|---:|---:|---:|---:|---:|
| `test1.csv` | `attack` | 5 | 629 | 60 | 98 | 192 |
| `test1.csv` | `attack_P1` | 3 | 480 | 98 | 190 | 192 |
| `test1.csv` | `attack_P2` | 2 | 149 | 60 | 74.5 | 89 |
| `test2.csv` | `attack` | 20 | 3,449 | 17 | 152 | 422 |
| `test2.csv` | `attack_P1` | 30 | 2,414 | 1 | 5.5 | 422 |
| `test2.csv` | `attack_P2` | 11 | 643 | 3 | 68 | 152 |
| `test2.csv` | `attack_P3` | 2 | 238 | 119 | 119 | 119 |
| `test3.csv` | `attack` | 8 | 1,535 | 84 | 160 | 421 |
| `test3.csv` | `attack_P1` | 9 | 1,337 | 2 | 127 | 421 |
| `test3.csv` | `attack_P2` | 13 | 372 | 3 | 3 | 106 |
| `test4.csv` | `attack` | 5 | 1,157 | 120 | 258 | 263 |
| `test4.csv` | `attack_P1` | 4 | 1,035 | 254 | 260 | 262 |
| `test4.csv` | `attack_P2` | 8 | 175 | 3 | 3 | 152 |
| `test4.csv` | `attack_P3` | 3 | 360 | 120 | 120 | 120 |
| `test5.csv` | `attack` | 12 | 2,177 | 51 | 188 | 262 |
| `test5.csv` | `attack_P1` | 13 | 1,771 | 1 | 156 | 262 |
| `test5.csv` | `attack_P2` | 17 | 525 | 1 | 3 | 152 |
| `test5.csv` | `attack_P3` | 3 | 360 | 120 | 120 | 120 |

## Recommended Preprocessing and Balancing Corrections

- Predict a single binary label: attack. Drop time, attack, attack_P1, attack_P2, attack_P3, and any other attack-containing column from model inputs.
- Keep strict file-level split: train1/train2/train3/test1/test2 for training construction, test3 for validation and threshold/ratio selection, test4/test5 for held-out test only.
- Build windows per file only; never let a sequence window cross file boundaries.
- Fit imputer and scaler on training windows only, then transform validation and held-out test with those fitted artifacts.
- Balance only training windows. Do not balance validation or held-out test. Select ratio and threshold using validation metrics only.
- Because train1/train2/train3 are fully normal, attack-positive training windows can only come from test1 and test2 under the current defensible split.
- Review constant-in-training and highly correlated features before pruning. If pruning is used, decide it from training/validation evidence only, then freeze before held-out testing.

## Files Written

- `outputs_hai2103_deep_study/reports/hai2103_dataset_deep_study.json`
- `outputs_hai2103_deep_study/reports/hai2103_deep_study.md`
- `outputs_hai2103_deep_study/reports/hai2103_feature_profile.csv`
- `outputs_hai2103_deep_study/reports/hai2103_file_label_window_summary.csv`
- `outputs_hai2103_deep_study/reports/hai2103_high_correlation_pairs.csv`
- `outputs_hai2103_deep_study/reports/hai2103_attack_episodes.csv`
