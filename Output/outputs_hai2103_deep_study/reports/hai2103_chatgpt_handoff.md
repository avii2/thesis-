# HAI 21.03 ChatGPT Handoff

Use this as the short prompt context for preprocessing/balancing advice. The full generated study is in `hai2103_deep_study.md`; every feature is profiled in `hai2103_feature_profile.csv`.

## Non-Negotiable Setup

- Predict one binary target only: `attack`.
- Exclude from model inputs: `time`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`, and any column containing `attack`.
- Do not train a multi-class or multi-output attack-type model unless the thesis scope changes.
- Do not use `test4.csv` or `test5.csv` for training, validation, threshold tuning, scaling, imputation, clustering, feature pruning, or ratio selection.
- Build sequence windows inside each file only; no window may cross file boundaries.
- Fit imputation/scaling on training windows only.
- Balance training windows only. Do not balance validation or held-out test.

## Label Findings

- Total rows: 1,323,608
- Binary `attack=1` rows: 8,947 (0.676%)
- Auxiliary label columns present: `attack_P1`, `attack_P2`, `attack_P3`
- `attack == OR(attack_P1, attack_P2, attack_P3)`: False
- Mismatch rows: 179; these are `attack=1` rows with no auxiliary process flag.
- Rows with multiple auxiliary process flags active: 1,091
- Interpretation: auxiliary columns are not a clean one-hot attack-type target. Use `attack` for single-label detection.

Auxiliary combination counts:
- `attack_P1=0,attack_P2=0,attack_P3=0`: 1,314,840 rows
- `attack_P1=0,attack_P2=0,attack_P3=1`: 409 rows
- `attack_P1=0,attack_P2=1,attack_P3=0`: 1,110 rows
- `attack_P1=0,attack_P2=1,attack_P3=1`: 212 rows
- `attack_P1=1,attack_P2=0,attack_P3=0`: 6,158 rows
- `attack_P1=1,attack_P2=0,attack_P3=1`: 337 rows
- `attack_P1=1,attack_P2=1,attack_P3=0`: 542 rows

## File Split and Imbalance

| split | source files | rows | attack=1 | attack rate |
|---|---|---:|---:|---:|
| `train_source` | `train1.csv`, `train2.csv`, `train3.csv`, `test1.csv`, `test2.csv` | 1,083,605 | 4,078 | 0.3763% |
| `validation_test3` | `test3.csv` | 108,001 | 1,535 | 1.421% |
| `heldout_test4_test5` | `test4.csv`, `test5.csv` | 132,002 | 3,334 | 2.526% |
| `all_rows` | `train1.csv`, `train2.csv`, `train3.csv`, `test1.csv`, `test2.csv`, `test3.csv`, `test4.csv`, `test5.csv` | 1,323,608 | 8,947 | 0.676% |

Important: `train1.csv`, `train2.csv`, and `train3.csv` are fully normal. Positive training rows/windows only come from `test1.csv` and `test2.csv` under the current file-level split.

## Window Counts

Windowing used in current Cluster 1 work: length=32, stride=8, label rule=`any_positive_row`.

| set | positive windows | negative windows |
|---|---:|---:|
| training source before balancing | 605 | 134,830 |
| validation `test3.csv` | 224 | 13,273 |
| held-out `test4.csv`+`test5.csv` | 481 | 16,013 |

## Feature Findings

- Telemetry features after leakage removal: 79
- Process feature counts: P1=38, P2=22, P3=7, P4=12
- Feature type counts: binary_or_boolean=11, continuous_numeric=56, low_cardinality_numeric=12
- Missing/non-finite features: 0
- Constant globally and in training source: 19
- Highly correlated pairs with |r| >= 0.98: 19

Constant features in the current data:
`P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P2_ASD`, `P2_AutoGO`, `P2_MSD`, `P2_ManualGO`, `P2_RTR`, `P2_TripEx`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P3_LH`, `P3_LL`

Top diagnostic univariate features by directionless ROC-AUC:
- `P1_PCV02Z`: AUC=0.754049, AP=0.0858291, SMD=0.355387
- `P2_VYT03`: AUC=0.618943, AP=0.0538273, SMD=-0.275233
- `P2_VXT03`: AUC=0.602698, AP=0.0459533, SMD=0.0907988
- `P1_FCV03D`: AUC=0.599211, AP=0.103257, SMD=-0.302935
- `P1_FCV03Z`: AUC=0.598658, AP=0.102604, SMD=-0.30644
- `P2_VYT02`: AUC=0.583472, AP=0.0541761, SMD=-0.209224
- `P1_FT03Z`: AUC=0.577905, AP=0.0384418, SMD=-0.245681
- `P1_FT03`: AUC=0.577503, AP=0.0393945, SMD=-0.239288
- `P1_B3005`: AUC=0.576472, AP=0.00882355, SMD=-0.22329
- `P1_LCV01D`: AUC=0.570816, AP=0.278851, SMD=0.517843

Largest validation drift features vs training source, excluding constant training features:
- `P2_VYT03`: shift=-1.70838
- `P2_VYT02`: shift=-1.12472
- `P1_B2004`: shift=-0.749634
- `P1_B4002`: shift=-0.670949
- `P1_B4022`: shift=-0.564111
- `P2_SIT02`: shift=-0.395745
- `P2_SIT01`: shift=-0.395641
- `P1_FCV01Z`: shift=-0.386992
- `P1_FCV01D`: shift=-0.386377
- `P2_24Vdc`: shift=-0.38461

## Recommended Question for ChatGPT

Given the constraints above, suggest leakage-safe preprocessing corrections and training-window balancing strategies for binary HAI 21.03 intrusion detection. Do not use held-out test files for any selection step, and do not use auxiliary `attack_P*` columns as features or multi-class targets.

## Full Artifacts

- `outputs_hai2103_deep_study/reports/hai2103_deep_study.md`
- `outputs_hai2103_deep_study/reports/hai2103_dataset_deep_study.json`
- `outputs_hai2103_deep_study/reports/hai2103_feature_profile.csv`
- `outputs_hai2103_deep_study/reports/hai2103_file_label_window_summary.csv`
- `outputs_hai2103_deep_study/reports/hai2103_high_correlation_pairs.csv`
- `outputs_hai2103_deep_study/reports/hai2103_attack_episodes.csv`
