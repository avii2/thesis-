# RESULTS SUMMARY MEAN STD

## 1. Seed coverage
| experiment_id | status | successful_seeds | missing_seeds | failed_seeds | notes |
| --- | --- | --- | --- | --- | --- |
| A_C1 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| A_C2 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| A_C3 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| B_C1 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| B_C2 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| B_C3 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| P_C1 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| P_C2 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| P_C3 | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| AB_C1_FEDAVG_TCN | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| AB_C2_FEDAVG_MLP | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |
| AB_C3_FEDAVG_CNN1D | PARTIAL | 42 | 123,2025 | none | seed 123 MISSING; seed 2025 MISSING |

## 2. Mean ± std across seeds
| experiment_id | Accuracy | Precision | Recall | F1 | AUROC | PR-AUC | FPR | wall-clock time | communication cost | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_C1 | 0.9897 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8340 ± 0.0000 | 0.3552 ± 0.0000 | 0.0000 ± 0.0000 | 123.2374 ± 0.0000 | 6993600.0000 ± 0.0000 | PARTIAL |
| A_C2 | 0.6667 ± 0.0000 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.8000 ± 0.0000 | 0.7558 ± 0.0000 | 0.8241 ± 0.0000 | 1.0000 ± 0.0000 | 17.9786 ± 0.0000 | 246000.0000 ± 0.0000 | PARTIAL |
| A_C3 | 0.8978 ± 0.0000 | 0.0833 ± 0.0000 | 0.0044 ± 0.0000 | 0.0084 ± 0.0000 | 0.9836 ± 0.0000 | 0.7538 ± 0.0000 | 0.0053 ± 0.0000 | 236.3212 ± 0.0000 | 246000.0000 ± 0.0000 | PARTIAL |
| B_C1 | 0.9897 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8344 ± 0.0000 | 0.3551 ± 0.0000 | 0.0000 ± 0.0000 | 121.7367 ± 0.0000 | 8159200.0000 ± 0.0000 | PARTIAL |
| B_C2 | 0.6667 ± 0.0000 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.8000 ± 0.0000 | 0.7559 ± 0.0000 | 0.8236 ± 0.0000 | 1.0000 ± 0.0000 | 19.1217 ± 0.0000 | 295200.0000 ± 0.0000 | PARTIAL |
| B_C3 | 0.8981 ± 0.0000 | 0.0827 ± 0.0000 | 0.0041 ± 0.0000 | 0.0078 ± 0.0000 | 0.9827 ± 0.0000 | 0.7548 ± 0.0000 | 0.0049 ± 0.0000 | 260.8789 ± 0.0000 | 295200.0000 ± 0.0000 | PARTIAL |
| P_C1 | 0.9931 ± 0.0000 | 0.9293 ± 0.0000 | 0.3594 ± 0.0000 | 0.5183 ± 0.0000 | 0.7826 ± 0.0000 | 0.5190 ± 0.0000 | 0.0003 ± 0.0000 | 1122.7119 ± 0.0000 | 148204000.0000 ± 0.0000 | PARTIAL |
| P_C2 | 0.8000 ± 0.0000 | 0.7692 ± 0.0000 | 1.0000 ± 0.0000 | 0.8696 ± 0.0000 | 0.8110 ± 0.0000 | 0.8552 ± 0.0000 | 0.6000 ± 0.0000 | 11.0641 ± 0.0000 | 22125600.0000 ± 0.0000 | PARTIAL |
| P_C3 | 0.9550 ± 0.0000 | 0.7999 ± 0.0000 | 0.7206 ± 0.0000 | 0.7582 ± 0.0000 | 0.9837 ± 0.0000 | 0.7488 ± 0.0000 | 0.0195 ± 0.0000 | 251.0429 ± 0.0000 | 590400.0000 ± 0.0000 | PARTIAL |
| AB_C1_FEDAVG_TCN | 0.9919 ± 0.0000 | 0.9661 ± 0.0000 | 0.2227 ± 0.0000 | 0.3619 ± 0.0000 | 0.8395 ± 0.0000 | 0.5149 ± 0.0000 | 0.0001 ± 0.0000 | 1131.2413 ± 0.0000 | 151804800.0000 ± 0.0000 | PARTIAL |
| AB_C2_FEDAVG_MLP | 0.6667 ± 0.0000 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.8000 ± 0.0000 | 0.8139 ± 0.0000 | 0.8610 ± 0.0000 | 1.0000 ± 0.0000 | 10.0352 ± 0.0000 | 22125600.0000 ± 0.0000 | PARTIAL |
| AB_C3_FEDAVG_CNN1D | 0.8981 ± 0.0000 | 0.0827 ± 0.0000 | 0.0041 ± 0.0000 | 0.0078 ± 0.0000 | 0.9827 ± 0.0000 | 0.7548 ± 0.0000 | 0.0049 ± 0.0000 | 250.1714 ± 0.0000 | 295200.0000 ± 0.0000 | PARTIAL |

## 3. Missing or failed seeds
- `A_C1`: seed 123 MISSING; seed 2025 MISSING
- `A_C2`: seed 123 MISSING; seed 2025 MISSING
- `A_C3`: seed 123 MISSING; seed 2025 MISSING
- `B_C1`: seed 123 MISSING; seed 2025 MISSING
- `B_C2`: seed 123 MISSING; seed 2025 MISSING
- `B_C3`: seed 123 MISSING; seed 2025 MISSING
- `P_C1`: seed 123 MISSING; seed 2025 MISSING
- `P_C2`: seed 123 MISSING; seed 2025 MISSING
- `P_C3`: seed 123 MISSING; seed 2025 MISSING
- `AB_C1_FEDAVG_TCN`: seed 123 MISSING; seed 2025 MISSING
- `AB_C2_FEDAVG_MLP`: seed 123 MISSING; seed 2025 MISSING
- `AB_C3_FEDAVG_CNN1D`: seed 123 MISSING; seed 2025 MISSING

## 4. Notes
- Expected seeds: `42,123,2025`.
- Means and standard deviations are computed from the available seed-specific metrics files only.
- Missing or failed seeds are reported explicitly and excluded from the aggregated statistics.
