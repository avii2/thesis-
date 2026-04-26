# RESULTS SUMMARY

## 1. Available experiments discovered

| experiment_id | status | artifacts_present | notes |
| --- | --- | --- | --- |
| A_C1 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| A_C2 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| A_C3 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| B_C1 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| B_C2 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| B_C3 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| P_C1 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| P_C2 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| P_C3 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under `outputs/`. |
| AB_C1_FEDAVG_TCN | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | Standalone ablation control outputs are present under `outputs/`. |
| AB_C2_FEDAVG_MLP | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | Standalone ablation control outputs are present under `outputs/`. |
| AB_C3_FEDAVG_CNN1D | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | Standalone ablation control outputs are present under `outputs/`. |

## 2. Per-experiment metrics

Primary test metrics below use the current per-experiment `*_metrics.csv` files. When `threshold_used` is available, the reported `test_*` values correspond to the tuned validation-selected threshold. When `threshold_used` is `NOT AVAILABLE`, the file does not expose a tuned threshold and the available test metrics are reported as-is.

| experiment_id | cluster_id | dataset | model | fl_method | aggregation | best_validation_round | threshold_used | validation_f1 | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | total_comm_cost | wall_clock_s |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_C1 | 1 | HAI 21.03 | cnn1d | FedAvg | weighted_arithmetic_mean | 40 | 0.9642 | 0.7027 | 0.9935 | 0.9123 | 0.4062 | 0.5622 | 0.8998 | 0.5194 | 0.0004 | 6993600 | 99.2161 |
| A_C2 | 2 | TON IoT combined telemetry | cnn1d | FedAvg | weighted_arithmetic_mean | 1 | 0.5080 | 0.9116 | 0.8951 | 0.8687 | 0.9926 | 0.9265 | 0.7873 | 0.8337 | 0.2999 | 246000 | 9.0622 |
| A_C3 | 3 | WUSTL-IIOT-2021 | cnn1d | FedAvg | weighted_arithmetic_mean | 50 | 0.0075 | 0.8737 | 0.9711 | 0.7759 | 0.9906 | 0.8702 | 0.9784 | 0.6481 | 0.0310 | 246000 | 128.0022 |
| B_C1 | 1 | HAI 21.03 | cnn1d | FedAvg | weighted_arithmetic_mean | 39 | 0.9632 | 0.7027 | 0.9935 | 0.9123 | 0.4062 | 0.5622 | 0.8997 | 0.5197 | 0.0004 | 8159200 | 97.4484 |
| B_C2 | 2 | TON IoT combined telemetry | cnn1d | FedAvg | weighted_arithmetic_mean | 1 | 0.5080 | 0.9116 | 0.8951 | 0.8687 | 0.9926 | 0.9265 | 0.7873 | 0.8337 | 0.2999 | 295200 | 9.0481 |
| B_C3 | 3 | WUSTL-IIOT-2021 | cnn1d | FedAvg | weighted_arithmetic_mean | 42 | 0.0093 | 0.8745 | 0.9712 | 0.7762 | 0.9922 | 0.8710 | 0.9780 | 0.6438 | 0.0310 | 295200 | 128.5612 |
| P_C1 | 1 | HAI 21.03 | tcn | FedBN | weighted_non_bn_mean | 40 | 0.9930 | 0.5635 | 0.9925 | 0.8257 | 0.3516 | 0.4932 | 0.8389 | 0.4198 | 0.0008 | 148204000 | 823.2584 |
| P_C2 | 2 | TON IoT combined telemetry | compact_mlp | FedProx | weighted_arithmetic_mean | 50 | 0.6379 | 0.9270 | 0.8985 | 0.8679 | 1.0000 | 0.9293 | 0.8124 | 0.8557 | 0.3045 | 22125600 | 11.2596 |
| P_C3 | 3 | WUSTL-IIOT-2021 | cnn1d | SCAFFOLD | weighted_arithmetic_mean | 50 | 0.1317 | 0.8694 | 0.9708 | 0.7714 | 0.9965 | 0.8697 | 0.9813 | 0.6901 | 0.0320 | 590400 | 131.0444 |
| AB_C1_FEDAVG_TCN | 1 | HAI 21.03 | tcn | FedAvg | weighted_arithmetic_mean | 21 | 0.8068 | 0.6977 | 0.9931 | 0.8346 | 0.4141 | 0.5535 | 0.7855 | 0.5112 | 0.0009 | 151804800 | 911.6239 |
| AB_C2_FEDAVG_MLP | 2 | TON IoT combined telemetry | compact_mlp | FedAvg | weighted_arithmetic_mean | 50 | NOT AVAILABLE | 0.8000 | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.8139 | 0.8610 | 1.0000 | 22125600 | 9.8056 |
| AB_C3_FEDAVG_CNN1D | 3 | WUSTL-IIOT-2021 | cnn1d | FedAvg | weighted_arithmetic_mean | 18 | NOT AVAILABLE | 0.0101 | 0.8981 | 0.0827 | 0.0041 | 0.0078 | 0.9827 | 0.7548 | 0.0049 | 295200 | 246.0777 |

## 3. Baseline vs proposed results

### Cluster 1
- Observed tuned-threshold test F1: `A_C1 = 0.5622`, `B_C1 = 0.5622`, `P_C1 = 0.4932`.
- Observed tuned-threshold test accuracy: `A_C1 = 0.9935`, `B_C1 = 0.9935`, `P_C1 = 0.9925`.
- Hierarchy showed no observed change in tuned-threshold test F1.
- Specialization did not improve tuned-threshold test F1 for Cluster 1.

### Cluster 2
- Observed tuned-threshold test F1: `A_C2 = 0.9265`, `B_C2 = 0.9265`, `P_C2 = 0.9293`.
- Observed tuned-threshold test accuracy: `A_C2 = 0.8951`, `B_C2 = 0.8951`, `P_C2 = 0.8985`.
- Hierarchy showed no observed change in tuned-threshold test F1.
- Specialization gave a small observed improvement in tuned-threshold test F1 for Cluster 2.

### Cluster 3
- Observed tuned-threshold test F1: `A_C3 = 0.8702`, `B_C3 = 0.8710`, `P_C3 = 0.8697`.
- Observed tuned-threshold test accuracy: `A_C3 = 0.9711`, `B_C3 = 0.9712`, `P_C3 = 0.9708`.
- Hierarchy gave a small observed increase in tuned-threshold test F1.
- Specialization did not improve tuned-threshold test F1, although `P_C3` has the strongest observed AUROC (`0.9813`) and PR-AUC (`0.6901`) among Cluster 3 runs.

## 4. Ablation summary

### Hierarchy effect

| scope | compared_runs | ΔAccuracy (B - A) | ΔF1 (B - A) | ΔAUROC (B - A) |
| --- | --- | ---: | ---: | ---: |
| Cluster 1 | A_C1 vs B_C1 | +0.0000 | +0.0000 | -0.0001 |
| Cluster 2 | A_C2 vs B_C2 | +0.0000 | +0.0000 | -0.0000 |
| Cluster 3 | A_C3 vs B_C3 | +0.0002 | +0.0008 | -0.0003 |

### Cluster 1 FedAvg vs FedBN

| compared_runs | ΔAccuracy (P - AB) | ΔF1 (P - AB) | ΔAUROC (P - AB) | ΔPR-AUC (P - AB) |
| --- | ---: | ---: | ---: | ---: |
| AB_C1_FEDAVG_TCN vs P_C1 | -0.0006 | -0.0604 | +0.0534 | -0.0914 |

Observed interpretation:
- Under the current tuned-threshold metrics, `FedBN` reduced Cluster 1 test F1 relative to `TCN + FedAvg`.
- `FedBN` increased AUROC but reduced PR-AUC and recall-weighted F1 in the currently saved results.

### Cluster 2 FedAvg vs FedProx

| compared_runs | ΔAccuracy (P - AB) | ΔF1 (P - AB) | ΔAUROC (P - AB) | ΔPR-AUC (P - AB) |
| --- | ---: | ---: | ---: | ---: |
| AB_C2_FEDAVG_MLP vs P_C2 | +0.2318 | +0.1293 | -0.0015 | -0.0053 |

Observed interpretation:
- `FedProx` improved the saved Cluster 2 test F1 substantially relative to the `compact_mlp + FedAvg` ablation control.
- In the currently saved outputs, AUROC and PR-AUC are slightly lower for `FedProx`, while F1 and accuracy are higher.

### Cluster 3 FedAvg vs SCAFFOLD

| compared_runs | ΔAccuracy (P - AB) | ΔF1 (P - AB) | ΔAUROC (P - AB) | ΔPR-AUC (P - AB) |
| --- | ---: | ---: | ---: | ---: |
| AB_C3_FEDAVG_CNN1D vs P_C3 | +0.0727 | +0.8618 | -0.0014 | -0.0647 |

Observed interpretation:
- In the current saved metrics, `SCAFFOLD` greatly improves Cluster 3 test F1 over the standalone FedAvg control.
- The same saved outputs show slightly lower AUROC and PR-AUC for `SCAFFOLD` than for the FedAvg control.

## 5. Ledger outputs summary

All complete experiments have standalone metadata-only ledger files. No raw data or model weights were inspected in this summary; the ledger rows are counted and summarized as stored.

| experiment_id | record_count | avg_logging_latency_s | ledger_size_bytes |
| --- | ---: | ---: | ---: |
| A_C1 | 50 | 0.0010 | 44522 |
| A_C2 | 50 | 0.0010 | 44472 |
| A_C3 | 50 | 0.0010 | 44522 |
| B_C1 | 50 | 0.0010 | 44922 |
| B_C2 | 50 | 0.0010 | 44872 |
| B_C3 | 50 | 0.0010 | 44922 |
| P_C1 | 50 | 0.0010 | 44672 |
| P_C2 | 50 | 0.0010 | 44922 |
| P_C3 | 50 | 0.0010 | 45022 |
| AB_C1_FEDAVG_TCN | 50 | 0.0010 | 44922 |
| AB_C2_FEDAVG_MLP | 50 | 0.0010 | 44872 |
| AB_C3_FEDAVG_CNN1D | 50 | 0.0010 | 44922 |

## 6. Missing experiments or blocked items

- No experiment ids from `docs/EXPERIMENT_MATRIX.csv` are currently missing; all 12 listed experiment ids have standalone outputs under `outputs/`.
- `AB_C2_FEDAVG_MLP` and `AB_C3_FEDAVG_CNN1D` currently have `threshold_used = NOT AVAILABLE` in their saved metrics file, so no tuned-threshold value can be reported for those two rows from the current artifacts.

## 7. Notes

- This report is generated from the current per-experiment artifacts already present under `outputs/`; no training was rerun to create it.
- The report uses per-experiment `*_metrics.csv`, `run_summary.json`, `round_metrics.csv`, `convergence_*.svg`, and `*_ledger.jsonl` files.
- `outputs/metrics/summary_all_experiments.csv` currently contains only one row and was not used as the source of truth for this report.
- Seed-specific files such as `*_seed_42_metrics.csv` exist, but this report summarizes the current non-seed per-experiment artifact set.
