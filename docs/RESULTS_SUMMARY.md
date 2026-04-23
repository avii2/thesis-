# RESULTS SUMMARY

## 1. Available experiments discovered
| experiment_id | status | artifacts_present | notes |
| --- | --- | --- | --- |
| A_C1 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| A_C2 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| A_C3 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| B_C1 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| B_C2 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| B_C3 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| P_C1 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| P_C2 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| P_C3 | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| AB_C1_FEDAVG_TCN | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| AB_C2_FEDAVG_MLP | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |
| AB_C3_FEDAVG_CNN1D | COMPLETE | run_summary, round_metrics, metrics_csv, ledger, convergence_plot | All expected per-experiment run/metrics/ledger/plot artifacts are present under outputs/. |

## 2. Per-experiment metrics
| experiment_id | status | cluster_id | dataset | model | fl_method | aggregation | hierarchy_type | clustering_type | best_validation_round | validation_f1 | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | total_comm_cost | wall_clock_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_C1 | COMPLETE | 1 | HAI_2103 | cnn1d | FedAvg | weighted_arithmetic_mean | flat | none | 50 | 0.0000 | 0.9897 | 0.0000 | 0.0000 | 0.0000 | 0.8340 | 0.3552 | 0.0000 | 6993600 | 96.1245 |
| A_C2 | COMPLETE | 2 | TON_IoT_combined_telemetry | cnn1d | FedAvg | weighted_arithmetic_mean | flat | none | 50 | 0.8000 | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.7558 | 0.8241 | 1.0000 | 246000 | 14.6612 |
| A_C3 | COMPLETE | 3 | WUSTL_IIOT_2021 | cnn1d | FedAvg | weighted_arithmetic_mean | flat | none | 28 | 0.0106 | 0.8978 | 0.0833 | 0.0044 | 0.0084 | 0.9836 | 0.7538 | 0.0053 | 246000 | 180.3744 |
| B_C1 | COMPLETE | 1 | HAI_2103 | cnn1d | FedAvg | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 50 | 0.0000 | 0.9897 | 0.0000 | 0.0000 | 0.0000 | 0.8344 | 0.3551 | 0.0000 | 8159200 | 96.9241 |
| B_C2 | COMPLETE | 2 | TON_IoT_combined_telemetry | cnn1d | FedAvg | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 50 | 0.8000 | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.7559 | 0.8236 | 1.0000 | 295200 | 14.7901 |
| B_C3 | COMPLETE | 3 | WUSTL_IIOT_2021 | cnn1d | FedAvg | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 18 | 0.0101 | 0.8981 | 0.0827 | 0.0041 | 0.0078 | 0.9827 | 0.7548 | 0.0049 | 295200 | 180.0806 |
| P_C1 | COMPLETE | 1 | HAI_2103 | tcn | FedBN | weighted_non_bn_mean | hierarchical_fixed | agglomerative | 31 | 0.6272 | 0.9931 | 0.9293 | 0.3594 | 0.5183 | 0.7826 | 0.5190 | 0.0003 | 148204000 | 928.9196 |
| P_C2 | COMPLETE | 2 | TON_IoT_combined_telemetry | compact_mlp | FedProx | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 50 | 0.8649 | 0.8000 | 0.7692 | 1.0000 | 0.8696 | 0.8110 | 0.8552 | 0.6000 | 22125600 | 10.7308 |
| P_C3 | COMPLETE | 3 | WUSTL_IIOT_2021 | cnn1d | SCAFFOLD | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 42 | 0.6889 | 0.9550 | 0.7999 | 0.7206 | 0.7582 | 0.9837 | 0.7488 | 0.0195 | 590400 | 182.4572 |
| AB_C1_FEDAVG_TCN | COMPLETE | 1 | HAI_2103 | tcn | FedAvg | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 50 | 0.4000 | 0.9919 | 0.9661 | 0.2227 | 0.3619 | 0.8395 | 0.5149 | 0.0001 | 151804800 | 1139.4502 |
| AB_C2_FEDAVG_MLP | COMPLETE | 2 | TON_IoT_combined_telemetry | compact_mlp | FedAvg | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 50 | 0.8000 | 0.6667 | 0.6667 | 1.0000 | 0.8000 | 0.8139 | 0.8610 | 1.0000 | 22125600 | 9.8056 |
| AB_C3_FEDAVG_CNN1D | COMPLETE | 3 | WUSTL_IIOT_2021 | cnn1d | FedAvg | weighted_arithmetic_mean | hierarchical_fixed | agglomerative | 18 | 0.0101 | 0.8981 | 0.0827 | 0.0041 | 0.0078 | 0.9827 | 0.7548 | 0.0049 | 295200 | 246.0777 |

## 3. Baseline vs proposed results
### Cluster 1
Observed test F1: `A_C1=0.0000`, `B_C1=0.0000`, `P_C1=0.5183`.
Observed test accuracy: `A_C1=0.9897`, `B_C1=0.9897`, `P_C1=0.9931`.
Hierarchy showed no observed change in test F1.
Specialization helped on observed test F1.
### Cluster 2
Observed test F1: `A_C2=0.8000`, `B_C2=0.8000`, `P_C2=0.8696`.
Observed test accuracy: `A_C2=0.6667`, `B_C2=0.6667`, `P_C2=0.8000`.
Hierarchy showed no observed change in test F1.
Specialization helped on observed test F1.
### Cluster 3
Observed test F1: `A_C3=0.0084`, `B_C3=0.0078`, `P_C3=0.7582`.
Observed test accuracy: `A_C3=0.8978`, `B_C3=0.8981`, `P_C3=0.9550`.
Hierarchy did not help on observed test F1.
Specialization helped on observed test F1.

## 4. Ablation summary
### hierarchy effect
| scope | compared_runs | ΔAccuracy | ΔF1 | ΔAUROC |
| --- | --- | --- | --- | --- |
| Cluster 1 | A_C1 vs B_C1 | 0.0000 | 0.0000 | 0.0004 |
| Cluster 2 | A_C2 vs B_C2 | 0.0000 | 0.0000 | 0.0001 |
| Cluster 3 | A_C3 vs B_C3 | 0.0003 | -0.0006 | -0.0009 |
### Cluster 1 FedAvg vs FedBN
| compared_runs | ΔAccuracy | ΔF1 | ΔAUROC |
| --- | --- | --- | --- |
| AB_C1_FEDAVG_TCN vs P_C1 | 0.0012 | 0.1564 | -0.0569 |
### Cluster 2 FedAvg vs FedProx
| compared_runs | ΔAccuracy | ΔF1 | ΔAUROC |
| --- | --- | --- | --- |
| AB_C2_FEDAVG_MLP vs P_C2 | 0.1333 | 0.0696 | -0.0029 |
### Cluster 3 FedAvg vs SCAFFOLD
| compared_runs | ΔAccuracy | ΔF1 | ΔAUROC |
| --- | --- | --- | --- |
| AB_C3_FEDAVG_CNN1D vs P_C3 | 0.0569 | 0.7504 | 0.0009 |

## 5. Ledger outputs summary
Ledger files discovered: `12`.
Average logging latency across discovered ledgers: `0.0010` seconds.
No illegal raw data or full model weights were found in the discovered ledger records.

| experiment_id | record_count | avg_logging_latency_s | ledger_size_bytes | illegal_raw_data_or_weights_found |
| --- | --- | --- | --- | --- |
| A_C1 | 50 | 0.0010 | 44522 | NO |
| A_C2 | 50 | 0.0010 | 44472 | NO |
| A_C3 | 50 | 0.0010 | 44522 | NO |
| B_C1 | 50 | 0.0010 | 44922 | NO |
| B_C2 | 50 | 0.0010 | 44872 | NO |
| B_C3 | 50 | 0.0010 | 44922 | NO |
| P_C1 | 50 | 0.0010 | 44672 | NO |
| P_C2 | 50 | 0.0010 | 44922 | NO |
| P_C3 | 50 | 0.0010 | 45022 | NO |
| AB_C1_FEDAVG_TCN | 50 | 0.0010 | 44922 | NO |
| AB_C2_FEDAVG_MLP | 50 | 0.0010 | 44872 | NO |
| AB_C3_FEDAVG_CNN1D | 50 | 0.0010 | 44922 | NO |

## 6. Missing experiments or blocked items
- None. All experiment IDs in `docs/EXPERIMENT_MATRIX.csv` have standalone outputs under `outputs/`.

## 7. Notes
- The report uses only files currently present under `outputs/metrics/`, `outputs/plots/`, `outputs/ledgers/`, and `outputs/runs/`.
- Only the three missing dedicated ablation experiments were run in this update; existing `A_*`, `B_*`, and `P_*` outputs were reused as-is.
- `run_repeats=3` is still specified in `docs/EXPERIMENT_MATRIX.csv`, but the discovered outputs are one finalized artifact set per experiment id, not three separately labeled repeat directories.
