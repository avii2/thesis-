# Results Summary

Generated from per-experiment metrics CSV files under `outputs/metrics/` only. No training was rerun and no missing metric values were fabricated.

## Audit Note

- No current source-metric mismatch was found before this regeneration; the summary was rebuilt from source metrics anyway.
- Historical note: A prior audit note observed that an earlier summary file was stale/incomplete; this regeneration uses only the per-experiment metrics CSV files for metric values.
- Metrics files found: 12 of 12 expected experiments.
- Missing experiments: none.
- `docs/EXPERIMENT_MATRIX.csv` was used only for expected experiment order and non-metric metadata such as run category.

## Source Files

- `outputs/metrics/A_C1_metrics.csv`: OK
- `outputs/metrics/A_C2_metrics.csv`: OK
- `outputs/metrics/A_C3_metrics.csv`: OK
- `outputs/metrics/B_C1_metrics.csv`: OK
- `outputs/metrics/B_C2_metrics.csv`: OK
- `outputs/metrics/B_C3_metrics.csv`: OK
- `outputs/metrics/P_C1_metrics.csv`: OK
- `outputs/metrics/P_C2_metrics.csv`: OK
- `outputs/metrics/P_C3_metrics.csv`: OK
- `outputs/metrics/AB_C1_FEDAVG_TCN_metrics.csv`: OK
- `outputs/metrics/AB_C2_FEDAVG_MLP_metrics.csv`: OK
- `outputs/metrics/AB_C3_FEDAVG_CNN1D_metrics.csv`: OK

## Master Metrics

| Experiment | Status | Category | Cluster | Model | FL | Threshold | Best Val F1 | Accuracy | Precision | Recall | F1 | AUROC | PR-AUC | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_C1 | OK | baseline_flat | 1 | cnn1d | FedAvg | 0.9642 | 0.7027 | 0.9935 | 0.9123 | 0.4062 | 0.5622 | 0.8998 | 0.5194 | 0.0004 |
| A_C2 | OK | baseline_flat | 2 | cnn1d | FedAvg | 0.5080 | 0.9116 | 0.8951 | 0.8687 | 0.9926 | 0.9265 | 0.7873 | 0.8337 | 0.2999 |
| A_C3 | OK | baseline_flat | 3 | cnn1d | FedAvg | 0.0075 | 0.8737 | 0.9711 | 0.7759 | 0.9906 | 0.8702 | 0.9784 | 0.6481 | 0.0310 |
| B_C1 | OK | baseline_uniform_hierarchical | 1 | cnn1d | FedAvg | 0.9632 | 0.7027 | 0.9935 | 0.9123 | 0.4062 | 0.5622 | 0.8997 | 0.5197 | 0.0004 |
| B_C2 | OK | baseline_uniform_hierarchical | 2 | cnn1d | FedAvg | 0.5080 | 0.9116 | 0.8951 | 0.8687 | 0.9926 | 0.9265 | 0.7873 | 0.8337 | 0.2999 |
| B_C3 | OK | baseline_uniform_hierarchical | 3 | cnn1d | FedAvg | 0.0093 | 0.8745 | 0.9712 | 0.7762 | 0.9922 | 0.8710 | 0.9780 | 0.6438 | 0.0310 |
| P_C1 | OK | proposed_specialized_hierarchical | 1 | tcn | FedBN | 0.9930 | 0.5635 | 0.9925 | 0.8257 | 0.3516 | 0.4932 | 0.8389 | 0.4198 | 0.0008 |
| P_C2 | OK | proposed_specialized_hierarchical | 2 | compact_mlp | FedProx | 0.6379 | 0.9270 | 0.8985 | 0.8679 | 1.0000 | 0.9293 | 0.8124 | 0.8557 | 0.3045 |
| P_C3 | OK | proposed_specialized_hierarchical | 3 | cnn1d | SCAFFOLD | 0.1317 | 0.8694 | 0.9708 | 0.7714 | 0.9965 | 0.8697 | 0.9813 | 0.6901 | 0.0320 |
| AB_C1_FEDAVG_TCN | OK | ablation_fl_method | 1 | tcn | FedAvg | 0.8068 | 0.6977 | 0.9931 | 0.8346 | 0.4141 | 0.5535 | 0.7855 | 0.5112 | 0.0009 |
| AB_C2_FEDAVG_MLP | OK | ablation_fl_method | 2 | compact_mlp | FedAvg | 0.8571 | 0.9270 | 0.8984 | 0.8678 | 1.0000 | 0.9292 | 0.8139 | 0.8610 | 0.3048 |
| AB_C3_FEDAVG_CNN1D | OK | ablation_fl_method | 3 | cnn1d | FedAvg | 0.0093 | 0.8745 | 0.9712 | 0.7762 | 0.9922 | 0.8710 | 0.9780 | 0.6438 | 0.0310 |

## Communication and Runtime

| Experiment | Status | Comm/Round Bytes | Control Variate Comm/Round Bytes | Total Comm Bytes | Control Variate Bytes | Wall Clock Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| A_C1 | OK | 139872 |  | 6993600 |  | 100.389 |
| A_C2 | OK | 4920 |  | 246000 |  | 9.295 |
| A_C3 | OK | 4920 |  | 246000 |  | 130.327 |
| B_C1 | OK | 163184 |  | 8159200 |  | 97.512 |
| B_C2 | OK | 5904 |  | 295200 |  | 9.064 |
| B_C3 | OK | 5904 |  | 295200 |  | 132.339 |
| P_C1 | OK | 2964080 |  | 148204000 |  | 1128.825 |
| P_C2 | OK | 442512 |  | 22125600 |  | 12.088 |
| P_C3 | OK | 11808 | 5904 | 590400 | 164 | 166.917 |
| AB_C1_FEDAVG_TCN | OK | 3036096 |  | 151804800 |  | 872.230 |
| AB_C2_FEDAVG_MLP | OK | 442512 |  | 22125600 |  | 10.436 |
| AB_C3_FEDAVG_CNN1D | OK | 5904 |  | 295200 |  | 134.844 |

## Default-Threshold Metrics

The master CSV preserves default-threshold metrics in columns ending with `_default_threshold`. The table above reports the tuned-threshold metrics stored in each per-experiment source CSV.

## Reproducibility

- Regenerated at: `2026-04-27T03:08:23+00:00`.
- Source of truth: per-experiment `*_metrics.csv` files only.
- Training code, model code, configs, datasets, and existing experiment outputs were not modified by this summary regeneration task.
