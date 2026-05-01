# P_C1_BATADAL_TCN_FEDYOGI Report

This is a Cluster 1 BATADAL alternative ablation. The existing P_C1_BATADAL experiment is unchanged.

## Protocol

- Dataset: BATADAL
- Model: TCN-GN
- FL method: FedYogi
- Aggregation: hierarchical_fedyogi_delta
- Leaf clients: 12
- Fixed sub-clusters: 2
- Frozen membership hash: `c605adadbd648c8a0c946c56e1bd185aec1f492c4c5368d6e9d4de2698f7400f`
- Thresholds are selected using validation predictions only.
- Held-out test predictions are evaluated only after validation threshold selection.
- No Cluster 2, Cluster 3, or ledger logic is modified by this ablation.

## Data Guardrails

- Input adapter: `sliding_window_feature_channels`
- Input channels: 36
- Input length: 48
- Test used for training: False
- Test used for validation: False
- Test used for threshold tuning: False
- Test used for preprocessing fit: False
- Test used for clustering: False
- Test used for descriptor computation: False
- Class-weight scales are reported as an ablation; any preferred scale must be chosen from validation metrics only.

## Best Validation Row

- positive_class_weight_scale: 0.25
- threshold_selection_mode: `max_validation_f1`
- best_validation_round: 45
- selected_threshold: 0.9635933041572571
- validation_f1: 0.6012024048096193
- validation_fpr: 0.39295392953929537
- test_f1: 0.3619402985074627
- test_fpr: 0.28807339449541286

## Metrics CSV

`outputs_c1_batadal_alt/metrics/P_C1_BATADAL_TCN_FEDYOGI_metrics.csv`

## Results

| scale | threshold mode | best round | threshold | test accuracy | test precision | test recall | test F1 | test AUROC | test PR-AUC | test FPR | confusion matrix | communication bytes | training seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.25 | `max_validation_f1` | 45 | 0.963593 | 0.665034 | 0.291729 | 0.476658 | 0.361940 | 0.644279 | 0.376332 | 0.288073 | `[[1164, 471], [213, 194]]` | 374533600 | 221.110 |
| 0.25 | `max_validation_f1_with_validation_fpr_le_0.10` | 8 | 0.064701 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.470515 | 0.213505 | 0.000000 | `[[1635, 0], [407, 0]]` | 374533600 | 221.110 |
| 0.25 | `max_validation_f1_with_validation_fpr_le_0.05` | 8 | 0.064701 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.470515 | 0.213505 | 0.000000 | `[[1635, 0], [407, 0]]` | 374533600 | 221.110 |
| 0.5 | `max_validation_f1` | 23 | 0.830939 | 0.301665 | 0.195821 | 0.805897 | 0.315082 | 0.624665 | 0.462132 | 0.823853 | `[[288, 1347], [79, 328]]` | 374533600 | 229.499 |
| 0.5 | `max_validation_f1_with_validation_fpr_le_0.10` | 4 | 0.250353 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.560924 | 0.286224 | 0.000000 | `[[1635, 0], [407, 0]]` | 374533600 | 229.499 |
| 0.5 | `max_validation_f1_with_validation_fpr_le_0.05` | 4 | 0.250353 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.560924 | 0.286224 | 0.000000 | `[[1635, 0], [407, 0]]` | 374533600 | 229.499 |
| 1.0 | `max_validation_f1` | 32 | 0.002392 | 0.499021 | 0.231239 | 0.651106 | 0.341275 | 0.618436 | 0.349844 | 0.538838 | `[[754, 881], [142, 265]]` | 374533600 | 230.420 |
| 1.0 | `max_validation_f1_with_validation_fpr_le_0.10` | 30 | 0.999999 | 0.817826 | 1.000000 | 0.085995 | 0.158371 | 0.622967 | 0.409962 | 0.000000 | `[[1635, 0], [372, 35]]` | 374533600 | 230.420 |
| 1.0 | `max_validation_f1_with_validation_fpr_le_0.05` | 3 | 0.569356 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.547207 | 0.300938 | 0.000000 | `[[1635, 0], [407, 0]]` | 374533600 | 230.420 |
