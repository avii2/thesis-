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
- Test used for validation or threshold tuning: False

## Best Validation Row

- positive_class_weight_scale: 0.25
- threshold_selection_mode: `max_validation_f1`
- best_validation_round: 1
- selected_threshold: 0.49468979239463806
- validation_f1: 0.5614035087719298
- validation_fpr: 0.12698412698412698
- test_f1: 0.0
- test_fpr: 0.1

## Metrics CSV

`outputs_c1_batadal_alt_smoke/metrics/P_C1_BATADAL_TCN_FEDYOGI_metrics.csv`

## Results

| scale | threshold mode | best round | threshold | val F1 | val FPR | test F1 | test FPR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.25 | `max_validation_f1` | 1 | 0.494690 | 0.561404 | 0.126984 | 0.000000 | 0.100000 |
| 0.25 | `max_validation_f1_with_validation_fpr_le_0.10` | 1 | 0.990000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | `max_validation_f1_with_validation_fpr_le_0.05` | 1 | 0.990000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
