# P_C1_BATADAL Positive-Window Oversampling Selection

Scope: Cluster 1 P_C1_BATADAL only. The BATADAL split, window length, stride, last-row labeling, model family, FedBN method, weighted non-BN aggregation, and fixed sub-clusters are unchanged.

Training change: positive-window oversampling is applied only inside local client training. Validation and held-out test windows are not resampled.

CSV: `outputs_c1_batadal_tuned/reports/p_c1_batadal_positive_oversampling_selection.csv`

Selection rule: highest validation F1 among candidates with validation FPR <= 0.10. Test metrics are reported only for the selected candidate.

| run_id | scale | validation_f1 | validation_fpr | threshold | selected | test_f1 | test_fpr | test_confusion_matrix |
|---|---:|---:|---:|---:|---|---:|---:|---|
| scale_010 | 0.100000 | 0.490566 | 0.097561 | 0.960233 | True | 0.080178 | 0.014679 | `[[1611, 24], [389, 18]]` |
| scale_025 | 0.250000 | 0.485804 | 0.097561 | 0.995022 | False |  |  |  |
| scale_050 | 0.500000 | 0.485804 | 0.097561 | 0.997479 | False |  |  |  |

## Final Selection

Selected run: `scale_010`
positive_class_weight_scale: `0.1`
training_resampling: `positive_window_oversampling`
target_positive_fraction: `0.5`
threshold_mode: `max_validation_f1_with_validation_fpr_le_0.10`
selected_threshold: `0.960233`

Selected held-out test metrics:
- accuracy: 0.797747
- precision: 0.428571
- recall: 0.044226
- F1: 0.080178
- AUROC: 0.634132
- PR-AUC: 0.363982
- FPR: 0.014679
- confusion matrix: `[[1611, 24], [389, 18]]`
