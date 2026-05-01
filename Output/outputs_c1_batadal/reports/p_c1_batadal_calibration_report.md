# P_C1_BATADAL Calibration Report

This report is limited to Cluster 1 P_C1_BATADAL. Cluster 2, Cluster 3, and ledger logic are not modified.

Protocol:
- Thresholds are selected on validation predictions only.
- Held-out BATADAL test predictions are used only after each validation-selected threshold is fixed.
- The raw train/validation/test split is unchanged.
- The threshold selection modes are `max_validation_f1`, `max_validation_f1_with_validation_fpr_le_0.10`, and `max_validation_f1_with_validation_fpr_le_0.05`.
- Positive class weight scale ablation uses `0.25`, `0.50`, and `1.00` for P_C1_BATADAL only.

Threshold sweep CSV: `outputs_c1_batadal/calibration/p_c1_batadal_threshold_sweep.csv`
Class-weight ablation CSV: `outputs_c1_batadal/calibration/p_c1_batadal_class_weight_ablation.csv`

## Threshold Selection

| mode | threshold | validation_f1 | validation_fpr | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | test_confusion_matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| max_validation_f1 | 0.500888 | 0.558140 | 0.791328 | 0.304114 | 0.216760 | 0.953317 | 0.353209 | 0.659979 | 0.340694 | 0.857492 | `[[233, 1402], [19, 388]]` |
| max_validation_f1_with_validation_fpr_le_0.10 | 0.857115 | 0.451613 | 0.097561 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.659979 | 0.340694 | 0.000000 | `[[1635, 0], [407, 0]]` |
| max_validation_f1_with_validation_fpr_le_0.05 | 0.980583 | 0.391304 | 0.048780 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.659979 | 0.340694 | 0.000000 | `[[1635, 0], [407, 0]]` |

## Positive Class Weight Ablation

| positive_class_weight_scale | mode | threshold | validation_f1 | validation_fpr | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | test_confusion_matrix |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.250000 | max_validation_f1 | 0.362629 | 0.554645 | 0.880759 | 0.203722 | 0.190766 | 0.923833 | 0.316232 | 0.382077 | 0.154601 | 0.975535 | `[[40, 1595], [31, 376]]` |
| 0.250000 | max_validation_f1_with_validation_fpr_le_0.10 | 0.385479 | 0.072289 | 0.097561 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.382077 | 0.154601 | 0.000000 | `[[1635, 0], [407, 0]]` |
| 0.250000 | max_validation_f1_with_validation_fpr_le_0.05 | 0.391883 | 0.026667 | 0.048780 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.382077 | 0.154601 | 0.000000 | `[[1635, 0], [407, 0]]` |
| 0.500000 | max_validation_f1 | 0.494844 | 0.546185 | 0.918699 | 0.212537 | 0.198695 | 0.972973 | 0.330000 | 0.619633 | 0.406723 | 0.976758 | `[[38, 1597], [11, 396]]` |
| 0.500000 | max_validation_f1_with_validation_fpr_le_0.10 | 0.499538 | 0.481013 | 0.097561 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.619633 | 0.406723 | 0.000000 | `[[1635, 0], [407, 0]]` |
| 0.500000 | max_validation_f1_with_validation_fpr_le_0.05 | 0.503697 | 0.425532 | 0.048780 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.619633 | 0.406723 | 0.000000 | `[[1635, 0], [407, 0]]` |
| 1.000000 | max_validation_f1 | 0.500888 | 0.558140 | 0.791328 | 0.304114 | 0.216760 | 0.953317 | 0.353209 | 0.659979 | 0.340694 | 0.857492 | `[[233, 1402], [19, 388]]` |
| 1.000000 | max_validation_f1_with_validation_fpr_le_0.10 | 0.857115 | 0.451613 | 0.097561 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.659979 | 0.340694 | 0.000000 | `[[1635, 0], [407, 0]]` |
| 1.000000 | max_validation_f1_with_validation_fpr_le_0.05 | 0.980583 | 0.391304 | 0.048780 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.659979 | 0.340694 | 0.000000 | `[[1635, 0], [407, 0]]` |

Confusion matrix layout is `[[TN, FP], [FN, TP]]`.
