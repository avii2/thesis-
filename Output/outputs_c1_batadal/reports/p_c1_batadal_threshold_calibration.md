# P_C1_BATADAL Threshold Calibration

This report is limited to Cluster 1 P_C1_BATADAL threshold calibration.

Protocol:
- Thresholds are selected on validation predictions only.
- Held-out BATADAL test predictions are evaluated only after each threshold is fixed.
- The BATADAL train/validation/test split, windowing, and preprocessing are unchanged.
- Test data is not used for threshold tuning.

Threshold sweep CSV: `outputs_c1_batadal/calibration/p_c1_batadal_threshold_sweep.csv`

| mode | threshold | validation_f1 | validation_fpr | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | test_confusion_matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| max_validation_f1 | 0.500888 | 0.558140 | 0.791328 | 0.304114 | 0.216760 | 0.953317 | 0.353209 | 0.659979 | 0.340694 | 0.857492 | `[[233, 1402], [19, 388]]` |
| max_validation_f1_with_validation_fpr_le_0.10 | 0.857115 | 0.451613 | 0.097561 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.659979 | 0.340694 | 0.000000 | `[[1635, 0], [407, 0]]` |
| max_validation_f1_with_validation_fpr_le_0.05 | 0.980583 | 0.391304 | 0.048780 | 0.800686 | 0.000000 | 0.000000 | 0.000000 | 0.659979 | 0.340694 | 0.000000 | `[[1635, 0], [407, 0]]` |

Confusion matrix layout is `[[TN, FP], [FN, TP]]`.
