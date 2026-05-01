# P_C1_BATADAL Tuning Comparison

Scope: Cluster 1 BATADAL only. The FCFL architecture, fixed sub-clusters, FedBN method, weighted non-BN aggregation, and BATADAL split are unchanged.

Comparison CSV: `outputs_c1_batadal_tuned/reports/p_c1_batadal_tuning_comparison.csv`

Decision rule: select using validation metrics only.
Applied rule: highest validation F1 with validation FPR <= 0.10.

## Comparison

| label | validation_f1 | validation_fpr | threshold | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | confusion_matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A_C1_BATADAL | 0.633929 | 0.276423 | 0.502414 | 0.681195 | 0.221461 | 0.238329 | 0.229586 | 0.563899 | 0.283908 | 0.208563 | `[[1294, 341], [310, 97]]` |
| B_C1_BATADAL | 0.634146 | 0.281843 | 0.501894 | 0.671890 | 0.219616 | 0.253071 | 0.235160 | 0.566460 | 0.285721 | 0.223853 | `[[1269, 366], [304, 103]]` |
| original P_C1_BATADAL | 0.558140 | 0.791328 | 0.500888 | 0.304114 | 0.216760 | 0.953317 | 0.353209 | 0.659979 | 0.340694 | 0.857492 | `[[233, 1402], [19, 388]]` |
| tuned P_C1_BATADAL scale_010_lr_0001_dropout_020 | 0.378378 | 0.097561 | 0.497906 | 0.814887 | 0.628319 | 0.174447 | 0.273077 | 0.628142 | 0.406597 | 0.025688 | `[[1593, 42], [336, 71]]` |
| tuned P_C1_BATADAL scale_025_lr_0001_dropout_010 | 0.484076 | 0.092141 | 0.499067 | 0.810970 | 0.769231 | 0.073710 | 0.134529 | 0.625072 | 0.410231 | 0.005505 | `[[1626, 9], [377, 30]]` |

## Final Selection

Selected run: `tuned P_C1_BATADAL scale_025_lr_0001_dropout_010`
Best hyperparameters: `{"batch_size": 64, "dropout": 0.1, "learning_rate": 0.001, "local_epochs": 1, "positive_class_weight_scale": 0.25, "rounds": 50, "seed": 42}`
Selected threshold mode: `max_validation_f1_with_validation_fpr_le_0.10`
Selected threshold: `0.499067`

Final test metrics:
- accuracy: 0.810970
- precision: 0.769231
- recall: 0.073710
- F1: 0.134529
- AUROC: 0.625072
- PR-AUC: 0.410231
- FPR: 0.005505
- confusion matrix: `[[1626, 9], [377, 30]]`

Beats A_C1_BATADAL on test F1: False.
Beats B_C1_BATADAL on test F1: False.
FPR acceptable under test FPR <= 0.20: True.

Exact final recommendation: Use `tuned P_C1_BATADAL scale_025_lr_0001_dropout_010` with threshold mode `max_validation_f1_with_validation_fpr_le_0.10` and threshold `0.499067`.
