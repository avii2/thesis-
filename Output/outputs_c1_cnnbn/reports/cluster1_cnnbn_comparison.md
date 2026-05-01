# Cluster 1 CNN1D-BN Comparison

This report was generated from the metrics CSV files listed below. No conclusions are fabricated beyond the stated decision rule.

## Source files

- Old A_C1: `outputs/metrics/A_C1_metrics.csv`
- Old B_C1: `outputs/metrics/B_C1_metrics.csv`
- Old P_C1: `outputs/metrics/P_C1_metrics.csv`
- New P_C1: `outputs_c1_cnnbn/metrics/P_C1_metrics.csv`
- New AB_C1_FEDAVG_CNNBN: `outputs_c1_cnnbn/metrics/AB_C1_FEDAVG_CNNBN_metrics.csv`

## Metrics

| run | model_family | best_validation_f1 | threshold_used | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | confusion_matrix | wall_clock_training_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Old A_C1 | cnn1d | 0.702703 | 0.964214 | 0.993462 | 0.912281 | 0.406250 | 0.562162 | 0.899771 | 0.519445 | 0.000408 | [[24514, 10], [152, 104]] | 106.074 |
| Old B_C1 | cnn1d | 0.702703 | 0.963231 | 0.993462 | 0.912281 | 0.406250 | 0.562162 | 0.899705 | 0.519672 | 0.000408 | [[24514, 10], [152, 104]] | 101.491 |
| Old P_C1 | tcn | 0.563536 | 0.992987 | 0.992534 | 0.825688 | 0.351562 | 0.493151 | 0.838881 | 0.419831 | 0.000775 | [[24505, 19], [166, 90]] | 977.387 |
| New P_C1 | cnn1d_bn | 0.598870 | 0.948396 | 0.992211 | 0.748031 | 0.371094 | 0.496084 | 0.764713 | 0.403029 | 0.001305 | [[24492, 32], [161, 95]] | 1221.018 |
| New AB_C1_FEDAVG_CNNBN | cnn1d_bn | 0.719101 | 0.804828 | 0.993705 | 0.852113 | 0.472656 | 0.608040 | 0.780115 | 0.524537 | 0.000856 | [[24503, 21], [135, 121]] | 1172.430 |

## Old P_C1 vs New P_C1

- Old P_C1 model_family: `tcn`
- New P_C1 model_family: `cnn1d_bn`
- Test F1 delta: +0.002933
- PR-AUC delta: -0.016803
- Wall-clock delta seconds: +243.631

## FedBN vs FedAvg With CNN1D-BN

- FedBN helped over FedAvg: no
- FedBN P_C1 test F1=0.496084, PR-AUC=0.403029; FedAvg ablation test F1=0.608040, PR-AUC=0.524537.
- Test F1 delta, FedBN minus FedAvg: -0.111957
- PR-AUC delta, FedBN minus FedAvg: -0.121509

## Decision

- Decision: Partial improvement
- Reason: New P_C1 test F1=0.496084 improves over old P_C1=0.493151, but remains below B_C1=0.562162; PR-AUC=0.403029 also remains below the success threshold.
- Replacement recommendation: do not replace TCN with CNN1D-BN as the proposed FedBN P_C1 under these hyperparameters; it is only a partial improvement over old P_C1 and does not beat B_C1.

## Run Configuration Notes

- New P_C1: CNN1D-BN + FedBN + weighted non-BN mean.
- New AB_C1_FEDAVG_CNNBN: CNN1D-BN + FedAvg + weighted arithmetic mean.
- Learning rate: 0.003 for both new runs.
- CNN1D-BN channels: [32, 64, 64]; kernel sizes: [5, 3, 3]; hidden dim: 32; dropout: 0.1.
- positive_class_weight_scale: 1.0 for P_C1; the ablation uses the same computed Cluster 1 positive class weight scale of 1.0.

