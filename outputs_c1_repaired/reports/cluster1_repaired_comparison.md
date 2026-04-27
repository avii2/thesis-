# Cluster 1 Repaired Experiment Comparison

Source files for original metrics are `outputs/metrics/*_C1_metrics.csv`; source files for repaired metrics are `outputs_c1_repaired/metrics/*_C1_REPAIRED_metrics.csv`.
The repaired variant uses `test4.csv` and `test5.csv` only for held-out testing.

## Metrics

| method | dataset pipeline | experiment_id | best_validation_round | validation_f1 | threshold_used | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | confusion_matrix | wall_clock_training_seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A | original | `A_C1` | 40.000000 | 0.702703 | 0.964214 | 0.993462 | 0.912281 | 0.406250 | 0.562162 | 0.899771 | 0.519445 | 0.000408 | [[24514, 10], [152, 104]] | 106.073551 |
| A | repaired | `A_C1_REPAIRED` | 27.000000 | 0.773585 | 0.999996 | 0.919668 | 0.204482 | 0.607069 | 0.305919 | 0.818402 | 0.178932 | 0.070942 | [[14877, 1136], [189, 292]] | 228.544650 |
| B | original | `B_C1` | 39.000000 | 0.702703 | 0.963231 | 0.993462 | 0.912281 | 0.406250 | 0.562162 | 0.899705 | 0.519672 | 0.000408 | [[24514, 10], [152, 104]] | 101.491112 |
| B | repaired | `B_C1_REPAIRED` | 2.000000 | 0.736318 | 0.998605 | 0.982600 | 0.921739 | 0.440748 | 0.596343 | 0.851439 | 0.640553 | 0.001124 | [[15995, 18], [269, 212]] | 186.757588 |
| P | original | `P_C1` | 40.000000 | 0.563536 | 0.992987 | 0.992534 | 0.825688 | 0.351562 | 0.493151 | 0.838881 | 0.419831 | 0.000775 | [[24505, 19], [166, 90]] | 977.387329 |
| P | repaired | `P_C1_REPAIRED` | 49.000000 | 0.726316 | 0.997828 | 0.778707 | 0.087692 | 0.700624 | 0.155874 | 0.823702 | 0.201311 | 0.218947 | [[12507, 3506], [144, 337]] | 1188.463370 |

## Deltas

| method | test_f1_delta_repaired_minus_original | test_recall_delta | test_precision_delta | test_fpr_delta |
|---|---:|---:|---:|---:|
| A | -0.256243 | 0.200819 | -0.707799 | 0.070535 |
| B | 0.034181 | 0.034498 | 0.009458 | 0.000716 |
| P | -0.337276 | 0.349061 | -0.737996 | 0.218172 |

## Required Answers

1. Did the repaired dataset pipeline improve P_C1? NO. P_C1 test F1 delta: -0.337276.
2. Did it improve the proposed-vs-baseline gap? NO. Original gap: -0.069011; repaired gap: -0.440469.
3. Did it reduce the extreme precision/recall tradeoff? NO. Original |precision-recall|: 0.474126; repaired |precision-recall|: 0.612932.
4. Should this repaired Cluster 1 dataset pipeline replace the original one? NO based on this single-seed comparison.
5. If results are still weak, is the next recommended step a model-family change? NO: repaired P_C1 underperforms the repaired baseline on F1, but AUROC is high enough that calibration/thresholding and imbalance should be reviewed before a model-family change.

## Notes

- Selection uses validation F1; test metrics are reported after the selected validation round and threshold are fixed.
- This report does not use screenshots or manually entered results.
