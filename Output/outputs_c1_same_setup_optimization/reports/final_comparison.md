# Cluster 1 Same-Setup Optimization Final Comparison

This is a same-setup optimized Cluster 1 run. It keeps the current proposed Cluster 1 model family, FedBN method, weighted non-BN aggregation, fixed hierarchy, agglomerative membership role, no-cross-cluster averaging, and ledger logic unchanged.

Selection used validation F1 only. Held-out `test4.csv` and `test5.csv` were attached only for the final selected candidate evaluation.

## Selected Candidate

- Candidate: `no_constants_no_corr_w64_s8_r20to1_pcw0p5_lr0p003_do0p05`
- Feature variant: `no_constants_no_corr`
- Balance ratio: `20:1`
- Positive class weight scale: `0.5`
- Window: `64` rows, stride `8`
- Learning rate/dropout: `0.003` / `0.05`
- Validation F1 used for selection: `0.786469`

## Final Held-Out Test Metrics

- test_accuracy: `0.980650`
- test_precision: `0.790404`
- test_recall: `0.570128`
- test_f1: `0.662434`
- test_auroc: `0.855590`
- test_pr_auc: `0.645929`
- test_fpr: `0.005208`
- confusion_matrix: `[[15854, 83], [236, 313]]`
- threshold_used: `0.519757`

## Comparisons

| row | validation_f1 | test_f1 | test_recall | test_precision | test_fpr | confusion_matrix |
|---|---:|---:|---:|---:|---:|---|
| `same_setup_optimized_P_C1` | 0.786469 | 0.662434 | 0.570128 | 0.790404 | 0.005208 | `[[15854, 83], [236, 313]]` |
| `original_A_C1` | 0.702703 | 0.562162 | 0.406250 | 0.912281 | 0.000408 | `[[24514, 10], [152, 104]]` |
| `original_B_C1` | 0.702703 | 0.562162 | 0.406250 | 0.912281 | 0.000408 | `[[24514, 10], [152, 104]]` |
| `original_P_C1` | 0.563536 | 0.493151 | 0.351562 | 0.825688 | 0.000775 | `[[24505, 19], [166, 90]]` |
| `best_balanced_B_C1` | 0.752475 | 0.627936 | 0.528067 | 0.774390 | 0.004621 | `[[15939, 74], [227, 254]]` |
| `best_balanced_P_C1` | 0.655827 | 0.421669 | 0.299376 | 0.712871 | 0.003622 | `[[15955, 58], [337, 144]]` |

## Required Answers

1. Did optimized same-setup `P_C1` beat original `P_C1`? `YES`.
2. Did it beat original `B_C1`? `YES`.
3. Did it beat balanced `B_C1` if available? `YES`.
4. Did it improve recall without exploding FPR? `YES`. Recall improved=`True`, FPR exploded=`False`.
5. Is this strong enough to keep as final Cluster 1 proposed method? `YES`.

## Search Scope

- Search rows written: `116`
- Full-search rows considered for selection: `10`
- Selection rule: validation F1, then validation PR-AUC, then lower validation FPR, then lower wall-clock time.
