# Cluster 1 Balanced Training Comparison

All balanced rows are marked as Cluster 1 balanced-training variants. Ratio selection uses validation F1 only; held-out test metrics are reported after that selection.

## Metrics

| experiment | ratio | validation_f1 | validation_pr_auc | validation_fpr | threshold | test_f1 | test_recall | test_precision | test_fpr | confusion_matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `A_C1` | original | 0.702703 | 0.610110 | 0.000648 | 0.964214 | 0.562162 | 0.406250 | 0.912281 | 0.000408 | `[[24514, 10], [152, 104]]` |
| `B_C1` | original | 0.702703 | 0.602385 | 0.000648 | 0.963231 | 0.562162 | 0.406250 | 0.912281 | 0.000408 | `[[24514, 10], [152, 104]]` |
| `P_C1` | original | 0.563536 | 0.486787 | 0.001054 | 0.992987 | 0.493151 | 0.351562 | 0.825688 | 0.000775 | `[[24505, 19], [166, 90]]` |
| `B_C1_BAL_1TO1` | ratio_1to1 | 0.293706 | 0.221326 | 0.001507 | 0.530884 | 0.134875 | 0.072765 | 0.921053 | 0.000187 | `[[16010, 3], [446, 35]]` |
| `P_C1_BAL_1TO1` | ratio_1to1 | 0.050267 | 0.020331 | 0.079560 | 0.505472 | 0.072299 | 0.191268 | 0.044574 | 0.123150 | `[[14041, 1972], [389, 92]]` |
| `B_C1_BAL_2TO1` | ratio_2to1 | 0.726316 | 0.641584 | 0.001356 | 0.564688 | 0.303211 | 0.422037 | 0.236597 | 0.040904 | `[[15358, 655], [278, 203]]` |
| `P_C1_BAL_2TO1` | ratio_2to1 | 0.231511 | 0.141193 | 0.003842 | 0.516010 | 0.170330 | 0.128898 | 0.251012 | 0.011553 | `[[15828, 185], [419, 62]]` |
| `B_C1_BAL_3TO1` | ratio_3to1 | 0.731579 | 0.656879 | 0.001281 | 0.665357 | 0.549512 | 0.409563 | 0.834746 | 0.002436 | `[[15974, 39], [284, 197]]` |
| `P_C1_BAL_3TO1` | ratio_3to1 | 0.278912 | 0.229268 | 0.002185 | 0.522559 | 0.212257 | 0.147609 | 0.377660 | 0.007307 | `[[15896, 117], [410, 71]]` |
| `B_C1_BAL_5TO1` | ratio_5to1 | 0.740000 | 0.677531 | 0.002110 | 0.622630 | 0.594796 | 0.498960 | 0.736196 | 0.005371 | `[[15927, 86], [241, 240]]` |
| `P_C1_BAL_5TO1` | ratio_5to1 | 0.437870 | 0.400160 | 0.003014 | 0.532368 | 0.293436 | 0.237006 | 0.385135 | 0.011366 | `[[15831, 182], [367, 114]]` |
| `B_C1_BAL_8TO1` | ratio_8to1 | 0.750630 | 0.691700 | 0.001808 | 0.663367 | 0.604651 | 0.486486 | 0.798635 | 0.003685 | `[[15954, 59], [247, 234]]` |
| `P_C1_BAL_8TO1` | ratio_8to1 | 0.633431 | 0.556506 | 0.000678 | 0.565156 | 0.383436 | 0.259875 | 0.730994 | 0.002873 | `[[15967, 46], [356, 125]]` |
| `B_C1_BAL_10TO1` | ratio_10to1 | 0.752475 | 0.700841 | 0.002110 | 0.658564 | 0.627936 | 0.528067 | 0.774390 | 0.004621 | `[[15939, 74], [227, 254]]` |
| `P_C1_BAL_10TO1` | ratio_10to1 | 0.655827 | 0.596174 | 0.001808 | 0.589798 | 0.421669 | 0.299376 | 0.712871 | 0.003622 | `[[15955, 58], [337, 144]]` |

## Decisions

1. Best proposed validation F1 ratio: `ratio_10to1` (`validation_f1=0.655827`).
2. Balanced proposed Cluster 1 improved over original `P_C1` on selected held-out test F1: `NO`.
3. Balanced proposed Cluster 1 beat original `B_C1` on selected held-out test F1: `NO`.
4. Balanced proposed Cluster 1 beat the balanced hierarchical baseline using the same ratio: `NO`.
5. Did balancing reduce false negatives versus original `P_C1`? `NO`. Raw FN delta is `171`; original and balanced rows use different test-window pools, so recall/F1 are the safer comparison.
6. Did balancing create too many false positives versus original `P_C1`? `YES`. Raw FP delta is `39` and selected FPR is `0.003622`.
7. Validation-selected ratio for the balanced proposed variant: `ratio_10to1`. Do not use it as a replacement for original `P_C1` because it did not improve held-out test F1.
8. The improvement is defensible without test leakage because ratio and threshold selection used validation only, and `test4.csv`/`test5.csv` were held out from training, preprocessing fit, clustering, and threshold tuning.
