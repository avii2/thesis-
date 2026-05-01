# Cluster 1 Same-Setup Optimization

This document tracks a same-setup optimization sweep for proposed Cluster 1. The sweep changes only leakage-safe dataset construction, feature selection, window setting, training-window balance, positive-class weighting scale, learning rate, and dropout.

Unchanged architecture and protocol:
- Exactly the current proposed Cluster 1 model family from `configs/proposed.yaml` is used: CNN1D-BN.
- FL method remains FedBN.
- Aggregation remains weighted non-BN mean.
- The fixed sub-cluster hierarchy, agglomerative membership role, no-cross-cluster averaging, and ledger metadata-only rules are unchanged.
- Cluster 2 and Cluster 3 are not touched.

Leakage controls:
- Training source: `train1.csv`, `train2.csv`, `train3.csv`, `test1.csv`, `test2.csv`.
- Validation and selection: `test3.csv`.
- Held-out test: `test4.csv`, `test5.csv`.
- Held-out test windows are omitted during fast and full candidate selection and attached only to the final selected rerun.
- Feature pruning and ranking use training source plus validation only.
- Imputation and scaling are fitted on balanced training windows only.
- Validation and held-out test windows are not balanced.

Generated reports:
- `outputs_c1_same_setup_optimization/reports/search_results.csv`
- `outputs_c1_same_setup_optimization/reports/best_config.json`
- `outputs_c1_same_setup_optimization/reports/final_comparison.md`
