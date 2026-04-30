# Exploratory Cluster 1 Repaired Dataset Note

This note documents a separate exploratory Cluster 1 data-construction variant. It does not replace the original Cluster 1 pipeline or outputs.

## File Split

Training/validation source files:

- `train1.csv`
- `train2.csv`
- `train3.csv`
- `test1.csv`
- `test2.csv`
- `test3.csv`

Held-out test source files:

- `test4.csv`
- `test5.csv`

The held-out files `test4.csv` and `test5.csv` are not used for training, validation, preprocessing-fit, descriptor computation, or agglomerative membership formation.

## Reason For The Variant

The official HAI `train1.csv`, `train2.csv`, and `train3.csv` files are normal-only in the current local dataset snapshot. The original all-file contiguous partitioning pipeline creates many attack-free clients, which weakens Cluster 1 supervised binary training and threshold selection.

This repaired variant creates windows first from the explicit training/validation source pool, then forms candidate clients with deterministic attack-aware positive-window distribution. The goal is to test whether a supervised Cluster 1 construction with attack coverage across clients improves the fixed architecture without changing the Cluster 1 model family, FL method, aggregation rule, or hierarchy.

## Scope

This is Cluster 1-only and exploratory. Cluster 2 and Cluster 3 are unchanged.

The architecture remains:

- Cluster 1 dataset: HAI 21.03
- Proposed Cluster 1 model: TCN
- Proposed Cluster 1 FL method: FedBN
- Proposed Cluster 1 aggregation: weighted non-BN mean
- Hierarchical runs: fixed K1 = 2 agglomerative sub-clusters

Generated pre-training artifacts for this variant live under:

```text
outputs_c1_repaired/
```
