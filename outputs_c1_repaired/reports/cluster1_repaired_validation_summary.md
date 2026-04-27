# Cluster 1 Repaired Validation Summary

This summary validates only the exploratory repaired Cluster 1 dataset-construction variant.

## Window Counts

- Repaired training/validation source windows: `{'0': 148103, '1': 829}`
- Repaired held-out test windows: `{'0': 16013, '1': 481}`

## Training Pool Class Counts

- Negative windows: `148103`
- Positive windows: `829`

## Held-Out Test Pool Class Counts

- Negative windows: `16013`
- Positive windows: `481`

## Client Attack Coverage

- Clients with at least one positive training window: `12/12`
- Every client received at least one positive training window: `YES`
- Attack-free training clients: `0`
- Attack-free training client IDs: `[]`

## Leakage Check

- `test4.csv` and `test5.csv` are used only for held-out test-window construction.
- They are not used for training, validation, preprocessing fit, or descriptor computation.
