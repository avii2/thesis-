# Cluster 1 Repaired File Split Report

This report is for the exploratory Cluster 1 repaired supervised variant only.

## Train/Validation Source Files

| file | rows | attack=0 | attack=1 |
|---|---:|---:|---:|
| train1.csv | 216001 | 216001 | 0 |
| train2.csv | 226801 | 226801 | 0 |
| train3.csv | 478801 | 478801 | 0 |
| test1.csv | 43201 | 42572 | 629 |
| test2.csv | 118801 | 115352 | 3449 |
| test3.csv | 108001 | 106466 | 1535 |

## Held-Out Test Source Files

| file | rows | attack=0 | attack=1 |
|---|---:|---:|---:|
| test4.csv | 39601 | 38444 | 1157 |
| test5.csv | 92401 | 90224 | 2177 |

## Window Counts

- Train/validation source windows: `{'0': 148103, '1': 829}`
- Held-out test windows: `{'0': 16013, '1': 481}`

`test4.csv` and `test5.csv` are not used for training, validation, preprocessing-fit, or descriptor computation.
