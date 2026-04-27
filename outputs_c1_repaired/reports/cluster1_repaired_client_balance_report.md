# Cluster 1 Repaired Client Balance Report

Positive windows are distributed round-robin across the 12 candidate clients before negative windows are used to fill deterministic client quotas.

| client | train 0 | train 1 | validation 0 | validation 1 | held-out test 0 | held-out test 1 |
|---|---:|---:|---:|---:|---:|---:|
| C1_L001 | 10490 | 60 | 1851 | 10 | 1302 | 73 |
| C1_L002 | 10491 | 59 | 1851 | 10 | 1302 | 73 |
| C1_L003 | 10491 | 59 | 1851 | 10 | 1356 | 19 |
| C1_L004 | 10491 | 59 | 1851 | 10 | 1350 | 25 |
| C1_L005 | 10491 | 59 | 1851 | 10 | 1339 | 36 |
| C1_L006 | 10491 | 59 | 1851 | 10 | 1320 | 55 |
| C1_L007 | 10491 | 59 | 1851 | 10 | 1322 | 52 |
| C1_L008 | 10491 | 59 | 1851 | 10 | 1361 | 13 |
| C1_L009 | 10491 | 59 | 1851 | 10 | 1330 | 44 |
| C1_L010 | 10491 | 59 | 1851 | 10 | 1374 | 0 |
| C1_L011 | 10491 | 59 | 1851 | 10 | 1374 | 0 |
| C1_L012 | 10491 | 59 | 1851 | 10 | 1283 | 91 |

Clients with at least one positive train-or-validation window: `12/12`
