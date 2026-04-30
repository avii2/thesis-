# Report Patch: Cluster 1 CNN1D-BN Replacement

Apply these replacements in the thesis/report source if it exists outside this repository.

| Old text | Replacement |
|---|---|
| `Cluster 1: HAI 21.03, TCN, FedBN, weighted non-BN aggregation` | `Cluster 1: HAI 21.03, CNN1D-BN, FedBN, weighted non-BN aggregation` |
| `TCN + FedBN` | `CNN1D-BN + FedBN` |
| `TCN/FedBN` | `CNN1D-BN/FedBN` |
| `P_C1 model = TCN` | `P_C1 model = CNN1D-BN` |
| `P_C1, HAI 21.03, TCN, FedBN` | `P_C1, HAI 21.03, CNN1D-BN, FedBN` |
| `AB_C1_FEDAVG_TCN` | `AB_C1_FEDAVG_CNNBN` |
| `Cluster 1 FedAvg TCN control` | `Cluster 1 FedAvg CNN1D-BN control` |
| `Cluster 1 FedAvg vs FedBN: TCN` | `Cluster 1 FedAvg vs FedBN: CNN1D-BN` |

Do not change the Cluster 1 dataset, hierarchy, clustering method, fixed memberships, FedBN method, weighted non-BN aggregation rule, positive-class weighting, or threshold-tuning description.
