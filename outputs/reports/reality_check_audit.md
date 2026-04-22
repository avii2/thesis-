# Dataset Audit Reality Check

Generated at UTC: 2026-04-21T17:01:55.115532+00:00

## Confirmed

- Cluster 1 (HAI 21.03): 8 CSV file(s), 1323608 row(s), schema_consistent=true.
- Cluster 1 confirmed label column: `attack` with mapped counts {'0': 1314661, '1': 8947}.
- Cluster 2 (TON IoT): 3 CSV file(s), 118491 row(s), schema_consistent=false.
- Cluster 3 (WUSTL-IIOT-2021): 1 CSV file(s), 1194464 row(s), schema_consistent=true.

## Requires User Confirmation / Follow-Up

- Cluster 2: CSV schemas differ across files in this cluster.
- Cluster 2: The combined telemetry training table is not present. A deterministic build step is required before Cluster 2 training.
- Cluster 2: label column still requires user confirmation. Observed candidates: label, type.
- Cluster 3: label column still requires user confirmation. Observed candidates: Target, Traffic.

## Local Layout Notes

- The audited repo uses repo-local data under `data/raw/`, not an external `desktop/thesis/data/` root.
- The current local TON and WUSTL directories have leading spaces in their names: `data/raw/ ton_iot/` and `data/raw/ wustl_iiot_2021/`.
- Cluster 2 does not yet have a `combined_telemetry/` directory or a unified combined telemetry CSV in the audited local repo.
