# VALIDATION_CHECKS.md

## 1. Purpose

This document defines the validation and acceptance checks for the FCFL CPS/IIoT intrusion-detection implementation.

Every major implementation stage must be checked against these rules. The system is not acceptable unless all mandatory checks pass.

This validation checklist is aligned with the current report architecture and implementation specification.

---

## 2. Validation philosophy

Use fail-fast validation.

If a mandatory check fails:
- stop execution,
- raise a clear error,
- print the relevant diagnostic information,
- do not continue training.

---

## 3. Pre-run checks

## 3.1 Environment checks

Mandatory:
- Python environment available
- required packages import successfully
- output directories writable
- dataset root path exists

Pass condition:

```text
All configured paths and required imports succeed.
```

## 3.2 Configuration checks

Mandatory:
- all config files load successfully
- cluster-to-dataset mapping matches architecture contract
- clustering parameters exist for each main cluster
- ledger metadata-only flag is enabled

Pass condition:

```text
No missing required config field.
```

---

## 4. Data checks

## 4.1 Raw data availability

Mandatory:
- Cluster 1 raw folder exists and contains at least one CSV
- Cluster 2 deterministic combined telemetry CSV exists at the configured processed path
- Cluster 3 raw folder exists and contains at least one CSV

Pass condition:

```text
At least one valid CSV file exists in each required raw dataset folder.
```

## 4.2 Schema profiling

Mandatory:
- print available column names for each dataset
- detect configured label column
- detect excluded leakage columns

Pass condition:

```text
Configured label column exists and forbidden columns are identified.
```

## 4.3 Binary label validation

Mandatory:
- labels map to exactly `{0,1}`
- at least one normal sample exists
- at least one attack sample exists

Pass condition:

```text
Unique mapped labels == {0,1}
```

## 4.4 Feature exclusion validation

Mandatory:
- excluded columns are absent from model-input matrices
- WUSTL leakage fields are absent from model-input matrices
- process-specific attack labels are absent from Cluster 1 model input if present

Pass condition:

```text
No forbidden column remains in model features.
```

## 4.5 Split leakage validation

Mandatory:
- validation/test samples not used to fit imputers
- validation/test samples not used to fit scalers or encoders
- validation/test samples not used for clustering descriptors
- timestamp/order columns are absent from model-input matrices and descriptor matrices

Pass condition:

```text
All fitted preprocessing artifacts are training-only.
```

---

## 5. Leaf-client and clustering checks

## 5.1 Leaf-client count validation

Mandatory defaults:
- Cluster 1: 12 candidate leaf clients
- Cluster 2: 15 candidate leaf clients
- Cluster 3: 15 candidate leaf clients

Pass condition:

```text
Observed candidate leaf-client counts match configured values.
```

## 5.2 Local split validation

Mandatory:
- every candidate leaf client has a training split
- empty validation/test splits are logged if unavoidable
- no leaf client uses samples outside its assigned partition

Pass condition:

```text
Every client has valid local split metadata.
```

## 5.3 Descriptor validation

Mandatory:
- descriptor is computed only from training features
- descriptor shape is consistent inside each main cluster
- no labels included in descriptor vectors
- no forbidden columns included in descriptor vectors

Pass condition:

```text
Descriptor matrix shape == [num_clients, descriptor_dim]
```

## 5.4 Agglomerative membership validation

Mandatory:
- Cluster 1 has exactly 2 sub-clusters
- Cluster 2 has exactly 3 sub-clusters
- Cluster 3 has exactly 3 sub-clusters
- no sub-cluster is empty
- each leaf client belongs to exactly one sub-cluster
- memberships are saved to disk

Pass condition:

```text
Membership mapping is complete, unique, non-empty, and matches K_m.
```

## 5.5 Fixed-membership validation

Mandatory:
- membership files are created before round 1
- membership files are reused for hierarchical baseline and proposed method
- membership files do not change across training rounds
- no clustering code is called after the first clustering stage

Pass condition:

```text
Membership hash before round 1 == membership hash after final round.
```

---

## 6. Architecture checks

## 6.1 No-cross-cluster-averaging validation

Mandatory:
- Cluster 1 parameters are never averaged with Cluster 2 or Cluster 3
- Cluster 2 parameters are never averaged with Cluster 1 or Cluster 3
- Cluster 3 parameters are never averaged with Cluster 1 or Cluster 2

Pass condition:

```text
No aggregation function is called with inputs from different main clusters.
```

## 6.2 Hierarchy validation

Mandatory:
- all leaf clients are under sub-clusters in hierarchical runs
- sub-cluster heads receive parent model from main-cluster head
- main-cluster head aggregates only sub-cluster outputs

Pass condition:

```text
Training graph follows main-cluster head -> sub-cluster head -> leaf clients and back upward.
```

---

## 7. Method-specific checks

## 7.1 FedBN checks (Cluster 1)

Mandatory:
- BN-related parameters are excluded from aggregation
- BN state remains local
- only non-BN parameters are weighted-averaged

Pass condition:

```text
No BatchNorm running statistics appear in aggregated parameter set.
```

## 7.2 FedProx checks (Cluster 2)

Mandatory:
- local loss includes proximal penalty
- proximal reference uses the incoming sub-cluster model
- aggregation remains weighted arithmetic mean

Pass condition:

```text
Local loss == base loss + proximal term.
```

## 7.3 SCAFFOLD checks (Cluster 3)

Mandatory:
- main-cluster server control variate exists
- each participating leaf client has local control variate state
- control-variate increment is computed
- sub-cluster heads forward control-variate increments upward
- main-cluster head updates control variate

Pass condition:

```text
Control variates exist, update each round, and are not all zero after training starts.
```

---

## 8. Aggregation checks

## 8.1 Weighted mean correctness

Mandatory:
- sub-cluster aggregation equals hand-computed weighted average on a small deterministic test case
- main-cluster aggregation equals hand-computed weighted average on a small deterministic test case

Pass condition:

```text
Numerical difference <= tolerance.
```

Recommended tolerance:

```text
1e-6
```

## 8.2 Hierarchical aggregation integrity

Mandatory:
- parent model for round `t+1` equals the aggregated result from current sub-cluster models
- sub-cluster initialization for round `t` uses current parent model

Pass condition:

```text
w_{m,s}^t is initialized from w_m^t and w_m^{t+1} matches sub-cluster aggregate.
```

---

## 9. Ledger / blockchain checks

## 9.1 Ledger-writer validation

Mandatory:
- only main-cluster heads create ledger records
- sub-cluster heads do not create ledger records
- leaf clients do not create ledger records

Pass condition:

```text
All ledger records are attributed to main-cluster heads only.
```

## 9.2 Metadata-only validation

Mandatory:
- ledger records contain only allowed metadata fields
- raw records are absent
- full model weights are absent
- optimizer state is absent
- raw descriptor vectors are absent

Pass condition:

```text
Ledger payload schema matches approved metadata schema only.
```

## 9.3 Required ledger-field validation

Mandatory fields:
- round id
- cluster id
- cluster head id
- model version
- previous main model hash
- new main model hash
- clustering method
- clustering configuration hash
- subcluster count
- subcluster membership hash
- FL method
- aggregation rule
- effective sample count
- participant count
- timestamp start
- timestamp end
- submitter identity

Pass condition:

```text
Every ledger record contains all mandatory fields.
```

---

## 10. Experiment checks

## 10.1 Baseline coverage

Mandatory:
- Baseline A runs for Cluster 1, Cluster 2, Cluster 3
- Baseline B runs for Cluster 1, Cluster 2, Cluster 3
- proposed specialized run exists for Cluster 1, Cluster 2, Cluster 3

Pass condition:

```text
All required experiment IDs have output directories and metrics files.
```

## 10.2 Ablation coverage

Mandatory:
- hierarchy ablation available (flat vs hierarchical)
- method ablation available for Cluster 1 (FedAvg vs FedBN)
- method ablation available for Cluster 2 (FedAvg vs FedProx)
- method ablation available for Cluster 3 (FedAvg vs SCAFFOLD), using `AB_C3_FEDAVG_CNN1D` or an equivalent reuse of `B_C3` against `P_C3`

Pass condition:

```text
All required ablation comparisons are computable from saved runs.
```

## 10.3 Metrics output validation

Mandatory:
- metrics CSV exists for every executed experiment
- summary CSV exists
- no experiment silently skips metrics export

Pass condition:

```text
All expected metrics files exist and are non-empty.
```

---

## 11. Output checks

## 11.1 Required artifacts

Mandatory outputs:
- preprocessing scalers
- client metadata files
- clustering membership files
- model checkpoints
- ledger files
- metrics CSVs
- summary CSV
- plots

Pass condition:

```text
All required artifact paths exist after run completion.
```

## 11.2 Reproducibility checks

Mandatory:
- seed is logged
- config file copy is saved in run directory
- clustering membership hash is logged

Pass condition:

```text
Run directory contains reproducibility metadata.
```

---

## 12. Acceptance criteria summary

The implementation is acceptable only if:

1. all mandatory schema checks pass,
2. labels are correctly binary,
3. fixed sub-clusters are formed once and frozen,
4. no cross-cluster averaging occurs,
5. FedBN excludes BN state from aggregation,
6. FedProx includes the proximal term,
7. SCAFFOLD maintains and updates control variates,
8. only main-cluster heads log metadata,
9. ledger contains metadata only,
10. all required experiment outputs are generated.

If any of these fail, the implementation is not ready for final experiments.
