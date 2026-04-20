# Project Rules for Codex

This repository implements a simulation-only FCFL intrusion-detection system.

Non-negotiable architecture:
- Exactly 3 main clusters.
- One fixed sub-cluster layer per main cluster.
- Agglomerative Clustering is run once before training.
- No reclustering during FL training.
- No cross-cluster parameter averaging.
- Raw data remain local to leaf clients.
- Only main-cluster heads write metadata to the ledger.
- Ledger stores metadata only, never raw data and never full model weights.

Cluster definitions:
- Cluster 1: HAI 21.03, TCN, FedBN, weighted non-BN aggregation.
- Cluster 2: TON IoT combined telemetry, compact MLP, FedProx, weighted arithmetic aggregation.
- Cluster 3: WUSTL-IIOT-2021, 1D-CNN, SCAFFOLD, weighted arithmetic aggregation.

Do not invent dataset columns.
Do not invent labels.
Do not download datasets unless explicitly instructed.
If required files or columns are missing, raise a clear error.
Do not change architecture without user approval.
Every implementation must include tests.