# Cluster 1 BATADAL Client Balance Report

This is controlled federated emulation. BATADAL is one water-distribution SCADA dataset; the 12 leaf clients are deterministic emulated clients, not physical independent water utilities.

Positive training windows are assigned round-robin across clients. Negative windows are then used to fill client quotas while preserving chronological order as much as practical.

| client | train 0 | train 1 | validation 0 | validation 1 | test 0 | test 1 |
|---|---:|---:|---:|---:|---:|---:|
| C1_L001 | 995 | 24 | 48 | 0 | 171 | 0 |
| C1_L002 | 995 | 24 | 48 | 0 | 101 | 70 |
| C1_L003 | 995 | 24 | 27 | 21 | 170 | 0 |
| C1_L004 | 995 | 24 | 0 | 48 | 105 | 65 |
| C1_L005 | 995 | 24 | 23 | 25 | 139 | 31 |
| C1_L006 | 995 | 24 | 48 | 0 | 139 | 31 |
| C1_L007 | 995 | 24 | 35 | 13 | 160 | 10 |
| C1_L008 | 995 | 24 | 0 | 48 | 80 | 90 |
| C1_L009 | 994 | 24 | 0 | 48 | 165 | 5 |
| C1_L010 | 994 | 24 | 46 | 1 | 95 | 75 |
| C1_L011 | 994 | 24 | 47 | 0 | 170 | 0 |
| C1_L012 | 994 | 24 | 47 | 0 | 140 | 30 |

Clients with at least one positive training window: `12/12`
Every client has positive training windows: `YES`
