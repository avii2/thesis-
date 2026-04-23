# PLOTS README

## Regeneration
`cd /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids && python3 scripts/generate_ieee_plots.py`

## Plot Inputs

- The plots use only already-generated files under `outputs/`.
- `outputs/predictions/` present: `YES`.

## Per-Plot Sources

### fig_f1_comparison
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/metrics/summary_all_experiments_mean_std.csv
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_auc_comparison
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/metrics/summary_all_experiments_mean_std.csv
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_fpr_comparison
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/metrics/summary_all_experiments_mean_std.csv
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_ablation_delta_f1
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/metrics/summary_all_experiments_mean_std.csv
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_convergence_cluster1
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/A_C1/seed_42/round_metrics.csv, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/B_C1/seed_42/round_metrics.csv, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/P_C1/seed_42/round_metrics.csv
- Seed mode: Cluster 1: single available seed only; no shaded std band.

### fig_convergence_cluster2
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/A_C2/seed_42/round_metrics.csv, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/B_C2/seed_42/round_metrics.csv, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/P_C2/seed_42/round_metrics.csv
- Seed mode: Cluster 2: single available seed only; no shaded std band.

### fig_convergence_cluster3
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/A_C3/seed_42/round_metrics.csv, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/B_C3/seed_42/round_metrics.csv, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/runs/P_C3/seed_42/round_metrics.csv
- Seed mode: Cluster 3: single available seed only; no shaded std band.

### fig_communication_cost
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/metrics/summary_all_experiments_mean_std.csv
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_ledger_overhead
- Data files used: /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/AB_C1_FEDAVG_TCN_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/AB_C2_FEDAVG_MLP_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/AB_C3_FEDAVG_CNN1D_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/A_C1_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/A_C2_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/A_C3_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/B_C1_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/B_C2_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/B_C3_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/P_C1_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/P_C2_ledger.jsonl, /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids/outputs/ledgers/P_C3_ledger.jsonl
- Seed mode: Ledger overhead computed from real JSONL metadata ledgers only; size from file bytes and latency from timestamp_end - timestamp_start.

## Missing Inputs
- AB_C1_FEDAVG_TCN: only 42 available; missing seeds 123, 2025.
- AB_C2_FEDAVG_MLP: only 42 available; missing seeds 123, 2025.
- AB_C3_FEDAVG_CNN1D: only 42 available; missing seeds 123, 2025.
- A_C1: only 42 available; missing seeds 123, 2025.
- A_C2: only 42 available; missing seeds 123, 2025.
- A_C3: only 42 available; missing seeds 123, 2025.
- B_C1: only 42 available; missing seeds 123, 2025.
- B_C2: only 42 available; missing seeds 123, 2025.
- B_C3: only 42 available; missing seeds 123, 2025.
- P_C1: only 42 available; missing seeds 123, 2025.
- P_C2: only 42 available; missing seeds 123, 2025.
- P_C3: only 42 available; missing seeds 123, 2025.

## Generated Files
- `outputs/plots_ieee/fig_f1_comparison.pdf`
- `outputs/plots_ieee/fig_f1_comparison.png`
- `outputs/plots_ieee/fig_auc_comparison.pdf`
- `outputs/plots_ieee/fig_auc_comparison.png`
- `outputs/plots_ieee/fig_fpr_comparison.pdf`
- `outputs/plots_ieee/fig_fpr_comparison.png`
- `outputs/plots_ieee/fig_ablation_delta_f1.pdf`
- `outputs/plots_ieee/fig_ablation_delta_f1.png`
- `outputs/plots_ieee/fig_convergence_cluster1.pdf`
- `outputs/plots_ieee/fig_convergence_cluster1.png`
- `outputs/plots_ieee/fig_convergence_cluster2.pdf`
- `outputs/plots_ieee/fig_convergence_cluster2.png`
- `outputs/plots_ieee/fig_convergence_cluster3.pdf`
- `outputs/plots_ieee/fig_convergence_cluster3.png`
- `outputs/plots_ieee/fig_communication_cost.pdf`
- `outputs/plots_ieee/fig_communication_cost.png`
- `outputs/plots_ieee/fig_ledger_overhead.pdf`
- `outputs/plots_ieee/fig_ledger_overhead.png`
