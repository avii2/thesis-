# PLOTS README

## Regeneration
`cd /Users/anilkumar/Desktop/Thesis/fcfl-cps-ids && python3 scripts/generate_ieee_plots.py`

## Plot Inputs

- The plots use only already-generated files under `outputs/`.
- Screenshots are not used as data.
- Metric comparison plots use generated metrics CSVs; convergence plots use `round_metrics.csv`; ledger overhead uses JSONL ledgers.
- `outputs/predictions/` present: `YES`.

## Per-Plot Sources

### fig_f1_comparison
- Data files used: `outputs/metrics/A_C1_metrics.csv`, `outputs/metrics/A_C2_metrics.csv`, `outputs/metrics/A_C3_metrics.csv`, `outputs/metrics/B_C1_metrics.csv`, `outputs/metrics/B_C2_metrics.csv`, `outputs/metrics/B_C3_metrics.csv`, `outputs/metrics/P_C1_metrics.csv`, `outputs/metrics/P_C2_metrics.csv`, `outputs/metrics/P_C3_metrics.csv`
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_auroc_comparison
- Data files used: `outputs/metrics/A_C1_metrics.csv`, `outputs/metrics/A_C2_metrics.csv`, `outputs/metrics/A_C3_metrics.csv`, `outputs/metrics/B_C1_metrics.csv`, `outputs/metrics/B_C2_metrics.csv`, `outputs/metrics/B_C3_metrics.csv`, `outputs/metrics/P_C1_metrics.csv`, `outputs/metrics/P_C2_metrics.csv`, `outputs/metrics/P_C3_metrics.csv`
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_pr_auc_comparison
- Data files used: `outputs/metrics/A_C1_metrics.csv`, `outputs/metrics/A_C2_metrics.csv`, `outputs/metrics/A_C3_metrics.csv`, `outputs/metrics/B_C1_metrics.csv`, `outputs/metrics/B_C2_metrics.csv`, `outputs/metrics/B_C3_metrics.csv`, `outputs/metrics/P_C1_metrics.csv`, `outputs/metrics/P_C2_metrics.csv`, `outputs/metrics/P_C3_metrics.csv`
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_fpr_comparison
- Data files used: `outputs/metrics/A_C1_metrics.csv`, `outputs/metrics/A_C2_metrics.csv`, `outputs/metrics/A_C3_metrics.csv`, `outputs/metrics/B_C1_metrics.csv`, `outputs/metrics/B_C2_metrics.csv`, `outputs/metrics/B_C3_metrics.csv`, `outputs/metrics/P_C1_metrics.csv`, `outputs/metrics/P_C2_metrics.csv`, `outputs/metrics/P_C3_metrics.csv`
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_ablation_delta_f1
- Data files used: `outputs/metrics/P_C1_metrics.csv`, `outputs/metrics/P_C2_metrics.csv`, `outputs/metrics/P_C3_metrics.csv`, `outputs/metrics/summary_all_experiments.csv`
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_convergence_cluster1
- Data files used: `outputs/runs/A_C1/round_metrics.csv`, `outputs/runs/B_C1/round_metrics.csv`, `outputs/runs/P_C1/round_metrics.csv`
- Seed mode: Cluster 1: single available seed only; no shaded std band.

### fig_convergence_cluster2
- Data files used: `outputs/runs/A_C2/round_metrics.csv`, `outputs/runs/B_C2/round_metrics.csv`, `outputs/runs/P_C2/round_metrics.csv`
- Seed mode: Cluster 2: single available seed only; no shaded std band.

### fig_convergence_cluster3
- Data files used: `outputs/runs/A_C3/round_metrics.csv`, `outputs/runs/B_C3/round_metrics.csv`, `outputs/runs/P_C3/round_metrics.csv`
- Seed mode: Cluster 3: single available seed only; no shaded std band.

### fig_communication_cost
- Data files used: `outputs/metrics/A_C1_metrics.csv`, `outputs/metrics/A_C2_metrics.csv`, `outputs/metrics/A_C3_metrics.csv`, `outputs/metrics/B_C1_metrics.csv`, `outputs/metrics/B_C2_metrics.csv`, `outputs/metrics/B_C3_metrics.csv`, `outputs/metrics/P_C1_metrics.csv`, `outputs/metrics/P_C2_metrics.csv`, `outputs/metrics/P_C3_metrics.csv`
- Seed mode: Single available seed only; bars/curves shown without error bars.

### fig_ledger_overhead
- Data files used: `outputs/ledgers/AB_C1_FEDAVG_TCN_ledger.jsonl`, `outputs/ledgers/AB_C2_FEDAVG_MLP_ledger.jsonl`, `outputs/ledgers/AB_C3_FEDAVG_CNN1D_ledger.jsonl`, `outputs/ledgers/A_C1_ledger.jsonl`, `outputs/ledgers/A_C2_ledger.jsonl`, `outputs/ledgers/A_C3_ledger.jsonl`, `outputs/ledgers/B_C1_ledger.jsonl`, `outputs/ledgers/B_C2_ledger.jsonl`, `outputs/ledgers/B_C3_ledger.jsonl`, `outputs/ledgers/P_C1_ledger.jsonl`, `outputs/ledgers/P_C2_ledger.jsonl`, `outputs/ledgers/P_C3_ledger.jsonl`
- Seed mode: Ledger overhead computed from real JSONL metadata ledgers only; size from file bytes and latency from timestamp_end - timestamp_start.

## Missing Inputs
- None.

## Generated Files
- `outputs/plots_ieee/fig_f1_comparison.pdf`
- `outputs/plots_ieee/fig_f1_comparison.png`
- `outputs/plots_ieee/fig_auroc_comparison.pdf`
- `outputs/plots_ieee/fig_auroc_comparison.png`
- `outputs/plots_ieee/fig_pr_auc_comparison.pdf`
- `outputs/plots_ieee/fig_pr_auc_comparison.png`
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
