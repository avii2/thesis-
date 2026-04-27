from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train import (
    ExperimentSpec,
    _finalize_run_outputs,
    _load_metrics_row,
    _metric_as_float,
    _write_confusion_matrices_report,
)
from src.train_cluster1_proposed import run_cluster1_proposed
from src.train_flat_baseline import run_flat_baseline_experiments
from src.train_hierarchical_baseline import run_hierarchical_baseline_experiments


REPAIRED_OUTPUT_ROOT = Path("outputs_c1_repaired")
ORIGINAL_METRICS_ROOT = Path("outputs/metrics")

COMPARISON_COLUMNS = (
    "best_validation_round",
    "validation_f1",
    "threshold_used",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_auroc",
    "test_pr_auc",
    "test_fpr",
    "confusion_matrix",
    "wall_clock_training_seconds",
)


def _repaired_specs() -> dict[str, ExperimentSpec]:
    return {
        "A_C1_REPAIRED": ExperimentSpec(
            experiment_id="A_C1_REPAIRED",
            run_category="baseline_flat",
            cluster_id=1,
            dataset="HAI 21.03 repaired supervised variant",
            model="cnn1d",
            fl_method="FedAvg",
            aggregation="weighted_arithmetic_mean",
            hierarchy="flat",
            clustering_method="none",
            n_subclusters=0,
            descriptor="none",
            run_repeats=1,
            notes="Exploratory repaired Cluster 1 split.",
        ),
        "B_C1_REPAIRED": ExperimentSpec(
            experiment_id="B_C1_REPAIRED",
            run_category="baseline_uniform_hierarchical",
            cluster_id=1,
            dataset="HAI 21.03 repaired supervised variant",
            model="cnn1d",
            fl_method="FedAvg",
            aggregation="weighted_arithmetic_mean",
            hierarchy="hierarchical_fixed",
            clustering_method="agglomerative",
            n_subclusters=2,
            descriptor="feature_mean_std_window_flattened",
            run_repeats=1,
            notes="Exploratory repaired Cluster 1 split.",
        ),
        "P_C1_REPAIRED": ExperimentSpec(
            experiment_id="P_C1_REPAIRED",
            run_category="proposed_specialized_hierarchical",
            cluster_id=1,
            dataset="HAI 21.03 repaired supervised variant",
            model="tcn",
            fl_method="FedBN",
            aggregation="weighted_non_bn_mean",
            hierarchy="hierarchical_fixed",
            clustering_method="agglomerative",
            n_subclusters=2,
            descriptor="feature_mean_std_window_flattened",
            run_repeats=1,
            notes="Exploratory repaired Cluster 1 split.",
        ),
    }


def _format_value(value: Any) -> str:
    numeric = _metric_as_float(value)
    if numeric is not None:
        return f"{numeric:.6f}"
    if value is None or value == "":
        return "MISSING"
    return str(value)


def _metrics_value(row: Mapping[str, Any], column: str) -> Any:
    if column == "validation_f1":
        return row.get("best_validation_f1")
    if column == "confusion_matrix":
        return row.get("test_confusion_matrix")
    return row.get(column)


def _load_optional_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_metrics_row(path)


def _metric_delta(new_row: Mapping[str, Any] | None, old_row: Mapping[str, Any] | None, key: str) -> float | None:
    if new_row is None or old_row is None:
        return None
    new_value = _metric_as_float(new_row.get(key))
    old_value = _metric_as_float(old_row.get(key))
    if new_value is None or old_value is None:
        return None
    return new_value - old_value


def _precision_recall_gap(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    precision = _metric_as_float(row.get("test_precision"))
    recall = _metric_as_float(row.get("test_recall"))
    if precision is None or recall is None:
        return None
    return abs(precision - recall)


def _comparison_answer(value: bool | None, true_text: str, false_text: str) -> str:
    if value is None:
        return "INSUFFICIENT DATA"
    return true_text if value else false_text


def _write_comparison_report(
    *,
    output_root: Path,
    report_path: Path | None = None,
) -> Path:
    if report_path is None:
        report_path = output_root / "reports" / "cluster1_repaired_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_pairs = (
        ("A", "A_C1", "A_C1_REPAIRED"),
        ("B", "B_C1", "B_C1_REPAIRED"),
        ("P", "P_C1", "P_C1_REPAIRED"),
    )
    original_rows = {
        original_id: _load_optional_metrics(ORIGINAL_METRICS_ROOT / f"{original_id}_metrics.csv")
        for _, original_id, _ in experiment_pairs
    }
    repaired_rows = {
        repaired_id: _load_optional_metrics(output_root / "metrics" / f"{repaired_id}_metrics.csv")
        for _, _, repaired_id in experiment_pairs
    }

    p_original = original_rows["P_C1"]
    p_repaired = repaired_rows["P_C1_REPAIRED"]
    p_f1_delta = _metric_delta(p_repaired, p_original, "test_f1")
    p_improved = None if p_f1_delta is None else p_f1_delta > 0.0

    original_baseline_f1_values = [
        _metric_as_float(original_rows[experiment_id].get("test_f1"))
        for experiment_id in ("A_C1", "B_C1")
        if original_rows[experiment_id] is not None
    ]
    repaired_baseline_f1_values = [
        _metric_as_float(repaired_rows[experiment_id].get("test_f1"))
        for experiment_id in ("A_C1_REPAIRED", "B_C1_REPAIRED")
        if repaired_rows[experiment_id] is not None
    ]
    original_p_f1 = _metric_as_float(p_original.get("test_f1")) if p_original is not None else None
    repaired_p_f1 = _metric_as_float(p_repaired.get("test_f1")) if p_repaired is not None else None
    original_gap = (
        original_p_f1 - max(value for value in original_baseline_f1_values if value is not None)
        if original_p_f1 is not None and any(value is not None for value in original_baseline_f1_values)
        else None
    )
    repaired_gap = (
        repaired_p_f1 - max(value for value in repaired_baseline_f1_values if value is not None)
        if repaired_p_f1 is not None and any(value is not None for value in repaired_baseline_f1_values)
        else None
    )
    gap_improved = (
        None
        if original_gap is None or repaired_gap is None
        else repaired_gap > original_gap
    )

    original_tradeoff = _precision_recall_gap(p_original)
    repaired_tradeoff = _precision_recall_gap(p_repaired)
    tradeoff_reduced = (
        None
        if original_tradeoff is None or repaired_tradeoff is None
        else repaired_tradeoff < original_tradeoff
    )

    should_replace = (
        None
        if p_improved is None or gap_improved is None
        else bool(p_improved and gap_improved)
    )

    repaired_p_auroc = _metric_as_float(p_repaired.get("test_auroc")) if p_repaired is not None else None
    if repaired_p_f1 is None:
        model_family_recommendation = "INSUFFICIENT DATA: repaired P_C1 metrics are missing."
    elif repaired_baseline_f1_values and any(value is not None for value in repaired_baseline_f1_values):
        best_repaired_baseline = max(value for value in repaired_baseline_f1_values if value is not None)
        if repaired_p_f1 < best_repaired_baseline and repaired_p_auroc is not None and repaired_p_auroc >= 0.80:
            model_family_recommendation = (
                "NO: repaired P_C1 underperforms the repaired baseline on F1, but AUROC is high enough that "
                "calibration/thresholding and imbalance should be reviewed before a model-family change."
            )
        elif repaired_p_f1 < best_repaired_baseline:
            model_family_recommendation = (
                "NOT YET: repaired P_C1 still trails the repaired baseline, but this single-seed run should be "
                "confirmed with the existing TCN family before changing model family."
            )
        else:
            model_family_recommendation = (
                "NO: repaired P_C1 is not weaker than the repaired baselines on test F1 in this run."
            )
    else:
        model_family_recommendation = "INSUFFICIENT DATA: repaired baseline metrics are missing."

    lines = [
        "# Cluster 1 Repaired Experiment Comparison",
        "",
        "Source files for original metrics are `outputs/metrics/*_C1_metrics.csv`; source files for repaired metrics are `outputs_c1_repaired/metrics/*_C1_REPAIRED_metrics.csv`.",
        "The repaired variant uses `test4.csv` and `test5.csv` only for held-out testing.",
        "",
        "## Metrics",
        "",
        "| method | dataset pipeline | experiment_id | best_validation_round | validation_f1 | threshold_used | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | confusion_matrix | wall_clock_training_seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for method, original_id, repaired_id in experiment_pairs:
        for pipeline_name, experiment_id, row in (
            ("original", original_id, original_rows[original_id]),
            ("repaired", repaired_id, repaired_rows[repaired_id]),
        ):
            values = [
                _format_value(_metrics_value(row, column)) if row is not None else "MISSING"
                for column in COMPARISON_COLUMNS
            ]
            lines.append(
                f"| {method} | {pipeline_name} | `{experiment_id}` | "
                + " | ".join(values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| method | test_f1_delta_repaired_minus_original | test_recall_delta | test_precision_delta | test_fpr_delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, original_id, repaired_id in experiment_pairs:
        deltas = [
            _metric_delta(repaired_rows[repaired_id], original_rows[original_id], key)
            for key in ("test_f1", "test_recall", "test_precision", "test_fpr")
        ]
        lines.append(
            f"| {method} | "
            + " | ".join("MISSING" if value is None else f"{value:.6f}" for value in deltas)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Did the repaired dataset pipeline improve P_C1? {_comparison_answer(p_improved, 'YES', 'NO')}. "
            f"P_C1 test F1 delta: {_format_value(p_f1_delta)}.",
            f"2. Did it improve the proposed-vs-baseline gap? {_comparison_answer(gap_improved, 'YES', 'NO')}. "
            f"Original gap: {_format_value(original_gap)}; repaired gap: {_format_value(repaired_gap)}.",
            f"3. Did it reduce the extreme precision/recall tradeoff? {_comparison_answer(tradeoff_reduced, 'YES', 'NO')}. "
            f"Original |precision-recall|: {_format_value(original_tradeoff)}; repaired |precision-recall|: {_format_value(repaired_tradeoff)}.",
            f"4. Should this repaired Cluster 1 dataset pipeline replace the original one? {_comparison_answer(should_replace, 'YES for the repaired Cluster 1 variant, pending final multi-seed reruns', 'NO based on this single-seed comparison')}.",
            f"5. If results are still weak, is the next recommended step a model-family change? {model_family_recommendation}",
            "",
            "## Notes",
            "",
            "- Selection uses validation F1; test metrics are reported after the selected validation round and threshold are fixed.",
            "- This report does not use screenshots or manually entered results.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_repaired_summary_csv(artifacts: Sequence[Any], output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        row = _load_metrics_row(artifact.metrics_csv_path)
        row["summary_path"] = str(artifact.summary_path)
        row["round_metrics_path"] = str(artifact.round_metrics_path)
        row["metrics_csv_path"] = str(artifact.metrics_csv_path)
        row["model_manifest_path"] = str(artifact.model_manifest_path)
        row["ledger_path"] = str(artifact.ledger_path)
        row["convergence_plot_path"] = str(artifact.convergence_plot_path)
        rows.append(row)
    output_path = output_root / "metrics" / "summary_cluster1_repaired.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def run_cluster1_repaired_experiments(
    *,
    output_root: str | Path = REPAIRED_OUTPUT_ROOT,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float = 0.05,
    seed: int | None = None,
) -> dict[str, Path]:
    output_root = Path(output_root)
    for relative in ("runs", "metrics", "plots", "ledgers", "models", "predictions", "reports"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    specs = _repaired_specs()
    raw_results: list[tuple[str, Any]] = []

    flat_results = run_flat_baseline_experiments(
        baseline_config_path="configs/baseline_flat_cluster1_repaired.yaml",
        experiment_ids=["A_C1_REPAIRED"],
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        output_root=output_root,
    )
    if len(flat_results) != 1:
        raise ValueError(f"A_C1_REPAIRED: expected one result, observed {len(flat_results)}.")
    raw_results.append(("A_C1_REPAIRED", flat_results[0]))

    hierarchical_results = run_hierarchical_baseline_experiments(
        baseline_config_path="configs/baseline_hierarchical_cluster1_repaired.yaml",
        experiment_ids=["B_C1_REPAIRED"],
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        output_root=output_root,
    )
    if len(hierarchical_results) != 1:
        raise ValueError(f"B_C1_REPAIRED: expected one result, observed {len(hierarchical_results)}.")
    raw_results.append(("B_C1_REPAIRED", hierarchical_results[0]))

    raw_results.append(
        (
            "P_C1_REPAIRED",
            run_cluster1_proposed(
                proposed_config_path="configs/proposed_cluster1_repaired.yaml",
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                output_root=output_root,
            ),
        )
    )

    artifacts = [
        _finalize_run_outputs(
            experiment_id=experiment_id,
            spec=specs[experiment_id],
            raw_result=raw_result,
            output_root=output_root,
        )
        for experiment_id, raw_result in raw_results
    ]
    summary_path = _write_repaired_summary_csv(artifacts, output_root)
    summary_rows = [_load_metrics_row(artifact.metrics_csv_path) for artifact in artifacts]
    confusion_report_path = _write_confusion_matrices_report(summary_rows, output_root)
    comparison_report_path = _write_comparison_report(output_root=output_root)
    return {
        "summary": summary_path,
        "confusion_report": confusion_report_path,
        "comparison_report": comparison_report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repaired Cluster 1 A/B/P experiments.")
    parser.add_argument("--output-root", default=str(REPAIRED_OUTPUT_ROOT))
    parser.add_argument("--rounds", type=int, help="Optional override for FL rounds.")
    parser.add_argument("--local-epochs", type=int, help="Optional override for local epochs.")
    parser.add_argument("--batch-size", type=int, help="Optional override for local batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, help="Optional override for run seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_cluster1_repaired_experiments(
        output_root=args.output_root,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
