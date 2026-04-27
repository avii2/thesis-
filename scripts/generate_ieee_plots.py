from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs"
PLOTS_DIR = OUTPUT_ROOT / "plots_ieee"
README_PATH = REPO_ROOT / "docs" / "PLOTS_README.md"
MATRIX_PATH = REPO_ROOT / "docs" / "EXPERIMENT_MATRIX.csv"
SUMMARY_MEAN_STD_PATH = OUTPUT_ROOT / "metrics" / "summary_all_experiments_mean_std.csv"
SUMMARY_SINGLE_PATH = OUTPUT_ROOT / "metrics" / "summary_all_experiments.csv"

A_B_P_GROUPS = {
    "Cluster 1": ("A_C1", "B_C1", "P_C1"),
    "Cluster 2": ("A_C2", "B_C2", "P_C2"),
    "Cluster 3": ("A_C3", "B_C3", "P_C3"),
}

ABLATION_GROUPS = {
    "Cluster 1": ("AB_C1_FEDAVG_TCN", "P_C1"),
    "Cluster 2": ("AB_C2_FEDAVG_MLP", "P_C2"),
    "Cluster 3": ("AB_C3_FEDAVG_CNN1D", "P_C3"),
}

METHOD_LABELS = {
    "A": "Baseline A",
    "B": "Baseline B",
    "P": "Proposed",
    "AB": "Ablation Control",
}

METHOD_COLORS = {
    "A": "#1F1F1F",
    "B": "#7A7A7A",
    "P": "#D9D9D9",
    "AB": "#B3B3B3",
}

METHOD_HATCHES = {
    "A": "",
    "B": "///",
    "P": "\\\\\\",
    "AB": "xx",
}

METHOD_LINESTYLES = {
    "A": "-",
    "B": "--",
    "P": "-.",
    "AB": ":",
}


@dataclass(frozen=True)
class MetricAggregate:
    experiment_id: str
    status: str
    successful_seed_count: int
    successful_seeds: tuple[str, ...]
    missing_seeds: tuple[str, ...]
    notes: str
    source_path: str
    metrics: dict[str, float]
    stds: dict[str, float]


@dataclass(frozen=True)
class CurveSeries:
    experiment_id: str
    rounds: list[int]
    mean_values: list[float]
    std_values: list[float]
    successful_seed_count: int
    source_paths: tuple[str, ...]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_experiment_matrix() -> dict[str, dict[str, str]]:
    rows = _read_csv_rows(MATRIX_PATH)
    return {row["experiment_id"]: row for row in rows}


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"MISSING", "NOT AVAILABLE"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _comma_split(value: str) -> tuple[str, ...]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return tuple(items)


def _method_key(experiment_id: str) -> str:
    if experiment_id.startswith("AB_"):
        return "AB"
    return experiment_id.split("_", 1)[0]


def _build_metric_aggregates() -> dict[str, MetricAggregate]:
    aggregates: dict[str, MetricAggregate] = {}
    if SUMMARY_MEAN_STD_PATH.exists():
        for row in _read_csv_rows(SUMMARY_MEAN_STD_PATH):
            metrics = {
                "test_accuracy": _parse_float(row.get("test_accuracy_mean")) or math.nan,
                "test_precision": _parse_float(row.get("test_precision_mean")) or math.nan,
                "test_recall": _parse_float(row.get("test_recall_mean")) or math.nan,
                "test_f1": _parse_float(row.get("test_f1_mean")) or math.nan,
                "test_auroc": _parse_float(row.get("test_auroc_mean")) or math.nan,
                "test_pr_auc": _parse_float(row.get("test_pr_auc_mean")) or math.nan,
                "test_fpr": _parse_float(row.get("test_fpr_mean")) or math.nan,
                "wall_clock_training_seconds": _parse_float(row.get("wall_clock_training_seconds_mean")) or math.nan,
                "total_communication_cost_bytes": _parse_float(row.get("total_communication_cost_bytes_mean")) or math.nan,
            }
            stds = {
                "test_accuracy": _parse_float(row.get("test_accuracy_std")) or 0.0,
                "test_precision": _parse_float(row.get("test_precision_std")) or 0.0,
                "test_recall": _parse_float(row.get("test_recall_std")) or 0.0,
                "test_f1": _parse_float(row.get("test_f1_std")) or 0.0,
                "test_auroc": _parse_float(row.get("test_auroc_std")) or 0.0,
                "test_pr_auc": _parse_float(row.get("test_pr_auc_std")) or 0.0,
                "test_fpr": _parse_float(row.get("test_fpr_std")) or 0.0,
                "wall_clock_training_seconds": _parse_float(row.get("wall_clock_training_seconds_std")) or 0.0,
                "total_communication_cost_bytes": _parse_float(row.get("total_communication_cost_bytes_std")) or 0.0,
            }
            aggregates[row["experiment_id"]] = MetricAggregate(
                experiment_id=row["experiment_id"],
                status=row.get("status", "UNKNOWN"),
                successful_seed_count=int(row.get("successful_seed_count", "0") or 0),
                successful_seeds=_comma_split(row.get("successful_seeds", "")),
                missing_seeds=_comma_split(row.get("missing_seeds", "")),
                notes=row.get("notes", "").strip(),
                source_path=str(SUMMARY_MEAN_STD_PATH),
                metrics=metrics,
                stds=stds,
            )
    if SUMMARY_SINGLE_PATH.exists():
        for row in _read_csv_rows(SUMMARY_SINGLE_PATH):
            if row["experiment_id"] in aggregates:
                continue
            aggregates[row["experiment_id"]] = MetricAggregate(
                experiment_id=row["experiment_id"],
                status="COMPLETE",
                successful_seed_count=1,
                successful_seeds=tuple(),
                missing_seeds=tuple(),
                notes="No multiseed summary file found; using single generated metrics row.",
                source_path=str(SUMMARY_SINGLE_PATH),
                metrics={
                    "test_accuracy": _parse_float(row.get("test_accuracy")) or math.nan,
                    "test_precision": _parse_float(row.get("test_precision")) or math.nan,
                    "test_recall": _parse_float(row.get("test_recall")) or math.nan,
                    "test_f1": _parse_float(row.get("test_f1")) or math.nan,
                    "test_auroc": _parse_float(row.get("test_auroc")) or math.nan,
                    "test_pr_auc": _parse_float(row.get("test_pr_auc")) or math.nan,
                    "test_fpr": _parse_float(row.get("test_fpr")) or math.nan,
                    "wall_clock_training_seconds": _parse_float(row.get("wall_clock_training_seconds")) or math.nan,
                    "total_communication_cost_bytes": _parse_float(row.get("total_communication_cost_bytes")) or math.nan,
                },
                stds={key: 0.0 for key in (
                    "test_accuracy",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_auroc",
                    "test_pr_auc",
                    "test_fpr",
                    "wall_clock_training_seconds",
                    "total_communication_cost_bytes",
                )},
            )
    metrics_dir = OUTPUT_ROOT / "metrics"
    for path in sorted(metrics_dir.glob("*_metrics.csv")):
        experiment_id = path.name.removesuffix("_metrics.csv")
        if experiment_id in aggregates or "_seed_" in experiment_id:
            continue
        rows = _read_csv_rows(path)
        if not rows:
            continue
        row = rows[0]
        aggregates[experiment_id] = MetricAggregate(
            experiment_id=experiment_id,
            status="COMPLETE",
            successful_seed_count=1,
            successful_seeds=tuple(),
            missing_seeds=tuple(),
            notes="Using single generated per-experiment metrics CSV.",
            source_path=str(path),
            metrics={
                "test_accuracy": _parse_float(row.get("test_accuracy")) or math.nan,
                "test_precision": _parse_float(row.get("test_precision")) or math.nan,
                "test_recall": _parse_float(row.get("test_recall")) or math.nan,
                "test_f1": _parse_float(row.get("test_f1")) or math.nan,
                "test_auroc": _parse_float(row.get("test_auroc")) or math.nan,
                "test_pr_auc": _parse_float(row.get("test_pr_auc")) or math.nan,
                "test_fpr": _parse_float(row.get("test_fpr")) or math.nan,
                "wall_clock_training_seconds": _parse_float(row.get("wall_clock_training_seconds")) or math.nan,
                "total_communication_cost_bytes": _parse_float(row.get("total_communication_cost_bytes")) or math.nan,
            },
            stds={key: 0.0 for key in (
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_auroc",
                "test_pr_auc",
                "test_fpr",
                "wall_clock_training_seconds",
                "total_communication_cost_bytes",
            )},
        )
    return aggregates


def _discover_curve_series(experiment_id: str) -> CurveSeries | None:
    run_root = OUTPUT_ROOT / "runs" / experiment_id
    seed_paths = sorted(run_root.glob("seed_*/round_metrics.csv"))
    source_paths: list[str] = []
    series_by_seed: list[tuple[list[int], list[float]]] = []

    if seed_paths:
        for path in seed_paths:
            rows = _read_csv_rows(path)
            rounds = [int(row["round"]) for row in rows]
            values = [float(row["validation_f1"]) for row in rows]
            series_by_seed.append((rounds, values))
            source_paths.append(str(path))
    else:
        fallback = run_root / "round_metrics.csv"
        if not fallback.exists():
            return None
        rows = _read_csv_rows(fallback)
        series_by_seed.append(([int(row["round"]) for row in rows], [float(row["validation_f1"]) for row in rows]))
        source_paths.append(str(fallback))

    reference_rounds = series_by_seed[0][0]
    for rounds, _values in series_by_seed[1:]:
        if rounds != reference_rounds:
            raise ValueError(
                f"{experiment_id}: round alignment mismatch across seeds. "
                f"Observed {rounds[:5]}... vs {reference_rounds[:5]}..."
            )

    values_matrix = list(zip(*(values for _rounds, values in series_by_seed)))
    mean_values = [sum(points) / len(points) for points in values_matrix]
    if len(series_by_seed) > 1:
        std_values = []
        for points in values_matrix:
            mean_value = sum(points) / len(points)
            variance = sum((point - mean_value) ** 2 for point in points) / len(points)
            std_values.append(math.sqrt(variance))
    else:
        std_values = [0.0 for _ in mean_values]

    return CurveSeries(
        experiment_id=experiment_id,
        rounds=reference_rounds,
        mean_values=mean_values,
        std_values=std_values,
        successful_seed_count=len(series_by_seed),
        source_paths=tuple(source_paths),
    )


def _load_ledger_overhead() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for path in sorted((OUTPUT_ROOT / "ledgers").glob("*_ledger.jsonl")):
        experiment_id = path.name.removesuffix("_ledger.jsonl")
        if "_seed_" in experiment_id:
            continue
        sizes = path.stat().st_size
        latencies_ms: list[float] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                start = datetime.fromisoformat(record["timestamp_start"])
                end = datetime.fromisoformat(record["timestamp_end"])
                latencies_ms.append((end - start).total_seconds() * 1000.0)
        rows.append(
            {
                "experiment_id": experiment_id,
                "ledger_size_bytes": float(sizes),
                "average_logging_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else math.nan,
            }
        )
        source_paths.append(str(path))
    return rows, source_paths


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, stem: str, *, metadata_note: str, sources: Iterable[str]) -> tuple[Path, Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PLOTS_DIR / f"{stem}.pdf"
    png_path = PLOTS_DIR / f"{stem}.png"
    source_note = "; ".join(sorted(set(sources)))
    pdf_metadata = {
        "Title": stem,
        "Author": "Codex",
        "Subject": metadata_note,
        "Keywords": source_note,
        "Creator": "scripts/generate_ieee_plots.py",
    }
    png_metadata = {
        "Title": stem,
        "Description": f"{metadata_note} Sources: {source_note}",
        "Author": "Codex",
        "Software": "scripts/generate_ieee_plots.py",
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(png_path, bbox_inches="tight", dpi=600, metadata=png_metadata)
    plt.close(fig)
    return pdf_path, png_path


def _format_value(value: float) -> str:
    return "MISSING" if math.isnan(value) else f"{value:.3f}"


def _add_bar_label(ax: plt.Axes, x: float, y: float) -> None:
    if math.isnan(y):
        ax.text(x, 0.01, "MISSING", ha="center", va="bottom", rotation=90, fontsize=7, color="#111827")
        return
    ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7, color="#111827", rotation=90)


def _metric_note(aggregates: Iterable[MetricAggregate]) -> str:
    counts = {aggregate.successful_seed_count for aggregate in aggregates}
    if not counts:
        return "No metrics available."
    if counts == {1}:
        return "Single available seed only; bars/curves shown without error bars."
    return "Mean ± std shown where multiple successful seeds are available; single-seed series are plotted without error bars."


def _bar_panel(
    ax: plt.Axes,
    *,
    cluster_groups: dict[str, tuple[str, ...]],
    aggregates: dict[str, MetricAggregate],
    metric_key: str,
    ylabel: str,
    title: str | None = None,
    scale_transform=lambda value: value,
    y_log: bool = False,
) -> list[str]:
    cluster_labels = list(cluster_groups.keys())
    x_positions = list(range(len(cluster_labels)))
    width = 0.22
    offsets = (-width, 0.0, width)
    all_sources: list[str] = []
    plotted_values: list[float] = []

    for method_index, method_id in enumerate(("A", "B", "P")):
        xs: list[float] = []
        ys: list[float] = []
        yerrs: list[float] = []
        any_error = False
        for cluster_index, cluster_label in enumerate(cluster_labels):
            experiment_id = cluster_groups[cluster_label][method_index]
            aggregate = aggregates.get(experiment_id)
            value = math.nan
            std = 0.0
            if aggregate is not None:
                all_sources.append(aggregate.source_path)
                raw_value = aggregate.metrics.get(metric_key, math.nan)
                value = scale_transform(raw_value) if not math.isnan(raw_value) else math.nan
                raw_std = aggregate.stds.get(metric_key, 0.0)
                std = scale_transform(raw_std) if aggregate.successful_seed_count > 1 else 0.0
                any_error = any_error or aggregate.successful_seed_count > 1
            xs.append(x_positions[cluster_index] + offsets[method_index])
            ys.append(value)
            yerrs.append(std)
            if not math.isnan(value):
                plotted_values.append(value)

        if any_error:
            bars = ax.bar(
                xs,
                ys,
                width=width,
                label=METHOD_LABELS[method_id],
                color=METHOD_COLORS[method_id],
                edgecolor="#111827",
                hatch=METHOD_HATCHES[method_id],
                linewidth=0.6,
                yerr=yerrs,
                capsize=3,
            )
        else:
            bars = ax.bar(
                xs,
                ys,
                width=width,
                label=METHOD_LABELS[method_id],
                color=METHOD_COLORS[method_id],
                edgecolor="#111827",
                hatch=METHOD_HATCHES[method_id],
                linewidth=0.6,
            )
        for bar, value in zip(bars, ys):
            _add_bar_label(ax, bar.get_x() + bar.get_width() / 2.0, value)

    ax.set_xticks(x_positions, cluster_labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if y_log:
        if any(value > 0 for value in plotted_values):
            ax.set_yscale("log")
    ax.legend(frameon=False, ncol=3, loc="best")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    return all_sources


def _plot_f1_comparison(aggregates: dict[str, MetricAggregate]) -> tuple[list[Path], list[str], str]:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    sources = _bar_panel(
        ax,
        cluster_groups=A_B_P_GROUPS,
        aggregates=aggregates,
        metric_key="test_f1",
        ylabel="Test F1",
    )
    ax.set_ylim(bottom=0.0)
    note = _metric_note(aggregates[exp_id] for group in A_B_P_GROUPS.values() for exp_id in group if exp_id in aggregates)
    pdf_path, png_path = _save_figure(fig, "fig_f1_comparison", metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _plot_auroc_comparison(aggregates: dict[str, MetricAggregate]) -> tuple[list[Path], list[str], str]:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    sources = _bar_panel(
        ax,
        cluster_groups=A_B_P_GROUPS,
        aggregates=aggregates,
        metric_key="test_auroc",
        ylabel="AUROC",
    )
    ax.set_ylim(bottom=0.0)
    note = _metric_note(aggregates[exp_id] for group in A_B_P_GROUPS.values() for exp_id in group if exp_id in aggregates)
    pdf_path, png_path = _save_figure(fig, "fig_auroc_comparison", metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _plot_pr_auc_comparison(aggregates: dict[str, MetricAggregate]) -> tuple[list[Path], list[str], str]:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    sources = _bar_panel(
        ax,
        cluster_groups=A_B_P_GROUPS,
        aggregates=aggregates,
        metric_key="test_pr_auc",
        ylabel="PR-AUC",
    )
    ax.set_ylim(bottom=0.0)
    note = _metric_note(aggregates[exp_id] for group in A_B_P_GROUPS.values() for exp_id in group if exp_id in aggregates)
    pdf_path, png_path = _save_figure(fig, "fig_pr_auc_comparison", metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _plot_fpr_comparison(aggregates: dict[str, MetricAggregate]) -> tuple[list[Path], list[str], str]:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    sources = _bar_panel(
        ax,
        cluster_groups=A_B_P_GROUPS,
        aggregates=aggregates,
        metric_key="test_fpr",
        ylabel="False Positive Rate",
    )
    ax.set_ylim(bottom=0.0)
    note = _metric_note(aggregates[exp_id] for group in A_B_P_GROUPS.values() for exp_id in group if exp_id in aggregates)
    pdf_path, png_path = _save_figure(fig, "fig_fpr_comparison", metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _plot_ablation_delta_f1(aggregates: dict[str, MetricAggregate]) -> tuple[list[Path], list[str], str]:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    cluster_labels = list(ABLATION_GROUPS.keys())
    xs = list(range(len(cluster_labels)))
    heights: list[float] = []
    errors: list[float] = []
    sources: list[str] = []
    any_error = False

    for cluster_label in cluster_labels:
        control_id, proposed_id = ABLATION_GROUPS[cluster_label]
        control = aggregates.get(control_id)
        proposed = aggregates.get(proposed_id)
        if control is None or proposed is None:
            heights.append(math.nan)
            errors.append(0.0)
            continue
        sources.extend([control.source_path, proposed.source_path])
        delta = proposed.metrics["test_f1"] - control.metrics["test_f1"]
        heights.append(delta)
        if control.successful_seed_count > 1 and proposed.successful_seed_count > 1:
            errors.append(math.sqrt(control.stds["test_f1"] ** 2 + proposed.stds["test_f1"] ** 2))
            any_error = True
        else:
            errors.append(0.0)

    if any_error:
        bars = ax.bar(
            xs,
            heights,
            color=METHOD_COLORS["P"],
            edgecolor="#111827",
            hatch=METHOD_HATCHES["P"],
            linewidth=0.6,
            yerr=errors,
            capsize=3,
        )
    else:
        bars = ax.bar(
            xs,
            heights,
            color=METHOD_COLORS["P"],
            edgecolor="#111827",
            hatch=METHOD_HATCHES["P"],
            linewidth=0.6,
        )
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(xs, cluster_labels)
    ax.set_ylabel(r"$\Delta$ Test F1 (Proposed - Ablation Control)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for bar, value in zip(bars, heights):
        _add_bar_label(ax, bar.get_x() + bar.get_width() / 2.0, value)
    note = _metric_note(
        aggregate
        for pair in ABLATION_GROUPS.values()
        for exp_id in pair
        if (aggregate := aggregates.get(exp_id)) is not None
    )
    pdf_path, png_path = _save_figure(fig, "fig_ablation_delta_f1", metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _plot_convergence(cluster_label: str, experiment_ids: tuple[str, str, str]) -> tuple[list[Path], list[str], str]:
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    sources: list[str] = []
    counts: list[int] = []

    for experiment_id in experiment_ids:
        series = _discover_curve_series(experiment_id)
        if series is None:
            continue
        method_id = _method_key(experiment_id)
        ax.plot(
            series.rounds,
            series.mean_values,
            label=METHOD_LABELS[method_id],
            color=METHOD_COLORS[method_id],
            linestyle=METHOD_LINESTYLES[method_id],
            linewidth=1.6,
        )
        if series.successful_seed_count > 1:
            lower = [mean - std for mean, std in zip(series.mean_values, series.std_values)]
            upper = [mean + std for mean, std in zip(series.mean_values, series.std_values)]
            ax.fill_between(series.rounds, lower, upper, color=METHOD_COLORS[method_id], alpha=0.16, linewidth=0)
        counts.append(series.successful_seed_count)
        sources.extend(series.source_paths)

    ax.set_xlabel("Round")
    ax.set_ylabel("Validation F1")
    ax.set_title(cluster_label)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0.0)
    if not counts:
        note = f"{cluster_label}: no round metrics found."
        fig.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=12)
    elif set(counts) == {1}:
        note = f"{cluster_label}: single available seed only; no shaded std band."
    else:
        note = f"{cluster_label}: mean validation F1 curves with shaded std for experiments having multiple successful seeds."

    stem = f"fig_convergence_cluster{cluster_label.split()[-1]}"
    pdf_path, png_path = _save_figure(fig, stem, metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _plot_communication_cost(aggregates: dict[str, MetricAggregate]) -> tuple[list[Path], list[str], str]:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    sources = _bar_panel(
        ax,
        cluster_groups=A_B_P_GROUPS,
        aggregates=aggregates,
        metric_key="total_communication_cost_bytes",
        ylabel="Total Communication Cost (MiB)",
        scale_transform=lambda value: value / (1024.0 * 1024.0),
        y_log=True,
    )
    note = _metric_note(aggregates[exp_id] for group in A_B_P_GROUPS.values() for exp_id in group if exp_id in aggregates)
    pdf_path, png_path = _save_figure(fig, "fig_communication_cost", metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _plot_ledger_overhead() -> tuple[list[Path], list[str], str]:
    rows, sources = _load_ledger_overhead()
    rows = [row for row in rows if row["experiment_id"] in _load_experiment_matrix()]
    rows.sort(key=lambda row: (_load_experiment_matrix()[row["experiment_id"]]["cluster_id"], row["experiment_id"]))
    experiment_ids = [row["experiment_id"] for row in rows]
    sizes_kib = [row["ledger_size_bytes"] / 1024.0 for row in rows]
    latencies_ms = [row["average_logging_latency_ms"] for row in rows]
    colors = [METHOD_COLORS[_method_key(experiment_id)] for experiment_id in experiment_ids]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    bars_top = axes[0].bar(range(len(experiment_ids)), sizes_kib, color=colors, edgecolor="#111827", linewidth=0.6)
    for bar, experiment_id in zip(bars_top, experiment_ids):
        bar.set_hatch(METHOD_HATCHES[_method_key(experiment_id)])
    axes[0].set_ylabel("Ledger Size (KiB)")
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=6))
    for x, value in enumerate(sizes_kib):
        _add_bar_label(axes[0], x, value)

    bars_bottom = axes[1].bar(range(len(experiment_ids)), latencies_ms, color=colors, edgecolor="#111827", linewidth=0.6)
    for bar, experiment_id in zip(bars_bottom, experiment_ids):
        bar.set_hatch(METHOD_HATCHES[_method_key(experiment_id)])
    axes[1].set_ylabel("Avg. Logging Latency (ms)")
    axes[1].set_xticks(range(len(experiment_ids)), experiment_ids, rotation=45, ha="right")
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
    for x, value in enumerate(latencies_ms):
        _add_bar_label(axes[1], x, value)

    note = "Ledger overhead computed from real JSONL metadata ledgers only; size from file bytes and latency from timestamp_end - timestamp_start."
    pdf_path, png_path = _save_figure(fig, "fig_ledger_overhead", metadata_note=note, sources=sources)
    return [pdf_path, png_path], sources, note


def _display_path(path_text: str) -> str:
    path = Path(path_text)
    try:
        return f"`{path.resolve().relative_to(REPO_ROOT.resolve())}`"
    except ValueError:
        return f"`{path_text}`"


def _write_readme(
    *,
    generated_files: list[Path],
    plot_sources: dict[str, list[str]],
    plot_notes: dict[str, str],
    missing_inputs: list[str],
    predictions_present: bool,
) -> None:
    lines = [
        "# PLOTS README",
        "",
        "## Regeneration",
        f"`cd {REPO_ROOT} && python3 scripts/generate_ieee_plots.py`",
        "",
        "## Plot Inputs",
        "",
        "- The plots use only already-generated files under `outputs/`.",
        "- Screenshots are not used as data.",
        "- Metric comparison plots use generated metrics CSVs; convergence plots use `round_metrics.csv`; ledger overhead uses JSONL ledgers.",
        f"- `outputs/predictions/` present: `{'YES' if predictions_present else 'NO'}`.",
    ]
    if not predictions_present:
        lines.append("- No prediction files were available, so no plot depends on saved probability outputs.")
    lines.extend(["", "## Per-Plot Sources"])

    for plot_name, sources in plot_sources.items():
        lines.extend(
            [
                "",
                f"### {plot_name}",
                f"- Data files used: {', '.join(_display_path(source) for source in sorted(set(sources))) if sources else 'NONE'}",
                f"- Seed mode: {plot_notes[plot_name]}",
            ]
        )

    lines.extend(["", "## Missing Inputs"])
    if missing_inputs:
        for item in missing_inputs:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Generated Files",
        ]
    )
    for path in generated_files:
        lines.append(f"- `{path.relative_to(REPO_ROOT)}`")

    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_plots() -> dict[str, Any]:
    _configure_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = _load_experiment_matrix()
    aggregates = _build_metric_aggregates()
    missing_inputs: list[str] = []
    plot_sources: dict[str, list[str]] = {}
    plot_notes: dict[str, str] = {}
    generated_files: list[Path] = []

    required_for_main = [exp_id for group in A_B_P_GROUPS.values() for exp_id in group]
    for exp_id in required_for_main:
        if exp_id not in aggregates:
            missing_inputs.append(f"Missing aggregated metrics for {exp_id}.")

    required_for_ablations = [exp_id for pair in ABLATION_GROUPS.values() for exp_id in pair]
    for exp_id in required_for_ablations:
        if exp_id not in aggregates:
            missing_inputs.append(f"Missing aggregated metrics for {exp_id}.")

    files, sources, note = _plot_f1_comparison(aggregates)
    generated_files.extend(files)
    plot_sources["fig_f1_comparison"] = sources
    plot_notes["fig_f1_comparison"] = note

    files, sources, note = _plot_auroc_comparison(aggregates)
    generated_files.extend(files)
    plot_sources["fig_auroc_comparison"] = sources
    plot_notes["fig_auroc_comparison"] = note

    files, sources, note = _plot_pr_auc_comparison(aggregates)
    generated_files.extend(files)
    plot_sources["fig_pr_auc_comparison"] = sources
    plot_notes["fig_pr_auc_comparison"] = note

    files, sources, note = _plot_fpr_comparison(aggregates)
    generated_files.extend(files)
    plot_sources["fig_fpr_comparison"] = sources
    plot_notes["fig_fpr_comparison"] = note

    files, sources, note = _plot_ablation_delta_f1(aggregates)
    generated_files.extend(files)
    plot_sources["fig_ablation_delta_f1"] = sources
    plot_notes["fig_ablation_delta_f1"] = note

    for cluster_label, experiment_ids in A_B_P_GROUPS.items():
        files, sources, note = _plot_convergence(cluster_label, experiment_ids)
        generated_files.extend(files)
        key = f"fig_convergence_cluster{cluster_label.split()[-1]}"
        plot_sources[key] = sources
        plot_notes[key] = note

    files, sources, note = _plot_communication_cost(aggregates)
    generated_files.extend(files)
    plot_sources["fig_communication_cost"] = sources
    plot_notes["fig_communication_cost"] = note

    files, sources, note = _plot_ledger_overhead()
    generated_files.extend(files)
    plot_sources["fig_ledger_overhead"] = sources
    plot_notes["fig_ledger_overhead"] = note

    predictions_present = (OUTPUT_ROOT / "predictions").exists()
    if not predictions_present:
        missing_inputs.append("`outputs/predictions/` directory is absent; no prediction-based artifacts were used.")

    for experiment_id, aggregate in aggregates.items():
        if aggregate.successful_seed_count <= 1 and aggregate.missing_seeds:
            missing_inputs.append(
                f"{experiment_id}: only {', '.join(aggregate.successful_seeds) or 'one unlabelled run'} available; "
                f"missing seeds {', '.join(aggregate.missing_seeds)}."
            )

    _write_readme(
        generated_files=generated_files,
        plot_sources=plot_sources,
        plot_notes=plot_notes,
        missing_inputs=sorted(set(missing_inputs)),
        predictions_present=predictions_present,
    )

    return {
        "generated_files": [str(path) for path in generated_files],
        "missing_inputs": sorted(set(missing_inputs)),
        "plots_use_real_metrics_only": True,
        "known_experiments": sorted(matrix.keys()),
    }


def main() -> None:
    report = generate_plots()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
