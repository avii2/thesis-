from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis.cluster1_batadal_calibration import (
    EXPERIMENT_ID,
    evaluate_threshold_modes,
    load_prediction_set,
    write_csv_rows,
)
from src.train_cluster1_proposed import run_cluster1_proposed


DEFAULT_OUTPUT_ROOT = Path("outputs_c1_batadal_tuned") / "positive_window_oversampling"
DEFAULT_PROPOSED_CONFIG_PATH = Path("configs/proposed_cluster1_batadal.yaml")
DEFAULT_THRESHOLD_MODE = "max_validation_f1_with_validation_fpr_le_0.10"
POSITIVE_CLASS_WEIGHT_SCALES = (0.10, 0.25, 0.50)
TARGET_POSITIVE_FRACTION = 0.5
COMPARISON_CSV = "p_c1_batadal_positive_oversampling_selection.csv"
COMPARISON_REPORT = "p_c1_batadal_positive_oversampling_selection.md"


@dataclass(frozen=True)
class ResamplingRunSpec:
    run_id: str
    output_root: Path
    positive_class_weight_scale: float
    threshold_mode: str = DEFAULT_THRESHOLD_MODE
    training_resampling: str = "positive_window_oversampling"
    target_positive_fraction: float = TARGET_POSITIVE_FRACTION
    seed: int = 42


@dataclass(frozen=True)
class ResamplingArtifacts:
    comparison_csv_path: Path
    comparison_report_path: Path
    selected_row: Mapping[str, Any]
    run_roots: tuple[Path, ...]


def _scale_name(scale: float) -> str:
    return f"scale_{int(round(scale * 100)):03d}"


def _prediction_suffix(seed: int | None) -> str:
    return f"_seed_{int(seed)}" if seed is not None else ""


def _has_prediction_outputs(output_root: Path, *, seed: int) -> bool:
    suffix = _prediction_suffix(seed)
    prediction_dir = output_root / "predictions" / EXPERIMENT_ID
    return (
        (prediction_dir / f"validation_predictions{suffix}.npz").exists()
        and (prediction_dir / f"test_predictions{suffix}.npz").exists()
    )


def _read_run_summary(output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "runs" / EXPERIMENT_ID / "run_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_path}: expected JSON object.")
    return payload


def _metric_float(value: Any, *, default: float = float("-inf")) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def build_resampling_specs(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    scales: Sequence[float] = POSITIVE_CLASS_WEIGHT_SCALES,
    seed: int = 42,
) -> tuple[ResamplingRunSpec, ...]:
    output_root = Path(output_root)
    return tuple(
        ResamplingRunSpec(
            run_id=_scale_name(float(scale)),
            output_root=output_root / _scale_name(float(scale)),
            positive_class_weight_scale=float(scale),
            seed=seed,
        )
        for scale in scales
    )


def run_or_reuse_spec(
    spec: ResamplingRunSpec,
    *,
    proposed_config_path: str | Path = DEFAULT_PROPOSED_CONFIG_PATH,
    force_rerun: bool = False,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    dropout: float | None = None,
) -> bool:
    if not force_rerun and _has_prediction_outputs(spec.output_root, seed=spec.seed):
        return True
    run_cluster1_proposed(
        proposed_config_path=proposed_config_path,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=spec.seed,
        output_root=spec.output_root,
        cnn_bn_dropout=dropout,
        positive_class_weight_scale=spec.positive_class_weight_scale,
        training_resampling=spec.training_resampling,
        target_positive_fraction=spec.target_positive_fraction,
    )
    return False


def _row_for_spec(spec: ResamplingRunSpec, *, reused_existing_run: bool) -> dict[str, Any]:
    predictions = load_prediction_set(spec.output_root, experiment_id=EXPERIMENT_ID, seed=spec.seed)
    metrics = evaluate_threshold_modes(
        predictions,
        experiment_id=EXPERIMENT_ID,
        seed=spec.seed,
        source_output_root=spec.output_root,
        positive_class_weight_scale=spec.positive_class_weight_scale,
        threshold_modes=(spec.threshold_mode,),
    )[0]
    summary = _read_run_summary(spec.output_root)
    return {
        "run_id": spec.run_id,
        "experiment_id": EXPERIMENT_ID,
        "output_root": str(spec.output_root),
        "seed": spec.seed,
        "rounds": summary.get("rounds", ""),
        "local_epochs": summary.get("local_epochs", ""),
        "batch_size": summary.get("batch_size", ""),
        "learning_rate": summary.get("learning_rate", ""),
        "dropout": summary.get("cnn_bn_dropout", ""),
        "positive_class_weight_scale": spec.positive_class_weight_scale,
        "training_resampling": spec.training_resampling,
        "target_positive_fraction": spec.target_positive_fraction,
        "threshold_mode": spec.threshold_mode,
        "selected_threshold": metrics["selected_threshold"],
        "validation_f1": metrics["validation_f1"],
        "validation_fpr": metrics["validation_fpr"],
        "validation_precision": metrics["validation_precision"],
        "validation_recall": metrics["validation_recall"],
        "validation_confusion_matrix": metrics["validation_confusion_matrix"],
        "best_validation_round": summary.get("best_validation_round", ""),
        "is_selected": False,
        "test_accuracy": "",
        "test_precision": "",
        "test_recall": "",
        "test_f1": "",
        "test_auroc": "",
        "test_pr_auc": "",
        "test_fpr": "",
        "test_confusion_matrix": "",
        "reused_existing_run": reused_existing_run,
        "_heldout_test_metrics": {
            "test_accuracy": metrics["test_accuracy"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "test_f1": metrics["test_f1"],
            "test_auroc": metrics["test_auroc"],
            "test_pr_auc": metrics["test_pr_auc"],
            "test_fpr": metrics["test_fpr"],
            "test_confusion_matrix": metrics["test_confusion_matrix"],
        },
    }


def select_by_validation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if _metric_float(row.get("validation_fpr")) <= 0.10]
    if not eligible:
        raise ValueError("No positive-window-oversampling candidate satisfies validation FPR <= 0.10.")
    return dict(
        max(
            eligible,
            key=lambda row: (
                _metric_float(row.get("validation_f1")),
                -_metric_float(row.get("validation_fpr"), default=1.0),
            ),
        )
    )


def _publish_selected_test_metrics(rows: Sequence[dict[str, Any]], selected_run_id: str) -> list[dict[str, Any]]:
    published: list[dict[str, Any]] = []
    for row in rows:
        cleaned = dict(row)
        heldout = cleaned.pop("_heldout_test_metrics")
        if cleaned["run_id"] == selected_run_id:
            cleaned["is_selected"] = True
            cleaned.update(heldout)
        published.append(cleaned)
    return published


def _format_metric(value: Any) -> str:
    numeric = _metric_float(value, default=float("nan"))
    if numeric == numeric:
        return f"{numeric:.6f}"
    return str(value)


def write_report(
    path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    selected_row: Mapping[str, Any],
    csv_path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P_C1_BATADAL Positive-Window Oversampling Selection",
        "",
        "Scope: Cluster 1 P_C1_BATADAL only. The BATADAL split, window length, stride, last-row labeling, model family, FedBN method, weighted non-BN aggregation, and fixed sub-clusters are unchanged.",
        "",
        "Training change: positive-window oversampling is applied only inside local client training. Validation and held-out test windows are not resampled.",
        "",
        f"CSV: `{csv_path}`",
        "",
        "Selection rule: highest validation F1 among candidates with validation FPR <= 0.10. Test metrics are reported only for the selected candidate.",
        "",
        "| run_id | scale | validation_f1 | validation_fpr | threshold | selected | test_f1 | test_fpr | test_confusion_matrix |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {scale} | {validation_f1} | {validation_fpr} | {threshold} | {selected} | {test_f1} | {test_fpr} | {matrix} |".format(
                run_id=row["run_id"],
                scale=_format_metric(row["positive_class_weight_scale"]),
                validation_f1=_format_metric(row["validation_f1"]),
                validation_fpr=_format_metric(row["validation_fpr"]),
                threshold=_format_metric(row["selected_threshold"]),
                selected=row["is_selected"],
                test_f1=_format_metric(row["test_f1"]) if row["test_f1"] != "" else "",
                test_fpr=_format_metric(row["test_fpr"]) if row["test_fpr"] != "" else "",
                matrix=f"`{row['test_confusion_matrix']}`" if row["test_confusion_matrix"] else "",
            )
        )

    lines.extend(
        [
            "",
            "## Final Selection",
            "",
            f"Selected run: `{selected_row['run_id']}`",
            f"positive_class_weight_scale: `{selected_row['positive_class_weight_scale']}`",
            f"training_resampling: `{selected_row['training_resampling']}`",
            f"target_positive_fraction: `{selected_row['target_positive_fraction']}`",
            f"threshold_mode: `{selected_row['threshold_mode']}`",
            f"selected_threshold: `{_format_metric(selected_row['selected_threshold'])}`",
            "",
            "Selected held-out test metrics:",
            f"- accuracy: {_format_metric(selected_row['test_accuracy'])}",
            f"- precision: {_format_metric(selected_row['test_precision'])}",
            f"- recall: {_format_metric(selected_row['test_recall'])}",
            f"- F1: {_format_metric(selected_row['test_f1'])}",
            f"- AUROC: {_format_metric(selected_row['test_auroc'])}",
            f"- PR-AUC: {_format_metric(selected_row['test_pr_auc'])}",
            f"- FPR: {_format_metric(selected_row['test_fpr'])}",
            f"- confusion matrix: `{selected_row['test_confusion_matrix']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_resampling_workflow(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    proposed_config_path: str | Path = DEFAULT_PROPOSED_CONFIG_PATH,
    scales: Sequence[float] = POSITIVE_CLASS_WEIGHT_SCALES,
    seed: int = 42,
    force_rerun: bool = False,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    dropout: float | None = None,
) -> ResamplingArtifacts:
    output_root = Path(output_root)
    specs = build_resampling_specs(output_root=output_root, scales=scales, seed=seed)
    rows: list[dict[str, Any]] = []
    run_roots: list[Path] = []
    for spec in specs:
        current_spec = replace(spec, output_root=output_root / spec.run_id)
        reused = run_or_reuse_spec(
            current_spec,
            proposed_config_path=proposed_config_path,
            force_rerun=force_rerun,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout=dropout,
        )
        rows.append(_row_for_spec(current_spec, reused_existing_run=reused))
        run_roots.append(current_spec.output_root)

    selected_private = select_by_validation(rows)
    published_rows = _publish_selected_test_metrics(rows, str(selected_private["run_id"]))
    selected_row = next(row for row in published_rows if row["run_id"] == selected_private["run_id"])
    reports_dir = output_root.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv_rows(reports_dir / COMPARISON_CSV, published_rows)
    report_path = write_report(
        reports_dir / COMPARISON_REPORT,
        rows=published_rows,
        selected_row=selected_row,
        csv_path=csv_path,
    )
    return ResamplingArtifacts(
        comparison_csv_path=csv_path,
        comparison_report_path=report_path,
        selected_row=selected_row,
        run_roots=tuple(run_roots),
    )
