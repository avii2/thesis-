from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.fl.maincluster import _split_metrics, validation_threshold_sweep
from src.train_cluster1_proposed import run_cluster1_proposed


EXPERIMENT_ID = "P_C1_BATADAL"
DEFAULT_OUTPUT_ROOT = Path("outputs_c1_batadal")
DEFAULT_PROPOSED_CONFIG_PATH = Path("configs/proposed_cluster1_batadal.yaml")
THRESHOLD_SELECTION_MODES = (
    "max_validation_f1",
    "max_validation_f1_with_validation_fpr_le_0.10",
    "max_validation_f1_with_validation_fpr_le_0.05",
)
THRESHOLD_MODE_FPR_LIMITS: Mapping[str, float | None] = {
    "max_validation_f1": None,
    "max_validation_f1_with_validation_fpr_le_0.10": 0.10,
    "max_validation_f1_with_validation_fpr_le_0.05": 0.05,
}
POSITIVE_CLASS_WEIGHT_SCALES = (0.25, 0.50, 1.00)

THRESHOLD_SWEEP_FILENAME = "p_c1_batadal_threshold_sweep.csv"
CLASS_WEIGHT_ABLATION_FILENAME = "p_c1_batadal_class_weight_ablation.csv"
CALIBRATION_REPORT_FILENAME = "p_c1_batadal_calibration_report.md"
THRESHOLD_CALIBRATION_REPORT_FILENAME = "p_c1_batadal_threshold_calibration.md"


@dataclass(frozen=True)
class PredictionSet:
    validation_labels: np.ndarray
    validation_probabilities: np.ndarray
    test_labels: np.ndarray
    test_probabilities: np.ndarray
    validation_predictions_path: Path
    test_predictions_path: Path


@dataclass(frozen=True)
class CalibrationArtifacts:
    threshold_sweep_csv_path: Path
    class_weight_ablation_csv_path: Path
    report_path: Path
    threshold_report_path: Path
    class_weight_run_roots: tuple[Path, ...]


def _prediction_suffix(seed: int | None) -> str:
    return f"_seed_{int(seed)}" if seed is not None else ""


def load_prediction_set(
    output_root: str | Path,
    *,
    experiment_id: str = EXPERIMENT_ID,
    seed: int | None = 42,
) -> PredictionSet:
    prediction_dir = Path(output_root) / "predictions" / experiment_id
    suffix = _prediction_suffix(seed)
    validation_path = prediction_dir / f"validation_predictions{suffix}.npz"
    test_path = prediction_dir / f"test_predictions{suffix}.npz"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Missing validation predictions for threshold tuning: {validation_path}. "
            "Run P_C1_BATADAL first."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing held-out test predictions for final evaluation: {test_path}. "
            "Run P_C1_BATADAL first."
        )

    with np.load(validation_path) as validation_npz:
        validation_labels = validation_npz["labels"].astype(np.int8, copy=True).reshape(-1)
        validation_probabilities = validation_npz["probabilities"].astype(np.float32, copy=True).reshape(-1)
    with np.load(test_path) as test_npz:
        test_labels = test_npz["labels"].astype(np.int8, copy=True).reshape(-1)
        test_probabilities = test_npz["probabilities"].astype(np.float32, copy=True).reshape(-1)

    if validation_labels.shape[0] != validation_probabilities.shape[0]:
        raise ValueError(f"{validation_path}: labels and probabilities have different lengths.")
    if test_labels.shape[0] != test_probabilities.shape[0]:
        raise ValueError(f"{test_path}: labels and probabilities have different lengths.")

    return PredictionSet(
        validation_labels=validation_labels,
        validation_probabilities=validation_probabilities,
        test_labels=test_labels,
        test_probabilities=test_probabilities,
        validation_predictions_path=validation_path,
        test_predictions_path=test_path,
    )


def _numeric_metric(value: Any, *, missing_value: float = float("-inf")) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return missing_value
    return missing_value


def _select_validation_threshold_row(
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    *,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in THRESHOLD_MODE_FPR_LIMITS:
        raise ValueError(f"Unsupported threshold selection mode: {mode}")

    sweep_rows = validation_threshold_sweep(validation_labels, validation_probabilities)
    if not sweep_rows:
        raise ValueError("No validation thresholds are available; test data was not used.")

    fpr_limit = THRESHOLD_MODE_FPR_LIMITS[mode]
    eligible_rows = [
        row
        for row in sweep_rows
        if fpr_limit is None or float(row["validation_fpr"]) <= fpr_limit + 1e-12
    ]
    if not eligible_rows:
        raise ValueError(f"No validation threshold satisfies {mode}; test data was not used for fallback tuning.")

    return max(
        eligible_rows,
        key=lambda row: (
            _numeric_metric(row.get("validation_f1")),
            _numeric_metric(row.get("validation_pr_auc")),
            -_numeric_metric(row.get("validation_fpr"), missing_value=float("inf")),
            _numeric_metric(row.get("threshold")),
        ),
    )


def _confusion_parts(metrics: Mapping[str, Any]) -> dict[str, int]:
    matrix = metrics["confusion_matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def evaluate_threshold_modes(
    prediction_set: PredictionSet,
    *,
    experiment_id: str = EXPERIMENT_ID,
    seed: int | None = 42,
    source_output_root: str | Path | None = None,
    positive_class_weight_scale: float | None = None,
    run_summary: Mapping[str, Any] | None = None,
    threshold_modes: Sequence[str] = THRESHOLD_SELECTION_MODES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in threshold_modes:
        selected = _select_validation_threshold_row(
            prediction_set.validation_labels,
            prediction_set.validation_probabilities,
            mode=mode,
        )
        threshold = float(selected["threshold"])
        validation_metrics = _split_metrics(
            prediction_set.validation_labels,
            prediction_set.validation_probabilities,
            threshold=threshold,
        )
        test_metrics = _split_metrics(
            prediction_set.test_labels,
            prediction_set.test_probabilities,
            threshold=threshold,
        )
        test_confusion = _confusion_parts(test_metrics)
        validation_confusion = _confusion_parts(validation_metrics)

        row = {
            "experiment_id": experiment_id,
            "seed": int(seed) if seed is not None else "",
            "positive_class_weight_scale": positive_class_weight_scale
            if positive_class_weight_scale is not None
            else "",
            "computed_positive_class_weight": "",
            "positive_class_weight": "",
            "best_validation_round": "",
            "threshold_selection_mode": mode,
            "validation_fpr_limit": THRESHOLD_MODE_FPR_LIMITS[mode]
            if THRESHOLD_MODE_FPR_LIMITS[mode] is not None
            else "",
            "threshold_selected_on": "validation",
            "test_dataset_used_for_threshold_tuning": False,
            "selected_threshold": threshold,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1": validation_metrics["f1"],
            "validation_auroc": validation_metrics["auroc"],
            "validation_pr_auc": validation_metrics["pr_auc"],
            "validation_fpr": validation_metrics["fpr"],
            "validation_support": validation_metrics["support"],
            "validation_tn": validation_confusion["tn"],
            "validation_fp": validation_confusion["fp"],
            "validation_fn": validation_confusion["fn"],
            "validation_tp": validation_confusion["tp"],
            "validation_confusion_matrix": _json_value(validation_metrics["confusion_matrix"]),
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auroc": test_metrics["auroc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_fpr": test_metrics["fpr"],
            "test_support": test_metrics["support"],
            "test_tn": test_confusion["tn"],
            "test_fp": test_confusion["fp"],
            "test_fn": test_confusion["fn"],
            "test_tp": test_confusion["tp"],
            "test_confusion_matrix": _json_value(test_metrics["confusion_matrix"]),
            "validation_predictions_path": str(prediction_set.validation_predictions_path),
            "test_predictions_path": str(prediction_set.test_predictions_path),
            "source_output_root": str(source_output_root) if source_output_root is not None else "",
        }
        if run_summary is not None:
            row.update(
                {
                    "computed_positive_class_weight": run_summary.get("computed_positive_class_weight", ""),
                    "positive_class_weight": run_summary.get("positive_class_weight", ""),
                    "best_validation_round": run_summary.get("best_validation_round", ""),
                }
            )
            if not row["positive_class_weight_scale"]:
                row["positive_class_weight_scale"] = run_summary.get("positive_class_weight_scale", "")
        rows.append(row)
    return rows


def write_threshold_calibration_report(
    report_path: str | Path,
    *,
    threshold_rows: Sequence[Mapping[str, Any]],
    threshold_csv_path: str | Path,
) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P_C1_BATADAL Threshold Calibration",
        "",
        "This report is limited to Cluster 1 P_C1_BATADAL threshold calibration.",
        "",
        "Protocol:",
        "- Thresholds are selected on validation predictions only.",
        "- Held-out BATADAL test predictions are evaluated only after each threshold is fixed.",
        "- The BATADAL train/validation/test split, windowing, and preprocessing are unchanged.",
        "- Test data is not used for threshold tuning.",
        "",
        f"Threshold sweep CSV: `{threshold_csv_path}`",
        "",
        *_threshold_report_table(threshold_rows),
        "",
        "Confusion matrix layout is `[[TN, FP], [FN, TP]]`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _load_run_summary(output_root: str | Path, *, experiment_id: str = EXPERIMENT_ID) -> Mapping[str, Any]:
    summary_path = Path(output_root) / "runs" / experiment_id / "run_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{summary_path}: expected JSON object.")
    return payload


def _scale_name(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "_")


def _class_weight_run_root(calibration_root: Path, scale: float) -> Path:
    return calibration_root / "class_weight_runs" / f"positive_class_weight_scale_{_scale_name(scale)}"


def _has_prediction_outputs(output_root: Path, *, seed: int | None) -> bool:
    suffix = _prediction_suffix(seed)
    prediction_dir = output_root / "predictions" / EXPERIMENT_ID
    return (
        (prediction_dir / f"validation_predictions{suffix}.npz").exists()
        and (prediction_dir / f"test_predictions{suffix}.npz").exists()
    )


def _run_or_reuse_class_weight_variant(
    *,
    main_output_root: Path,
    calibration_root: Path,
    proposed_config_path: Path,
    scale: float,
    seed: int,
    force_rerun: bool,
    rounds: int | None,
    local_epochs: int | None,
    batch_size: int | None,
    smoke_test: bool,
    max_train_examples_per_client: int | None,
    max_eval_examples_per_client: int | None,
) -> tuple[Path, bool]:
    if scale == 1.0 and not force_rerun and _has_prediction_outputs(main_output_root, seed=seed):
        return main_output_root, True

    run_root = _class_weight_run_root(calibration_root, scale)
    if not force_rerun and _has_prediction_outputs(run_root, seed=seed):
        return run_root, True

    run_cluster1_proposed(
        proposed_config_path=proposed_config_path,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
        smoke_test=smoke_test,
        max_train_examples_per_client=max_train_examples_per_client,
        max_eval_examples_per_client=max_eval_examples_per_client,
        output_root=run_root,
        positive_class_weight_scale=scale,
    )
    return run_root, False


def _report_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _threshold_report_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| mode | threshold | validation_f1 | validation_fpr | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | test_confusion_matrix |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {threshold} | {validation_f1} | {validation_fpr} | {test_accuracy} | {test_precision} | {test_recall} | {test_f1} | {test_auroc} | {test_pr_auc} | {test_fpr} | `{matrix}` |".format(
                mode=row["threshold_selection_mode"],
                threshold=_report_metric(row["selected_threshold"]),
                validation_f1=_report_metric(row["validation_f1"]),
                validation_fpr=_report_metric(row["validation_fpr"]),
                test_accuracy=_report_metric(row["test_accuracy"]),
                test_precision=_report_metric(row["test_precision"]),
                test_recall=_report_metric(row["test_recall"]),
                test_f1=_report_metric(row["test_f1"]),
                test_auroc=_report_metric(row["test_auroc"]),
                test_pr_auc=_report_metric(row["test_pr_auc"]),
                test_fpr=_report_metric(row["test_fpr"]),
                matrix=row["test_confusion_matrix"],
            )
        )
    return lines


def _class_weight_report_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| positive_class_weight_scale | mode | threshold | validation_f1 | validation_fpr | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | test_confusion_matrix |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scale} | {mode} | {threshold} | {validation_f1} | {validation_fpr} | {test_accuracy} | {test_precision} | {test_recall} | {test_f1} | {test_auroc} | {test_pr_auc} | {test_fpr} | `{matrix}` |".format(
                scale=_report_metric(row["positive_class_weight_scale"]),
                mode=row["threshold_selection_mode"],
                threshold=_report_metric(row["selected_threshold"]),
                validation_f1=_report_metric(row["validation_f1"]),
                validation_fpr=_report_metric(row["validation_fpr"]),
                test_accuracy=_report_metric(row["test_accuracy"]),
                test_precision=_report_metric(row["test_precision"]),
                test_recall=_report_metric(row["test_recall"]),
                test_f1=_report_metric(row["test_f1"]),
                test_auroc=_report_metric(row["test_auroc"]),
                test_pr_auc=_report_metric(row["test_pr_auc"]),
                test_fpr=_report_metric(row["test_fpr"]),
                matrix=row["test_confusion_matrix"],
            )
        )
    return lines


def write_calibration_report(
    report_path: str | Path,
    *,
    threshold_rows: Sequence[Mapping[str, Any]],
    class_weight_rows: Sequence[Mapping[str, Any]],
    threshold_csv_path: str | Path,
    class_weight_csv_path: str | Path,
) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P_C1_BATADAL Calibration Report",
        "",
        "This report is limited to Cluster 1 P_C1_BATADAL. Cluster 2, Cluster 3, and ledger logic are not modified.",
        "",
        "Protocol:",
        "- Thresholds are selected on validation predictions only.",
        "- Held-out BATADAL test predictions are used only after each validation-selected threshold is fixed.",
        "- The raw train/validation/test split is unchanged.",
        "- The threshold selection modes are `max_validation_f1`, `max_validation_f1_with_validation_fpr_le_0.10`, and `max_validation_f1_with_validation_fpr_le_0.05`.",
        "- Positive class weight scale ablation uses `0.25`, `0.50`, and `1.00` for P_C1_BATADAL only.",
        "",
        f"Threshold sweep CSV: `{threshold_csv_path}`",
        f"Class-weight ablation CSV: `{class_weight_csv_path}`",
        "",
        "## Threshold Selection",
        "",
        *_threshold_report_table(threshold_rows),
        "",
        "## Positive Class Weight Ablation",
        "",
        *_class_weight_report_table(class_weight_rows),
        "",
        "Confusion matrix layout is `[[TN, FP], [FN, TP]]`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_calibration_analysis(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    proposed_config_path: str | Path = DEFAULT_PROPOSED_CONFIG_PATH,
    seed: int = 42,
    positive_class_weight_scales: Sequence[float] = POSITIVE_CLASS_WEIGHT_SCALES,
    threshold_modes: Sequence[str] = THRESHOLD_SELECTION_MODES,
    force_rerun_class_weight: bool = False,
    rounds: int | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
    smoke_test: bool = False,
    max_train_examples_per_client: int | None = None,
    max_eval_examples_per_client: int | None = None,
) -> CalibrationArtifacts:
    output_root = Path(output_root)
    proposed_config_path = Path(proposed_config_path)
    calibration_root = output_root / "calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)

    main_predictions = load_prediction_set(output_root, seed=seed)
    main_summary = _load_run_summary(output_root)
    threshold_rows = evaluate_threshold_modes(
        main_predictions,
        seed=seed,
        source_output_root=output_root,
        run_summary=main_summary,
        threshold_modes=threshold_modes,
    )
    threshold_csv_path = write_csv_rows(calibration_root / THRESHOLD_SWEEP_FILENAME, threshold_rows)
    threshold_report_path = write_threshold_calibration_report(
        output_root / "reports" / THRESHOLD_CALIBRATION_REPORT_FILENAME,
        threshold_rows=threshold_rows,
        threshold_csv_path=threshold_csv_path,
    )

    class_weight_rows: list[dict[str, Any]] = []
    class_weight_run_roots: list[Path] = []
    for scale in positive_class_weight_scales:
        resolved_scale = float(scale)
        run_root, reused_existing = _run_or_reuse_class_weight_variant(
            main_output_root=output_root,
            calibration_root=calibration_root,
            proposed_config_path=proposed_config_path,
            scale=resolved_scale,
            seed=seed,
            force_rerun=force_rerun_class_weight,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            smoke_test=smoke_test,
            max_train_examples_per_client=max_train_examples_per_client,
            max_eval_examples_per_client=max_eval_examples_per_client,
        )
        class_weight_run_roots.append(run_root)
        predictions = load_prediction_set(run_root, seed=seed)
        summary = _load_run_summary(run_root)
        rows = evaluate_threshold_modes(
            predictions,
            seed=seed,
            source_output_root=run_root,
            positive_class_weight_scale=resolved_scale,
            run_summary=summary,
            threshold_modes=threshold_modes,
        )
        for row in rows:
            row["reused_existing_run"] = reused_existing
        class_weight_rows.extend(rows)

    class_weight_csv_path = write_csv_rows(calibration_root / CLASS_WEIGHT_ABLATION_FILENAME, class_weight_rows)
    report_path = write_calibration_report(
        output_root / "reports" / CALIBRATION_REPORT_FILENAME,
        threshold_rows=threshold_rows,
        class_weight_rows=class_weight_rows,
        threshold_csv_path=threshold_csv_path,
        class_weight_csv_path=class_weight_csv_path,
    )
    return CalibrationArtifacts(
        threshold_sweep_csv_path=threshold_csv_path,
        class_weight_ablation_csv_path=class_weight_csv_path,
        report_path=report_path,
        threshold_report_path=threshold_report_path,
        class_weight_run_roots=tuple(class_weight_run_roots),
    )


def iter_metric_rows_for_report(rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    yield from rows
