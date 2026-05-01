from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis.cluster1_batadal_calibration import (
    EXPERIMENT_ID,
    THRESHOLD_SELECTION_MODES,
    evaluate_threshold_modes,
    load_prediction_set,
    write_csv_rows,
)
from src.train_cluster1_proposed import run_cluster1_proposed


DEFAULT_SOURCE_OUTPUT_ROOT = Path("outputs_c1_batadal")
DEFAULT_TUNED_OUTPUT_ROOT = Path("outputs_c1_batadal_tuned")
DEFAULT_PROPOSED_CONFIG_PATH = Path("configs/proposed_cluster1_batadal.yaml")
COMPARISON_CSV_FILENAME = "p_c1_batadal_tuning_comparison.csv"
COMPARISON_REPORT_FILENAME = "p_c1_batadal_tuning_comparison.md"


@dataclass(frozen=True)
class TuningRunSpec:
    run_id: str
    output_root: Path
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    dropout: float
    positive_class_weight_scale: float
    threshold_mode: str
    seed: int


@dataclass(frozen=True)
class TuningArtifacts:
    threshold_sweep_csv_path: Path
    threshold_report_path: Path
    comparison_csv_path: Path
    comparison_report_path: Path
    executed_run_roots: tuple[Path, ...]
    final_row: Mapping[str, Any]


FIRST_TUNING_SPEC = TuningRunSpec(
    run_id="scale_010_lr_0001_dropout_020",
    output_root=DEFAULT_TUNED_OUTPUT_ROOT / "scale_010_lr_0001_dropout_020",
    rounds=50,
    local_epochs=1,
    batch_size=64,
    learning_rate=0.001,
    dropout=0.20,
    positive_class_weight_scale=0.10,
    threshold_mode="max_validation_f1_with_validation_fpr_le_0.10",
    seed=42,
)
FPR_REDUCTION_SPEC = TuningRunSpec(
    run_id="scale_005_lr_0001_dropout_020",
    output_root=DEFAULT_TUNED_OUTPUT_ROOT / "scale_005_lr_0001_dropout_020",
    rounds=50,
    local_epochs=1,
    batch_size=64,
    learning_rate=0.001,
    dropout=0.20,
    positive_class_weight_scale=0.05,
    threshold_mode="max_validation_f1_with_validation_fpr_le_0.05",
    seed=42,
)
RECALL_RECOVERY_SPEC = TuningRunSpec(
    run_id="scale_025_lr_0001_dropout_010",
    output_root=DEFAULT_TUNED_OUTPUT_ROOT / "scale_025_lr_0001_dropout_010",
    rounds=50,
    local_epochs=1,
    batch_size=64,
    learning_rate=0.001,
    dropout=0.10,
    positive_class_weight_scale=0.25,
    threshold_mode="max_validation_f1_with_validation_fpr_le_0.10",
    seed=42,
)


def _prediction_suffix(seed: int | None) -> str:
    return f"_seed_{int(seed)}" if seed is not None else ""


def _has_prediction_outputs(output_root: Path, *, experiment_id: str, seed: int) -> bool:
    suffix = _prediction_suffix(seed)
    prediction_dir = output_root / "predictions" / experiment_id
    return (
        (prediction_dir / f"validation_predictions{suffix}.npz").exists()
        and (prediction_dir / f"test_predictions{suffix}.npz").exists()
    )


def run_or_reuse_tuning_spec(
    spec: TuningRunSpec,
    *,
    proposed_config_path: str | Path = DEFAULT_PROPOSED_CONFIG_PATH,
    force_rerun: bool = False,
) -> bool:
    if not force_rerun and _has_prediction_outputs(spec.output_root, experiment_id=EXPERIMENT_ID, seed=spec.seed):
        return True

    run_cluster1_proposed(
        proposed_config_path=proposed_config_path,
        rounds=spec.rounds,
        local_epochs=spec.local_epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        seed=spec.seed,
        output_root=spec.output_root,
        cnn_bn_dropout=spec.dropout,
        positive_class_weight_scale=spec.positive_class_weight_scale,
    )
    return False


def _read_single_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[0]) if rows else {}


def _read_run_summary(output_root: Path, *, experiment_id: str) -> dict[str, Any]:
    summary_path = output_root / "runs" / experiment_id / "run_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_path}: expected JSON object.")
    return payload


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _best_validation_metric_name(row: Mapping[str, Any]) -> float:
    value = _float_or_none(row.get("validation_f1"))
    return float("-inf") if value is None else value


def _comparison_row_from_predictions(
    *,
    label: str,
    source_type: str,
    output_root: Path,
    experiment_id: str,
    threshold_mode: str,
    seed: int,
    training_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prediction_set = load_prediction_set(output_root, experiment_id=experiment_id, seed=seed)
    metrics_row = evaluate_threshold_modes(
        prediction_set,
        experiment_id=experiment_id,
        seed=seed,
        source_output_root=output_root,
        threshold_modes=(threshold_mode,),
    )[0]
    summary = _read_run_summary(output_root, experiment_id=experiment_id)
    csv_row = _read_single_csv_row(output_root / "metrics" / f"{experiment_id}_metrics.csv")
    training_overrides = dict(training_overrides or {})

    row = {
        "label": label,
        "source_type": source_type,
        "experiment_id": experiment_id,
        "output_root": str(output_root),
        "threshold_mode": threshold_mode,
        "selected_threshold": metrics_row["selected_threshold"],
        "validation_f1": metrics_row["validation_f1"],
        "validation_fpr": metrics_row["validation_fpr"],
        "test_accuracy": metrics_row["test_accuracy"],
        "test_precision": metrics_row["test_precision"],
        "test_recall": metrics_row["test_recall"],
        "test_f1": metrics_row["test_f1"],
        "test_auroc": metrics_row["test_auroc"],
        "test_pr_auc": metrics_row["test_pr_auc"],
        "test_fpr": metrics_row["test_fpr"],
        "test_confusion_matrix": metrics_row["test_confusion_matrix"],
        "seed": seed,
        "rounds": training_overrides.get("rounds", summary.get("rounds", csv_row.get("rounds", ""))),
        "local_epochs": training_overrides.get("local_epochs", summary.get("local_epochs", "")),
        "batch_size": training_overrides.get("batch_size", summary.get("batch_size", "")),
        "learning_rate": training_overrides.get("learning_rate", summary.get("learning_rate", "")),
        "dropout": training_overrides.get("dropout", summary.get("cnn_bn_dropout", csv_row.get("cnn_bn_dropout", ""))),
        "positive_class_weight_scale": training_overrides.get(
            "positive_class_weight_scale",
            summary.get("positive_class_weight_scale", csv_row.get("positive_class_weight_scale", "")),
        ),
        "best_validation_round": summary.get("best_validation_round", csv_row.get("best_validation_round", "")),
        "model_family": summary.get("model_family", csv_row.get("model_family", "")),
        "fl_method": summary.get("fl_method", csv_row.get("fl_method", "")),
        "aggregation": summary.get("aggregation", csv_row.get("aggregation", "")),
    }
    return row


def row_for_tuning_spec(spec: TuningRunSpec, *, reused_existing_run: bool) -> dict[str, Any]:
    row = _comparison_row_from_predictions(
        label=f"tuned P_C1_BATADAL {spec.run_id}",
        source_type="tuned_proposed",
        output_root=spec.output_root,
        experiment_id=EXPERIMENT_ID,
        threshold_mode=spec.threshold_mode,
        seed=spec.seed,
        training_overrides={
            "rounds": spec.rounds,
            "local_epochs": spec.local_epochs,
            "batch_size": spec.batch_size,
            "learning_rate": spec.learning_rate,
            "dropout": spec.dropout,
            "positive_class_weight_scale": spec.positive_class_weight_scale,
        },
    )
    row["run_id"] = spec.run_id
    row["reused_existing_run"] = reused_existing_run
    return row


def original_comparison_rows(*, source_output_root: Path, seed: int) -> list[dict[str, Any]]:
    return [
        _comparison_row_from_predictions(
            label="A_C1_BATADAL",
            source_type="baseline_flat",
            output_root=source_output_root,
            experiment_id="A_C1_BATADAL",
            threshold_mode="max_validation_f1",
            seed=seed,
        ),
        _comparison_row_from_predictions(
            label="B_C1_BATADAL",
            source_type="baseline_hierarchical",
            output_root=source_output_root,
            experiment_id="B_C1_BATADAL",
            threshold_mode="max_validation_f1",
            seed=seed,
        ),
        _comparison_row_from_predictions(
            label="original P_C1_BATADAL",
            source_type="original_proposed",
            output_root=source_output_root,
            experiment_id=EXPERIMENT_ID,
            threshold_mode="max_validation_f1",
            seed=seed,
        ),
    ]


def select_final_candidate(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    p_rows = [
        dict(row)
        for row in rows
        if row.get("experiment_id") == EXPERIMENT_ID
        and row.get("source_type") in {"original_proposed", "tuned_proposed"}
    ]
    if not p_rows:
        raise ValueError("No P_C1_BATADAL candidates are available for final selection.")

    eligible_010 = [
        row
        for row in p_rows
        if (_float_or_none(row.get("validation_fpr")) is not None and float(row["validation_fpr"]) <= 0.10)
    ]
    if eligible_010:
        return max(
            eligible_010,
            key=lambda row: (_best_validation_metric_name(row), -float(row["validation_fpr"])),
        ), "highest validation F1 with validation FPR <= 0.10"

    eligible_020 = [
        row
        for row in p_rows
        if (_float_or_none(row.get("validation_fpr")) is not None and float(row["validation_fpr"]) <= 0.20)
    ]
    if eligible_020:
        return max(
            eligible_020,
            key=lambda row: (_best_validation_metric_name(row), -float(row["validation_fpr"])),
        ), "highest validation F1 with validation FPR <= 0.20"

    return max(
        p_rows,
        key=lambda row: (_best_validation_metric_name(row), -float(row.get("validation_fpr", 1.0))),
    ), "fallback: no P_C1_BATADAL candidate satisfied validation FPR <= 0.20"


def _metric_text(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.6f}"


def _comparison_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| label | validation_f1 | validation_fpr | threshold | test_accuracy | test_precision | test_recall | test_f1 | test_auroc | test_pr_auc | test_fpr | confusion_matrix |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {validation_f1} | {validation_fpr} | {threshold} | {accuracy} | {precision} | {recall} | {f1} | {auroc} | {pr_auc} | {fpr} | `{matrix}` |".format(
                label=row["label"],
                validation_f1=_metric_text(row["validation_f1"]),
                validation_fpr=_metric_text(row["validation_fpr"]),
                threshold=_metric_text(row["selected_threshold"]),
                accuracy=_metric_text(row["test_accuracy"]),
                precision=_metric_text(row["test_precision"]),
                recall=_metric_text(row["test_recall"]),
                f1=_metric_text(row["test_f1"]),
                auroc=_metric_text(row["test_auroc"]),
                pr_auc=_metric_text(row["test_pr_auc"]),
                fpr=_metric_text(row["test_fpr"]),
                matrix=row["test_confusion_matrix"],
            )
        )
    return lines


def write_comparison_report(
    path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    final_row: Mapping[str, Any],
    decision_rule: str,
    comparison_csv_path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_a = next(row for row in rows if row["experiment_id"] == "A_C1_BATADAL")
    baseline_b = next(row for row in rows if row["experiment_id"] == "B_C1_BATADAL")
    final_test_f1 = float(final_row["test_f1"])
    beats_a = final_test_f1 > float(baseline_a["test_f1"])
    beats_b = final_test_f1 > float(baseline_b["test_f1"])
    fpr_acceptable = float(final_row["test_fpr"]) <= 0.20

    best_hyperparameters = {
        "rounds": final_row.get("rounds", ""),
        "local_epochs": final_row.get("local_epochs", ""),
        "batch_size": final_row.get("batch_size", ""),
        "learning_rate": final_row.get("learning_rate", ""),
        "dropout": final_row.get("dropout", ""),
        "positive_class_weight_scale": final_row.get("positive_class_weight_scale", ""),
        "seed": final_row.get("seed", ""),
    }
    recommendation = (
        f"Use `{final_row['label']}` with threshold mode `{final_row['threshold_mode']}` "
        f"and threshold `{_metric_text(final_row['selected_threshold'])}`."
    )
    if not fpr_acceptable:
        recommendation += " Test FPR remains above 0.20, so this should be reported as a recall/FPR trade-off rather than an acceptable operating point."

    lines = [
        "# P_C1_BATADAL Tuning Comparison",
        "",
        "Scope: Cluster 1 BATADAL only. The FCFL architecture, fixed sub-clusters, FedBN method, weighted non-BN aggregation, and BATADAL split are unchanged.",
        "",
        f"Comparison CSV: `{comparison_csv_path}`",
        "",
        "Decision rule: select using validation metrics only.",
        f"Applied rule: {decision_rule}.",
        "",
        "## Comparison",
        "",
        *_comparison_table(rows),
        "",
        "## Final Selection",
        "",
        f"Selected run: `{final_row['label']}`",
        f"Best hyperparameters: `{json.dumps(best_hyperparameters, sort_keys=True)}`",
        f"Selected threshold mode: `{final_row['threshold_mode']}`",
        f"Selected threshold: `{_metric_text(final_row['selected_threshold'])}`",
        "",
        "Final test metrics:",
        f"- accuracy: {_metric_text(final_row['test_accuracy'])}",
        f"- precision: {_metric_text(final_row['test_precision'])}",
        f"- recall: {_metric_text(final_row['test_recall'])}",
        f"- F1: {_metric_text(final_row['test_f1'])}",
        f"- AUROC: {_metric_text(final_row['test_auroc'])}",
        f"- PR-AUC: {_metric_text(final_row['test_pr_auc'])}",
        f"- FPR: {_metric_text(final_row['test_fpr'])}",
        f"- confusion matrix: `{final_row['test_confusion_matrix']}`",
        "",
        f"Beats A_C1_BATADAL on test F1: {beats_a}.",
        f"Beats B_C1_BATADAL on test F1: {beats_b}.",
        f"FPR acceptable under test FPR <= 0.20: {fpr_acceptable}.",
        "",
        f"Exact final recommendation: {recommendation}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _spec_to_run_metadata(spec: TuningRunSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["output_root"] = str(spec.output_root)
    return payload


def run_tuning_workflow(
    *,
    source_output_root: str | Path = DEFAULT_SOURCE_OUTPUT_ROOT,
    tuned_output_root: str | Path = DEFAULT_TUNED_OUTPUT_ROOT,
    proposed_config_path: str | Path = DEFAULT_PROPOSED_CONFIG_PATH,
    seed: int = 42,
    force_rerun: bool = False,
) -> TuningArtifacts:
    source_output_root = Path(source_output_root)
    tuned_output_root = Path(tuned_output_root)
    proposed_config_path = Path(proposed_config_path)
    reports_dir = tuned_output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    first_spec = replace(FIRST_TUNING_SPEC, output_root=tuned_output_root / FIRST_TUNING_SPEC.run_id)
    fpr_reduction_spec = replace(FPR_REDUCTION_SPEC, output_root=tuned_output_root / FPR_REDUCTION_SPEC.run_id)
    recall_recovery_spec = replace(RECALL_RECOVERY_SPEC, output_root=tuned_output_root / RECALL_RECOVERY_SPEC.run_id)

    # Task 1 threshold calibration for the original P_C1_BATADAL run.
    original_predictions = load_prediction_set(source_output_root, experiment_id=EXPERIMENT_ID, seed=seed)
    threshold_rows = evaluate_threshold_modes(
        original_predictions,
        experiment_id=EXPERIMENT_ID,
        seed=seed,
        source_output_root=source_output_root,
        threshold_modes=THRESHOLD_SELECTION_MODES,
    )
    from src.analysis.cluster1_batadal_calibration import (
        THRESHOLD_CALIBRATION_REPORT_FILENAME,
        THRESHOLD_SWEEP_FILENAME,
        write_threshold_calibration_report,
    )

    threshold_sweep_csv_path = write_csv_rows(
        source_output_root / "calibration" / THRESHOLD_SWEEP_FILENAME,
        threshold_rows,
    )
    threshold_report_path = write_threshold_calibration_report(
        source_output_root / "reports" / THRESHOLD_CALIBRATION_REPORT_FILENAME,
        threshold_rows=threshold_rows,
        threshold_csv_path=threshold_sweep_csv_path,
    )

    executed_run_roots: list[Path] = []
    tuned_rows: list[dict[str, Any]] = []

    first_reused = run_or_reuse_tuning_spec(
        first_spec,
        proposed_config_path=proposed_config_path,
        force_rerun=force_rerun,
    )
    executed_run_roots.append(first_spec.output_root)
    first_row = row_for_tuning_spec(first_spec, reused_existing_run=first_reused)
    first_row["run_metadata"] = json.dumps(_spec_to_run_metadata(first_spec), sort_keys=True)
    tuned_rows.append(first_row)

    conditional_rows = [first_row]
    if float(first_row["test_fpr"]) > 0.20:
        fpr_reused = run_or_reuse_tuning_spec(
            fpr_reduction_spec,
            proposed_config_path=proposed_config_path,
            force_rerun=force_rerun,
        )
        executed_run_roots.append(fpr_reduction_spec.output_root)
        fpr_row = row_for_tuning_spec(fpr_reduction_spec, reused_existing_run=fpr_reused)
        fpr_row["run_metadata"] = json.dumps(_spec_to_run_metadata(fpr_reduction_spec), sort_keys=True)
        tuned_rows.append(fpr_row)
        conditional_rows.append(fpr_row)

    if any(float(row["test_recall"]) < 0.50 for row in conditional_rows):
        recall_reused = run_or_reuse_tuning_spec(
            recall_recovery_spec,
            proposed_config_path=proposed_config_path,
            force_rerun=force_rerun,
        )
        executed_run_roots.append(recall_recovery_spec.output_root)
        recall_row = row_for_tuning_spec(recall_recovery_spec, reused_existing_run=recall_reused)
        recall_row["run_metadata"] = json.dumps(_spec_to_run_metadata(recall_recovery_spec), sort_keys=True)
        tuned_rows.append(recall_row)

    comparison_rows = original_comparison_rows(source_output_root=source_output_root, seed=seed) + tuned_rows
    final_row, decision_rule = select_final_candidate(comparison_rows)
    comparison_csv_path = write_csv_rows(reports_dir / COMPARISON_CSV_FILENAME, comparison_rows)
    comparison_report_path = write_comparison_report(
        reports_dir / COMPARISON_REPORT_FILENAME,
        rows=comparison_rows,
        final_row=final_row,
        decision_rule=decision_rule,
        comparison_csv_path=comparison_csv_path,
    )
    return TuningArtifacts(
        threshold_sweep_csv_path=threshold_sweep_csv_path,
        threshold_report_path=threshold_report_path,
        comparison_csv_path=comparison_csv_path,
        comparison_report_path=comparison_report_path,
        executed_run_roots=tuple(executed_run_roots),
        final_row=final_row,
    )
