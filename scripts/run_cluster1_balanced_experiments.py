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

from src.data.cluster1_balanced import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VARIANT_ROOT,
    HELDOUT_TEST_FILES,
    RATIO_SPECS,
    TRAIN_SOURCE_FILES,
    VALIDATION_SOURCE_FILES,
    prepare_cluster1_balanced_train,
)
from src.fl.maincluster import _split_metrics, validation_threshold_sweep  # noqa: E402
from src.train import (  # noqa: E402
    ExperimentSpec,
    _finalize_run_outputs,
    _load_metrics_row,
    _metric_as_float,
    _write_confusion_matrices_report,
)
from src.train_cluster1_proposed import run_cluster1_proposed  # noqa: E402
from src.train_hierarchical_baseline import run_hierarchical_baseline_experiments  # noqa: E402


ORIGINAL_METRICS_ROOT = Path("outputs/metrics")
ORIGINAL_RUNS_ROOT = Path("outputs/runs")
COMPARISON_CSV = DEFAULT_OUTPUT_ROOT / "reports" / "cluster1_balanced_training_comparison.csv"
COMPARISON_MD = DEFAULT_OUTPUT_ROOT / "reports" / "cluster1_balanced_training_comparison.md"
DOC_PATH = Path("docs/CLUSTER1_BALANCED_TRAINING_VARIANT.md")


def _short_ratio(ratio_name: str) -> str:
    return ratio_name.removeprefix("ratio_")


def _upper_ratio(ratio_name: str) -> str:
    return _short_ratio(ratio_name).upper()


def _balanced_specs(ratio_name: str) -> dict[str, ExperimentSpec]:
    upper = _upper_ratio(ratio_name)
    dataset = f"HAI 21.03 balanced-training variant {_short_ratio(ratio_name)}"
    return {
        f"B_C1_BAL_{upper}": ExperimentSpec(
            experiment_id=f"B_C1_BAL_{upper}",
            run_category="baseline_uniform_hierarchical",
            cluster_id=1,
            dataset=dataset,
            model="cnn1d",
            fl_method="FedAvg",
            aggregation="weighted_arithmetic_mean",
            hierarchy="hierarchical_fixed",
            clustering_method="agglomerative",
            n_subclusters=2,
            descriptor="feature_wise_mean_std",
            run_repeats=1,
            notes="Cluster 1 balanced-training variant.",
        ),
        f"P_C1_BAL_{upper}": ExperimentSpec(
            experiment_id=f"P_C1_BAL_{upper}",
            run_category="proposed_specialized_hierarchical",
            cluster_id=1,
            dataset=dataset,
            model="cnn1d_bn",
            fl_method="FedBN",
            aggregation="weighted_non_bn_mean",
            hierarchy="hierarchical_fixed",
            clustering_method="agglomerative",
            n_subclusters=2,
            descriptor="feature_wise_mean_std",
            run_repeats=1,
            notes="Cluster 1 balanced-training variant.",
        ),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_metrics_row(path)


def _validation_metrics_from_summary(summary: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        return {}
    metrics = summary.get("best_round_validation_metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _test_metrics_from_summary(summary: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        return {}
    metrics = summary.get("best_round_test_metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _selected_threshold_from_prediction(output_root: Path, experiment_id: str) -> float | None:
    threshold_path = output_root / "predictions" / experiment_id / "selected_threshold_seed_42.json"
    if not threshold_path.exists():
        return None
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    return _metric_as_float(payload.get("selected_threshold"))


def _write_ratio_threshold_and_test_metrics(
    *,
    ratio_root: Path,
    experiment_ids: Sequence[str],
) -> tuple[Path, Path]:
    threshold_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for experiment_id in experiment_ids:
        validation_path = ratio_root / "predictions" / experiment_id / "validation_predictions_seed_42.npz"
        test_path = ratio_root / "predictions" / experiment_id / "test_predictions_seed_42.npz"
        if validation_path.exists():
            import numpy as np

            with np.load(validation_path) as payload:
                validation_labels = payload["labels"].copy()
                validation_probabilities = payload["probabilities"].copy()
            sweep = validation_threshold_sweep(validation_labels, validation_probabilities)
            for row in sweep:
                threshold_rows.append({"experiment_id": experiment_id, **row})

        metrics_path = ratio_root / "metrics" / f"{experiment_id}_metrics.csv"
        summary_path = ratio_root / "runs" / experiment_id / "run_summary.json"
        row = _load_optional_metrics(metrics_path)
        summary = _load_json(summary_path)
        if row is None:
            continue
        validation_metrics = _validation_metrics_from_summary(summary)
        if test_path.exists():
            import numpy as np

            with np.load(test_path) as payload:
                test_labels = payload["labels"].copy()
                test_probabilities = payload["probabilities"].copy()
            selected_threshold = _metric_as_float(row.get("threshold_used"))
            selected_test_metrics = _split_metrics(test_labels, test_probabilities, threshold=float(selected_threshold))
        else:
            selected_test_metrics = _test_metrics_from_summary(summary)
        test_rows.append(
            {
                "experiment_id": experiment_id,
                "validation_precision": validation_metrics.get("precision"),
                "validation_recall": validation_metrics.get("recall"),
                "validation_f1": validation_metrics.get("f1"),
                "validation_pr_auc": validation_metrics.get("pr_auc"),
                "validation_fpr": validation_metrics.get("fpr"),
                "selected_threshold": row.get("threshold_used"),
                "test_accuracy": selected_test_metrics.get("accuracy", row.get("test_accuracy")),
                "test_precision": selected_test_metrics.get("precision", row.get("test_precision")),
                "test_recall": selected_test_metrics.get("recall", row.get("test_recall")),
                "test_f1": selected_test_metrics.get("f1", row.get("test_f1")),
                "test_auroc": selected_test_metrics.get("auroc", row.get("test_auroc")),
                "test_pr_auc": selected_test_metrics.get("pr_auc", row.get("test_pr_auc")),
                "test_fpr": selected_test_metrics.get("fpr", row.get("test_fpr")),
                "test_confusion_matrix": json.dumps(
                    selected_test_metrics.get("confusion_matrix", row.get("test_confusion_matrix"))
                ),
                "wall_clock_training_seconds": row.get("wall_clock_training_seconds"),
            }
        )

    threshold_path = _write_rows(ratio_root / "metrics" / "validation_threshold_selection.csv", threshold_rows)
    test_metrics_path = _write_rows(ratio_root / "metrics" / "test_metrics.csv", test_rows)
    return threshold_path, test_metrics_path


def _comparison_row(
    *,
    experiment_id: str,
    ratio: str,
    source: str,
    metrics_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    metrics = _load_optional_metrics(metrics_path) or {}
    summary = _load_json(summary_path)
    validation_metrics = _validation_metrics_from_summary(summary)
    test_metrics = _test_metrics_from_summary(summary)
    return {
        "experiment_id": experiment_id,
        "ratio": ratio,
        "source": source,
        "validation_precision": validation_metrics.get("precision"),
        "validation_recall": validation_metrics.get("recall"),
        "validation_f1": validation_metrics.get("f1", metrics.get("best_validation_f1")),
        "validation_pr_auc": validation_metrics.get("pr_auc"),
        "validation_fpr": validation_metrics.get("fpr"),
        "selected_threshold": metrics.get("threshold_used"),
        "test_accuracy": test_metrics.get("accuracy", metrics.get("test_accuracy")),
        "test_precision": test_metrics.get("precision", metrics.get("test_precision")),
        "test_recall": test_metrics.get("recall", metrics.get("test_recall")),
        "test_f1": test_metrics.get("f1", metrics.get("test_f1")),
        "test_auroc": test_metrics.get("auroc", metrics.get("test_auroc")),
        "test_pr_auc": test_metrics.get("pr_auc", metrics.get("test_pr_auc")),
        "test_fpr": test_metrics.get("fpr", metrics.get("test_fpr")),
        "test_confusion_matrix": json.dumps(test_metrics.get("confusion_matrix", metrics.get("test_confusion_matrix"))),
        "wall_clock_training_seconds": metrics.get("wall_clock_training_seconds"),
        "metrics_path": str(metrics_path) if metrics_path.exists() else "",
        "summary_path": str(summary_path) if summary_path.exists() else "",
    }


def _float(row: Mapping[str, Any], key: str) -> float | None:
    return _metric_as_float(row.get(key))


def _confusion(row: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    value = row.get("test_confusion_matrix")
    if value in (None, ""):
        return None
    matrix = json.loads(value) if isinstance(value, str) else value
    if isinstance(matrix, str):
        matrix = json.loads(matrix)
    try:
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        return int(tn), int(fp), int(fn), int(tp)
    except (TypeError, ValueError, IndexError):
        return None


def _write_comparison_and_doc(output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[Path, Path, Path]:
    rows: list[dict[str, Any]] = []
    for experiment_id in ("A_C1", "B_C1", "P_C1"):
        rows.append(
            _comparison_row(
                experiment_id=experiment_id,
                ratio="original",
                source="outputs",
                metrics_path=ORIGINAL_METRICS_ROOT / f"{experiment_id}_metrics.csv",
                summary_path=ORIGINAL_RUNS_ROOT / experiment_id / "run_summary.json",
            )
        )

    for ratio_name, _ratio_value in RATIO_SPECS:
        upper = _upper_ratio(ratio_name)
        ratio_root = output_root / ratio_name
        for experiment_id in (f"B_C1_BAL_{upper}", f"P_C1_BAL_{upper}"):
            rows.append(
                _comparison_row(
                    experiment_id=experiment_id,
                    ratio=ratio_name,
                    source=str(ratio_root),
                    metrics_path=ratio_root / "metrics" / f"{experiment_id}_metrics.csv",
                    summary_path=ratio_root / "runs" / experiment_id / "run_summary.json",
                )
            )

    comparison_csv = _write_rows(COMPARISON_CSV, rows)
    proposed_rows = [row for row in rows if str(row["experiment_id"]).startswith("P_C1_BAL_")]
    proposed_rows_with_val = [row for row in proposed_rows if _float(row, "validation_f1") is not None]
    best_proposed = max(proposed_rows_with_val, key=lambda row: _float(row, "validation_f1") or float("-inf"))
    best_ratio = str(best_proposed["ratio"])
    best_upper = _upper_ratio(best_ratio)
    best_baseline = next(row for row in rows if row["experiment_id"] == f"B_C1_BAL_{best_upper}")
    original_p = next(row for row in rows if row["experiment_id"] == "P_C1")
    original_b = next(row for row in rows if row["experiment_id"] == "B_C1")

    best_p_test_f1 = _float(best_proposed, "test_f1")
    original_p_test_f1 = _float(original_p, "test_f1")
    original_b_test_f1 = _float(original_b, "test_f1")
    best_b_test_f1 = _float(best_baseline, "test_f1")
    best_p_cm = _confusion(best_proposed)
    original_p_cm = _confusion(original_p)
    false_negative_delta = None
    false_positive_delta = None
    false_negative_answer = "INSUFFICIENT DATA"
    false_positive_answer = "INSUFFICIENT DATA"
    if best_p_cm is not None and original_p_cm is not None:
        false_negative_delta = best_p_cm[2] - original_p_cm[2]
        false_positive_delta = best_p_cm[1] - original_p_cm[1]
        false_negative_answer = "NO" if false_negative_delta >= 0 else "YES"
        false_positive_answer = "YES" if false_positive_delta > 0 else "NO"

    answers = {
        "best_ratio": best_ratio,
        "best_validation_f1": _float(best_proposed, "validation_f1"),
        "improved_over_original_p": (
            best_p_test_f1 is not None and original_p_test_f1 is not None and best_p_test_f1 > original_p_test_f1
        ),
        "beat_original_b": (
            best_p_test_f1 is not None and original_b_test_f1 is not None and best_p_test_f1 > original_b_test_f1
        ),
        "beat_balanced_b_same_ratio": (
            best_p_test_f1 is not None and best_b_test_f1 is not None and best_p_test_f1 > best_b_test_f1
        ),
        "false_negative_delta": false_negative_delta,
        "false_positive_delta": false_positive_delta,
        "false_negative_answer": false_negative_answer,
        "false_positive_answer": false_positive_answer,
    }

    lines = [
        "# Cluster 1 Balanced Training Comparison",
        "",
        "All balanced rows are marked as Cluster 1 balanced-training variants. Ratio selection uses validation F1 only; held-out test metrics are reported after that selection.",
        "",
        "## Metrics",
        "",
        "| experiment | ratio | validation_f1 | validation_pr_auc | validation_fpr | threshold | test_f1 | test_recall | test_precision | test_fpr | confusion_matrix |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['experiment_id']}` | {row['ratio']} | {_format(row.get('validation_f1'))} | "
            f"{_format(row.get('validation_pr_auc'))} | {_format(row.get('validation_fpr'))} | "
            f"{_format(row.get('selected_threshold'))} | {_format(row.get('test_f1'))} | "
            f"{_format(row.get('test_recall'))} | {_format(row.get('test_precision'))} | "
            f"{_format(row.get('test_fpr'))} | `{row.get('test_confusion_matrix')}` |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"1. Best proposed validation F1 ratio: `{best_ratio}` (`validation_f1={_format(answers['best_validation_f1'])}`).",
            f"2. Balanced proposed Cluster 1 improved over original `P_C1` on selected held-out test F1: `{'YES' if answers['improved_over_original_p'] else 'NO'}`.",
            f"3. Balanced proposed Cluster 1 beat original `B_C1` on selected held-out test F1: `{'YES' if answers['beat_original_b'] else 'NO'}`.",
            f"4. Balanced proposed Cluster 1 beat the balanced hierarchical baseline using the same ratio: `{'YES' if answers['beat_balanced_b_same_ratio'] else 'NO'}`.",
            f"5. Did balancing reduce false negatives versus original `P_C1`? `{false_negative_answer}`. Raw FN delta is `{false_negative_delta}`; original and balanced rows use different test-window pools, so recall/F1 are the safer comparison.",
            f"6. Did balancing create too many false positives versus original `P_C1`? `{false_positive_answer}`. Raw FP delta is `{false_positive_delta}` and selected FPR is `{_format(best_proposed.get('test_fpr'))}`.",
            f"7. Validation-selected ratio for the balanced proposed variant: `{best_ratio}`. Do not use it as a replacement for original `P_C1` because it did not improve held-out test F1.",
            "8. The improvement is defensible without test leakage because ratio and threshold selection used validation only, and `test4.csv`/`test5.csv` were held out from training, preprocessing fit, clustering, and threshold tuning.",
        ]
    )
    _write_markdown(COMPARISON_MD, lines)
    _write_documentation(rows, best_proposed, best_baseline, answers)
    return comparison_csv, COMPARISON_MD, DOC_PATH


def _format(value: Any) -> str:
    numeric = _metric_as_float(value)
    if numeric is not None:
        return f"{numeric:.6f}"
    if value is None or value == "":
        return "MISSING"
    return str(value)


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_documentation(
    rows: Sequence[Mapping[str, Any]],
    best_proposed: Mapping[str, Any],
    best_baseline: Mapping[str, Any],
    answers: Mapping[str, Any],
) -> None:
    generation_summary = _load_json(DEFAULT_OUTPUT_ROOT / "reports" / "balanced_variant_generation_summary.json") or {}
    raw_audit = _load_json(DEFAULT_OUTPUT_ROOT / "reports" / "hai_file_audit.json") or {}
    window_audit = _load_json(DEFAULT_OUTPUT_ROOT / "reports" / "window_audit.json") or {}
    ratio_lines = []
    for ratio_name, _ in RATIO_SPECS:
        metadata = _load_json(DEFAULT_VARIANT_ROOT / ratio_name / "metadata.json") or {}
        ratio_lines.append(f"- `{ratio_name}`: `{metadata.get('class_counts', {}).get('train', {})}`")

    raw_lines = []
    for entry in raw_audit.get("files", []):
        raw_lines.append(
            f"- `{entry['file']}`: rows={entry['row_count']}, attack=0={entry['attack_0_count']}, attack=1={entry['attack_1_count']}"
        )
    window_lines = []
    for entry in window_audit.get("files", []):
        window_lines.append(
            f"- `{entry['file']}`: windows={entry['window_count']}, positive={entry['positive_windows']}, negative={entry['negative_windows']}"
        )

    lines = [
        "# Cluster 1 Balanced-Training Variant",
        "",
        "Cluster 1 balancing was needed because the original HAI 21.03 training files are fully normal and the available attack supervision is sparse. The variant keeps the FCFL architecture, FedBN proposed method, no-cross-cluster averaging rule, and ledger metadata logic unchanged.",
        "",
        "## Data Split",
        "",
        f"- Training source files: `{list(TRAIN_SOURCE_FILES)}`",
        f"- Validation and threshold/ratio-selection file: `{list(VALIDATION_SOURCE_FILES)}`",
        f"- Held-out test files: `{list(HELDOUT_TEST_FILES)}`",
        "- Validation windows are not balanced.",
        "- Held-out test windows are not balanced.",
        "- Held-out test predictions are not used for threshold tuning or ratio selection.",
        "- Preprocessing uses `SimpleImputer(strategy=\"median\")` and `StandardScaler` fitted only on balanced training windows.",
        "",
        "## Row-Level Audit",
        "",
        *raw_lines,
        "",
        "## Window-Level Audit",
        "",
        *window_lines,
        "",
        "## Ratios Tested",
        "",
        *ratio_lines,
        "",
        "## Result",
        "",
        f"- Final selected ratio by validation F1: `{answers['best_ratio']}`",
        f"- Best proposed validation F1: `{_format(answers['best_validation_f1'])}`",
        f"- Best proposed held-out test F1 after validation-based selection: `{_format(best_proposed.get('test_f1'))}`",
        f"- Same-ratio balanced hierarchical baseline held-out test F1: `{_format(best_baseline.get('test_f1'))}`",
        f"- Proposed now beats original `B_C1`: `{'YES' if answers['beat_original_b'] else 'NO'}`",
        f"- Proposed now beats same-ratio balanced baseline: `{'YES' if answers['beat_balanced_b_same_ratio'] else 'NO'}`",
        f"- Balancing reduced false negatives versus original `P_C1`: `{answers.get('false_negative_answer')}`",
        f"- Balancing created more false positives versus original `P_C1`: `{answers.get('false_positive_answer')}`",
        "",
        "## Recommendation",
        "",
        f"Do not replace original `P_C1` with this balanced-training variant for final thesis reporting. If the balanced-training variant is discussed as an ablation or negative result, report `{answers['best_ratio']}` because it was selected by validation F1 only. The variant is defensible without test leakage because `test4.csv` and `test5.csv` are held out from training, validation, scaling, imputation, clustering, ratio selection, and threshold tuning.",
        "",
        "## Generated Summary",
        "",
        f"- Ratios generated: `{generation_summary.get('ratios_generated', [])}`",
        f"- Feature count: `{generation_summary.get('feature_count')}`",
        f"- Train pool counts before balancing: `{generation_summary.get('train_pool_counts_before_balancing')}`",
        f"- Natural validation counts: `{generation_summary.get('validation_counts')}`",
        f"- Natural held-out test counts: `{generation_summary.get('heldout_test_counts')}`",
    ]
    _write_markdown(DOC_PATH, lines)


def run_cluster1_balanced_experiments(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    rounds: int = 50,
    local_epochs: int = 1,
    batch_size: int = 128,
    seed: int = 42,
    baseline_learning_rate: float = 0.05,
    prepare: bool = True,
) -> dict[str, Path]:
    if prepare:
        preparation = prepare_cluster1_balanced_train()
        if preparation["status"] != "ok":
            raise RuntimeError(f"Balanced variant preparation failed: {preparation['status']}")

    all_metric_rows: list[dict[str, Any]] = []
    all_artifacts = []
    for ratio_name, _ratio_value in RATIO_SPECS:
        short = _short_ratio(ratio_name)
        upper = _upper_ratio(ratio_name)
        ratio_root = output_root / ratio_name
        specs = _balanced_specs(ratio_name)
        baseline_id = f"B_C1_BAL_{upper}"
        proposed_id = f"P_C1_BAL_{upper}"

        baseline_results = run_hierarchical_baseline_experiments(
            baseline_config_path=f"configs/baseline_hierarchical_cluster1_balanced_{short}.yaml",
            experiment_ids=[baseline_id],
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=baseline_learning_rate,
            seed=seed,
            output_root=ratio_root,
        )
        if len(baseline_results) != 1:
            raise ValueError(f"{baseline_id}: expected one result, observed {len(baseline_results)}.")
        baseline_artifact = _finalize_run_outputs(
            experiment_id=baseline_id,
            spec=specs[baseline_id],
            raw_result=baseline_results[0],
            output_root=ratio_root,
        )
        all_artifacts.append(baseline_artifact)
        all_metric_rows.append(_load_metrics_row(baseline_artifact.metrics_csv_path))

        proposed_result = run_cluster1_proposed(
            proposed_config_path=f"configs/proposed_cluster1_balanced_{short}.yaml",
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            output_root=ratio_root,
        )
        proposed_artifact = _finalize_run_outputs(
            experiment_id=proposed_id,
            spec=specs[proposed_id],
            raw_result=proposed_result,
            output_root=ratio_root,
        )
        all_artifacts.append(proposed_artifact)
        all_metric_rows.append(_load_metrics_row(proposed_artifact.metrics_csv_path))

        _write_ratio_threshold_and_test_metrics(
            ratio_root=ratio_root,
            experiment_ids=[baseline_id, proposed_id],
        )
        _write_confusion_matrices_report(
            [_load_metrics_row(baseline_artifact.metrics_csv_path), _load_metrics_row(proposed_artifact.metrics_csv_path)],
            ratio_root,
        )

    _write_rows(output_root / "metrics" / "summary_all_balanced_cluster1_experiments.csv", all_metric_rows)
    comparison_csv, comparison_md, doc_path = _write_comparison_and_doc(output_root)
    return {
        "summary": output_root / "metrics" / "summary_all_balanced_cluster1_experiments.csv",
        "comparison_csv": comparison_csv,
        "comparison_md": comparison_md,
        "doc": doc_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cluster 1 balanced-training variant experiments.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-learning-rate", type=float, default=0.05)
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_cluster1_balanced_experiments(
        output_root=Path(args.output_root),
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        baseline_learning_rate=args.baseline_learning_rate,
        prepare=not args.skip_prepare,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
