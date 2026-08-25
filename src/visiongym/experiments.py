from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from visiongym.evaluation import compare_metric_files, evaluate_files, read_jsonl
from visiongym.reporting import create_report


def _prediction_file(path: Path) -> bool:
    if path.name in {"failures.jsonl", "benchmark.jsonl", "train_sft.jsonl"}:
        return False
    try:
        records = read_jsonl(path)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    if not records:
        return False
    first = records[0]
    return "sample_id" in first and "prediction" in first


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(output: Path, path: Path) -> str:
    return path.resolve().relative_to(output.resolve()).as_posix()


def _discover_prediction_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if _prediction_file(path))


def _validate_predictions(dataset_ids: set[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_ids = [str(record.get("sample_id", "")) for record in records]
    nonempty_ids = [sample_id for sample_id in prediction_ids if sample_id]
    unique_ids = set(nonempty_ids)
    duplicate_count = len(nonempty_ids) - len(unique_ids)
    missing_ids = sorted(dataset_ids - unique_ids)
    unexpected_ids = sorted(unique_ids - dataset_ids)
    return {
        "prediction_records": len(records),
        "unique_prediction_ids": len(unique_ids),
        "duplicate_ids": duplicate_count,
        "missing_ids": len(missing_ids),
        "unexpected_ids": len(unexpected_ids),
        "missing_id_examples": missing_ids[:10],
        "unexpected_id_examples": unexpected_ids[:10],
    }


def _run_name(path: Path, used: set[str]) -> str:
    base = path.stem.replace(" ", "-").replace("_", "-")
    name = base
    suffix = 2
    while name in used:
        name = f"{base}-{suffix}"
        suffix += 1
    used.add(name)
    return name


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _write_experiment_summary(output_dir: Path, runs: list[dict[str, Any]], comparison: pd.DataFrame) -> Path:
    summary = output_dir / "summary.md"
    lines = [
        "# VisionGym Experiment Bundle",
        "",
        f"- Imported at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Runs: {len(runs)}",
        "",
        "## Validation",
        "",
        "| Run | Records | Missing | Duplicate IDs | Unexpected IDs |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        validation = run["validation"]
        lines.append(
            f"| {run['name']} | {validation['prediction_records']} | {validation['missing_ids']} | "
            f"{validation['duplicate_ids']} | {validation['unexpected_ids']} |"
        )

    if not comparison.empty:
        lines.extend(
            [
                "",
                "## Accuracy comparison",
                "",
                "| Run | Model | Prompt | Overall | ID | OOD | OOD gap | Multi-hop | Avg latency | Samples/sec | Peak VRAM |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run, (_, row) in zip(runs, comparison.iterrows()):
            latency = row.get("average_latency_seconds")
            latency_text = "n/a" if latency is None or pd.isna(latency) else f"{float(latency):.3f}s"
            throughput = row.get("throughput_samples_per_second")
            throughput_text = "n/a" if throughput is None or pd.isna(throughput) else f"{float(throughput):.2f}"
            peak_vram = row.get("peak_vram_gb")
            peak_vram_text = "n/a" if peak_vram is None or pd.isna(peak_vram) else f"{float(peak_vram):.2f} GB"
            lines.append(
                f"| {run['name']} | {row.get('model') or 'unknown'} | {row.get('prompt_mode') or 'unknown'} | "
                f"{_format_pct(row.get('overall_accuracy'))} | {_format_pct(row.get('id_accuracy'))} | "
                f"{_format_pct(row.get('ood_accuracy'))} | {_format_pct(row.get('ood_gap'))} | "
                f"{_format_pct(row.get('task_multi_hop'))} | {latency_text} | {throughput_text} | {peak_vram_text} |"
            )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _find_baseline_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    for run in runs:
        if run.get("prompt_mode") == "direct" and not run.get("adapter_path"):
            return run
    for run in runs:
        if run.get("prompt_mode") == "direct":
            return run
    return runs[0]


def _paired_analysis(output: Path, runs: list[dict[str, Any]]) -> dict[str, str] | None:
    baseline = _find_baseline_run(runs)
    if baseline is None or len(runs) < 2:
        return None

    baseline_frame = pd.read_csv(output / baseline["scored_predictions"])
    summary_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    for candidate in runs:
        if candidate["name"] == baseline["name"]:
            continue
        candidate_frame = pd.read_csv(output / candidate["scored_predictions"])
        candidate_columns = candidate_frame[
            ["sample_id", "prediction", "correct", "normalized_prediction"]
        ].rename(
            columns={
                "prediction": "candidate_prediction",
                "correct": "candidate_correct",
                "normalized_prediction": "candidate_normalized_prediction",
            }
        )
        baseline_columns = baseline_frame[
            [
                "sample_id",
                "question",
                "answer",
                "prediction",
                "correct",
                "task",
                "difficulty",
                "split",
                "ood_type",
                "image",
            ]
        ].rename(columns={"prediction": "baseline_prediction", "correct": "baseline_correct"})
        paired = baseline_columns.merge(candidate_columns, on="sample_id", how="inner")
        paired["baseline_correct"] = paired["baseline_correct"].astype(bool)
        paired["candidate_correct"] = paired["candidate_correct"].astype(bool)
        paired["transition"] = "both_wrong"
        paired.loc[paired["baseline_correct"] & paired["candidate_correct"], "transition"] = "both_correct"
        paired.loc[paired["baseline_correct"] & ~paired["candidate_correct"], "transition"] = "regressed"
        paired.loc[~paired["baseline_correct"] & paired["candidate_correct"], "transition"] = "improved"

        transition_counts = paired["transition"].value_counts().to_dict()
        baseline_accuracy = float(paired["baseline_correct"].mean()) if len(paired) else 0.0
        candidate_accuracy = float(paired["candidate_correct"].mean()) if len(paired) else 0.0
        summary_rows.append(
            {
                "baseline": baseline["name"],
                "candidate": candidate["name"],
                "paired_samples": len(paired),
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": candidate_accuracy,
                "accuracy_delta": candidate_accuracy - baseline_accuracy,
                "improved": int(transition_counts.get("improved", 0)),
                "regressed": int(transition_counts.get("regressed", 0)),
                "both_correct": int(transition_counts.get("both_correct", 0)),
                "both_wrong": int(transition_counts.get("both_wrong", 0)),
                "net_fixed_samples": int(transition_counts.get("improved", 0) - transition_counts.get("regressed", 0)),
            }
        )

        for task, group in paired.groupby("task", dropna=False):
            base_task_accuracy = float(group["baseline_correct"].mean())
            candidate_task_accuracy = float(group["candidate_correct"].mean())
            task_rows.append(
                {
                    "baseline": baseline["name"],
                    "candidate": candidate["name"],
                    "task": task,
                    "samples": len(group),
                    "baseline_accuracy": base_task_accuracy,
                    "candidate_accuracy": candidate_task_accuracy,
                    "accuracy_delta": candidate_task_accuracy - base_task_accuracy,
                    "improved": int((group["transition"] == "improved").sum()),
                    "regressed": int((group["transition"] == "regressed").sum()),
                }
            )

        flips = paired[paired["transition"].isin(["improved", "regressed"])].copy()
        flips = flips.sort_values(["transition", "task", "sample_id"])
        for _, row in flips.head(48).iterrows():
            example_rows.append(
                {
                    "baseline": baseline["name"],
                    "candidate": candidate["name"],
                    "transition": row["transition"],
                    "sample_id": row["sample_id"],
                    "image": row.get("image"),
                    "question": row.get("question"),
                    "ground_truth": row.get("answer"),
                    "baseline_prediction": row.get("baseline_prediction"),
                    "candidate_prediction": row.get("candidate_prediction"),
                    "task": row.get("task"),
                    "difficulty": row.get("difficulty"),
                    "split": row.get("split"),
                    "ood_type": row.get("ood_type") if pd.notna(row.get("ood_type")) else "ID",
                }
            )

    summary_path = output / "pairwise_summary.csv"
    task_path = output / "pairwise_task_delta.csv"
    examples_path = output / "pairwise_examples.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(task_rows).to_csv(task_path, index=False)
    pd.DataFrame(example_rows).to_csv(examples_path, index=False)
    return {
        "baseline": baseline["name"],
        "summary": _artifact_path(output, summary_path),
        "task_delta": _artifact_path(output, task_path),
        "examples": _artifact_path(output, examples_path),
    }


def ingest_results(
    bundle_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    strict: bool = False,
) -> dict[str, Any]:
    """Ingest a Colab result directory/ZIP and rebuild validated reports locally."""
    bundle = Path(bundle_path).expanduser().resolve()
    dataset = Path(dataset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset_records = read_jsonl(dataset)
    dataset_ids = {str(record["sample_id"]) for record in dataset_records}
    if len(dataset_ids) != len(dataset_records):
        raise ValueError("Dataset contains duplicate sample_id values; refusing to ingest results.")

    with tempfile.TemporaryDirectory(prefix="visiongym-ingest-") as temp_dir:
        if bundle.is_file() and bundle.suffix.lower() == ".zip":
            with zipfile.ZipFile(bundle) as archive:
                archive.extractall(temp_dir)
            root = Path(temp_dir)
        elif bundle.is_dir():
            root = bundle
        else:
            raise ValueError("--bundle must point to a directory or .zip file")

        prediction_files = _discover_prediction_files(root)
        if not prediction_files:
            raise ValueError(f"No prediction JSONL files found under {bundle}")

        runs: list[dict[str, Any]] = []
        metric_paths: list[Path] = []
        gallery_rows: list[dict[str, Any]] = []
        used_names: set[str] = set()
        predictions_dir = output / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for source_path in prediction_files:
            prediction_records = read_jsonl(source_path)
            validation = _validate_predictions(dataset_ids, prediction_records)
            if strict and any(validation[key] for key in ("duplicate_ids", "missing_ids", "unexpected_ids")):
                raise ValueError(f"Prediction validation failed for {source_path}: {validation}")

            name = _run_name(source_path, used_names)
            copied_predictions = predictions_dir / f"{name}.jsonl"
            shutil.copy2(source_path, copied_predictions)
            first = prediction_records[0] if prediction_records else {}
            model_name = first.get("model")
            prompt_mode = first.get("prompt_mode")
            adapter_path = first.get("adapter_path")

            run_output = output / "runs" / name
            metrics = evaluate_files(
                dataset_path=dataset,
                predictions_path=copied_predictions,
                output_dir=run_output,
                model_name=str(model_name) if model_name else None,
                prompt_mode=str(prompt_mode) if prompt_mode else None,
            )
            create_report(run_output / "metrics.json", run_output)
            metric_paths.append(run_output / "metrics.json")

            runtime_metadata: dict[str, Any] = {}
            source_runtime = source_path.with_suffix(".meta.json")
            runtime_artifact: str | None = None
            if source_runtime.exists():
                runtime_metadata = json.loads(source_runtime.read_text(encoding="utf-8"))
                runtime_copy = run_output / "runtime.json"
                shutil.copy2(source_runtime, runtime_copy)
                runtime_artifact = _artifact_path(output, runtime_copy)

            failures = read_jsonl(run_output / "failures.jsonl")
            for failure in failures[:24]:
                gallery_rows.append(
                    {
                        "run": name,
                        "sample_id": failure.get("sample_id"),
                        "image": failure.get("image"),
                        "question": failure.get("question"),
                        "ground_truth": failure.get("answer"),
                        "prediction": failure.get("prediction"),
                        "task": failure.get("task"),
                        "difficulty": failure.get("difficulty"),
                        "split": failure.get("split"),
                        "ood_type": failure.get("ood_type") or "ID",
                        "error_type": failure.get("error_type"),
                    }
                )

            runs.append(
                {
                    "name": name,
                    "source": source_path.relative_to(root).as_posix(),
                    "prediction_sha256": _sha256(copied_predictions),
                    "predictions": _artifact_path(output, copied_predictions),
                    "metrics": _artifact_path(output, run_output / "metrics.json"),
                    "scored_predictions": _artifact_path(output, run_output / "predictions_scored.csv"),
                    "model": model_name,
                    "prompt_mode": prompt_mode,
                    "adapter_path": adapter_path,
                    "runtime": runtime_artifact,
                    "runtime_metadata": runtime_metadata,
                    "validation": validation,
                    "overall_accuracy": metrics.get("overall_accuracy"),
                    "id_accuracy": metrics.get("id_accuracy"),
                    "ood_accuracy": metrics.get("ood_accuracy"),
                }
            )

    comparison_path = output / "comparison.csv"
    comparison = compare_metric_files(metric_paths, comparison_path)
    for index, run in enumerate(runs):
        runtime = run.get("runtime_metadata") or {}
        comparison.loc[index, "configured_batch_size"] = runtime.get("configured_batch_size")
        comparison.loc[index, "throughput_samples_per_second"] = runtime.get("throughput_samples_per_second")
        comparison.loc[index, "peak_vram_gb"] = runtime.get("peak_vram_gb")
        comparison.loc[index, "wall_seconds"] = runtime.get("wall_seconds")
    comparison.to_csv(comparison_path, index=False)
    gallery_path = output / "failure_gallery.csv"
    pd.DataFrame(gallery_rows).to_csv(gallery_path, index=False)
    pairwise = _paired_analysis(output, runs)
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "dataset_samples": len(dataset_records),
        "bundle": str(bundle),
        "runs": runs,
        "comparison": _artifact_path(output, comparison_path),
        "failure_gallery": _artifact_path(output, gallery_path),
        "pairwise": pairwise,
    }
    manifest_path = output / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = _write_experiment_summary(output, runs, comparison)
    manifest["summary"] = _artifact_path(output, summary_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
