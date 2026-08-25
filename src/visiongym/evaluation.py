from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_answer(text: Any) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"])
    except json.JSONDecodeError:
        pass
    json_match = re.search(r'"answer"\s*:\s*"?([^"}\n]+)', raw, flags=re.IGNORECASE)
    if json_match:
        return json_match.group(1).strip()
    answer_match = re.search(r"(?:final\s+answer|answer)\s*[:=-]\s*(.+)", raw, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().splitlines()[0]
    return raw.splitlines()[-1].strip()


def normalize_answer(text: Any) -> str:
    answer = extract_answer(text).lower().strip()
    answer = ARTICLES.sub(" ", answer)
    answer = NON_ALNUM.sub(" ", answer)
    return " ".join(answer.split())


def classify_error(sample: dict[str, Any], prediction: str) -> str:
    normalized_prediction = normalize_answer(prediction)
    normalized_answer = normalize_answer(sample["answer"])
    if not normalized_prediction:
        return "answer_format_failure"
    if sample.get("ood_type"):
        return "ood_failure"
    task = sample.get("task", "")
    if task == "counting":
        return "counting_error"
    if task in {"left_right", "above_below", "relative_ordering"}:
        return "spatial_inversion"
    if task == "multi_hop":
        return "relation_chain_failure"
    if task in {"nearest_farthest", "center_proximity"}:
        return "distance_reasoning_error"
    if task in {"overlap", "inside_outside"}:
        return "relation_error"
    if normalized_answer and any(token in normalized_prediction for token in ("circle", "triangle", "rectangle", "star", "hexagon", "diamond")):
        return "object_confusion"
    return "other"


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def evaluate_records(
    dataset_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    model_name: str | None = None,
    prompt_mode: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = {record["sample_id"]: record for record in prediction_records}
    rows: list[dict[str, Any]] = []
    for sample in dataset_records:
        prediction_record = predictions.get(sample["sample_id"], {})
        prediction = prediction_record.get("prediction", "")
        normalized_ground_truth = normalize_answer(sample["answer"])
        normalized_prediction = normalize_answer(prediction)
        correct = normalized_ground_truth == normalized_prediction
        row = {
            **sample,
            "prediction": prediction,
            "normalized_ground_truth": normalized_ground_truth,
            "normalized_prediction": normalized_prediction,
            "correct": correct,
            "latency_seconds": prediction_record.get("latency_seconds"),
            "error_type": None if correct else classify_error(sample, prediction),
        }
        rows.append(row)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ood: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
        by_difficulty[str(row["difficulty"])].append(row)
        by_split[row["split"]].append(row)
        if row.get("ood_type"):
            by_ood[row["ood_type"]].append(row)

    id_rows = [row for row in rows if not row.get("ood_type")]
    ood_rows = [row for row in rows if row.get("ood_type")]
    id_accuracy = _accuracy(id_rows)
    ood_accuracy = _accuracy(ood_rows)
    latency_values = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    error_counts = Counter(row["error_type"] for row in rows if row["error_type"])
    missing_count = sum(row["sample_id"] not in predictions for row in rows)

    metrics: dict[str, Any] = {
        "model": model_name,
        "prompt_mode": prompt_mode,
        "samples": len(rows),
        "predictions_missing": missing_count,
        "overall_accuracy": _accuracy(rows),
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "ood_gap": (id_accuracy - ood_accuracy) if id_accuracy is not None and ood_accuracy is not None else None,
        "invalid_output_rate": sum(not row["normalized_prediction"] for row in rows) / len(rows) if rows else None,
        "average_latency_seconds": sum(latency_values) / len(latency_values) if latency_values else None,
        "by_task": {key: {"accuracy": _accuracy(value), "samples": len(value)} for key, value in sorted(by_task.items())},
        "by_difficulty": {key: {"accuracy": _accuracy(value), "samples": len(value)} for key, value in sorted(by_difficulty.items())},
        "by_split": {key: {"accuracy": _accuracy(value), "samples": len(value)} for key, value in sorted(by_split.items())},
        "by_ood_type": {key: {"accuracy": _accuracy(value), "samples": len(value)} for key, value in sorted(by_ood.items())},
        "error_distribution": dict(error_counts.most_common()),
    }
    return metrics, rows


def evaluate_files(
    dataset_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    model_name: str | None = None,
    prompt_mode: str | None = None,
) -> dict[str, Any]:
    dataset_records = read_jsonl(dataset_path)
    prediction_records = read_jsonl(predictions_path)
    metrics, rows = evaluate_records(dataset_records, prediction_records, model_name=model_name, prompt_mode=prompt_mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(output / "predictions_scored.csv", index=False)
    failures = [row for row in rows if not row["correct"]]
    failures.sort(key=lambda row: (row.get("task", ""), row.get("sample_id", "")))
    write_jsonl(output / "failures.jsonl", failures[:100])
    return metrics


def compare_metric_files(metric_paths: list[str | Path], output_path: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in metric_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            metric = json.load(handle)
        row = {
            "source": str(path),
            "model": metric.get("model"),
            "prompt_mode": metric.get("prompt_mode"),
            "overall_accuracy": metric.get("overall_accuracy"),
            "id_accuracy": metric.get("id_accuracy"),
            "ood_accuracy": metric.get("ood_accuracy"),
            "ood_gap": metric.get("ood_gap"),
            "average_latency_seconds": metric.get("average_latency_seconds"),
        }
        for task, values in metric.get("by_task", {}).items():
            row[f"task_{task}"] = values.get("accuracy")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    return frame

