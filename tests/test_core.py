from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from visiongym.dataset import generate_single_split
from visiongym.evaluation import evaluate_records, normalize_answer, read_jsonl
from visiongym.experiments import ingest_results
from visiongym.geometry import overlaps
from visiongym.inference import resolve_device


def test_answer_normalization() -> None:
    assert normalize_answer("The red circle.") == "red circle"
    assert normalize_answer('{"answer": "Blue Triangle"}') == "blue triangle"
    assert normalize_answer("Answer: 3") == "3"


def test_generation_is_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_single_split(3, "test", first, seed=123, questions_per_scene=3)
    generate_single_split(3, "test", second, seed=123, questions_per_scene=3)
    assert read_jsonl(first / "test.jsonl") == read_jsonl(second / "test.jsonl")
    assert read_jsonl(first / "scenes" / "test.jsonl") == read_jsonl(second / "scenes" / "test.jsonl")


def test_oracle_evaluation_is_perfect(tmp_path: Path) -> None:
    generate_single_split(2, "test", tmp_path, seed=55, questions_per_scene=4)
    dataset = read_jsonl(tmp_path / "test.jsonl")
    predictions = [{"sample_id": sample["sample_id"], "prediction": sample["answer"]} for sample in dataset]
    metrics, rows = evaluate_records(dataset, predictions, model_name="oracle")
    assert metrics["overall_accuracy"] == 1.0
    assert all(row["correct"] for row in rows)


def test_occlusion_split_contains_overlap(tmp_path: Path) -> None:
    generate_single_split(3, "ood_occlusion", tmp_path, seed=17, questions_per_scene=2)
    scenes = read_jsonl(tmp_path / "scenes" / "ood_occlusion.jsonl")
    for scene in scenes:
        objects = scene["objects"]
        first = objects[0]
        last = objects[-1]
        ax1, ay1, ax2, ay2 = first["bbox"]
        bx1, by1, bx2, by2 = last["bbox"]
        assert ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def test_cpu_device_rejects_bitsandbytes_4bit() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    assert resolve_device(fake_torch, requested="cpu", load_in_4bit=False) == "cpu"
    assert resolve_device(fake_torch, requested="auto", load_in_4bit=False) == "cpu"
    try:
        resolve_device(fake_torch, requested="cpu", load_in_4bit=True)
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("4-bit CPU loading must be rejected")


def test_ood_error_keeps_reasoning_category() -> None:
    dataset = [
        {
            "sample_id": "ood_1",
            "image": "images/ood_1.png",
            "question": "How many red objects are there?",
            "answer": "2",
            "task": "counting",
            "difficulty": 2,
            "split": "ood_count",
            "ood_type": "count",
        }
    ]
    metrics, rows = evaluate_records(dataset, [{"sample_id": "ood_1", "prediction": "3"}])
    assert rows[0]["error_type"] == "counting_error"
    assert rows[0]["ood_failure"] is True
    assert metrics["ood_error_distribution"] == {"counting_error": 1}


def test_ingest_results_builds_comparison_and_failure_gallery(tmp_path: Path) -> None:
    dataset_path = tmp_path / "benchmark.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"sample_id":"s1","image":"images/s1.png","question":"Count","answer":"2","task":"counting","difficulty":1,"split":"test","ood_type":null}',
                '{"sample_id":"s2","image":"images/s2.png","question":"Count","answer":"1","task":"counting","difficulty":1,"split":"ood_count","ood_type":"count"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "base-direct.jsonl").write_text(
        "\n".join(
            [
                '{"sample_id":"s1","prediction":"2","model":"demo","prompt_mode":"direct","latency_seconds":0.2}',
                '{"sample_id":"s2","prediction":"2","model":"demo","prompt_mode":"direct","latency_seconds":0.2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "base-reasoning.jsonl").write_text(
        "\n".join(
            [
                '{"sample_id":"s1","prediction":"2","model":"demo","prompt_mode":"reasoning","latency_seconds":0.3}',
                '{"sample_id":"s2","prediction":"1","model":"demo","prompt_mode":"reasoning","latency_seconds":0.3}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = ingest_results(bundle, dataset_path, tmp_path / "experiment", strict=True)
    assert len(result["runs"]) == 2
    experiment_dir = tmp_path / "experiment"
    assert (experiment_dir / result["comparison"]).exists()
    assert (experiment_dir / result["failure_gallery"]).exists()
    assert (tmp_path / "experiment" / "runs" / "base-direct" / "task_domain_accuracy.csv").exists()
    assert result["pairwise"] is not None
    pairwise = pd.read_csv(experiment_dir / result["pairwise"]["summary"])
    assert pairwise.loc[0, "improved"] == 1
    assert pairwise.loc[0, "regressed"] == 0
    assert len(result["dataset_sha256"]) == 64
    assert len(result["runs"][0]["prediction_sha256"]) == 64
