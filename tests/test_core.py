from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from visiongym.dataset import generate_single_split
from visiongym.evaluation import evaluate_records, normalize_answer, read_jsonl
from visiongym.geometry import overlaps
from visiongym.inference import resolve_device, run_inference


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


def test_inference_rejects_invalid_batch_size(tmp_path: Path) -> None:
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")
    try:
        run_inference(dataset, tmp_path / "predictions.jsonl", "unused", batch_size=0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("batch_size=0 must be rejected")
