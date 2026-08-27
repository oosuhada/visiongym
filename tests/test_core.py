from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from visiongym.dataset import generate_single_split
from visiongym.evaluation import evaluate_records, normalize_answer, read_jsonl
from visiongym.experiments import build_analysis_from_reports, ingest_results
from visiongym.geometry import overlaps
from visiongym.inference import _iter_batches, _run_with_runner, resolve_device, run_inference, run_prompt_sweep


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


def test_inference_batches_preserve_order_and_validate_size() -> None:
    records = [{"sample_id": f"s{i}"} for i in range(5)]
    batches = _iter_batches(records, 2)
    assert [[row["sample_id"] for row in batch] for batch in batches] == [["s0", "s1"], ["s2", "s3"], ["s4"]]
    try:
        _iter_batches(records, 0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("batch_size=0 must be rejected")


def test_batched_runner_writes_ordered_predictions_and_runtime_metadata(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    records = []
    for index in range(5):
        image_path = image_dir / f"s{index}.png"
        Image.new("RGB", (16, 16), "white").save(image_path)
        records.append(
            {
                "sample_id": f"s{index}",
                "image": f"images/s{index}.png",
                "question": f"question-{index}",
            }
        )

    class FakeRunner:
        device = "cpu"
        calls: list[int] = []

        def predict_batch(self, images, questions, prompt_mode="direct", max_new_tokens=48):
            self.calls.append(len(images))
            return [f"answer-{question.split('-')[-1]}" for question in questions]

    output = tmp_path / "predictions.jsonl"
    runner = FakeRunner()
    predictions = _run_with_runner(
        runner=runner,
        records=records,
        dataset_path=tmp_path / "benchmark.jsonl",
        output_path=output,
        model_name="fake",
        prompt_mode="direct",
        adapter_path=None,
        batch_size=2,
        max_new_tokens=12,
    )
    assert runner.calls == [2, 2, 1]
    assert [record["sample_id"] for record in predictions] == [f"s{i}" for i in range(5)]
    assert [record["batch_size"] for record in predictions] == [2, 2, 2, 2, 1]
    metadata = json.loads((tmp_path / "predictions.meta.json").read_text(encoding="utf-8"))
    assert metadata["configured_batch_size"] == 2
    assert metadata["samples"] == 5
    assert metadata["max_new_tokens"] == 12
    assert metadata["throughput_samples_per_second"] > 0


def test_prompt_sweep_loads_runner_once(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "white").save(image_dir / "s1.png")
    dataset_path = tmp_path / "benchmark.jsonl"
    dataset_path.write_text(
        '{"sample_id":"s1","image":"images/s1.png","question":"Where?","answer":"left","task":"left_right","difficulty":1,"split":"test","ood_type":null}\n',
        encoding="utf-8",
    )

    class FakeSweepRunner:
        init_count = 0
        device = "cpu"

        def __init__(self, *args, **kwargs):
            type(self).init_count += 1

        def predict_batch(self, images, questions, prompt_mode="direct", max_new_tokens=48):
            return [prompt_mode for _ in questions]

    monkeypatch.setattr("visiongym.inference.TransformersVLM", FakeSweepRunner)
    outputs = run_prompt_sweep(
        dataset_path=dataset_path,
        output_dir=tmp_path / "outputs",
        model_name="fake",
        prompt_modes=["direct", "reasoning", "fewshot"],
        batch_size=2,
    )
    assert FakeSweepRunner.init_count == 1
    assert set(outputs) == {"direct", "reasoning", "fewshot"}
    for mode, path in outputs.items():
        records = read_jsonl(path)
        assert records[0]["prediction"] == mode


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
    (bundle / "base-reasoning.meta.json").write_text(
        json.dumps(
            {
                "configured_batch_size": 8,
                "throughput_samples_per_second": 12.5,
                "peak_vram_gb": 9.75,
                "wall_seconds": 0.16,
            }
        ),
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
    comparison = pd.read_csv(experiment_dir / result["comparison"])
    reasoning = comparison[comparison["prompt_mode"] == "reasoning"].iloc[0]
    assert reasoning["configured_batch_size"] == 8
    assert reasoning["throughput_samples_per_second"] == 12.5
    assert reasoning["peak_vram_gb"] == 9.75


def test_build_analysis_from_existing_reports(tmp_path: Path) -> None:
    report_dirs: list[Path] = []
    for name, accuracy, prompt, model in [
        ("base-direct", 0.5, "direct", "base"),
        ("lora-direct", 0.75, "direct", "base+LoRA"),
    ]:
        report_dir = tmp_path / name
        report_dir.mkdir()
        metrics = {
            "model": model,
            "prompt_mode": prompt,
            "overall_accuracy": accuracy,
            "id_accuracy": accuracy,
            "ood_accuracy": accuracy,
            "ood_gap": 0.0,
            "average_latency_seconds": 0.1,
            "by_task": {"multi_hop": {"accuracy": accuracy, "samples": 2}},
        }
        (report_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "image": "images/s1.png",
                    "question": "Where?",
                    "answer": "left",
                    "prediction": "left" if accuracy > 0.5 else "right",
                    "normalized_prediction": "left" if accuracy > 0.5 else "right",
                    "correct": accuracy > 0.5,
                    "task": "multi_hop",
                    "difficulty": 4,
                    "split": "test",
                    "ood_type": None,
                }
            ]
        ).to_csv(report_dir / "predictions_scored.csv", index=False)
        (report_dir / "failures.jsonl").write_text("", encoding="utf-8")
        report_dirs.append(report_dir)

    output = tmp_path / "measured"
    manifest = build_analysis_from_reports(report_dirs, output)
    comparison = pd.read_csv(output / "comparison.csv")
    pairwise = pd.read_csv(output / "pairwise_summary.csv")
    assert manifest["source"] == "checked-in evaluated reports"
    assert comparison["run"].tolist() == ["base-direct", "lora-direct"]
    assert pairwise.loc[0, "improved"] == 1
    assert pairwise.loc[0, "regressed"] == 0
