from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visiongym.question_generator import generate_questions
from visiongym.renderer import render_scene
from visiongym.scene_generator import SceneGenerator


OOD_BY_LABEL = {
    "ID": None,
    "OOD · unseen shapes": "shape",
    "OOD · new palette": "palette",
    "OOD · more objects": "count",
    "OOD · background shift": "background",
    "OOD · occlusion": "occlusion",
}

BASE_LABEL = os.getenv("VISIONGYM_BASE_LABEL", "Base few-shot")
FINETUNED_LABEL = os.getenv("VISIONGYM_FINETUNED_LABEL", "LoRA direct")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _prediction_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    return {record["sample_id"]: str(record.get("prediction", "")) for record in _read_jsonl(path)}


def _resolve_optional_path(env_name: str) -> Path | None:
    value = os.getenv(env_name)
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_experiment_data() -> tuple[Path | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment_dir = _resolve_optional_path("VISIONGYM_EXPERIMENT_DIR") or REPO_ROOT / "reports" / "measured"
    if experiment_dir is None or not experiment_dir.is_dir():
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    comparison = _read_csv(experiment_dir / "comparison.csv")
    pairwise = _read_csv(experiment_dir / "pairwise_summary.csv")
    gallery = _read_csv(experiment_dir / "failure_gallery.csv")
    return experiment_dir, comparison, pairwise, gallery


def generate_problem(condition: str, seed: int) -> tuple[str, str, str, str, str]:
    ood_type = OOD_BY_LABEL[condition]
    object_count = random.Random(seed).randint(6, 8) if ood_type == "count" else random.Random(seed).randint(3, 5)
    split = f"demo_{ood_type or 'id'}"
    generator = SceneGenerator(seed=seed)
    scene = generator.generate(scene_index=0, split=split, object_count=object_count, ood_type=ood_type)
    output_dir = REPO_ROOT / "tmp" / "demo"
    image_path = output_dir / f"scene_{condition.replace(' ', '_').replace('·', '')}_{seed}.png"
    render_scene(scene, image_path, seed=seed)
    questions = generate_questions(scene, image_path.as_posix(), count=5, seed=seed)
    if not questions:
        return str(image_path), "No valid question generated.", "", "", ""
    sample = questions[seed % len(questions)]
    return str(image_path), sample.question, sample.answer, sample.task, str(sample.difficulty)


def _build_viewer_data() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    showcase = REPO_ROOT / "data" / "showcase"
    dataset_path = _resolve_optional_path("VISIONGYM_DATASET") or showcase / "benchmark.jsonl"
    base_path = _resolve_optional_path("VISIONGYM_BASE_PREDICTIONS") or showcase / "base-fewshot.jsonl"
    finetuned_path = _resolve_optional_path("VISIONGYM_FINETUNED_PREDICTIONS") or showcase / "lora-direct.jsonl"
    return _read_jsonl(dataset_path), _prediction_map(base_path), _prediction_map(finetuned_path)


VIEWER_RECORDS, BASE_PREDICTIONS, FINETUNED_PREDICTIONS = _build_viewer_data()
VIEWER_BY_ID = {record["sample_id"]: record for record in VIEWER_RECORDS}
EXPERIMENT_DIR, EXPERIMENT_COMPARISON, EXPERIMENT_PAIRWISE, EXPERIMENT_FAILURES = _load_experiment_data()
EXPERIMENT_FAILURE_BY_KEY: dict[str, dict[str, Any]] = {}
if not EXPERIMENT_FAILURES.empty:
    for row in EXPERIMENT_FAILURES.to_dict(orient="records"):
        key = f"{row.get('run', 'run')} · {row.get('sample_id', 'sample')}"
        EXPERIMENT_FAILURE_BY_KEY[key] = row


def show_result(sample_id: str) -> tuple[str | None, str, str, str, str, str, str]:
    sample = VIEWER_BY_ID.get(sample_id)
    if sample is None:
        return None, "", "", "", "", "", ""
    dataset_path = _resolve_optional_path("VISIONGYM_DATASET") or REPO_ROOT / "data" / "showcase" / "benchmark.jsonl"
    image_path = dataset_path.parent / sample["image"]
    ground_truth = str(sample["answer"])
    base = BASE_PREDICTIONS.get(sample_id, "No base prediction loaded")
    finetuned = FINETUNED_PREDICTIONS.get(sample_id, "No fine-tuned prediction loaded")
    status = (
        f"{BASE_LABEL}: {'correct' if base.strip().lower() == ground_truth.strip().lower() else 'incorrect'} · "
        f"{FINETUNED_LABEL}: {'correct' if finetuned.strip().lower() == ground_truth.strip().lower() else 'incorrect'}"
    )
    metadata = f"task={sample['task']} · difficulty={sample['difficulty']} · split={sample['split']} · ood={sample.get('ood_type') or 'ID'}"
    return str(image_path), sample["question"], ground_truth, base, finetuned, status, metadata


def show_experiment_failure(key: str) -> tuple[str | None, str, str, str, str]:
    row = EXPERIMENT_FAILURE_BY_KEY.get(key)
    if row is None:
        return None, "", "", "", ""
    dataset_path = _resolve_optional_path("VISIONGYM_DATASET") or REPO_ROOT / "data" / "sample" / "benchmark.jsonl"
    image_value = row.get("image")
    image_path = dataset_path.parent / str(image_value) if image_value else None
    if image_path is not None and not image_path.exists():
        image_path = None
    metadata = (
        f"run={row.get('run')} · task={row.get('task')} · difficulty={row.get('difficulty')} · "
        f"split={row.get('split')} · ood={row.get('ood_type')} · error={row.get('error_type')}"
    )
    return (
        str(image_path) if image_path is not None else None,
        str(row.get("question") or ""),
        str(row.get("ground_truth") or ""),
        str(row.get("prediction") or ""),
        metadata,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VLM Reasoning Lab") as demo:
        gr.Markdown(
            "# VLM Reasoning Lab\n"
            "Synthetic VLM reasoning benchmark, fine-tuning, and measured failure analysis."
        )

        with gr.Tab("Generate Problem"):
            with gr.Row():
                condition = gr.Dropdown(list(OOD_BY_LABEL), value="ID", label="Distribution")
                seed = gr.Slider(0, 10000, value=42, step=1, label="Seed")
                generate_button = gr.Button("Generate Problem", variant="primary")
            generated_image = gr.Image(label="Synthetic scene", type="filepath")
            generated_question = gr.Textbox(label="Question")
            with gr.Row():
                generated_answer = gr.Textbox(label="Ground truth")
                generated_task = gr.Textbox(label="Task")
                generated_difficulty = gr.Textbox(label="Difficulty")
            generate_button.click(
                generate_problem,
                inputs=[condition, seed],
                outputs=[generated_image, generated_question, generated_answer, generated_task, generated_difficulty],
            )

        with gr.Tab("Result Viewer"):
            sample_ids = list(VIEWER_BY_ID)
            sample_dropdown = gr.Dropdown(sample_ids, value=sample_ids[0] if sample_ids else None, label="Benchmark sample")
            result_image = gr.Image(label="Image", type="filepath")
            result_question = gr.Textbox(label="Question")
            with gr.Row():
                result_ground_truth = gr.Textbox(label="Ground truth")
                result_base = gr.Textbox(label=BASE_LABEL)
                result_finetuned = gr.Textbox(label=FINETUNED_LABEL)
            result_status = gr.Textbox(label="Correct / Incorrect")
            result_metadata = gr.Textbox(label="Task metadata")
            sample_dropdown.change(
                show_result,
                inputs=sample_dropdown,
                outputs=[
                    result_image,
                    result_question,
                    result_ground_truth,
                    result_base,
                    result_finetuned,
                    result_status,
                    result_metadata,
                ],
            )
            if sample_ids:
                demo.load(
                    show_result,
                    inputs=sample_dropdown,
                    outputs=[
                        result_image,
                        result_question,
                        result_ground_truth,
                        result_base,
                        result_finetuned,
                        result_status,
                        result_metadata,
                    ],
                )

        with gr.Tab("Experiment Analysis"):
            experiment_status = (
                f"Loaded experiment bundle: `{EXPERIMENT_DIR}`"
                if EXPERIMENT_DIR is not None
                else "No experiment bundle loaded. Set `VISIONGYM_EXPERIMENT_DIR` to an ingested result directory."
            )
            gr.Markdown(experiment_status)
            gr.Markdown("### Run comparison")
            gr.Dataframe(value=EXPERIMENT_COMPARISON, interactive=False, wrap=True)
            gr.Markdown("### Paired improvement / regression")
            gr.Dataframe(value=EXPERIMENT_PAIRWISE, interactive=False, wrap=True)
            gr.Markdown("### Representative failures")
            failure_keys = list(EXPERIMENT_FAILURE_BY_KEY)
            failure_dropdown = gr.Dropdown(
                failure_keys,
                value=failure_keys[0] if failure_keys else None,
                label="Failure sample",
            )
            failure_image = gr.Image(label="Image", type="filepath")
            failure_question = gr.Textbox(label="Question")
            with gr.Row():
                failure_ground_truth = gr.Textbox(label="Ground truth")
                failure_prediction = gr.Textbox(label="Prediction")
            failure_metadata = gr.Textbox(label="Failure metadata")
            failure_dropdown.change(
                show_experiment_failure,
                inputs=failure_dropdown,
                outputs=[
                    failure_image,
                    failure_question,
                    failure_ground_truth,
                    failure_prediction,
                    failure_metadata,
                ],
            )
            if failure_keys:
                demo.load(
                    show_experiment_failure,
                    inputs=failure_dropdown,
                    outputs=[
                        failure_image,
                        failure_question,
                        failure_ground_truth,
                        failure_prediction,
                        failure_metadata,
                    ],
                )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Reasoning Lab Gradio demo")
    parser.add_argument("--host", default=os.getenv("VISIONGYM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VISIONGYM_PORT", "7860")))
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
