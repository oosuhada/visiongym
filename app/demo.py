from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import gradio as gr


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
        f"Base: {'correct' if base.strip().lower() == ground_truth.strip().lower() else 'incorrect'} · "
        f"Fine-tuned: {'correct' if finetuned.strip().lower() == ground_truth.strip().lower() else 'incorrect'}"
    )
    metadata = f"task={sample['task']} · difficulty={sample['difficulty']} · split={sample['split']} · ood={sample.get('ood_type') or 'ID'}"
    return str(image_path), sample["question"], ground_truth, base, finetuned, status, metadata


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VisionGym") as demo:
        gr.Markdown("# VisionGym\nSynthetic visual reasoning generator and benchmark result viewer.")

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
                result_base = gr.Textbox(label="Base model")
                result_finetuned = gr.Textbox(label="Fine-tuned model")
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
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionGym Gradio demo")
    parser.add_argument("--host", default=os.getenv("VISIONGYM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VISIONGYM_PORT", "7860")))
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
