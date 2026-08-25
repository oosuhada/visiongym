from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from visiongym.question_generator import generate_questions
from visiongym.renderer import render_scene
from visiongym.scene_generator import SceneGenerator


OOD_SPLITS = {
    "ood_shape": "shape",
    "ood_palette": "palette",
    "ood_count": "count",
    "ood_background": "background",
    "ood_occlusion": "occlusion",
}


def _jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_split(
    output_dir: Path,
    split: str,
    scene_count: int,
    seed: int,
    width: int,
    height: int,
    questions_per_scene: int,
    object_range: tuple[int, int],
    enabled_tasks: list[str],
    ood_type: str | None = None,
) -> dict[str, Any]:
    generator = SceneGenerator(width=width, height=height, seed=seed)
    rng = random.Random(seed + sum(ord(char) for char in split))
    qa_records: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    image_dir = output_dir / "images" / split

    for scene_index in tqdm(range(scene_count), desc=f"Generating {split}"):
        object_count = rng.randint(object_range[0], object_range[1])
        scene = generator.generate(scene_index=scene_index, split=split, object_count=object_count, ood_type=ood_type)
        relative_image_path = Path("images") / split / f"{scene.scene_id}.png"
        render_scene(scene, image_dir / f"{scene.scene_id}.png", seed=seed + scene_index)
        questions = generate_questions(
            scene=scene,
            image_path=relative_image_path.as_posix(),
            count=questions_per_scene,
            seed=seed,
            enabled_tasks=enabled_tasks,
        )
        scene_record = scene.to_dict()
        scene_record["image"] = relative_image_path.as_posix()
        scene_records.append(scene_record)
        qa_records.extend(question.to_dict() for question in questions)

    _jsonl_write(output_dir / f"{split}.jsonl", qa_records)
    _jsonl_write(output_dir / "scenes" / f"{split}.jsonl", scene_records)
    task_counts = Counter(record["task"] for record in qa_records)
    return {
        "split": split,
        "ood_type": ood_type,
        "scenes": scene_count,
        "qa_pairs": len(qa_records),
        "task_counts": dict(sorted(task_counts.items())),
    }


def generate_from_config(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = load_dataset_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    canvas = config.get("canvas", {})
    width = int(canvas.get("width", 512))
    height = int(canvas.get("height", 512))
    questions_per_scene = int(config.get("questions_per_scene", 4))
    object_ranges = config.get("objects_per_scene", {})
    id_range = tuple(object_ranges.get("id", [3, 5]))
    count_range = tuple(object_ranges.get("ood_count", [6, 9]))
    enabled_tasks = list(config.get("task_families", []))

    summaries: list[dict[str, Any]] = []
    for split, scene_count in config.get("splits", {}).items():
        scene_count = int(scene_count)
        if scene_count <= 0:
            continue
        ood_type = OOD_SPLITS.get(split)
        object_range = count_range if ood_type == "count" else id_range
        summaries.append(
            generate_split(
                output_dir=output,
                split=split,
                scene_count=scene_count,
                seed=seed,
                width=width,
                height=height,
                questions_per_scene=questions_per_scene,
                object_range=(int(object_range[0]), int(object_range[1])),
                enabled_tasks=enabled_tasks,
                ood_type=ood_type,
            )
        )

    benchmark_records: list[dict[str, Any]] = []
    for summary in summaries:
        split = summary["split"]
        if split == "test" or split.startswith("ood_"):
            benchmark_records.extend(_read_jsonl(output / f"{split}.jsonl"))
    _jsonl_write(output / "benchmark.jsonl", benchmark_records)

    manifest = {"config": config, "splits": summaries, "benchmark_qa_pairs": len(benchmark_records)}
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_single_split(
    count: int,
    split: str,
    output_dir: str | Path,
    seed: int = 42,
    questions_per_scene: int = 4,
) -> dict[str, Any]:
    ood_type = OOD_SPLITS.get(split)
    object_range = (6, 9) if ood_type == "count" else (3, 5)
    return generate_split(
        output_dir=Path(output_dir),
        split=split,
        scene_count=count,
        seed=seed,
        width=512,
        height=512,
        questions_per_scene=questions_per_scene,
        object_range=object_range,
        enabled_tasks=[
            "counting",
            "left_right",
            "above_below",
            "nearest_farthest",
            "relative_size",
            "center_proximity",
            "relative_ordering",
            "between",
            "overlap",
            "inside_outside",
            "multi_hop",
        ],
        ood_type=ood_type,
    )

