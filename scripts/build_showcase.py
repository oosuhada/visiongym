from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def normalize(value: str) -> str:
    return value.strip().lower().rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact real-prediction showcase for the demo")
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/benchmark.jsonl"))
    parser.add_argument("--base", type=Path, default=Path("outputs/base-fewshot.jsonl"))
    parser.add_argument("--finetuned", type=Path, default=Path("outputs/lora-direct.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/showcase"))
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    base = {record["sample_id"]: record for record in read_jsonl(args.base)}
    finetuned = {record["sample_id"]: record for record in read_jsonl(args.finetuned)}
    candidates = []
    for sample in dataset:
        sample_id = sample["sample_id"]
        if sample_id not in base or sample_id not in finetuned:
            continue
        answer = normalize(str(sample["answer"]))
        base_correct = normalize(str(base[sample_id]["prediction"])) == answer
        finetuned_correct = normalize(str(finetuned[sample_id]["prediction"])) == answer
        priority = 0 if finetuned_correct and not base_correct else 1 if finetuned_correct else 2
        candidates.append((priority, sample["split"], sample["task"], sample_id, sample))

    selected: list[dict[str, Any]] = []
    best_by_group: dict[tuple[str, str], tuple[int, str, str, str, dict[str, Any]]] = {}
    for candidate in sorted(candidates):
        _, split, task, _, _ = candidate
        best_by_group.setdefault((split, task), candidate)
    groups = list(best_by_group.values())
    splits = sorted({split for _, split, _, _, _ in groups})
    tasks = sorted({task for _, _, task, _, _ in groups})
    selected_ids: set[str] = set()
    selected_splits: set[str] = set()
    for task in tasks:
        options = [candidate for candidate in groups if candidate[2] == task]
        candidate = min(options, key=lambda item: (item[0], item[1] in selected_splits, item[3]))
        selected.append(candidate[-1])
        selected_ids.add(candidate[3])
        selected_splits.add(candidate[1])
    for split in splits:
        if split in selected_splits:
            continue
        options = [candidate for candidate in groups if candidate[1] == split and candidate[3] not in selected_ids]
        candidate = min(options)
        selected.append(candidate[-1])
        selected_ids.add(candidate[3])
        selected_splits.add(split)
    while len(selected) < args.count:
        split_counts = {split: sum(sample["split"] == split for sample in selected) for split in splits}
        task_counts = {task: sum(sample["task"] == task for sample in selected) for task in tasks}
        options = [candidate for candidate in groups if candidate[3] not in selected_ids]
        candidate = min(
            options,
            key=lambda item: (item[0], split_counts[item[1]], task_counts[item[2]], item[3]),
        )
        selected.append(candidate[-1])
        selected_ids.add(candidate[3])
    if len(selected) < args.count:
        chosen = {sample["sample_id"] for sample in selected}
        remaining = [sample for *_, sample in sorted(candidates) if sample["sample_id"] not in chosen]
        selected.extend(remaining[: args.count - len(selected)])

    if args.output.exists():
        shutil.rmtree(args.output)
    for sample in selected:
        source = args.dataset.parent / sample["image"]
        target = args.output / sample["image"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    write_jsonl(args.output / "benchmark.jsonl", selected)
    write_jsonl(args.output / "base-fewshot.jsonl", [base[sample["sample_id"]] for sample in selected])
    write_jsonl(args.output / "lora-direct.jsonl", [finetuned[sample["sample_id"]] for sample in selected])
    print(json.dumps({"showcase_samples": len(selected), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
