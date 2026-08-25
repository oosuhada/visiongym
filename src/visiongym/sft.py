from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from visiongym.evaluation import read_jsonl, write_jsonl


SYSTEM_PROMPT = (
    "You are a visual spatial reasoning model. Inspect the image carefully and answer with only the short canonical answer. "
    "For object identity use '<color> <shape>', for counts use a number, and for binary questions use 'yes' or 'no'."
)


def to_sft_record(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "image": sample["image"],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
        ],
        "task": sample["task"],
        "difficulty": sample["difficulty"],
    }


def prepare_sft_dataset(input_path: str | Path, output_path: str | Path, limit: int | None = None) -> int:
    records = read_jsonl(input_path)
    if limit is not None:
        records = records[:limit]
    converted = [to_sft_record(record) for record in records]
    write_jsonl(output_path, converted)
    return len(converted)


def _load_training_dataset(
    path: str | Path,
    max_samples: int | None = None,
    include_tasks: list[str] | None = None,
    curriculum: bool = False,
):
    try:
        from datasets import Dataset, Image as HFImage
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -r requirements-train.txt") from exc

    dataset_path = Path(path)
    records = read_jsonl(dataset_path)
    if include_tasks:
        allowed = set(include_tasks)
        records = [record for record in records if record.get("task") in allowed]
    if curriculum:
        records.sort(key=lambda record: (int(record.get("difficulty", 1)), record.get("task", ""), record.get("sample_id", "")))
    if max_samples is not None:
        records = records[:max_samples]
    rows: list[dict[str, Any]] = []
    for sample in records:
        rows.append(
            {
                "image": str((dataset_path.parent / sample["image"]).resolve()),
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sample["question"]},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
                ],
            }
        )
    dataset = Dataset.from_list(rows)
    return dataset.cast_column("image", HFImage())


def train_lora(config_path: str | Path) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -r requirements-train.txt") from exc

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_name = config["model_name"]
    output_dir = config["output_dir"]
    max_samples = config.get("max_samples")
    include_tasks = list(config.get("include_tasks") or [])
    curriculum = bool(config.get("curriculum", False))
    train_dataset = _load_training_dataset(
        config["train_jsonl"],
        max_samples=max_samples,
        include_tasks=include_tasks,
        curriculum=curriculum,
    )
    eval_dataset = _load_training_dataset(
        config["validation_jsonl"],
        max_samples=min(int(max_samples or 1000000), 500),
        include_tasks=include_tasks,
        curriculum=False,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    quantization_config = None
    if bool(config.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    peft_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=float(config.get("epochs", 1)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        bf16=bool(config.get("bf16", True)),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        max_length=None,
        remove_unused_columns=False,
        model_init_kwargs={"dtype": "auto"},
    )

    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    train_result = trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    summary = {
        "model_name": model_name,
        "output_dir": output_dir,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "include_tasks": include_tasks,
        "curriculum": curriculum,
        "train_metrics": train_result.metrics,
    }
    summary_path = Path(output_dir) / "visiongym_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary

