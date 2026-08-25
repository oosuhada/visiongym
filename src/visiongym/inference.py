from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from visiongym.evaluation import read_jsonl, write_jsonl


PROMPT_MODES = {"direct", "json", "reasoning", "fewshot"}
DEVICE_MODES = {"auto", "cuda", "mps", "cpu"}


def build_prompt(question: str, mode: str) -> str:
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {mode}. Choose one of {sorted(PROMPT_MODES)}")
    if mode == "json":
        return (
            "Inspect the image carefully and answer the visual reasoning question. "
            "Return valid JSON only, exactly in the form {\"answer\": \"...\"}. "
            f"Question: {question}"
        )
    if mode == "reasoning":
        return (
            "Inspect object colors, shapes, positions, and sizes carefully. "
            "Solve the spatial reasoning internally, then output only the final short answer with no explanation. "
            f"Question: {question}"
        )
    if mode == "fewshot":
        return (
            "Answer visual questions with a short canonical answer. Examples of answer style: "
            "counting -> '3'; yes/no -> 'yes'; object identity -> 'red circle'. "
            "Do not add explanation. "
            f"Question: {question}"
        )
    return f"Look at the image and answer the question. Return only the short final answer. Question: {question}"


def resolve_device(torch_module: Any, requested: str = "auto", load_in_4bit: bool = False) -> str:
    if requested not in DEVICE_MODES:
        raise ValueError(f"Unsupported device: {requested}. Choose one of {sorted(DEVICE_MODES)}")

    if requested == "auto":
        if torch_module.cuda.is_available():
            resolved = "cuda"
        elif hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = requested

    if resolved == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if resolved == "mps" and not (hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()):
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    if load_in_4bit and resolved != "cuda":
        raise RuntimeError("4-bit bitsandbytes loading is supported by VisionGym only on CUDA. Disable --load-in-4bit for MPS/CPU.")
    return resolved


class TransformersVLM:
    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install VLM dependencies with: pip install -r requirements-vlm.txt") from exc

        self.torch = torch
        self.model_name = model_name
        self.device = resolve_device(torch, requested=device, load_in_4bit=load_in_4bit)
        self.processor = AutoProcessor.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {"dtype": "auto"}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        if "Qwen3-VL" in model_name:
            from transformers import Qwen3VLForConditionalGeneration

            base_model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        else:
            from transformers import AutoModelForImageTextToText

            base_model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load a LoRA adapter. Install requirements-train.txt") from exc
            base_model = PeftModel.from_pretrained(base_model, adapter_path)
        if self.device in {"mps", "cpu"}:
            base_model = base_model.to(self.device)
        self.model = base_model.eval()

        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    @staticmethod
    def _conversation(image: Image.Image, question: str, prompt_mode: str) -> list[dict[str, Any]]:
        prompt = build_prompt(question, prompt_mode)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def predict(self, image: Image.Image, question: str, prompt_mode: str = "direct", max_new_tokens: int = 48) -> str:
        messages = self._conversation(image, question, prompt_mode)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_length = inputs["input_ids"].shape[1]
        generated = generated[:, input_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def predict_batch(
        self,
        images: list[Image.Image],
        questions: list[str],
        prompt_mode: str = "direct",
        max_new_tokens: int = 48,
    ) -> list[str]:
        if not images:
            return []
        if len(images) != len(questions):
            raise ValueError("images and questions must have the same length")
        conversations = [
            self._conversation(image=image, question=question, prompt_mode=prompt_mode)
            for image, question in zip(images, questions)
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs.pop("token_type_ids", None)
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        except self.torch.cuda.OutOfMemoryError as exc:
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA out of memory during batch inference with batch size {len(images)}. "
                "Retry with a smaller --batch-size."
            ) from exc
        input_length = inputs["input_ids"].shape[1]
        generated = generated[:, input_length:]
        return [
            text.strip()
            for text in self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]


def _iter_batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def _run_with_runner(
    runner: TransformersVLM,
    records: list[dict[str, Any]],
    dataset_path: Path,
    output_path: str | Path,
    model_name: str,
    prompt_mode: str,
    adapter_path: str | None,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    output = Path(output_path)
    started_run = time.perf_counter()
    if runner.device == "cuda":
        runner.torch.cuda.reset_peak_memory_stats()

    progress = tqdm(total=len(records), desc=f"Inference {model_name} [{prompt_mode}] b{batch_size}")
    for batch in _iter_batches(records, batch_size):
        images: list[Image.Image] = []
        batch_predictions: list[str] = []
        batch_latency = 0.0
        try:
            for sample in batch:
                image_path = dataset_path.parent / sample["image"]
                with Image.open(image_path) as image_handle:
                    images.append(image_handle.convert("RGB").copy())
            batch_started = time.perf_counter()
            batch_predictions = runner.predict_batch(
                images=images,
                questions=[str(sample["question"]) for sample in batch],
                prompt_mode=prompt_mode,
                max_new_tokens=max_new_tokens,
            )
            batch_latency = time.perf_counter() - batch_started
        finally:
            for image in images:
                image.close()

        if len(batch_predictions) != len(batch):
            raise RuntimeError(
                f"Model returned {len(batch_predictions)} predictions for a batch of {len(batch)} samples."
            )
        sample_latency = batch_latency / len(batch)
        throughput = len(batch) / batch_latency if batch_latency > 0 else None
        for sample, prediction in zip(batch, batch_predictions):
            predictions.append(
                {
                    "sample_id": sample["sample_id"],
                    "prediction": prediction,
                    "latency_seconds": sample_latency,
                    "batch_latency_seconds": batch_latency,
                    "batch_size": len(batch),
                    "throughput_samples_per_second": throughput,
                    "model": model_name,
                    "prompt_mode": prompt_mode,
                    "adapter_path": adapter_path,
                    "device": runner.device,
                }
            )
        progress.update(len(batch))
    progress.close()

    wall_seconds = time.perf_counter() - started_run
    peak_vram_gb = None
    if runner.device == "cuda":
        peak_vram_gb = runner.torch.cuda.max_memory_allocated() / (1024**3)
    write_jsonl(output, predictions)
    metadata = {
        "model": model_name,
        "prompt_mode": prompt_mode,
        "adapter_path": adapter_path,
        "device": runner.device,
        "configured_batch_size": batch_size,
        "samples": len(records),
        "max_new_tokens": max_new_tokens,
        "wall_seconds": wall_seconds,
        "throughput_samples_per_second": (len(records) / wall_seconds) if wall_seconds > 0 else None,
        "peak_vram_gb": peak_vram_gb,
    }
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return predictions


def run_inference(
    dataset_path: str | Path,
    output_path: str | Path,
    model_name: str,
    prompt_mode: str = "direct",
    adapter_path: str | None = None,
    limit: int | None = None,
    load_in_4bit: bool = False,
    device: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 48,
) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_path)
    records = read_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]
    runner = TransformersVLM(
        model_name=model_name,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        device=device,
    )
    return _run_with_runner(
        runner=runner,
        records=records,
        dataset_path=dataset_path,
        output_path=output_path,
        model_name=model_name,
        prompt_mode=prompt_mode,
        adapter_path=adapter_path,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


def run_prompt_sweep(
    dataset_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    prompt_modes: list[str],
    adapter_path: str | None = None,
    limit: int | None = None,
    load_in_4bit: bool = False,
    device: str = "auto",
    batch_size: int = 1,
    max_new_tokens: int = 48,
) -> dict[str, str]:
    if not prompt_modes:
        raise ValueError("prompt_modes must contain at least one mode")
    invalid = [mode for mode in prompt_modes if mode not in PROMPT_MODES]
    if invalid:
        raise ValueError(f"Unsupported prompt modes: {invalid}. Choose from {sorted(PROMPT_MODES)}")
    dataset = Path(dataset_path)
    records = read_jsonl(dataset)
    if limit is not None:
        records = records[:limit]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = TransformersVLM(
        model_name=model_name,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        device=device,
    )
    results: dict[str, str] = {}
    for mode in prompt_modes:
        prediction_path = output / f"{mode}.jsonl"
        _run_with_runner(
            runner=runner,
            records=records,
            dataset_path=dataset,
            output_path=prediction_path,
            model_name=model_name,
            prompt_mode=mode,
            adapter_path=adapter_path,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        results[mode] = str(prediction_path)
    return results


def write_oracle_predictions(dataset_path: str | Path, output_path: str | Path) -> None:
    records = read_jsonl(dataset_path)
    write_jsonl(
        output_path,
        [{"sample_id": record["sample_id"], "prediction": record["answer"], "latency_seconds": 0.0, "model": "oracle"} for record in records],
    )

