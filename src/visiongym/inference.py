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

    def predict(self, image: Image.Image, question: str, prompt_mode: str = "direct", max_new_tokens: int = 48) -> str:
        prompt = build_prompt(question, prompt_mode)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
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


def run_inference(
    dataset_path: str | Path,
    output_path: str | Path,
    model_name: str,
    prompt_mode: str = "direct",
    adapter_path: str | None = None,
    limit: int | None = None,
    load_in_4bit: bool = False,
    device: str = "auto",
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
    predictions: list[dict[str, Any]] = []
    for sample in tqdm(records, desc=f"Inference {model_name}"):
        image_path = dataset_path.parent / sample["image"]
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            started = time.perf_counter()
            prediction = runner.predict(image=image, question=sample["question"], prompt_mode=prompt_mode)
            latency = time.perf_counter() - started
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "prediction": prediction,
                "latency_seconds": latency,
                "model": model_name,
                "prompt_mode": prompt_mode,
                "adapter_path": adapter_path,
                "device": runner.device,
            }
        )
    write_jsonl(output_path, predictions)
    return predictions


def write_oracle_predictions(dataset_path: str | Path, output_path: str | Path) -> None:
    records = read_jsonl(dataset_path)
    write_jsonl(
        output_path,
        [{"sample_id": record["sample_id"], "prediction": record["answer"], "latency_seconds": 0.0, "model": "oracle"} for record in records],
    )

