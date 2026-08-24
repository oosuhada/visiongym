from __future__ import annotations

import argparse
import json
from pathlib import Path

from visiongym.dataset import generate_from_config, generate_single_split
from visiongym.evaluation import compare_metric_files, evaluate_files
from visiongym.inference import run_inference, write_oracle_predictions
from visiongym.reporting import create_report
from visiongym.sft import prepare_sft_dataset, train_lora


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visiongym", description="VisionGym synthetic visual reasoning lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a synthetic visual reasoning dataset")
    generate.add_argument("--config", default="configs/dataset.yaml", help="Dataset YAML config")
    generate.add_argument("--output", default="data/generated", help="Output directory")
    generate.add_argument("--count", type=int, default=None, help="Generate this many scenes for one split")
    generate.add_argument("--split", default="train", help="Split used with --count")
    generate.add_argument("--seed", type=int, default=42, help="Random seed used with --count")
    generate.add_argument("--questions-per-scene", type=int, default=4)

    infer = subparsers.add_parser("infer", help="Run VLM inference over a JSONL benchmark")
    infer.add_argument("--dataset", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    infer.add_argument("--prompt-mode", default="direct", choices=["direct", "json", "reasoning", "fewshot"])
    infer.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter path")
    infer.add_argument("--limit", type=int, default=None)
    infer.add_argument("--load-in-4bit", action="store_true")
    infer.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    oracle = subparsers.add_parser("oracle", help="Create perfect predictions for evaluation smoke tests")
    oracle.add_argument("--dataset", required=True)
    oracle.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Score prediction JSONL against benchmark JSONL")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--prompt-mode", default=None)

    prepare_sft = subparsers.add_parser("prepare-sft", help="Convert benchmark JSONL to multimodal SFT messages")
    prepare_sft.add_argument("--dataset", required=True)
    prepare_sft.add_argument("--output", required=True)
    prepare_sft.add_argument("--limit", type=int, default=None)

    train = subparsers.add_parser("train-lora", help="Fine-tune a VLM using LoRA/QLoRA")
    train.add_argument("--config", default="configs/training.yaml")

    compare = subparsers.add_parser("compare", help="Compare multiple evaluation metrics.json files")
    compare.add_argument("metrics", nargs="+")
    compare.add_argument("--output", default="reports/comparison.csv")

    report = subparsers.add_parser("report", help="Generate CSV/chart/markdown summaries from metrics.json")
    report.add_argument("--metrics", required=True)
    report.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "generate":
        if args.count is not None:
            summary = generate_single_split(
                count=args.count,
                split=args.split,
                output_dir=Path(args.output),
                seed=args.seed,
                questions_per_scene=args.questions_per_scene,
            )
        else:
            summary = generate_from_config(args.config, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "infer":
        records = run_inference(
            dataset_path=args.dataset,
            output_path=args.output,
            model_name=args.model,
            prompt_mode=args.prompt_mode,
            adapter_path=args.adapter,
            limit=args.limit,
            load_in_4bit=args.load_in_4bit,
            device=args.device,
        )
        print(json.dumps({"predictions": len(records), "output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "oracle":
        write_oracle_predictions(args.dataset, args.output)
        print(json.dumps({"output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        metrics = evaluate_files(
            dataset_path=args.dataset,
            predictions_path=args.predictions,
            output_dir=args.output,
            model_name=args.model,
            prompt_mode=args.prompt_mode,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.command == "prepare-sft":
        count = prepare_sft_dataset(args.dataset, args.output, limit=args.limit)
        print(json.dumps({"records": count, "output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "train-lora":
        summary = train_lora(args.config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "compare":
        frame = compare_metric_files(args.metrics, args.output)
        print(frame.to_string(index=False))
    elif args.command == "report":
        created = create_report(args.metrics, args.output)
        print(json.dumps({"created": [str(path) for path in created]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

