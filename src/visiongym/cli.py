from __future__ import annotations

import argparse
import json
from pathlib import Path

from visiongym.dataset import generate_from_config, generate_single_split
from visiongym.evaluation import compare_metric_files, evaluate_files
from visiongym.experiments import build_analysis_from_reports, ingest_results
from visiongym.inference import run_inference, run_prompt_sweep, write_oracle_predictions
from visiongym.reporting import create_report
from visiongym.sft import prepare_sft_dataset, train_lora


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm-reasoning-lab",
        description="Synthetic VLM reasoning benchmark, fine-tuning, and failure-analysis lab",
    )
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
    infer.add_argument("--batch-size", type=int, default=1, help="Inference batch size; increase on GPUs with headroom")
    infer.add_argument("--max-new-tokens", type=int, default=48)

    sweep = subparsers.add_parser("infer-prompts", help="Evaluate multiple prompt modes while loading the VLM only once")
    sweep.add_argument("--dataset", required=True)
    sweep.add_argument("--output-dir", required=True)
    sweep.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    sweep.add_argument(
        "--prompt-modes",
        nargs="+",
        default=["direct", "reasoning", "fewshot"],
        choices=["direct", "json", "reasoning", "fewshot"],
    )
    sweep.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter path")
    sweep.add_argument("--limit", type=int, default=None)
    sweep.add_argument("--load-in-4bit", action="store_true")
    sweep.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    sweep.add_argument("--batch-size", type=int, default=1)
    sweep.add_argument("--max-new-tokens", type=int, default=48)

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

    ingest = subparsers.add_parser("ingest-results", help="Ingest Colab result ZIP/directory and rebuild validated experiment reports")
    ingest.add_argument("--bundle", required=True, help="Colab result directory or ZIP")
    ingest.add_argument("--dataset", required=True, help="Benchmark JSONL used for the predictions")
    ingest.add_argument("--output", required=True, help="Experiment output directory")
    ingest.add_argument("--strict", action="store_true", help="Fail on missing, duplicate, or unexpected sample IDs")

    analyze = subparsers.add_parser("analyze-reports", help="Build compact cross-run analysis from evaluated report directories")
    analyze.add_argument("reports", nargs="+", help="Evaluated report directories containing metrics.json and predictions_scored.csv")
    analyze.add_argument("--output", required=True, help="Output directory for compact measured analysis")

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
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps({"predictions": len(records), "output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "infer-prompts":
        outputs = run_prompt_sweep(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            model_name=args.model,
            prompt_modes=args.prompt_modes,
            adapter_path=args.adapter,
            limit=args.limit,
            load_in_4bit=args.load_in_4bit,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))
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
    elif args.command == "ingest-results":
        manifest = ingest_results(args.bundle, args.dataset, args.output, strict=args.strict)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "analyze-reports":
        manifest = build_analysis_from_reports(args.reports, args.output)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
