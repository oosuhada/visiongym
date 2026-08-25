from __future__ import annotations

import argparse
import json

from visiongym.dataset import generate_single_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VisionGym scenes and QA pairs")
    parser.add_argument("--count", type=int, required=True, help="Number of scenes to generate")
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--output", default="data/generated", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--questions-per-scene", type=int, default=4)
    args = parser.parse_args()
    summary = generate_single_split(
        count=args.count,
        split=args.split,
        output_dir=args.output,
        seed=args.seed,
        questions_per_scene=args.questions_per_scene,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
