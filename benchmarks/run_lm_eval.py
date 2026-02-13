#!/usr/bin/env python3
"""
LM-Evaluation-Harness Runner for Unified Agentic Model

Runs standard LLM benchmarks:
- MMLU (Massive Multitask Language Understanding) - 5-shot
- GSM8K (Grade School Math) - 4-shot
- HumanEval (Code Generation) - 0-shot
- HellaSwag (Commonsense) - 10-shot
- ARC (AI2 Reasoning Challenge) - 25-shot

Usage:
    # Quick test
    python benchmarks/run_lm_eval.py --tasks hellaswag --limit 100

    # Full MMLU
    python benchmarks/run_lm_eval.py --tasks mmlu --num_fewshot 5

    # Multiple benchmarks
    python benchmarks/run_lm_eval.py --tasks mmlu,gsm8k,hellaswag

    # Using lm_eval CLI directly:
    lm_eval --model hf --model_args pretrained=Arjunvad/unified-model-stage1-action-tokens-v2 --tasks mmlu --num_fewshot 5 --batch_size 4
"""

import argparse
import json
import os
import subprocess
from datetime import datetime


# Standard benchmark configurations (matching Qwen's evaluation setup)
BENCHMARK_CONFIGS = {
    "mmlu": {
        "num_fewshot": 5,
        "description": "Massive Multitask Language Understanding",
    },
    "gsm8k": {
        "num_fewshot": 4,
        "description": "Grade School Math 8K",
    },
    "humaneval": {
        "num_fewshot": 0,
        "description": "Code Generation (HumanEval)",
    },
    "hellaswag": {
        "num_fewshot": 10,
        "description": "Commonsense Reasoning",
    },
    "arc_challenge": {
        "num_fewshot": 25,
        "description": "AI2 Reasoning Challenge",
    },
    "winogrande": {
        "num_fewshot": 5,
        "description": "Winograd Schema Challenge",
    },
    "truthfulqa_mc2": {
        "num_fewshot": 0,
        "description": "TruthfulQA Multiple Choice",
    },
}


def run_lm_eval(
    model_name: str,
    tasks: str,
    num_fewshot: int = None,
    batch_size: int = 4,
    limit: int = None,
    output_dir: str = "results/lm_eval",
    device: str = None,
):
    """Run lm-evaluation-harness on specified tasks."""

    # Build command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", str(batch_size),
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if limit:
        cmd.extend(["--limit", str(limit)])

    if device:
        cmd.extend(["--device", device])

    # Output path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{tasks.replace(',', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    cmd.extend(["--output_path", output_path])

    print("="*60)
    print("LM-EVALUATION-HARNESS")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Tasks: {tasks}")
    print(f"Num fewshot: {num_fewshot}")
    print(f"Batch size: {batch_size}")
    print(f"Limit: {limit}")
    print(f"Output: {output_path}")
    print("="*60)
    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\n✓ Evaluation complete. Results saved to: {output_path}")
        else:
            print(f"\n✗ Evaluation failed with return code {result.returncode}")
    except FileNotFoundError:
        print("\nERROR: lm_eval not found. Install with: pip install lm-eval")
        print("\nAlternatively, run directly:")
        print(f"  {' '.join(cmd)}")


def run_all_benchmarks(model_name: str, output_dir: str, batch_size: int = 4, limit: int = None):
    """Run all standard benchmarks."""
    print("="*60)
    print("RUNNING ALL STANDARD BENCHMARKS")
    print("="*60)

    for task, config in BENCHMARK_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running: {task} - {config['description']}")
        print(f"{'='*60}")

        run_lm_eval(
            model_name=model_name,
            tasks=task,
            num_fewshot=config["num_fewshot"],
            batch_size=batch_size,
            limit=limit,
            output_dir=output_dir,
        )


def main():
    parser = argparse.ArgumentParser(description="Run LM-Eval benchmarks on unified model")

    parser.add_argument(
        "--model",
        type=str,
        default="Arjunvad/unified-model-stage1-action-tokens-v2",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names (e.g., mmlu,gsm8k,hellaswag)"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (overrides default)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for quick testing)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/lm_eval",
        help="Output directory"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all standard benchmarks"
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List available benchmark configurations"
    )

    args = parser.parse_args()

    if args.list_tasks:
        print("Available benchmark configurations:")
        print("="*60)
        for task, config in BENCHMARK_CONFIGS.items():
            print(f"  {task:20} - {config['description']} ({config['num_fewshot']}-shot)")
        return

    if args.all:
        run_all_benchmarks(
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    elif args.tasks:
        # Use default fewshot if not specified
        if args.num_fewshot is None and args.tasks in BENCHMARK_CONFIGS:
            args.num_fewshot = BENCHMARK_CONFIGS[args.tasks]["num_fewshot"]

        run_lm_eval(
            model_name=args.model,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            output_dir=args.output_dir,
        )
    else:
        print("Usage:")
        print("  python benchmarks/run_lm_eval.py --tasks mmlu --limit 100")
        print("  python benchmarks/run_lm_eval.py --all")
        print("  python benchmarks/run_lm_eval.py --list_tasks")


if __name__ == "__main__":
    main()
