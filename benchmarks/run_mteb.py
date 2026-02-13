#!/usr/bin/env python3
"""
MTEB Benchmark Runner for Unified Agentic Model

Evaluates embedding quality using the Massive Text Embedding Benchmark.

Usage:
    # Quick test on single task
    python benchmarks/run_mteb.py --tasks Banking77Classification

    # Retrieval tasks only
    python benchmarks/run_mteb.py --task_types Retrieval

    # Compare with base Qwen
    python benchmarks/run_mteb.py --compare_base
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F

# Check if mteb is installed
try:
    import mteb
except ImportError:
    print("ERROR: mteb not installed. Run: pip install mteb>=2.0.0")
    exit(1)


class UnifiedModelEncoder:
    """
    Encoder wrapper to make our unified model compatible with MTEB.

    Implements the mteb Encoder protocol with encode() method.
    """

    def __init__(self, model_name: str, device: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer - use base Qwen if our model has tokenizer issues
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except:
            print("Loading tokenizer from base Qwen...")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
            # Add action tokens
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": ["<ACT:GEN>", "<ACT:RET>", "<ACT:TOOL>", "<ACT:CODE>"]
            })

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded. Vocab size: {len(self.tokenizer)}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences to embeddings using mean pooling.

        Returns numpy array as required by MTEB.
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )

            # Mean pooling over hidden states
            hidden_states = outputs.hidden_states[-1].float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts

            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Convert to numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

            if show_progress_bar and (i // batch_size) % 10 == 0:
                print(f"  Encoded {min(i + batch_size, len(sentences))}/{len(sentences)} sentences")

        # Concatenate all batches
        return np.vstack(all_embeddings)


def run_mteb_evaluation(
    model_name: str,
    tasks: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    output_dir: str = "results/mteb",
    batch_size: int = 32,
):
    """Run MTEB evaluation on specified tasks."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = UnifiedModelEncoder(model_name)

    # Get tasks
    if tasks:
        print(f"\nUsing specific tasks: {tasks}")
        task_list = mteb.get_tasks(tasks=tasks)
    elif task_types:
        print(f"\nUsing task types: {task_types}")
        task_list = mteb.get_tasks(task_types=task_types)
    else:
        # Default: quick test tasks
        print("\nUsing default quick test tasks")
        task_list = mteb.get_tasks(tasks=["Banking77Classification"])

    print(f"Running {len(task_list)} tasks")

    # Run evaluation using new API
    all_results = []
    for task in task_list:
        print(f"\n  Running task: {task.metadata.name}")
        result = mteb.evaluate(
            model,
            task,
            encode_kwargs={"batch_size": batch_size},
            prediction_folder=output_dir,
        )
        all_results.append(result)

    results = all_results

    # Print summary
    print("\n" + "="*60)
    print("MTEB EVALUATION RESULTS")
    print("="*60)

    summary_data = []
    for task_result in results:
        task_name = task_result.task_name
        scores = task_result.scores

        print(f"\n{task_name}:")
        for split, split_scores in scores.items():
            if isinstance(split_scores, list) and len(split_scores) > 0:
                main_score = split_scores[0].get("main_score", "N/A")
                if isinstance(main_score, float):
                    print(f"  {split}: {main_score:.4f}")
                    summary_data.append({
                        "task": task_name,
                        "split": split,
                        "main_score": main_score
                    })
                else:
                    print(f"  {split}: {main_score}")

    # Save summary
    summary = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "results": summary_data,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run MTEB benchmark on unified model")

    parser.add_argument(
        "--model",
        type=str,
        default="Arjunvad/unified-model-stage1-action-tokens-v2",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific MTEB tasks to run"
    )
    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=None,
        choices=["Classification", "Clustering", "PairClassification", "Reranking", "Retrieval", "STS", "Summarization"],
        help="Task types to run"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mteb",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="Also run on base Qwen model for comparison"
    )

    args = parser.parse_args()

    print("="*60)
    print("MTEB BENCHMARK RUNNER")
    print("="*60)

    # Run on unified model
    print(f"\n[1/{'2' if args.compare_base else '1'}] Evaluating: {args.model}")
    results = run_mteb_evaluation(
        model_name=args.model,
        tasks=args.tasks,
        task_types=args.task_types,
        output_dir=os.path.join(args.output_dir, "unified_model"),
        batch_size=args.batch_size,
    )

    # Optionally compare with base
    if args.compare_base:
        print(f"\n[2/2] Evaluating baseline: Qwen/Qwen2.5-3B-Instruct")
        base_results = run_mteb_evaluation(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            tasks=args.tasks,
            task_types=args.task_types,
            output_dir=os.path.join(args.output_dir, "base_qwen"),
            batch_size=args.batch_size,
        )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
