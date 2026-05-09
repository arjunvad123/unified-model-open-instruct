#!/usr/bin/env python3
"""
Generation Quality Smoke Test for Unified Agentic Model.

This is a SMOKE TEST, not a research-grade benchmark. It runs a tiny
hand-written prompt set against the model and checks substring / action-token
hits. Use it to catch regressions during development; for credible numbers run
the lm-evaluation-harness benchmarks reported in baseline_benchmark_report.md.

Tests:
1. Action token routing (does the model emit the expected trained tokens?)
2. Basic QA (substring match against gold keywords)
3. Math (substring match against gold answer)
4. Code generation (keyword presence heuristic)

Usage:
    python benchmarks/run_generation_eval.py
    python benchmarks/run_generation_eval.py --compare_base
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_instruct.action_tokens import ACTION_TOKENS, ROUTING_TOKENS  # noqa: E402

SCRIPT_VERSION = "2026-05-03-smoke-v2"


class GenerationModel:
    """Wrapper for generation tasks."""

    def __init__(self, model_name: str, device: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            print("Loading tokenizer from base Qwen...")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
            self.tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})

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

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate response for a prompt."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        return response.strip()


def _first_routing_token(response: str) -> str | None:
    """Return the earliest routing token (THINK/RET/GEN/STOP) found in the response."""
    earliest_idx = None
    earliest_token = None
    for token in ROUTING_TOKENS:
        idx = response.find(token)
        if idx == -1:
            continue
        if earliest_idx is None or idx < earliest_idx:
            earliest_idx = idx
            earliest_token = token
    return earliest_token


def eval_action_token_routing(model: GenerationModel) -> Dict:
    """Smoke-test action token routing.

    Trained tokens (per open_instruct.action_tokens) are
    GEN/RET/THINK/STOP. Synthetic trajectories teach two
    routes:
      - direct:    <ACT:THINK> ... <ACT:GEN> ... <ACT:STOP>
      - retrieval: <ACT:THINK> ... <ACT:RET> ... <ACT:GEN> ... <ACT:STOP>

    For each query we record the first routing token emitted and whether the
    response contains GEN (any path) and RET (only on retrieval-style queries).
    There is no TOOL/CODE route in training; do not test for those tokens.
    """
    print("\n" + "=" * 60)
    print("ACTION TOKEN ROUTING SMOKE TEST")
    print("=" * 60)

    # category -> list of prompts. "direct" expects THINK->GEN, "retrieval"
    # expects THINK->RET->GEN.
    test_cases: List[Dict] = [
        {"prompt": "What is machine learning?", "category": "direct"},
        {"prompt": "Explain the concept of recursion", "category": "direct"},
        {"prompt": "What are the benefits of exercise?", "category": "direct"},
        {"prompt": "How does photosynthesis work?", "category": "direct"},
        {"prompt": "Find information about RLHF", "category": "retrieval"},
        {"prompt": "Search for how transformers use attention", "category": "retrieval"},
        {"prompt": "Look up the latest news about AI", "category": "retrieval"},
        {"prompt": "Find documents about climate change", "category": "retrieval"},
    ]

    per_query: List[Dict] = []
    counters = {
        "any_routing_token_emitted": 0,
        "starts_with_think": 0,
        "contains_gen": 0,
        "direct_correct": 0,
        "retrieval_correct": 0,
    }
    direct_total = sum(1 for c in test_cases if c["category"] == "direct")
    retrieval_total = len(test_cases) - direct_total

    for case in test_cases:
        prompt = case["prompt"]
        category = case["category"]
        response = model.generate(prompt, max_new_tokens=80)

        first = _first_routing_token(response)
        contains_ret = "<ACT:RET>" in response
        contains_gen = "<ACT:GEN>" in response
        starts_with_think = first == "<ACT:THINK>"

        if first is not None:
            counters["any_routing_token_emitted"] += 1
        if starts_with_think:
            counters["starts_with_think"] += 1
        if contains_gen:
            counters["contains_gen"] += 1

        if category == "direct":
            # Direct path: GEN reached, RET was not used
            correct = contains_gen and not contains_ret
            if correct:
                counters["direct_correct"] += 1
        else:
            # Retrieval path: RET used and GEN reached
            correct = contains_ret and contains_gen
            if correct:
                counters["retrieval_correct"] += 1

        per_query.append({
            "prompt": prompt,
            "category": category,
            "first_routing_token": first,
            "contains_ret": contains_ret,
            "contains_gen": contains_gen,
            "starts_with_think": starts_with_think,
            "category_correct": correct,
        })

        print(f"  [{category}] {prompt[:50]}")
        print(f"    first={first} ret={contains_ret} gen={contains_gen} ok={correct}")

    total = len(test_cases)
    results = {
        "task": "ActionTokenRoutingSmoke",
        "trained_routing_tokens": ROUTING_TOKENS,
        "total": total,
        "fraction_emitted_any_routing_token": counters["any_routing_token_emitted"] / total,
        "fraction_starts_with_think": counters["starts_with_think"] / total,
        "fraction_contains_gen": counters["contains_gen"] / total,
        "direct_path_accuracy": counters["direct_correct"] / direct_total if direct_total else 0.0,
        "retrieval_path_accuracy": counters["retrieval_correct"] / retrieval_total if retrieval_total else 0.0,
        "per_query": per_query,
    }

    print("\nResults:")
    print(f"  emitted any routing token: {results['fraction_emitted_any_routing_token']:.2%}")
    print(f"  starts with <ACT:THINK>:   {results['fraction_starts_with_think']:.2%}")
    print(f"  contains <ACT:GEN>:        {results['fraction_contains_gen']:.2%}")
    print(f"  direct path accuracy:      {results['direct_path_accuracy']:.2%}")
    print(f"  retrieval path accuracy:   {results['retrieval_path_accuracy']:.2%}")

    return results


def eval_qa_quality(model: GenerationModel) -> Dict:
    """Evaluate QA generation quality."""
    print("\n" + "="*60)
    print("QA QUALITY EVALUATION")
    print("="*60)

    test_cases = [
        {
            "question": "What is the capital of France?",
            "keywords": ["paris"],
        },
        {
            "question": "What is 2 + 2?",
            "keywords": ["4", "four"],
        },
        {
            "question": "What planet is known as the Red Planet?",
            "keywords": ["mars"],
        },
        {
            "question": "What is the largest mammal?",
            "keywords": ["whale", "blue whale"],
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "keywords": ["shakespeare", "william shakespeare"],
        },
    ]

    correct = 0
    total = 0
    response_lengths = []

    for case in test_cases:
        response = model.generate(case["question"], max_new_tokens=100)
        response_lower = response.lower()

        # Check if any keyword is in response
        found_keyword = any(kw.lower() in response_lower for kw in case["keywords"])
        if found_keyword:
            correct += 1
        total += 1
        response_lengths.append(len(response))

        print(f"  Q: {case['question']}")
        print(f"  A: {response[:100]}...")
        print(f"  Keywords found: {found_keyword}")

    accuracy = correct / total if total > 0 else 0
    avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    results = {
        "task": "QA",
        "accuracy": float(accuracy),
        "correct": correct,
        "total": total,
        "avg_response_length": float(avg_length),
    }

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Avg Response Length: {avg_length:.1f} chars")

    return results


def eval_math(model: GenerationModel) -> Dict:
    """Evaluate math problem solving."""
    print("\n" + "="*60)
    print("MATH EVALUATION")
    print("="*60)

    test_cases = [
        {"question": "What is 15 + 27?", "answer": "42"},
        {"question": "What is 100 - 37?", "answer": "63"},
        {"question": "What is 12 * 8?", "answer": "96"},
        {"question": "What is 144 / 12?", "answer": "12"},
        {"question": "What is 7 squared?", "answer": "49"},
    ]

    correct = 0
    total = 0

    for case in test_cases:
        response = model.generate(case["question"], max_new_tokens=50)

        # Check if correct answer is in response
        found = case["answer"] in response
        if found:
            correct += 1
        total += 1

        print(f"  Q: {case['question']}")
        print(f"  A: {response[:80]}...")
        print(f"  Expected: {case['answer']}, Found: {found}")

    accuracy = correct / total if total > 0 else 0

    results = {
        "task": "Math",
        "accuracy": float(accuracy),
        "correct": correct,
        "total": total,
    }

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    return results


def eval_code_generation(model: GenerationModel) -> Dict:
    """Evaluate code generation quality."""
    print("\n" + "="*60)
    print("CODE GENERATION EVALUATION")
    print("="*60)

    test_cases = [
        {
            "prompt": "Write a Python function to check if a number is even",
            "keywords": ["def", "return", "%", "2", "=="],
            "min_keywords": 3,
        },
        {
            "prompt": "Write a Python function to calculate factorial",
            "keywords": ["def", "factorial", "return", "if", "*"],
            "min_keywords": 3,
        },
        {
            "prompt": "Write a Python function to find the maximum in a list",
            "keywords": ["def", "return", "max", "for", "if"],
            "min_keywords": 2,
        },
    ]

    passing = 0
    total = 0

    for case in test_cases:
        response = model.generate(case["prompt"], max_new_tokens=200)
        response_lower = response.lower()

        # Count keywords found
        found_keywords = sum(1 for kw in case["keywords"] if kw.lower() in response_lower)
        passed = found_keywords >= case["min_keywords"]

        if passed:
            passing += 1
        total += 1

        print(f"  Prompt: {case['prompt'][:50]}...")
        print(f"  Response: {response[:100]}...")
        print(f"  Keywords found: {found_keywords}/{len(case['keywords'])}, Passed: {passed}")

    pass_rate = passing / total if total > 0 else 0

    results = {
        "task": "CodeGeneration",
        "pass_rate": float(pass_rate),
        "passing": passing,
        "total": total,
    }

    print(f"\nResults:")
    print(f"  Pass Rate: {pass_rate:.4f} ({passing}/{total})")

    return results


def _git_commit() -> str | None:
    """Best-effort current git commit for run provenance."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _run_metadata(model_name: str) -> Dict:
    """Reproducibility metadata embedded in every result dict."""
    return {
        "model": model_name,
        "script": "benchmarks/run_generation_eval.py",
        "script_version": SCRIPT_VERSION,
        "kind": "smoke_test",
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "trained_action_tokens": ACTION_TOKENS,
    }


def main():
    parser = argparse.ArgumentParser(description="Generation quality smoke test")

    parser.add_argument(
        "--model",
        type=str,
        default="Arjunvad/unified-model-stage1-action-tokens-v2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/generation_eval",
        help="Output directory"
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="Compare with base Qwen model"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GENERATION QUALITY SMOKE TEST")
    print("(NOT a research benchmark — see baseline_benchmark_report.md)")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # Evaluate unified model
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {args.model}")
    print(f"{'=' * 60}")

    model = GenerationModel(args.model)

    results = {
        **_run_metadata(args.model),
        "tasks": {},
    }

    results["tasks"]["action_routing"] = eval_action_token_routing(model)
    results["tasks"]["qa"] = eval_qa_quality(model)
    results["tasks"]["math"] = eval_math(model)
    results["tasks"]["code"] = eval_code_generation(model)

    all_results["unified_model"] = results

    # Save
    output_path = os.path.join(args.output_dir, "unified_model_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Optionally compare with base
    if args.compare_base:
        print(f"\n{'=' * 60}")
        print("EVALUATING BASELINE: Qwen/Qwen2.5-3B-Instruct")
        print(f"{'=' * 60}")

        base_model = GenerationModel("Qwen/Qwen2.5-3B-Instruct")

        base_results = {
            **_run_metadata("Qwen/Qwen2.5-3B-Instruct"),
            "tasks": {},
        }

        # Base model wasn't trained on action tokens, so skip that test
        base_results["tasks"]["qa"] = eval_qa_quality(base_model)
        base_results["tasks"]["math"] = eval_math(base_model)
        base_results["tasks"]["code"] = eval_code_generation(base_model)

        all_results["base_qwen"] = base_results

        output_path = os.path.join(args.output_dir, "base_qwen_results.json")
        with open(output_path, "w") as f:
            json.dump(base_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model_name, res in all_results.items():
        print(f"\n{model_name}:")
        tasks = res["tasks"]

        if "action_routing" in tasks:
            ar = tasks["action_routing"]
            print(
                f"  Action Routing: direct={ar['direct_path_accuracy']:.2%}"
                f" retrieval={ar['retrieval_path_accuracy']:.2%}"
                f" any_routing_token={ar['fraction_emitted_any_routing_token']:.2%}"
            )

        if "qa" in tasks:
            print(f"  QA Accuracy: {tasks['qa']['accuracy']:.4f}")

        if "math" in tasks:
            print(f"  Math Accuracy: {tasks['math']['accuracy']:.4f}")

        if "code" in tasks:
            print(f"  Code Pass Rate: {tasks['code']['pass_rate']:.4f}")

    # Save combined summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
