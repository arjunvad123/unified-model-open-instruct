#!/usr/bin/env python3
"""
Generation Quality Evaluation for Unified Agentic Model

Tests generation capabilities:
1. Action Token Routing Accuracy
2. Basic QA Quality
3. Code Generation
4. Math Problem Solving

Usage:
    python benchmarks/run_generation_eval.py
    python benchmarks/run_generation_eval.py --compare_base
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Tuple

import torch


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
        except:
            print("Loading tokenizer from base Qwen...")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
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


def eval_action_token_routing(model: GenerationModel) -> Dict:
    """Evaluate action token routing accuracy."""
    print("\n" + "="*60)
    print("ACTION TOKEN ROUTING EVALUATION")
    print("="*60)

    test_cases = [
        # GEN queries
        ("What is machine learning?", "<ACT:GEN>"),
        ("Explain the concept of recursion", "<ACT:GEN>"),
        ("What are the benefits of exercise?", "<ACT:GEN>"),
        ("How does photosynthesis work?", "<ACT:GEN>"),

        # RET queries
        ("Find information about RLHF", "<ACT:RET>"),
        ("Search for how transformers use attention", "<ACT:RET>"),
        ("Look up the latest news about AI", "<ACT:RET>"),
        ("Find documents about climate change", "<ACT:RET>"),

        # TOOL queries
        ("What's the weather in San Francisco?", "<ACT:TOOL>"),
        ("Calculate 1234 * 5678", "<ACT:TOOL>"),
        ("What time is it in Tokyo?", "<ACT:TOOL>"),
        ("Convert 100 USD to EUR", "<ACT:TOOL>"),

        # CODE queries
        ("Write a Python function to check if a number is prime", "<ACT:CODE>"),
        ("Implement a binary search algorithm", "<ACT:CODE>"),
        ("Create a function to reverse a linked list", "<ACT:CODE>"),
        ("Write code to sort a list using quicksort", "<ACT:CODE>"),
    ]

    results_by_type = {"<ACT:GEN>": [], "<ACT:RET>": [], "<ACT:TOOL>": [], "<ACT:CODE>": []}
    correct = 0
    total = 0

    for query, expected in test_cases:
        response = model.generate(query, max_new_tokens=50)

        # Extract action token from response
        detected = None
        for token in ["<ACT:GEN>", "<ACT:RET>", "<ACT:TOOL>", "<ACT:CODE>"]:
            if token in response:
                detected = token
                break

        is_correct = detected == expected
        if is_correct:
            correct += 1
        total += 1

        results_by_type[expected].append(is_correct)

        print(f"  Query: {query[:40]}...")
        print(f"    Expected: {expected}, Detected: {detected}, Correct: {is_correct}")

    # Compute per-type accuracy
    type_accuracy = {}
    for action_type, results in results_by_type.items():
        if results:
            type_accuracy[action_type] = sum(results) / len(results)

    overall_accuracy = correct / total if total > 0 else 0

    results = {
        "task": "ActionTokenRouting",
        "overall_accuracy": float(overall_accuracy),
        "type_accuracy": type_accuracy,
        "correct": correct,
        "total": total,
    }

    print(f"\nResults:")
    print(f"  Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")
    for action_type, acc in type_accuracy.items():
        print(f"  {action_type}: {acc:.4f}")

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


def main():
    parser = argparse.ArgumentParser(description="Generation quality evaluation")

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

    print("="*60)
    print("GENERATION QUALITY EVALUATION")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # Evaluate unified model
    print(f"\n{'='*60}")
    print(f"EVALUATING: {args.model}")
    print(f"{'='*60}")

    model = GenerationModel(args.model)

    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
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
        print(f"\n{'='*60}")
        print(f"EVALUATING BASELINE: Qwen/Qwen2.5-3B-Instruct")
        print(f"{'='*60}")

        base_model = GenerationModel("Qwen/Qwen2.5-3B-Instruct")

        base_results = {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
        }

        # Base model won't have action tokens, so skip that test
        base_results["tasks"]["qa"] = eval_qa_quality(base_model)
        base_results["tasks"]["math"] = eval_math(base_model)
        base_results["tasks"]["code"] = eval_code_generation(base_model)

        all_results["base_qwen"] = base_results

        output_path = os.path.join(args.output_dir, "base_qwen_results.json")
        with open(output_path, "w") as f:
            json.dump(base_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for model_name, res in all_results.items():
        print(f"\n{model_name}:")
        tasks = res["tasks"]

        if "action_routing" in tasks:
            print(f"  Action Routing: {tasks['action_routing']['overall_accuracy']:.4f}")

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
