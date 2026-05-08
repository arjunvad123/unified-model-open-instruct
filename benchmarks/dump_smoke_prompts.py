#!/usr/bin/env python3
"""
Dump every hardcoded smoke-test prompt to a single JSONL.

This file is the input to `decontamination/search.py` so we can check
whether any eval prompts overlap with the training corpus before
publishing benchmark numbers.

Output: benchmarks/smoke_prompts.jsonl (gitignored — regenerate on demand).

Usage:
    python benchmarks/dump_smoke_prompts.py
    python decontamination/search.py \\
        --train_dataset_names allenai/tulu-3-sft-mixture ms_marco \\
                              GritLM/MEDI2 sentence-transformers/natural-questions \\
        --dataset benchmarks/smoke_prompts.jsonl --field prompt \\
        --ngram_size 13 --match_threshold 0.5 \\
        --output_dir decontamination_results/

If the eval scripts gain new hardcoded prompts, add them here so the
contamination check stays in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "smoke_prompts.jsonl"


# benchmarks/run_generation_eval.py
ROUTING_DIRECT = [
    "What is machine learning?",
    "Explain the concept of recursion",
    "What are the benefits of exercise?",
    "How does photosynthesis work?",
]
ROUTING_RETRIEVAL = [
    "Find information about RLHF",
    "Search for how transformers use attention",
    "Look up the latest news about AI",
    "Find documents about climate change",
]
QA = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "What planet is known as the Red Planet?",
    "What is the largest mammal?",
    "Who wrote Romeo and Juliet?",
]
MATH = [
    "What is 15 + 27?",
    "What is 100 - 37?",
    "What is 12 * 8?",
    "What is 144 / 12?",
    "What is 7 squared?",
]
CODE = [
    "Write a Python function to check if a number is even",
    "Write a Python function to calculate factorial",
    "Write a Python function to find the maximum in a list",
]

# benchmarks/run_ragas.py — RAG_TEST_CASES (questions only)
RAGAS_QUESTIONS = [
    "What is reinforcement learning from human feedback?",
    "How do transformers use attention mechanisms?",
    "What is retrieval-augmented generation?",
    "How does contrastive learning work for embeddings?",
    "What is the difference between quantum and classical computing?",
]

# benchmarks/run_embedding_eval.py — eval_synthetic_retrieval pairs
EMBEDDING_PAIRS = [
    ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data."),
    ("How do neural networks work?", "Neural networks process information through layers of interconnected nodes."),
    ("What is deep learning?", "Deep learning uses multi-layer neural networks to learn representations."),
    ("Explain gradient descent", "Gradient descent optimizes by iteratively moving toward the minimum of a function."),
    ("What is backpropagation?", "Backpropagation computes gradients by propagating errors backward through the network."),
    ("What is RLHF?", "RLHF trains models using human feedback to align with preferences."),
    ("How does attention work?", "Attention mechanisms allow models to focus on relevant parts of the input."),
    ("What are transformers?", "Transformers are architectures that use self-attention for sequence processing."),
    ("Explain fine-tuning", "Fine-tuning adapts a pre-trained model to a specific task."),
    ("What is transfer learning?", "Transfer learning reuses knowledge from one task to improve another."),
]
EMBEDDING_DISTRACTORS = [
    "The weather today is sunny and warm.",
    "Python is a popular programming language.",
    "Coffee is a widely consumed beverage.",
    "Mount Everest is the tallest mountain.",
    "The stock market opened higher today.",
]

# benchmarks/inference-scripts/01_generation_demo.py — qualitative prompts
DEMO_01_GENERATION = [
    "What causes the seasons on Earth?",
    "Explain how vaccines work.",
    "What is the greenhouse effect?",
    "How does gravity work?",
    "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? Explain your reasoning.",
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "There are three boxes. One contains only apples, one contains only oranges, and one contains both. The labels are all wrong. You can pick one fruit from one box. How do you label all boxes correctly?",
    "Write a Python function to find the nth Fibonacci number.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to merge two sorted lists into one sorted list.",
    "Write a Python class for a stack with push, pop, and peek methods.",
    "List exactly 5 benefits of regular exercise, numbered 1 through 5.",
    "Summarize the concept of photosynthesis in exactly 2 sentences.",
    "Write a haiku about the ocean.",
    "Explain recursion to a 10-year-old in 3 sentences or less.",
]

# benchmarks/inference-scripts/03_action_token_routing.py — routing demos
DEMO_03_ROUTING = [
    "What is quantum computing?",
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of World War I?",
    "Describe how the human immune system works.",
    "What is the difference between mitosis and meiosis?",
    "Find information about the latest developments in fusion energy.",
    "Search for research papers on transformer architectures.",
    "Look up statistics on global renewable energy adoption.",
    "Find articles about CRISPR gene editing applications.",
    "Search for the history of the internet.",
]


def main() -> None:
    rows: list[dict] = []

    for source, prompts in [
        ("run_generation_eval.routing.direct", ROUTING_DIRECT),
        ("run_generation_eval.routing.retrieval", ROUTING_RETRIEVAL),
        ("run_generation_eval.qa", QA),
        ("run_generation_eval.math", MATH),
        ("run_generation_eval.code", CODE),
        ("run_ragas.test_cases", RAGAS_QUESTIONS),
        ("inference_scripts.01_generation_demo", DEMO_01_GENERATION),
        ("inference_scripts.03_action_token_routing", DEMO_03_ROUTING),
    ]:
        for p in prompts:
            rows.append({"source": source, "prompt": p})

    for query, doc in EMBEDDING_PAIRS:
        rows.append({"source": "run_embedding_eval.synthetic_retrieval.query", "prompt": query})
        rows.append({"source": "run_embedding_eval.synthetic_retrieval.doc", "prompt": doc})
    for d in EMBEDDING_DISTRACTORS:
        rows.append({"source": "run_embedding_eval.synthetic_retrieval.distractor", "prompt": d})

    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} smoke prompts -> {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")


if __name__ == "__main__":
    main()
