#!/usr/bin/env python3
"""
RAGAS Benchmark Runner for Unified Agentic Model

Evaluates RAG pipeline quality using RAGAS metrics:
- Context Precision: Is the retrieved context relevant?
- Context Recall: Does context cover the ground truth?
- Faithfulness: Is the answer faithful to the context?
- Answer Relevancy: Is the answer relevant to the question?

Usage:
    python benchmarks/run_ragas.py
    python benchmarks/run_ragas.py --num_samples 50
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_instruct.action_tokens import ACTION_TOKENS  # noqa: E402

# Check dependencies
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Run: pip install ragas datasets")
    exit(1)


# Sample RAG test cases
RAG_TEST_CASES = [
    {
        "question": "What is RLHF and how is it used in language models?",
        "ground_truth": "RLHF (Reinforcement Learning from Human Feedback) is a technique to fine-tune language models using human preferences. It involves training a reward model on human comparisons and using PPO to optimize the language model.",
        "contexts": [
            "RLHF (Reinforcement Learning from Human Feedback) is a technique used to fine-tune language models. It involves training a reward model on human preferences and then using PPO or similar algorithms to optimize the language model against this reward.",
            "Transformers are a type of neural network architecture that uses self-attention mechanisms.",
        ],
    },
    {
        "question": "How do transformers use attention mechanisms?",
        "ground_truth": "Transformers use self-attention mechanisms to process input sequences in parallel, allowing each position to attend to all other positions in the sequence.",
        "contexts": [
            "Transformers are a type of neural network architecture that uses self-attention mechanisms. They were introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. Transformers have become the foundation for models like BERT, GPT, and T5.",
            "Self-attention allows each token in the sequence to attend to every other token, computing attention weights based on query, key, and value projections.",
        ],
    },
    {
        "question": "What is RAG (Retrieval-Augmented Generation)?",
        "ground_truth": "RAG combines retrieval systems with generative models. When a query comes in, relevant documents are retrieved from a knowledge base and provided as context to the generator.",
        "contexts": [
            "RAG (Retrieval-Augmented Generation) combines retrieval systems with generative models. When a query comes in, relevant documents are retrieved from a knowledge base and provided as context to the generator.",
            "Dense retrieval uses neural embeddings to find similar documents.",
        ],
    },
    {
        "question": "How does contrastive learning work?",
        "ground_truth": "Contrastive learning is a self-supervised learning technique where the model learns to distinguish between similar and dissimilar pairs using a contrastive loss function.",
        "contexts": [
            "Contrastive learning is a self-supervised learning technique where the model learns to distinguish between similar and dissimilar pairs. SimCLR and CLIP are popular contrastive learning methods.",
            "The InfoNCE loss function is commonly used in contrastive learning to maximize agreement between positive pairs while minimizing agreement with negative pairs.",
        ],
    },
    {
        "question": "What are quantum computers and how do they differ from classical computers?",
        "ground_truth": "Quantum computers use quantum bits (qubits) that can exist in superposition states, allowing them to perform certain calculations exponentially faster than classical computers.",
        "contexts": [
            "Quantum computing uses quantum bits (qubits) that can exist in superposition states. This allows quantum computers to perform certain calculations exponentially faster than classical computers.",
            "Classical computers use binary bits that are either 0 or 1, while quantum computers can represent multiple states simultaneously.",
        ],
    },
]


class UnifiedRAGPipeline:
    """
    RAG pipeline using our unified model for both retrieval and generation.
    """

    def __init__(self, model_name: str, device: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except:
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

    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding using mean pooling."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[-1].float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)

    def retrieve(self, query: str, documents: List[str], top_k: int = 2) -> List[str]:
        """Retrieve top-k relevant documents."""
        query_emb = self.get_embedding(query)

        doc_embs = []
        for doc in documents:
            doc_embs.append(self.get_embedding(doc))
        doc_embs = torch.cat(doc_embs, dim=0)

        scores = (query_emb @ doc_embs.T).squeeze()

        if len(documents) == 1:
            return documents

        top_indices = scores.argsort(descending=True)[:top_k]
        return [documents[i] for i in top_indices]

    def generate(self, question: str, contexts: List[str], max_tokens: int = 200) -> str:
        """Generate answer given question and contexts."""
        context_str = "\n".join([f"- {c}" for c in contexts])

        prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {question}

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()


def run_rag_pipeline(pipeline: UnifiedRAGPipeline, test_cases: List[Dict]) -> List[Dict]:
    """Run RAG pipeline on test cases."""
    results = []

    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Processing: {case['question'][:50]}...")

        # Retrieve relevant contexts
        retrieved = pipeline.retrieve(case["question"], case["contexts"])

        # Generate answer
        answer = pipeline.generate(case["question"], retrieved)

        results.append({
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "contexts": retrieved,
            "answer": answer,
        })

        print(f"  Retrieved {len(retrieved)} contexts")
        print(f"  Answer: {answer[:100]}...")

    return results


def evaluate_with_ragas(results: List[Dict], output_dir: str):
    """Evaluate results using RAGAS metrics."""
    print("\n" + "="*60)
    print("EVALUATING WITH RAGAS")
    print("="*60)

    # Convert to RAGAS format
    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }

    dataset = Dataset.from_dict(data)

    # Run RAGAS evaluation
    print("\nRunning RAGAS metrics (this may take a few minutes)...")

    try:
        scores = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )

        print("\n" + "="*60)
        print("RAGAS RESULTS")
        print("="*60)

        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "ragas_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "scores": {k: float(v) for k, v in scores.items()},
                "num_samples": len(results),
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        return scores

    except Exception as e:
        print(f"\nWarning: RAGAS evaluation failed: {e}")
        print("This often happens without an OpenAI API key (RAGAS uses LLM-as-judge)")
        print("\nFalling back to manual evaluation...")

        # Manual evaluation without LLM
        return manual_evaluate(results, output_dir)


def manual_evaluate(results: List[Dict], output_dir: str):
    """Simple manual evaluation without LLM-as-judge."""
    print("\n" + "="*60)
    print("MANUAL EVALUATION (No LLM Judge)")
    print("="*60)

    scores = {
        "has_answer": 0,
        "answer_length_avg": 0,
        "context_used": 0,
    }

    for r in results:
        # Check if answer is non-empty
        if len(r["answer"]) > 10:
            scores["has_answer"] += 1

        scores["answer_length_avg"] += len(r["answer"])

        # Check if answer mentions context keywords
        context_text = " ".join(r["contexts"]).lower()
        answer_lower = r["answer"].lower()
        keywords = context_text.split()[:20]  # First 20 words
        overlap = sum(1 for kw in keywords if kw in answer_lower)
        if overlap > 3:
            scores["context_used"] += 1

    n = len(results)
    final_scores = {
        "answer_rate": scores["has_answer"] / n,
        "avg_answer_length": scores["answer_length_avg"] / n,
        "context_usage_rate": scores["context_used"] / n,
    }

    for metric, score in final_scores.items():
        print(f"  {metric}: {score:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "manual_eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "scores": final_scores,
            "num_samples": n,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return final_scores


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS benchmark on unified model")

    parser.add_argument(
        "--model",
        type=str,
        default="Arjunvad/unified-model-stage1-action-tokens-v2",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ragas",
        help="Output directory for results"
    )

    args = parser.parse_args()

    print("="*60)
    print("RAGAS RAG EVALUATION")
    print("="*60)

    # Load model
    pipeline = UnifiedRAGPipeline(args.model)

    # Get test cases
    test_cases = RAG_TEST_CASES
    if args.num_samples:
        test_cases = test_cases[:args.num_samples]

    print(f"\nRunning on {len(test_cases)} test cases")

    # Run RAG pipeline
    results = run_rag_pipeline(pipeline, test_cases)

    # Evaluate
    scores = evaluate_with_ragas(results, args.output_dir)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
