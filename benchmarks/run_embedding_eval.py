#!/usr/bin/env python3
"""
Direct Embedding Evaluation for Unified Agentic Model

This script evaluates embedding quality directly without relying on MTEB's model loading.
It tests:
1. Classification (Banking77)
2. Retrieval (custom dataset)
3. Similarity (STS-style)

Usage:
    python benchmarks/run_embedding_eval.py
    python benchmarks/run_embedding_eval.py --compare_base
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_instruct.action_tokens import ACTION_TOKENS  # noqa: E402

# Try to import datasets
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets not installed. Run: pip install datasets")
    exit(1)


class EmbeddingModel:
    """Embedding model using mean pooling over hidden states."""

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

    def encode(self, texts: List[str], batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        for i in iterator:
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )

            hidden_states = outputs.hidden_states[-1].float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def eval_banking77_classification(model: EmbeddingModel, num_train: int = 1000, num_test: int = 500) -> Dict:
    """Evaluate on Banking77 classification task."""
    print("\n" + "="*60)
    print("BANKING77 CLASSIFICATION")
    print("="*60)

    # Load dataset
    print("Loading Banking77 dataset...")
    try:
        dataset = load_dataset("banking77", split="train", trust_remote_code=True)
    except:
        # Fallback to legacy load
        dataset = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)

    # Sample train/test
    dataset = dataset.shuffle(seed=42)
    train_data = dataset.select(range(min(num_train, len(dataset) // 2)))
    test_data = dataset.select(range(len(dataset) // 2, min(len(dataset) // 2 + num_test, len(dataset))))

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Encode
    print("Encoding train set...")
    train_embeddings = model.encode(train_data["text"], batch_size=16)
    print("Encoding test set...")
    test_embeddings = model.encode(test_data["text"], batch_size=16)

    train_labels = np.array(train_data["label"])
    test_labels = np.array(test_data["label"])

    # Logistic Regression
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embeddings, train_labels)
    lr_preds = clf.predict(test_embeddings)
    lr_acc = accuracy_score(test_labels, lr_preds)
    lr_f1 = f1_score(test_labels, lr_preds, average='macro')

    # KNN
    print("Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings, train_labels)
    knn_preds = knn.predict(test_embeddings)
    knn_acc = accuracy_score(test_labels, knn_preds)
    knn_f1 = f1_score(test_labels, knn_preds, average='macro')

    results = {
        "task": "Banking77Classification",
        "num_train": len(train_data),
        "num_test": len(test_data),
        "num_classes": len(set(train_labels)),
        "logistic_regression": {
            "accuracy": float(lr_acc),
            "f1_macro": float(lr_f1),
        },
        "knn": {
            "accuracy": float(knn_acc),
            "f1_macro": float(knn_f1),
        },
    }

    print(f"\nResults:")
    print(f"  Logistic Regression: Acc={lr_acc:.4f}, F1={lr_f1:.4f}")
    print(f"  KNN (k=5):           Acc={knn_acc:.4f}, F1={knn_f1:.4f}")

    return results


def eval_retrieval(model: EmbeddingModel, num_queries: int = 100) -> Dict:
    """Evaluate retrieval quality on NQ-like task."""
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION")
    print("="*60)

    # Load Natural Questions dataset
    print("Loading Natural Questions dataset...")
    # TODO(decontam): this loads sentence-transformers/natural-questions TRAIN
    # split, which is the same source the Stage-1.5-v3 contrastive trainer
    # samples from (scripts/data/create_contrastive_dataset.py). Any retrieval
    # number reported from this code path against a Stage-1.5-v3 checkpoint
    # is contaminated by construction. Swap to a held-out source before
    # publishing -- candidates: BEIR/nq dev, sentence-transformers/nq dev
    # split, or the MTEB NQ task. See decontamination/EVAL_CONTAMINATION_CHECK.md.
    try:
        dataset = load_dataset("sentence-transformers/natural-questions", split="train", streaming=True)
    except Exception:
        print("Could not load NQ dataset, using synthetic data...")
        return eval_synthetic_retrieval(model)

    # Collect query-answer pairs
    pairs = []
    for item in tqdm(dataset, total=num_queries * 2, desc="Loading data"):
        query = item.get("query", "")
        answer = item.get("answer", "")

        if query.strip() and answer.strip() and len(answer) > 20:
            pairs.append({"query": query, "answer": answer})

        if len(pairs) >= num_queries:
            break

    if len(pairs) < 10:
        print("Not enough data, using synthetic...")
        return eval_synthetic_retrieval(model)

    print(f"Loaded {len(pairs)} query-answer pairs")

    # Encode queries and answers
    queries = [p["query"] for p in pairs]
    answers = [p["answer"] for p in pairs]

    print("Encoding queries...")
    query_embs = model.encode(queries, batch_size=16)
    print("Encoding answers...")
    answer_embs = model.encode(answers, batch_size=16)

    # Compute similarities
    similarities = query_embs @ answer_embs.T  # [num_queries, num_answers]

    # Compute metrics
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0

    for i in range(len(queries)):
        # Get ranking (highest similarity first)
        ranking = np.argsort(-similarities[i])

        # Find position of correct answer
        correct_pos = np.where(ranking == i)[0][0]

        if correct_pos == 0:
            hits_at_1 += 1
        if correct_pos < 5:
            hits_at_5 += 1
        if correct_pos < 10:
            hits_at_10 += 1

        mrr += 1.0 / (correct_pos + 1)

    n = len(queries)
    results = {
        "task": "Retrieval",
        "num_queries": n,
        "num_documents": n,
        "hits@1": float(hits_at_1 / n),
        "hits@5": float(hits_at_5 / n),
        "hits@10": float(hits_at_10 / n),
        "mrr": float(mrr / n),
    }

    print(f"\nResults:")
    print(f"  Hits@1:  {results['hits@1']:.4f}")
    print(f"  Hits@5:  {results['hits@5']:.4f}")
    print(f"  Hits@10: {results['hits@10']:.4f}")
    print(f"  MRR:     {results['mrr']:.4f}")

    return results


def eval_synthetic_retrieval(model: EmbeddingModel) -> Dict:
    """Synthetic retrieval evaluation with hand-crafted examples."""
    print("Using synthetic retrieval data...")

    # Create synthetic query-document pairs
    pairs = [
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

    queries = [p[0] for p in pairs]
    docs = [p[1] for p in pairs]

    # Add some distractors
    distractors = [
        "The weather today is sunny and warm.",
        "Python is a popular programming language.",
        "Coffee is a widely consumed beverage.",
        "Mount Everest is the tallest mountain.",
        "The stock market opened higher today.",
    ]
    docs.extend(distractors)

    print("Encoding queries...")
    query_embs = model.encode(queries, batch_size=8, show_progress=False)
    print("Encoding documents...")
    doc_embs = model.encode(docs, batch_size=8, show_progress=False)

    # Compute similarities
    similarities = query_embs @ doc_embs.T

    # Compute metrics
    hits_at_1 = 0
    hits_at_3 = 0
    mrr = 0

    for i in range(len(queries)):
        ranking = np.argsort(-similarities[i])
        correct_pos = np.where(ranking == i)[0][0]

        if correct_pos == 0:
            hits_at_1 += 1
        if correct_pos < 3:
            hits_at_3 += 1
        mrr += 1.0 / (correct_pos + 1)

    n = len(queries)
    results = {
        "task": "SyntheticRetrieval",
        "num_queries": n,
        "num_documents": len(docs),
        "hits@1": float(hits_at_1 / n),
        "hits@3": float(hits_at_3 / n),
        "mrr": float(mrr / n),
    }

    print(f"\nResults:")
    print(f"  Hits@1: {results['hits@1']:.4f}")
    print(f"  Hits@3: {results['hits@3']:.4f}")
    print(f"  MRR:    {results['mrr']:.4f}")

    return results


def eval_similarity(model: EmbeddingModel) -> Dict:
    """Evaluate semantic similarity."""
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY EVALUATION")
    print("="*60)

    # High similarity pairs
    high_sim_pairs = [
        ("The cat sat on the mat.", "A cat is sitting on a mat."),
        ("Machine learning is a subset of AI.", "ML is part of artificial intelligence."),
        ("The stock market crashed today.", "Stocks fell sharply today."),
        ("She enjoys reading books.", "Reading is her favorite hobby."),
        ("The weather is beautiful.", "It's a lovely day outside."),
    ]

    # Low similarity pairs
    low_sim_pairs = [
        ("The cat sat on the mat.", "Quantum computing uses qubits."),
        ("Machine learning is powerful.", "Pizza is a popular food."),
        ("The weather is sunny.", "Database optimization is complex."),
        ("She enjoys reading.", "The car needs new tires."),
        ("Coffee is delicious.", "Mountains are very tall."),
    ]

    print("Computing similarities...")

    high_sims = []
    for s1, s2 in high_sim_pairs:
        emb1 = model.encode([s1], batch_size=1, show_progress=False)
        emb2 = model.encode([s2], batch_size=1, show_progress=False)
        sim = float((emb1 @ emb2.T)[0, 0])
        high_sims.append(sim)

    low_sims = []
    for s1, s2 in low_sim_pairs:
        emb1 = model.encode([s1], batch_size=1, show_progress=False)
        emb2 = model.encode([s2], batch_size=1, show_progress=False)
        sim = float((emb1 @ emb2.T)[0, 0])
        low_sims.append(sim)

    avg_high = np.mean(high_sims)
    avg_low = np.mean(low_sims)
    separation = avg_high - avg_low

    results = {
        "task": "SemanticSimilarity",
        "avg_high_similarity": float(avg_high),
        "avg_low_similarity": float(avg_low),
        "separation": float(separation),
        "high_sims": high_sims,
        "low_sims": low_sims,
    }

    print(f"\nResults:")
    print(f"  Avg High Similarity: {avg_high:.4f}")
    print(f"  Avg Low Similarity:  {avg_low:.4f}")
    print(f"  Separation:          {separation:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Direct embedding evaluation")

    parser.add_argument(
        "--model",
        type=str,
        default="Arjunvad/unified-model-stage1-action-tokens-v2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/embedding_eval",
        help="Output directory"
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="Compare with base Qwen model"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer samples"
    )

    args = parser.parse_args()

    print("="*60)
    print("EMBEDDING EVALUATION")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluation parameters
    num_train = 200 if args.quick else 1000
    num_test = 100 if args.quick else 500
    num_retrieval = 50 if args.quick else 100

    all_results = {}

    # Evaluate unified model
    print(f"\n{'='*60}")
    print(f"EVALUATING: {args.model}")
    print(f"{'='*60}")

    model = EmbeddingModel(args.model)

    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    # Run evaluations
    results["tasks"]["classification"] = eval_banking77_classification(model, num_train, num_test)
    results["tasks"]["retrieval"] = eval_retrieval(model, num_retrieval)
    results["tasks"]["similarity"] = eval_similarity(model)

    all_results["unified_model"] = results

    # Save
    output_path = os.path.join(args.output_dir, "unified_model_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Optionally compare with base
    if args.compare_base:
        print(f"\n{'='*60}")
        print(f"EVALUATING BASELINE: Qwen/Qwen2.5-3B-Instruct")
        print(f"{'='*60}")

        base_model = EmbeddingModel("Qwen/Qwen2.5-3B-Instruct")

        base_results = {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
        }

        base_results["tasks"]["classification"] = eval_banking77_classification(base_model, num_train, num_test)
        base_results["tasks"]["retrieval"] = eval_retrieval(base_model, num_retrieval)
        base_results["tasks"]["similarity"] = eval_similarity(base_model)

        all_results["base_qwen"] = base_results

        # Save
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

        if "classification" in tasks:
            clf = tasks["classification"]
            print(f"  Classification (Banking77): LR Acc={clf['logistic_regression']['accuracy']:.4f}")

        if "retrieval" in tasks:
            ret = tasks["retrieval"]
            print(f"  Retrieval: Hits@1={ret.get('hits@1', 'N/A'):.4f}, MRR={ret.get('mrr', 'N/A'):.4f}")

        if "similarity" in tasks:
            sim = tasks["similarity"]
            print(f"  Similarity: Separation={sim['separation']:.4f}")

    # Save combined summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
