#!/usr/bin/env python3
"""
MTEB Retrieval Benchmark Comparison

Compares our Unified Agentic Model against:
1. gte-Qwen2-7B-instruct (SOTA Qwen embedding model)
2. gte-Qwen2-1.5B-instruct (smaller Qwen embedding model)
3. Base Qwen-2.5-3B-Instruct (our base model)

Usage:
    python benchmarks/run_mteb_comparison.py --model unified
    python benchmarks/run_mteb_comparison.py --model all
    python benchmarks/run_mteb_comparison.py --model gte-qwen2-1.5b
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

# Try to import sentence_transformers for comparison models
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class UnifiedModelMTEB:
    """
    MTEB-compatible wrapper for our Unified Agentic Model.
    Uses mean pooling over last hidden states.
    """

    def __init__(
        self,
        model_name: str = "Arjunvad/unified-model-stage1-action-tokens-v2",
        device: str = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        # Load tokenizer from base Qwen (workaround for saved tokenizer issue)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            trust_remote_code=True
        )
        # Add action tokens
        action_tokens = ["<ACT:GEN>", "<ACT:RET>", "<ACT:TOOL>", "<ACT:CODE>"]
        self.tokenizer.add_tokens(action_tokens, special_tokens=True)

        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully. Embedding dim: {self.model.config.hidden_size}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 8,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences using mean pooling."""
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1].float()

                # Mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                summed = (hidden_states * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = summed / counts

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)
        if convert_to_numpy:
            return result.numpy()
        return result


class BaseQwenMTEB:
    """
    MTEB-compatible wrapper for base Qwen-2.5-3B-Instruct.
    Uses mean pooling over last hidden states.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully. Embedding dim: {self.model.config.hidden_size}")

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 8,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences using mean pooling."""
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1].float()

                # Mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                summed = (hidden_states * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = summed / counts

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)
        if convert_to_numpy:
            return result.numpy()
        return result


class GTEQwenMTEB:
    """
    MTEB-compatible wrapper for GTE-Qwen models.
    Uses last token pooling as per GTE methodology.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        device: str = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        # Query instruction for GTE models
        self.query_instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        print(f"Model loaded successfully. Embedding dim: {self.model.config.hidden_size}")

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Pool using the last token."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 4,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        prompt_name: str = None,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences using last token pooling."""
        all_embeddings = []

        # Add instruction for queries
        is_query = prompt_name == "query" if prompt_name else False

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            if is_query:
                batch = [self.query_instruction + s for s in batch]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._last_token_pool(
                    outputs.last_hidden_state.float(),
                    inputs["attention_mask"]
                )

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)
        if convert_to_numpy:
            return result.numpy()
        return result


# Standard MTEB retrieval benchmarks
RETRIEVAL_TASKS = [
    "NFCorpus",
    "SciFact",
    "ArguAna",
    "SCIDOCS",
    "FiQA2018",
    "TRECCOVID",
]

# Smaller subset for quick testing
QUICK_TASKS = [
    "NFCorpus",
    "SciFact",
]


def run_retrieval_benchmark(model, model_name: str, tasks: List[str], output_dir: str):
    """Run retrieval benchmark on specified tasks using MTEB."""
    import mteb

    print(f"\n{'='*60}")
    print(f"Running MTEB Retrieval on: {model_name}")
    print(f"Tasks: {tasks}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for task_name in tasks:
        print(f"\nEvaluating {task_name}...")
        try:
            # Get task
            task = mteb.get_task(task_name)

            # Load data
            task.load_data()

            # Get data from dataset structure (new MTEB API)
            # Structure: task.dataset[subset][split] -> {corpus, queries, relevant_docs}
            subset = list(task.dataset.keys())[0]  # Usually 'default'
            split_name = "test" if "test" in task.dataset[subset] else list(task.dataset[subset].keys())[0]
            split_data = task.dataset[subset][split_name]

            corpus_ds = split_data["corpus"]
            queries_ds = split_data["queries"]
            qrels = split_data["relevant_docs"]

            if corpus_ds is None or queries_ds is None or qrels is None:
                print(f"  Skipping {task_name}: missing data")
                continue

            # Convert HF datasets to dict format
            corpus = {row["id"]: {"title": row.get("title", ""), "text": row.get("text", "")} for row in corpus_ds}
            queries = {row["id"]: row["text"] for row in queries_ds}

            print(f"  Corpus size: {len(corpus)}")
            print(f"  Queries: {len(queries)}")
            print(f"  Qrels: {len(qrels)}")

            # Encode corpus
            corpus_ids = list(corpus.keys())
            corpus_texts = [
                f"{corpus[cid].get('title', '')} {corpus[cid].get('text', '')}".strip()
                for cid in corpus_ids
            ]
            print(f"  Encoding corpus...")
            corpus_embeddings = model.encode(corpus_texts, batch_size=8, show_progress_bar=True)

            # Encode queries
            query_ids = list(queries.keys())
            query_texts = [queries[qid] for qid in query_ids]
            print(f"  Encoding queries...")

            # For GTE models, use query prompt
            if isinstance(model, GTEQwenMTEB):
                query_embeddings = model.encode(query_texts, batch_size=8, prompt_name="query")
            else:
                query_embeddings = model.encode(query_texts, batch_size=8)

            # Compute scores
            print(f"  Computing similarities...")
            scores = np.dot(query_embeddings, corpus_embeddings.T)

            # Compute metrics
            hits_at_1 = 0
            hits_at_5 = 0
            hits_at_10 = 0
            mrr_sum = 0
            ndcg_at_10_sum = 0

            for i, qid in enumerate(query_ids):
                if qid not in qrels:
                    continue

                relevant_docs = set(qrels[qid].keys())
                ranked_indices = np.argsort(scores[i])[::-1]

                # Hits@k
                top_k_docs = [corpus_ids[idx] for idx in ranked_indices[:10]]
                if any(doc in relevant_docs for doc in top_k_docs[:1]):
                    hits_at_1 += 1
                if any(doc in relevant_docs for doc in top_k_docs[:5]):
                    hits_at_5 += 1
                if any(doc in relevant_docs for doc in top_k_docs[:10]):
                    hits_at_10 += 1

                # MRR
                for rank, idx in enumerate(ranked_indices):
                    if corpus_ids[idx] in relevant_docs:
                        mrr_sum += 1 / (rank + 1)
                        break

                # NDCG@10
                dcg = 0
                for rank, idx in enumerate(ranked_indices[:10]):
                    if corpus_ids[idx] in relevant_docs:
                        dcg += 1 / np.log2(rank + 2)
                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_docs), 10)))
                if idcg > 0:
                    ndcg_at_10_sum += dcg / idcg

            n_queries = len([qid for qid in query_ids if qid in qrels])
            task_results = {
                "hits@1": hits_at_1 / n_queries if n_queries > 0 else 0,
                "hits@5": hits_at_5 / n_queries if n_queries > 0 else 0,
                "hits@10": hits_at_10 / n_queries if n_queries > 0 else 0,
                "mrr": mrr_sum / n_queries if n_queries > 0 else 0,
                "ndcg@10": ndcg_at_10_sum / n_queries if n_queries > 0 else 0,
                "n_queries": n_queries,
                "corpus_size": len(corpus),
            }

            results[task_name] = task_results
            print(f"  Results: Hits@1={task_results['hits@1']:.4f}, NDCG@10={task_results['ndcg@10']:.4f}, MRR={task_results['mrr']:.4f}")

        except Exception as e:
            print(f"  Error on {task_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return results


def load_model(model_type: str):
    """Load the specified model."""
    if model_type == "unified":
        return UnifiedModelMTEB(), "unified_model"
    elif model_type == "base-qwen":
        return BaseQwenMTEB(), "base_qwen_3b"
    elif model_type == "gte-qwen2-1.5b":
        return GTEQwenMTEB("Alibaba-NLP/gte-Qwen2-1.5B-instruct"), "gte_qwen2_1.5b"
    elif model_type == "gte-qwen2-7b":
        return GTEQwenMTEB("Alibaba-NLP/gte-Qwen2-7B-instruct"), "gte_qwen2_7b"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_comparison_table(output_dir: str):
    """Create comparison table from all results."""
    all_results = {}

    # Load all result files
    for filename in os.listdir(output_dir):
        if filename.endswith("_results.json"):
            with open(os.path.join(output_dir, filename)) as f:
                data = json.load(f)
                model_name = data["model"]
                all_results[model_name] = data["results"]

    if not all_results:
        print("No results found")
        return

    # Get all tasks
    all_tasks = set()
    for results in all_results.values():
        all_tasks.update(results.keys())
    all_tasks = sorted(all_tasks)

    # Print comparison table
    print("\n" + "="*100)
    print("MTEB RETRIEVAL BENCHMARK COMPARISON")
    print("="*100)

    models = sorted(all_results.keys())

    # Print header
    print(f"\n{'Task':<20}" + "".join(f"{m:<25}" for m in models))
    print("-" * (20 + 25 * len(models)))

    # Metrics to show
    metrics = ["ndcg@10", "mrr", "hits@1"]

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for task in all_tasks:
            row = f"  {task:<18}"
            for model in models:
                score = all_results.get(model, {}).get(task, {}).get(metric, "-")
                if isinstance(score, (int, float)):
                    row += f"{score:.4f}".ljust(25)
                else:
                    row += "-".ljust(25)
            print(row)

        # Print average
        avg_row = f"  {'AVERAGE':<18}"
        for model in models:
            scores = [
                all_results.get(model, {}).get(task, {}).get(metric, None)
                for task in all_tasks
            ]
            valid_scores = [s for s in scores if isinstance(s, (int, float))]
            if valid_scores:
                avg = sum(valid_scores) / len(valid_scores)
                avg_row += f"{avg:.4f}".ljust(25)
            else:
                avg_row += "-".ljust(25)
        print("-" * (20 + 25 * len(models)))
        print(avg_row)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "tasks": list(all_tasks),
        "results": all_results
    }

    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary saved to: {output_dir}/comparison_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Run MTEB retrieval benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        default="unified",
        choices=["unified", "base-qwen", "gte-qwen2-1.5b", "gte-qwen2-7b", "all"],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Task set to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mteb",
        help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = QUICK_TASKS if args.tasks == "quick" else RETRIEVAL_TASKS

    if args.model == "all":
        # Run all models
        for model_type in ["unified", "base-qwen", "gte-qwen2-1.5b"]:
            try:
                model, model_name = load_model(model_type)
                run_retrieval_benchmark(model, model_name, tasks, args.output_dir)
                # Free memory
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                print(f"Error running {model_type}: {e}")
                import traceback
                traceback.print_exc()

        # Create comparison table
        create_comparison_table(args.output_dir)
    else:
        model, model_name = load_model(args.model)
        run_retrieval_benchmark(model, model_name, tasks, args.output_dir)

        # If results exist for multiple models, create comparison
        result_files = [f for f in os.listdir(args.output_dir) if f.endswith("_results.json")]
        if len(result_files) > 1:
            create_comparison_table(args.output_dir)


if __name__ == "__main__":
    main()
