#!/usr/bin/env python3
"""
Cross-Task Eval: Generation Models on Retrieval (MTEB)

Tests whether pure generation models can do retrieval via mean pooling.
PI hypothesis: "pure generation model might be even better than retrieval tuned model"

Models tested:
  1. Qwen/Qwen2.5-3B-Instruct  (gen model, mean pool)
  2. Qwen/Qwen2.5-3B-Instruct  (gen model, last-token pool — compare method)
  3. Qwen/Qwen2.5-3B           (gen model, mean pool)
  4. Qwen/Qwen3-4B             (gen model, mean pool)
  5. Qwen/Qwen3-4B             (gen model, last-token pool)

Reference (already benchmarked, included for side-by-side):
  6. Ours Stage 1               (mean pool)
  7. Ours Stage 1.5             (mean pool, contrastive-trained)

Compare against existing results:
  - Qwen3-Embedding-0.6B: 0.3669 avg NDCG@10
  - Qwen3-Embedding-4B:   0.4097 avg NDCG@10

Benchmarks: NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA2018
Metrics: NDCG@10, Hits@1, Hits@10, MRR

Usage:
  pip install torch transformers accelerate mteb numpy tqdm
  python cross_eval_gen_on_retrieval.py
"""
import json, os, gc, sys
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

TASKS = ["NFCorpus", "SciFact", "ArguAna", "SCIDOCS", "FiQA2018"]

# ============================================================
# Pooling helpers
# ============================================================
def mean_pool(hidden_states, attention_mask):
    """Mean pooling over non-padded tokens. Same method as our unified model."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def last_token_pool(hidden_states, attention_mask):
    """Last non-padded token. Used by dedicated embedding models."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lens]


# ============================================================
# Model wrapper: CausalLM with configurable pooling
# ============================================================
class CausalLMRetrieval:
    """Load a CausalLM and extract embeddings via mean or last-token pooling."""

    def __init__(self, model_path, pool_method="mean", device=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pool_method = pool_method
        print(f"  Loading CausalLM: {model_path}")
        print(f"  Pooling: {pool_method}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # For last-token pooling, left-pad so the last token is always meaningful
        if pool_method == "last_token":
            self.tokenizer.padding_side = "left"
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  Params: {params/1e9:.1f}B")

    def encode(self, sentences, batch_size=8, **kwargs):
        all_emb = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding", disable=len(sentences) < 100):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True, use_cache=False)
                h = out.hidden_states[-1].float()
                if self.pool_method == "last_token":
                    emb = last_token_pool(h, inputs["attention_mask"])
                else:
                    emb = mean_pool(h, inputs["attention_mask"])
                emb = F.normalize(emb, p=2, dim=1)
                all_emb.append(emb.cpu())
            del inputs, out, h, emb
            torch.cuda.empty_cache()
        return torch.cat(all_emb, dim=0).numpy()


# ============================================================
# Evaluation (same as baseline_embedding_sota.py)
# ============================================================
def evaluate_task(model, task_name):
    import mteb
    try:
        task = mteb.get_task(task_name)
        task.load_data()
        subset = list(task.dataset.keys())[0]
        split = "test" if "test" in task.dataset[subset] else list(task.dataset[subset].keys())[0]
        data = task.dataset[subset][split]

        corpus = {r["id"]: {"title": r.get("title", ""), "text": r.get("text", "")} for r in data["corpus"]}
        queries = {r["id"]: r["text"] for r in data["queries"]}
        qrels = data["relevant_docs"]
        print(f"    Corpus: {len(corpus)}, Queries: {len(queries)}")

        corpus_ids = list(corpus.keys())
        corpus_texts = [f"{corpus[c].get('title','')} {corpus[c].get('text','')}".strip() for c in corpus_ids]
        corpus_emb = model.encode(corpus_texts, batch_size=8)

        query_ids = list(queries.keys())
        query_texts = [queries[q] for q in query_ids]
        query_emb = model.encode(query_texts, batch_size=16)

        scores = np.dot(query_emb, corpus_emb.T)

        h1, h10, mrr, ndcg, n = 0, 0, 0, 0, 0
        for i, qid in enumerate(query_ids):
            if qid not in qrels:
                continue
            n += 1
            rel = set(qrels[qid].keys())
            ranked = np.argsort(scores[i])[::-1]
            top10 = [corpus_ids[idx] for idx in ranked[:10]]
            if any(d in rel for d in top10[:1]):
                h1 += 1
            if any(d in rel for d in top10[:10]):
                h10 += 1
            for rank, idx in enumerate(ranked):
                if corpus_ids[idx] in rel:
                    mrr += 1 / (rank + 1)
                    break
            dcg = sum(1 / np.log2(r + 2) for r, idx in enumerate(ranked[:10]) if corpus_ids[idx] in rel)
            idcg = sum(1 / np.log2(j + 2) for j in range(min(len(rel), 10)))
            if idcg > 0:
                ndcg += dcg / idcg

        return {"hits@1": h1 / n, "hits@10": h10 / n, "ndcg@10": ndcg / n, "mrr": mrr / n}
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    # Models to evaluate: (display_name, key, loader_fn)
    # Note: Each model is loaded, evaluated, then freed before loading the next.
    MODELS = [
        # --- Generation models (the cross-task experiment) ---
        ("Qwen2.5-3B-Inst (mean)",
         "qwen25_3b_inst_mean",
         lambda: CausalLMRetrieval("Qwen/Qwen2.5-3B-Instruct", pool_method="mean")),

        ("Qwen2.5-3B-Inst (last-tok)",
         "qwen25_3b_inst_last",
         lambda: CausalLMRetrieval("Qwen/Qwen2.5-3B-Instruct", pool_method="last_token")),

        ("Qwen2.5-3B (mean)",
         "qwen25_3b_base_mean",
         lambda: CausalLMRetrieval("Qwen/Qwen2.5-3B", pool_method="mean")),

        ("Qwen3-4B (mean)",
         "qwen3_4b_mean",
         lambda: CausalLMRetrieval("Qwen/Qwen3-4B", pool_method="mean")),

        ("Qwen3-4B (last-tok)",
         "qwen3_4b_last",
         lambda: CausalLMRetrieval("Qwen/Qwen3-4B", pool_method="last_token")),

        # --- Our models (reference, already benchmarked) ---
        ("Ours Stage 1 (mean)",
         "ours_stage1",
         lambda: CausalLMRetrieval("Arjunvad/unified-model-stage1-action-tokens-v2", pool_method="mean")),

        ("Ours Stage 1.5 (mean)",
         "ours_stage15",
         lambda: CausalLMRetrieval("Arjunvad/unified-model-stage1-5-embedding-v2", pool_method="mean")),
    ]

    all_results = {}
    for display_name, key, loader in MODELS:
        print("\n" + "=" * 70)
        print(f"EVALUATING: {display_name}")
        print("=" * 70)
        try:
            model = loader()
            results = {}
            for task in TASKS:
                print(f"\n  {task}...")
                r = evaluate_task(model, task)
                results[task] = r
                if "error" not in r:
                    print(f"    NDCG@10={r['ndcg@10']:.4f}  Hits@1={r['hits@1']:.4f}  MRR={r['mrr']:.4f}")
            all_results[key] = results
            del model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[key] = {"error": str(e)}

    # ============================================================
    # Print comparison tables
    # ============================================================
    keys = [k for _, k, _ in MODELS]
    names = {k: dn for dn, k, _ in MODELS}

    # Known embedding model baselines (from previous benchmarks)
    EMBEDDING_BASELINES = {
        "Qwen3-Emb-0.6B": {"NFCorpus": 0.2321, "SciFact": 0.5907, "ArguAna": 0.5515, "SCIDOCS": 0.1309, "FiQA2018": 0.3294},
        "Qwen3-Emb-4B":   {"NFCorpus": 0.2676, "SciFact": 0.6614, "ArguAna": 0.5694, "SCIDOCS": 0.1453, "FiQA2018": 0.4049},
    }

    print("\n" + "=" * 140)
    print("CROSS-TASK EVALUATION: Generation Models on Retrieval")
    print("Question: Can pure generation models match dedicated embedding models?")
    print("=" * 140)

    for metric in ["ndcg@10", "hits@1", "hits@10", "mrr"]:
        print(f"\n--- {metric.upper()} ---")

        # Build header
        all_keys = keys + list(EMBEDDING_BASELINES.keys())
        all_names = {**names, **{k: k for k in EMBEDDING_BASELINES}}
        header = f"{'Task':<15}" + "".join(f"{all_names[k][:18]:<20}" for k in all_keys)
        print(header)
        print("-" * len(header))

        avgs = {k: [] for k in all_keys}
        for task in TASKS:
            row = f"{task:<15}"
            for k in all_keys:
                if k in all_results and task in all_results.get(k, {}) and "error" not in all_results[k].get(task, {}):
                    val = all_results[k][task].get(metric, 0)
                    row += f"{val:<20.4f}"
                    avgs[k].append(val)
                elif k in EMBEDDING_BASELINES and metric == "ndcg@10":
                    val = EMBEDDING_BASELINES[k].get(task, 0)
                    row += f"{val:<20.4f}"
                    avgs[k].append(val)
                else:
                    row += f"{'N/A':<20}"
            print(row)

        print("-" * len(header))
        row = f"{'AVERAGE':<15}"
        for k in all_keys:
            if avgs[k]:
                row += f"{sum(avgs[k])/len(avgs[k]):<20.4f}"
            else:
                row += f"{'N/A':<20}"
        print(row)

    # ============================================================
    # Key findings summary
    # ============================================================
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    # Compare gen models vs embedding models on NDCG@10
    for k in keys:
        if k in all_results and "error" not in all_results.get(k, {}):
            task_scores = []
            for task in TASKS:
                if task in all_results[k] and "error" not in all_results[k][task]:
                    task_scores.append(all_results[k][task].get("ndcg@10", 0))
            if task_scores:
                avg = sum(task_scores) / len(task_scores)
                emb_06_avg = sum(EMBEDDING_BASELINES["Qwen3-Emb-0.6B"].values()) / 5
                emb_4b_avg = sum(EMBEDDING_BASELINES["Qwen3-Emb-4B"].values()) / 5
                vs_06 = ((avg - emb_06_avg) / emb_06_avg) * 100
                vs_4b = ((avg - emb_4b_avg) / emb_4b_avg) * 100
                print(f"\n  {names[k]}: NDCG@10 avg = {avg:.4f}")
                print(f"    vs Qwen3-Emb-0.6B ({emb_06_avg:.4f}): {'+' if vs_06 > 0 else ''}{vs_06:.1f}%")
                print(f"    vs Qwen3-Emb-4B   ({emb_4b_avg:.4f}): {'+' if vs_4b > 0 else ''}{vs_4b:.1f}%")

    print(f"""
  INTERPRETATION:
  - If gen models score CLOSE to or ABOVE embedding models → supports unified approach
  - If mean pooling ≈ last-token pooling → pooling method doesn't matter much
  - If Qwen2.5-3B-Instruct (our base) already retrieves well → our model has a strong starting point
  - If Qwen3-4B outperforms Qwen3-Emb-4B → retrieval tuning may hurt (PI's hypothesis)
    """)

    # Save results
    results_dir = "results/cross_eval"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "gen_on_retrieval.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiment": "generation_models_on_retrieval",
            "tasks": TASKS,
            "results": all_results,
            "embedding_baselines": EMBEDDING_BASELINES,
        }, f, indent=2)
    print(f"\nSaved to {results_dir}/gen_on_retrieval.json")
    print("=" * 140)


if __name__ == "__main__":
    main()
