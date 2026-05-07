#!/usr/bin/env python3
"""
Retrieval Demo: RAG Pipeline with Stage 1 vs Stage 1.5
Index a corpus, run queries, show ranked results with similarity scores.

Models:
  Stage 1:   Arjunvad/unified-model-stage1-action-tokens-v2 (3B)
  Stage 1.5: Arjunvad/unified-model-stage1-5-embedding-v2 (3B)

Usage:
  export HF_TOKEN="your-token"
  python 04_retrieval_demo.py
"""
import os
import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================
# Device Detection
# ============================================
DEVICE = (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
else:
    print("WARNING: HF_TOKEN not set.")

MODELS = [
    ("Stage 1", "Arjunvad/unified-model-stage1-action-tokens-v2"),
    ("Stage 1.5", "Arjunvad/unified-model-stage1-5-embedding-v2"),
]

# ============================================
# Document Corpus
# ============================================
CORPUS = [
    {"id": "doc01", "text": "Transformers are a type of neural network architecture that uses self-attention mechanisms. They were introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017."},
    {"id": "doc02", "text": "RLHF (Reinforcement Learning from Human Feedback) is a technique used to fine-tune language models. It involves training a reward model on human preferences and then using PPO to optimize the language model."},
    {"id": "doc03", "text": "RAG (Retrieval-Augmented Generation) combines retrieval systems with generative models. When a query comes in, relevant documents are retrieved from a knowledge base and provided as context to the generator."},
    {"id": "doc04", "text": "Contrastive learning is a self-supervised learning technique where the model learns to distinguish between similar and dissimilar pairs. SimCLR and CLIP are popular contrastive learning methods."},
    {"id": "doc05", "text": "Quantum computing uses quantum bits (qubits) that can exist in superposition states. This allows quantum computers to perform certain calculations exponentially faster than classical computers."},
    {"id": "doc06", "text": "CRISPR-Cas9 is a gene editing technology that allows scientists to modify DNA sequences. It has applications in treating genetic diseases, creating disease-resistant crops, and basic research."},
    {"id": "doc07", "text": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycle."},
    {"id": "doc08", "text": "Python's GIL (Global Interpreter Lock) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously."},
    {"id": "doc09", "text": "REST APIs use HTTP methods like GET, POST, PUT, DELETE to perform operations on resources. They are stateless and use standard HTTP status codes for responses."},
    {"id": "doc10", "text": "Docker containers package applications with their dependencies into isolated environments. Unlike VMs, containers share the host OS kernel, making them lightweight and fast to start."},
]

QUERIES = [
    ("How does retrieval-augmented generation work?", "doc03"),
    ("Tell me about reinforcement learning from human feedback", "doc02"),
    ("What is a quantum computer?", "doc05"),
    ("How do you edit genes with CRISPR?", "doc06"),
    ("Explain the Python Global Interpreter Lock", "doc08"),
    ("What are transformer neural networks?", "doc01"),
]


def get_embedding(model, tokenizer, text):
    """Mean-pooled embedding from last hidden state, L2 normalized."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1].float()
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def free_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("RETRIEVAL DEMO: RAG Pipeline")
    print("Stage 1 vs Stage 1.5")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    print(f"\nCorpus: {len(CORPUS)} documents")
    for doc in CORPUS:
        print(f"  {doc['id']}: {doc['text'][:60]}...")

    print(f"\nQueries: {len(QUERIES)}")
    for q, expected in QUERIES:
        print(f"  \"{q[:50]}...\" -> expected: {expected}")

    all_retrieval_results = {}

    for model_name, model_id in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Indexing with: {model_name} ({model_id})")
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DTYPE, trust_remote_code=True
        ).to(DEVICE).eval()
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        # Index corpus
        doc_embeddings = []
        for doc in CORPUS:
            emb = get_embedding(model, tokenizer, doc["text"])
            doc_embeddings.append(emb)
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        print(f"  Indexed {len(CORPUS)} documents. Embedding shape: {list(doc_embeddings.shape)}")

        # Run queries
        model_results = []
        for query, expected_id in QUERIES:
            query_emb = get_embedding(model, tokenizer, query)
            scores = (query_emb @ doc_embeddings.T).squeeze()
            ranked_indices = scores.argsort(descending=True)

            top3 = []
            for rank, idx in enumerate(ranked_indices[:3]):
                idx = idx.item()
                top3.append({
                    "rank": rank + 1,
                    "doc_id": CORPUS[idx]["id"],
                    "score": scores[idx].item(),
                    "text": CORPUS[idx]["text"][:80],
                })

            hit_at_1 = top3[0]["doc_id"] == expected_id
            model_results.append({
                "query": query,
                "expected": expected_id,
                "top3": top3,
                "hit_at_1": hit_at_1,
            })

        all_retrieval_results[model_name] = model_results

        del model, tokenizer, doc_embeddings
        free_memory()

    # ============================================================
    # Print Side-by-Side Results
    # ============================================================
    print(f"\n{'=' * 70}")
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for i, (query, expected_id) in enumerate(QUERIES):
        print(f"\n{'─' * 70}")
        print(f"Query {i+1}: \"{query}\"")
        print(f"Expected: {expected_id}")
        print(f"{'─' * 70}")

        for model_name, _ in MODELS:
            result = all_retrieval_results[model_name][i]
            status = "CORRECT" if result["hit_at_1"] else "WRONG"
            print(f"\n  {model_name} [{status}]:")
            for item in result["top3"]:
                marker = " <<" if item["doc_id"] == expected_id else ""
                print(f"    Rank {item['rank']}: [{item['score']:.4f}] {item['doc_id']} - {item['text']}...{marker}")

    # Summary
    print(f"\n{'=' * 70}")
    print("RETRIEVAL COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25}", end="")
    for name, _ in MODELS:
        print(f"{name:<20}", end="")
    print()
    print("-" * 65)

    # Hits@1
    print(f"{'Hits@1':<25}", end="")
    for name, _ in MODELS:
        results = all_retrieval_results[name]
        hits = sum(1 for r in results if r["hit_at_1"])
        print(f"{hits}/{len(results)} ({100*hits/len(results):.0f}%){'':<8}", end="")
    print()

    # Avg top-1 score
    print(f"{'Avg Top-1 Score':<25}", end="")
    for name, _ in MODELS:
        results = all_retrieval_results[name]
        avg = sum(r["top3"][0]["score"] for r in results) / len(results)
        print(f"{avg:<20.4f}", end="")
    print()

    # Avg score gap (1st - 2nd)
    print(f"{'Avg Score Gap (1-2)':<25}", end="")
    for name, _ in MODELS:
        results = all_retrieval_results[name]
        gaps = [r["top3"][0]["score"] - r["top3"][1]["score"] for r in results]
        avg_gap = sum(gaps) / len(gaps)
        print(f"{avg_gap:<20.4f}", end="")
    print()

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
