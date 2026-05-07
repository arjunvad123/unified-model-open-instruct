#!/usr/bin/env python3
"""
Embedding Demo: Stage 1 vs Stage 1.5 Embedding Quality
Shows cosine similarity for similar/dissimilar pairs and topic clustering.

Models:
  Stage 1:   Arjunvad/unified-model-stage1-action-tokens-v2 (3B)
  Stage 1.5: Arjunvad/unified-model-stage1-5-embedding-v2 (3B)

Usage:
  export HF_TOKEN="your-token"
  python 02_embedding_demo.py
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
# Test Pairs
# ============================================
SIMILAR_PAIRS = [
    ("The cat sat on the mat.", "A feline rested on the rug."),
    ("Machine learning is a subset of AI.", "Deep learning uses neural networks for prediction."),
    ("Python is a popular programming language.", "Python is widely used in software development."),
    ("The stock market crashed in 2008.", "There was a major financial crisis in 2008."),
    ("Quantum computing uses qubits.", "Quantum computers process information with quantum bits."),
    ("Exercise improves cardiovascular health.", "Physical activity strengthens the heart."),
    ("The Eiffel Tower is in Paris.", "Paris is famous for its iconic iron tower."),
    ("Climate change causes rising sea levels.", "Global warming leads to ocean level increases."),
    ("DNA stores genetic information.", "Genes are encoded in deoxyribonucleic acid."),
    ("The speed of light is approximately 300,000 km/s.", "Light travels at about three hundred thousand kilometers per second."),
]

DISSIMILAR_PAIRS = [
    ("The cat sat on the mat.", "The stock market crashed in 2008."),
    ("Machine learning is a subset of AI.", "The Eiffel Tower is in Paris."),
    ("Python is a programming language.", "Photosynthesis converts sunlight to energy."),
    ("Quantum computing uses qubits.", "Shakespeare wrote Hamlet in the early 1600s."),
    ("Exercise improves health.", "Docker containers package applications with dependencies."),
]

TOPIC_GROUPS = {
    "Programming": [
        "Python is great for data science.",
        "JavaScript runs in web browsers.",
        "C++ is used for system programming.",
    ],
    "Animals": [
        "Dogs are loyal pets.",
        "Cats are independent animals.",
        "Birds can fly through the air.",
    ],
    "Food": [
        "Pizza originated in Italy.",
        "Sushi is a Japanese dish.",
        "Tacos are popular in Mexico.",
    ],
}


def get_embedding(model, tokenizer, text):
    """Mean-pooled embedding from last hidden state, L2 normalized."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        output_hidden_states=True)
    hidden = outputs.hidden_states[-1].float()
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def cosine_sim(emb1, emb2):
    return (emb1 @ emb2.T).item()


def free_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("EMBEDDING DEMO: Stage 1 vs Stage 1.5")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    all_results = {}

    for model_name, model_id in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Loading: {model_name} ({model_id})")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DTYPE, trust_remote_code=True
        ).to(DEVICE).eval()
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

        results = {"similar": [], "dissimilar": [], "intra": [], "inter": []}

        # Similar pairs
        for a, b in SIMILAR_PAIRS:
            score = cosine_sim(get_embedding(model, tokenizer, a), get_embedding(model, tokenizer, b))
            results["similar"].append(score)

        # Dissimilar pairs
        for a, b in DISSIMILAR_PAIRS:
            score = cosine_sim(get_embedding(model, tokenizer, a), get_embedding(model, tokenizer, b))
            results["dissimilar"].append(score)

        # Topic clustering
        topic_embs = {}
        for topic, texts in TOPIC_GROUPS.items():
            topic_embs[topic] = [get_embedding(model, tokenizer, t) for t in texts]

        for topic, embs in topic_embs.items():
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    results["intra"].append(cosine_sim(embs[i], embs[j]))
            for other_topic, other_embs in topic_embs.items():
                if other_topic != topic:
                    for e1 in embs:
                        for e2 in other_embs:
                            results["inter"].append(cosine_sim(e1, e2))

        all_results[model_name] = results

        del model, tokenizer
        free_memory()

    # ============================================================
    # Print Results
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SIMILARITY PAIRS: Similar (should be high)")
    print("=" * 70)

    for i, (a, b) in enumerate(SIMILAR_PAIRS):
        scores = " | ".join(f"{name}: {all_results[name]['similar'][i]:.4f}" for name, _ in MODELS)
        print(f"\n  [{i+1}] \"{a[:40]}...\" <-> \"{b[:40]}...\"")
        print(f"      {scores}")

    print(f"\n{'=' * 70}")
    print("SIMILARITY PAIRS: Dissimilar (should be low)")
    print("=" * 70)

    for i, (a, b) in enumerate(DISSIMILAR_PAIRS):
        scores = " | ".join(f"{name}: {all_results[name]['dissimilar'][i]:.4f}" for name, _ in MODELS)
        print(f"\n  [{i+1}] \"{a[:40]}...\" <-> \"{b[:40]}...\"")
        print(f"      {scores}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("EMBEDDING QUALITY SUMMARY")
    print("=" * 70)

    header = f"{'Metric':<25}" + "".join(f"{name:<20}" for name, _ in MODELS)
    print(f"\n{header}")
    print("-" * len(header))

    for name, _ in MODELS:
        r = all_results[name]
        avg_sim = sum(r["similar"]) / len(r["similar"])
        avg_dis = sum(r["dissimilar"]) / len(r["dissimilar"])
        separation = avg_sim - avg_dis
        correct = sum(1 for s in r["similar"] if s > 0.5) + sum(1 for s in r["dissimilar"] if s < 0.5)
        total = len(r["similar"]) + len(r["dissimilar"])
        avg_intra = sum(r["intra"]) / len(r["intra"])
        avg_inter = sum(r["inter"]) / len(r["inter"])

    # Print rows
    metrics = [
        ("Avg Similar", lambda r: sum(r["similar"]) / len(r["similar"])),
        ("Avg Dissimilar", lambda r: sum(r["dissimilar"]) / len(r["dissimilar"])),
        ("Separation", lambda r: sum(r["similar"]) / len(r["similar"]) - sum(r["dissimilar"]) / len(r["dissimilar"])),
        ("Correct Pairs", lambda r: f"{sum(1 for s in r['similar'] if s > 0.5) + sum(1 for s in r['dissimilar'] if s < 0.5)}/{len(r['similar']) + len(r['dissimilar'])}"),
        ("Avg Intra-topic", lambda r: sum(r["intra"]) / len(r["intra"])),
        ("Avg Inter-topic", lambda r: sum(r["inter"]) / len(r["inter"])),
        ("Cluster Margin", lambda r: sum(r["intra"]) / len(r["intra"]) - sum(r["inter"]) / len(r["inter"])),
    ]

    for metric_name, fn in metrics:
        row = f"{metric_name:<25}"
        for name, _ in MODELS:
            val = fn(all_results[name])
            if isinstance(val, float):
                row += f"{val:<20.4f}"
            else:
                row += f"{val:<20}"
        print(row)

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
