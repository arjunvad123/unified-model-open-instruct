#!/usr/bin/env python3
"""
Quick test for the trained Unified Model on MPS (no bitsandbytes).
Tests: Generation, Embedding similarity.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

def mean_pool(hidden_states, attention_mask):
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-8)

def get_embedding(model, tokenizer, text, device):
    """Get embedding for a text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        emb = mean_pool(hidden, inputs["attention_mask"])
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("=" * 60)
    print("Testing Unified Model (MPS - no quantization)")
    print("=" * 60)

    base_model = "Qwen/Qwen2.5-7B"
    adapter_path = "/Users/arjunvad/Desktop/SVCL RESEARCH/unified-model-open-instruct/trained_model"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    print(f"Vocab size: {len(tokenizer)}")

    # Check action tokens
    action_tokens = ["<ACT:THINK>", "<ACT:RET>", "<ACT:GEN>", "<ACT:STOP>"]
    print("\nAction tokens:")
    for tok in action_tokens:
        print(f"  {tok}: {tokenizer.convert_tokens_to_ids(tok)}")

    # Load base model in fp16
    print("\nLoading base model (fp16)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

    # Resize embeddings
    target_vocab = 151672
    if base.config.vocab_size != target_vocab:
        print(f"Resizing embeddings: {base.config.vocab_size} -> {target_vocab}")
        base.resize_token_embeddings(target_vocab)

    # Load LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print("Model loaded!")

    # TEST 1: Generation
    print("\n" + "=" * 60)
    print("TEST 1: Generation")
    print("=" * 60)

    prompts = [
        "What is machine learning?",
        "Write a Python function to reverse a string:",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Response: {resp[len(prompt):].strip()[:300]}")

    # TEST 2: Embeddings
    print("\n" + "=" * 60)
    print("TEST 2: Embedding Similarity")
    print("=" * 60)

    pairs = [
        ("Capital of France?", "Which city is France's capital?", "HIGH"),
        ("How to learn coding?", "Best way to start programming?", "HIGH"),
        ("What is Python?", "Recipe for chocolate cake", "LOW"),
    ]

    for t1, t2, expected in pairs:
        e1 = get_embedding(model, tokenizer, t1, device)
        e2 = get_embedding(model, tokenizer, t2, device)
        sim = cosine_sim(e1, e2)
        status = "OK" if (expected == "HIGH" and sim > 0.7) or (expected == "LOW" and sim < 0.5) else "?"
        print(f"  [{status}] '{t1}' <-> '{t2}': {sim:.4f} (expected {expected})")

    # TEST 3: Retrieval
    print("\n" + "=" * 60)
    print("TEST 3: Retrieval Ranking")
    print("=" * 60)

    query = "Benefits of exercise?"
    docs = [
        "Exercise improves cardiovascular health and mental well-being.",
        "The stock market saw volatility this quarter.",
        "Working out reduces stress and improves sleep.",
        "New phones have better cameras.",
    ]

    print(f"Query: {query}")
    q_emb = get_embedding(model, tokenizer, query, device)

    scores = [(cosine_sim(q_emb, get_embedding(model, tokenizer, d, device)), d) for d in docs]
    scores.sort(reverse=True)

    print("Rankings:")
    for i, (s, d) in enumerate(scores, 1):
        print(f"  {i}. [{s:.4f}] {d[:50]}...")

    # Check if relevant docs ranked higher
    relevant = ["exercise", "workout", "health"]
    top_doc = scores[0][1].lower()
    success = any(r in top_doc for r in relevant)
    print(f"\nRetrieval quality: {'PASS' if success else 'FAIL'} - Top doc is {'relevant' if success else 'not relevant'}")

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
