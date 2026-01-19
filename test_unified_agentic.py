#!/usr/bin/env python3
"""
Comprehensive test for the Unified Agentic Model.
Tests all three core capabilities:
1. Embedding/Retrieval - semantic similarity and document retrieval
2. Generation - text generation and instruction following
3. Agentic Routing - action token usage for think/retrieve/generate/stop
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import numpy as np
from typing import List, Tuple
import json

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_PATH = "/Users/arjunvad/Desktop/SVCL RESEARCH/unified-model-open-instruct/trained_model_v100_step500"
BASE_MODEL = "Qwen/Qwen2.5-7B"

# Action tokens
ACTION_TOKENS = {
    "think": "<ACT:THINK>",
    "retrieve": "<ACT:RET>",
    "generate": "<ACT:GEN>",
    "stop": "<ACT:STOP>",
    "wait": "<WAIT>",
    "ret_result": "<RET_RESULT>"
}

# ============================================================================
# Helper Functions
# ============================================================================
def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-8)

def get_embedding(model, tokenizer, text: str, device: str) -> np.ndarray:
    """Get normalized embedding for a text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        emb = mean_pool(hidden, inputs["attention_mask"])
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def generate_text(model, tokenizer, prompt: str, device: str, max_tokens: int = 150) -> str:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)):]

# ============================================================================
# Test 1: Embedding Quality
# ============================================================================
def test_embeddings(model, tokenizer, device: str) -> dict:
    """Test embedding quality through similarity and retrieval tasks."""
    print("\n" + "="*70)
    print("TEST 1: EMBEDDING QUALITY")
    print("="*70)

    results = {"similarity": [], "retrieval": []}

    # 1a. Semantic Similarity Tests
    print("\n--- 1a. Semantic Similarity ---")
    similarity_pairs = [
        # Should be HIGH similarity
        ("What is the capital of France?", "Which city is France's capital?", "high"),
        ("How do I learn Python programming?", "Best way to start coding in Python?", "high"),
        ("Machine learning algorithms", "AI and deep learning techniques", "high"),
        ("The weather is nice today", "It's a beautiful sunny day", "high"),

        # Should be LOW similarity
        ("What is the capital of France?", "How to make chocolate cake?", "low"),
        ("Python programming tutorial", "Ancient Roman history", "low"),
        ("Stock market analysis", "Recipe for pasta", "low"),
        ("Quantum physics equations", "Dog training tips", "low"),
    ]

    high_sims = []
    low_sims = []

    for text1, text2, expected in similarity_pairs:
        emb1 = get_embedding(model, tokenizer, text1, device)
        emb2 = get_embedding(model, tokenizer, text2, device)
        sim = cosine_similarity(emb1, emb2)

        if expected == "high":
            high_sims.append(sim)
            status = "✓" if sim > 0.7 else "✗"
        else:
            low_sims.append(sim)
            status = "✓" if sim < 0.5 else "✗"

        print(f"  [{status}] '{text1[:30]}...' <-> '{text2[:30]}...'")
        print(f"      Similarity: {sim:.4f} (expected {expected})")
        results["similarity"].append({"text1": text1, "text2": text2, "sim": sim, "expected": expected})

    avg_high = np.mean(high_sims)
    avg_low = np.mean(low_sims)
    separation = avg_high - avg_low

    print(f"\n  Summary:")
    print(f"    Avg HIGH similarity: {avg_high:.4f}")
    print(f"    Avg LOW similarity: {avg_low:.4f}")
    print(f"    Separation: {separation:.4f} (want > 0.3)")

    # 1b. Retrieval Test
    print("\n--- 1b. Document Retrieval ---")

    queries_and_docs = [
        {
            "query": "What are the health benefits of regular exercise?",
            "docs": [
                "Regular physical activity improves cardiovascular health, reduces stress, and helps maintain healthy weight.",
                "The stock market experienced significant gains in the technology sector.",
                "Exercise releases endorphins which improve mood and reduce anxiety symptoms.",
                "New smartphone models feature improved camera capabilities.",
                "Studies show that 30 minutes of daily exercise can reduce heart disease risk by 40%.",
            ],
            "relevant": [0, 2, 4]  # indices of relevant docs
        },
        {
            "query": "How does machine learning work?",
            "docs": [
                "Machine learning algorithms learn patterns from data to make predictions.",
                "The best pizza in New York can be found in Brooklyn.",
                "Neural networks are inspired by biological neurons in the brain.",
                "Training ML models requires large datasets and computational resources.",
                "Gardening tips for growing tomatoes in your backyard.",
            ],
            "relevant": [0, 2, 3]
        },
    ]

    retrieval_scores = []
    for item in queries_and_docs:
        query = item["query"]
        docs = item["docs"]
        relevant_indices = set(item["relevant"])

        print(f"\n  Query: '{query[:50]}...'")

        query_emb = get_embedding(model, tokenizer, query, device)
        doc_scores = []
        for i, doc in enumerate(docs):
            doc_emb = get_embedding(model, tokenizer, doc, device)
            score = cosine_similarity(query_emb, doc_emb)
            doc_scores.append((score, i, doc))

        doc_scores.sort(reverse=True)

        # Calculate precision@3
        top3_indices = set([idx for _, idx, _ in doc_scores[:3]])
        precision = len(top3_indices & relevant_indices) / 3
        retrieval_scores.append(precision)

        print(f"  Rankings:")
        for rank, (score, idx, doc) in enumerate(doc_scores, 1):
            is_relevant = "✓" if idx in relevant_indices else " "
            print(f"    {rank}. [{is_relevant}] [{score:.4f}] {doc[:50]}...")
        print(f"  Precision@3: {precision:.2f}")

    avg_precision = np.mean(retrieval_scores)
    results["retrieval_precision"] = avg_precision
    print(f"\n  Average Precision@3: {avg_precision:.2f}")

    return results

# ============================================================================
# Test 2: Generation Quality
# ============================================================================
def test_generation(model, tokenizer, device: str) -> dict:
    """Test text generation quality."""
    print("\n" + "="*70)
    print("TEST 2: GENERATION QUALITY")
    print("="*70)

    results = {"generations": []}

    prompts = [
        # Instruction following
        "Write a short poem about artificial intelligence:",

        # Code generation
        "Write a Python function to check if a number is prime:\n```python",

        # Question answering
        "Explain what machine learning is in simple terms:",

        # Creative writing
        "Write a brief story opening about a robot discovering emotions:",

        # Reasoning
        "If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours? Show your reasoning:",
    ]

    for prompt in prompts:
        print(f"\n--- Prompt: {prompt[:50]}... ---")
        response = generate_text(model, tokenizer, prompt, device)

        # Clean up response for display
        response_clean = response.strip()[:500]
        print(f"Response: {response_clean}")

        results["generations"].append({
            "prompt": prompt,
            "response": response_clean
        })

    return results

# ============================================================================
# Test 3: Agentic Routing (Action Tokens)
# ============================================================================
def test_agentic_routing(model, tokenizer, device: str) -> dict:
    """Test if the model uses action tokens appropriately."""
    print("\n" + "="*70)
    print("TEST 3: AGENTIC ROUTING (Action Tokens)")
    print("="*70)

    results = {"routing": []}

    # Check if action tokens exist in vocabulary
    print("\n--- Action Token Check ---")
    for name, token in ACTION_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        exists = token_id != tokenizer.unk_token_id
        print(f"  {token}: ID={token_id}, exists={exists}")

    # Test scenarios that should trigger different actions
    print("\n--- Routing Scenarios ---")

    scenarios = [
        {
            "prompt": "I need to find information about the latest research on climate change. What should I do?",
            "expected_action": "retrieve",
            "description": "Should trigger retrieval"
        },
        {
            "prompt": "Let me think step by step about how to solve this math problem: 15 * 23 = ?",
            "expected_action": "think",
            "description": "Should trigger thinking"
        },
        {
            "prompt": "Write me a haiku about spring flowers.",
            "expected_action": "generate",
            "description": "Should trigger generation"
        },
        {
            "prompt": "Task completed. The answer is 42. Is there anything else?",
            "expected_action": "stop",
            "description": "Should trigger stop"
        },
    ]

    for scenario in scenarios:
        prompt = scenario["prompt"]
        expected = scenario["expected_action"]
        desc = scenario["description"]

        print(f"\n  Scenario: {desc}")
        print(f"  Prompt: {prompt[:60]}...")

        response = generate_text(model, tokenizer, prompt, device, max_tokens=100)

        # Check which action tokens appear in response
        found_actions = []
        for name, token in ACTION_TOKENS.items():
            if token in response:
                found_actions.append(name)

        print(f"  Response: {response[:200]}...")
        print(f"  Expected action: {expected}")
        print(f"  Found actions: {found_actions if found_actions else 'None'}")

        success = expected in found_actions
        print(f"  Result: {'✓ PASS' if success else '? (may still be valid)'}")

        results["routing"].append({
            "prompt": prompt,
            "expected": expected,
            "found": found_actions,
            "response": response[:200]
        })

    return results

# ============================================================================
# Main
# ============================================================================
def main():
    print("="*70)
    print("UNIFIED AGENTIC MODEL - COMPREHENSIVE TEST")
    print("="*70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Base Model: {BASE_MODEL}")

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load tokenizer from checkpoint (has action tokens)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    print(f"Vocab size: {len(tokenizer)}")

    # Load base model first, then load checkpoint weights
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else {"": device},
        trust_remote_code=True,
    )

    # Resize embeddings to match tokenizer (includes action tokens)
    print(f"Resizing embeddings from {model.config.vocab_size} to {len(tokenizer)}...")
    model.resize_token_embeddings(len(tokenizer))

    # Load checkpoint weights
    print("Loading checkpoint weights...")
    checkpoint_path = f"{CHECKPOINT_PATH}/model.safetensors"
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("Model loaded!")

    # Run tests
    all_results = {}

    # Test 1: Embeddings
    all_results["embeddings"] = test_embeddings(model, tokenizer, device)

    # Test 2: Generation
    all_results["generation"] = test_generation(model, tokenizer, device)

    # Test 3: Agentic Routing
    all_results["agentic"] = test_agentic_routing(model, tokenizer, device)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    emb_results = all_results["embeddings"]
    high_sims = [r["sim"] for r in emb_results["similarity"] if r["expected"] == "high"]
    low_sims = [r["sim"] for r in emb_results["similarity"] if r["expected"] == "low"]

    print(f"\n1. EMBEDDING QUALITY:")
    print(f"   - Avg similar pair score: {np.mean(high_sims):.4f} (want > 0.7)")
    print(f"   - Avg dissimilar pair score: {np.mean(low_sims):.4f} (want < 0.5)")
    print(f"   - Retrieval Precision@3: {emb_results.get('retrieval_precision', 0):.2f}")

    print(f"\n2. GENERATION QUALITY:")
    print(f"   - Generated {len(all_results['generation']['generations'])} responses")
    print(f"   - Review responses above for quality")

    print(f"\n3. AGENTIC ROUTING:")
    routing = all_results["agentic"]["routing"]
    correct = sum(1 for r in routing if r["expected"] in r["found"])
    print(f"   - Correct action tokens: {correct}/{len(routing)}")

    # Save results
    results_path = CHECKPOINT_PATH + "_test_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
