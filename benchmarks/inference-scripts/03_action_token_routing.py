#!/usr/bin/env python3
"""
Action Token Routing Demo: Stage 1 Unified Agentic Model
Shows the model routing queries to <ACT:GEN>, <ACT:RET>, <ACT:TOOL>, <ACT:CODE>.

Model: Arjunvad/unified-model-stage1-action-tokens-v2 (3B)

Usage:
  export HF_TOKEN="your-token"
  python 03_action_token_routing.py
"""
import os
import torch
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

MODEL_ID = "Arjunvad/unified-model-stage1-action-tokens-v2"
ACTION_TOKENS = ["<ACT:GEN>", "<ACT:RET>", "<ACT:TOOL>", "<ACT:CODE>"]

# ============================================
# Test Queries (5 per category)
# ============================================
QUERIES = {
    "GEN": [
        "What is quantum computing?",
        "Explain the theory of relativity in simple terms.",
        "What are the main causes of World War I?",
        "Describe how the human immune system works.",
        "What is the difference between mitosis and meiosis?",
    ],
    "RET": [
        "Find information about the latest developments in fusion energy.",
        "Search for research papers on transformer architectures.",
        "Look up statistics on global renewable energy adoption.",
        "Find articles about CRISPR gene editing applications.",
        "Search for the history of the internet.",
    ],
    "TOOL": [
        "What is the weather in San Francisco right now?",
        "Calculate 1847 * 293 + 456.",
        "What is the current price of Bitcoin?",
        "Convert 72 degrees Fahrenheit to Celsius.",
        "What time is it in Tokyo?",
    ],
    "CODE": [
        "Write a Python function to find all prime numbers up to n.",
        "Implement a binary search algorithm in Python.",
        "Create a Python class for a linked list with append and display methods.",
        "Write a Python function to calculate the edit distance between two strings.",
        "Implement merge sort in Python.",
    ],
}


def generate_response(model, tokenizer, prompt, max_tokens=100):
    """Generate response with action tokens visible."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False).strip()


def extract_action_token(response):
    """Extract action token label from response."""
    for token in ACTION_TOKENS:
        if token in response:
            return token.replace("<ACT:", "").replace(">", "")
    return None


def main():
    print("=" * 70)
    print("ACTION TOKEN ROUTING DEMO")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE).eval()

    # Show token IDs
    print("\nAction tokens in vocabulary:")
    for t in ACTION_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"  {t} -> Token ID {tid}")

    # Build confusion matrix
    confusion = {exp: {pred: 0 for pred in list(QUERIES.keys()) + ["NONE"]} for exp in QUERIES}
    results = []

    for expected, queries in QUERIES.items():
        print(f"\n{'=' * 70}")
        print(f"Category: {expected} (expected: <ACT:{expected}>)")
        print("=" * 70)

        for query in queries:
            response = generate_response(model, tokenizer, query)
            found = extract_action_token(response)
            is_correct = found == expected
            pred_key = found if found else "NONE"
            confusion[expected][pred_key] += 1

            status = "CORRECT" if is_correct else "WRONG"
            print(f"\n  Query: \"{query}\"")
            print(f"  Expected: <ACT:{expected}> | Found: {'<ACT:' + found + '>' if found else 'NONE'} [{status}]")
            print(f"  Response (raw):")
            print(f"  ---")
            # Show first 200 chars of response
            print(f"  {response[:200]}{'...' if len(response) > 200 else ''}")
            print(f"  ---")

            results.append({"query": query, "expected": expected, "found": found, "correct": is_correct})

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ROUTING ACCURACY SUMMARY")
    print("=" * 70)

    print(f"\n{'Category':<12} {'Correct':<10} {'Total':<8} {'Accuracy'}")
    print("-" * 45)

    total_correct = 0
    total_count = 0
    for category in QUERIES:
        cat_results = [r for r in results if r["expected"] == category]
        correct = sum(1 for r in cat_results if r["correct"])
        total = len(cat_results)
        total_correct += correct
        total_count += total
        print(f"{category:<12} {correct:<10} {total:<8} {100*correct/total:.0f}%")

    print("-" * 45)
    print(f"{'Overall':<12} {total_correct:<10} {total_count:<8} {100*total_correct/total_count:.0f}%")

    # Confusion Matrix
    all_labels = list(QUERIES.keys()) + ["NONE"]
    print(f"\nConfusion Matrix:")
    header = f"{'':>12}" + "".join(f"{p:>8}" for p in all_labels)
    print(f"{'':>12} {'Predicted':^{8*len(all_labels)}}")
    print(header)
    print("-" * (12 + 8 * len(all_labels)))

    for expected in QUERIES:
        row = f"{expected:>12}"
        for pred in all_labels:
            row += f"{confusion[expected][pred]:>8}"
        print(row)

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
