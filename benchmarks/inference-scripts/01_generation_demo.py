#!/usr/bin/env python3
"""
Generation Demo: Stage 1 Unified Agentic Model
Shows model generation across diverse prompt categories.

Model: Arjunvad/unified-model-stage1-action-tokens-v2 (3B)

Usage:
  export HF_TOKEN="your-token"
  python 01_generation_demo.py
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

# ============================================
# HuggingFace Auth
# ============================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
else:
    print("WARNING: HF_TOKEN not set. Set it with: export HF_TOKEN='your-token'")

MODEL_ID = "Arjunvad/unified-model-stage1-action-tokens-v2"

# ============================================
# Prompts by Category
# ============================================
PROMPTS = {
    "General Knowledge": [
        "What causes the seasons on Earth?",
        "Explain how vaccines work.",
        "What is the greenhouse effect?",
        "How does gravity work?",
    ],
    "Reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? Explain your reasoning.",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "There are three boxes. One contains only apples, one contains only oranges, and one contains both. The labels are all wrong. You can pick one fruit from one box. How do you label all boxes correctly?",
    ],
    "Code Generation": [
        "Write a Python function to find the nth Fibonacci number.",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python function to merge two sorted lists into one sorted list.",
        "Write a Python class for a stack with push, pop, and peek methods.",
    ],
    "Instruction Following": [
        "List exactly 5 benefits of regular exercise, numbered 1 through 5.",
        "Summarize the concept of photosynthesis in exactly 2 sentences.",
        "Write a haiku about the ocean.",
        "Explain recursion to a 10-year-old in 3 sentences or less.",
    ],
}


def generate_response(model, tokenizer, prompt, max_tokens=300):
    """Generate a response using chat template."""
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
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    return response.strip()


def extract_action_token(response):
    """Extract action token from response."""
    for token in ["<ACT:GEN>", "<ACT:RET>", "<ACT:TOOL>", "<ACT:CODE>"]:
        if token in response:
            return token
    return None


def main():
    print("=" * 70)
    print("GENERATION DEMO")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE} ({DTYPE})")
    print("=" * 70)

    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE).eval()
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    # Check action tokens
    action_tokens = ["<ACT:GEN>", "<ACT:RET>", "<ACT:TOOL>", "<ACT:CODE>"]
    print("\nAction tokens in vocabulary:")
    for t in action_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        print(f"  {t} -> ID {tid}")

    all_responses = []

    for category, prompts in PROMPTS.items():
        print(f"\n{'=' * 70}")
        print(f"Category: {category}")
        print("=" * 70)

        for prompt in prompts:
            response = generate_response(model, tokenizer, prompt)
            action = extract_action_token(response)

            print(f"\n{'─' * 60}")
            print(f"Prompt: \"{prompt}\"")
            if action:
                print(f"Action Token: {action}")
            print(f"{'─' * 60}")
            print(f"Response:\n{response}")

            all_responses.append({
                "category": category,
                "prompt": prompt,
                "action_token": action,
                "response": response,
                "length": len(response.split()),
            })

    # Summary
    print(f"\n{'=' * 70}")
    print("GENERATION DEMO SUMMARY")
    print("=" * 70)

    print(f"\n{'Category':<25} {'Prompts':<10} {'Avg Words':<12} {'Action Tokens'}")
    print("-" * 70)

    for category in PROMPTS:
        cat_responses = [r for r in all_responses if r["category"] == category]
        avg_len = sum(r["length"] for r in cat_responses) / len(cat_responses)
        actions = [r["action_token"] for r in cat_responses if r["action_token"]]
        action_str = ", ".join(set(a for a in actions)) if actions else "none"
        print(f"{category:<25} {len(cat_responses):<10} {avg_len:<12.0f} {action_str}")

    total = len(all_responses)
    with_action = sum(1 for r in all_responses if r["action_token"])
    print(f"\nTotal: {total} prompts, {with_action} with action tokens ({100*with_action/total:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
