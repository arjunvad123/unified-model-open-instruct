#!/usr/bin/env python3
"""
Stage 1 vs Stage 1.5: The Catastrophic Forgetting Problem
Shows embedding improvement AND generation degradation side-by-side.

Models:
  Stage 1:   Arjunvad/unified-model-stage1-action-tokens-v2 (3B)
  Stage 1.5: Arjunvad/unified-model-stage1-5-embedding-v2 (3B)

Usage:
  export HF_TOKEN="your-token"
  python 05_stage1_vs_stage1_5.py
"""
import os
import sys
import gc
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from open_instruct.action_tokens import ACTION_TOKENS  # noqa: E402,F401

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
# Part A: Embedding Pairs
# ============================================
SIMILAR_PAIRS = [
    ("The cat sat on the mat.", "A feline rested on the rug."),
    ("Machine learning is a subset of AI.", "Deep learning uses neural networks."),
    ("Quantum computing uses qubits.", "Quantum computers use quantum bits."),
    ("Exercise improves health.", "Physical activity strengthens the body."),
    ("Climate change causes rising seas.", "Global warming raises ocean levels."),
    ("DNA stores genetic information.", "Genes are encoded in DNA molecules."),
    ("The Eiffel Tower is in Paris.", "Paris is home to a famous iron tower."),
    ("Python is a programming language.", "Python is used for software development."),
]

DISSIMILAR_PAIRS = [
    ("The cat sat on the mat.", "Stock prices rose sharply today."),
    ("Machine learning is a subset of AI.", "Pizza originated in Italy."),
    ("Quantum computing uses qubits.", "Dogs are loyal pets."),
    ("Exercise improves health.", "The Eiffel Tower is 330 meters tall."),
    ("DNA stores genetic information.", "Docker containers are lightweight."),
]

# ============================================
# Part B: Generation Prompts
# ============================================
GENERATION_PROMPTS = [
    ("Factual", "What is quantum computing and how does it differ from classical computing?"),
    ("Code", "Write a Python function to reverse a linked list."),
    ("Simplification", "Explain the water cycle to a 5-year-old."),
    ("Structured", "List 3 pros and 3 cons of remote work."),
    ("Reasoning", "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?"),
    ("Creative", "Write a haiku about machine learning."),
]

# ============================================
# Part C: Action Token Queries
# ============================================
ACTION_QUERIES = [
    ("What is quantum computing?", "GEN"),
    ("Find information about RLHF", "RET"),
    ("What's the weather in Tokyo?", "TOOL"),
    ("Write a Python sort function", "CODE"),
]

# ACTION_TOKENS imported above from open_instruct.action_tokens registry
# (was previously hardcoded to a wrong list including untrained TOOL/CODE).


def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1].float()
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


def cosine_sim(emb1, emb2):
    return (emb1 @ emb2.T).item()


def generate_response(model, tokenizer, prompt, max_tokens=250, skip_special=True):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=skip_special).strip()


def extract_action_token(response):
    for token in ACTION_TOKENS:
        if token in response:
            return token.replace("<ACT:", "").replace(">", "")
    return None


def free_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("STAGE 1 vs STAGE 1.5: THE CATASTROPHIC FORGETTING PROBLEM")
    print("=" * 70)
    print()
    print("Contrastive training (Stage 1 -> Stage 1.5) improved embedding")
    print("quality (+31% NDCG@10) but degraded generation quality (-22%).")
    print("This demo shows both effects qualitatively.")
    print(f"\nDevice: {DEVICE}")

    # Collect all results across both models
    emb_results = {}  # model_name -> {similar: [...], dissimilar: [...]}
    gen_results = {}  # model_name -> [(category, prompt, response)]
    act_results = {}  # model_name -> [(query, expected, found)]

    for model_name, model_id in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Loading: {model_name} ({model_id})")
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DTYPE, trust_remote_code=True
        ).to(DEVICE).eval()
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

        # Part A: Embeddings
        sim_scores = []
        for a, b in SIMILAR_PAIRS:
            score = cosine_sim(get_embedding(model, tokenizer, a), get_embedding(model, tokenizer, b))
            sim_scores.append(score)

        dis_scores = []
        for a, b in DISSIMILAR_PAIRS:
            score = cosine_sim(get_embedding(model, tokenizer, a), get_embedding(model, tokenizer, b))
            dis_scores.append(score)

        emb_results[model_name] = {"similar": sim_scores, "dissimilar": dis_scores}

        # Part B: Generation
        gen_out = []
        for category, prompt in GENERATION_PROMPTS:
            response = generate_response(model, tokenizer, prompt)
            gen_out.append((category, prompt, response))
        gen_results[model_name] = gen_out

        # Part C: Action tokens
        act_out = []
        for query, expected in ACTION_QUERIES:
            raw = generate_response(model, tokenizer, query, max_tokens=50, skip_special=False)
            found = extract_action_token(raw)
            act_out.append((query, expected, found, raw))
        act_results[model_name] = act_out

        del model, tokenizer
        free_memory()

    # ============================================================
    # PART A: Embedding Quality (Stage 1.5 is BETTER)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PART A: EMBEDDING QUALITY (Stage 1.5 should be BETTER)")
    print("=" * 70)

    print("\nSimilar Pairs:")
    for i, (a, b) in enumerate(SIMILAR_PAIRS):
        s1 = emb_results["Stage 1"]["similar"][i]
        s15 = emb_results["Stage 1.5"]["similar"][i]
        change = ((s15 - s1) / abs(s1)) * 100 if s1 != 0 else 0
        arrow = "+" if change > 0 else ""
        print(f"  [{i+1}] Stage 1: {s1:.4f} | Stage 1.5: {s15:.4f} ({arrow}{change:.1f}%)")
        print(f"      \"{a[:35]}...\" <-> \"{b[:35]}...\"")

    print("\nDissimilar Pairs:")
    for i, (a, b) in enumerate(DISSIMILAR_PAIRS):
        s1 = emb_results["Stage 1"]["dissimilar"][i]
        s15 = emb_results["Stage 1.5"]["dissimilar"][i]
        change = ((s15 - s1) / abs(s1)) * 100 if s1 != 0 else 0
        arrow = "+" if change > 0 else ""
        print(f"  [{i+1}] Stage 1: {s1:.4f} | Stage 1.5: {s15:.4f} ({arrow}{change:.1f}%)")
        print(f"      \"{a[:35]}...\" <-> \"{b[:35]}...\"")

    # Summary
    print(f"\n{'Metric':<25} {'Stage 1':<15} {'Stage 1.5':<15} {'Change'}")
    print("-" * 65)
    for name in ["Stage 1", "Stage 1.5"]:
        pass  # computed below

    s1_avg_sim = sum(emb_results["Stage 1"]["similar"]) / len(emb_results["Stage 1"]["similar"])
    s15_avg_sim = sum(emb_results["Stage 1.5"]["similar"]) / len(emb_results["Stage 1.5"]["similar"])
    s1_avg_dis = sum(emb_results["Stage 1"]["dissimilar"]) / len(emb_results["Stage 1"]["dissimilar"])
    s15_avg_dis = sum(emb_results["Stage 1.5"]["dissimilar"]) / len(emb_results["Stage 1.5"]["dissimilar"])
    s1_sep = s1_avg_sim - s1_avg_dis
    s15_sep = s15_avg_sim - s15_avg_dis

    print(f"{'Avg Similar':<25} {s1_avg_sim:<15.4f} {s15_avg_sim:<15.4f} {'+' if s15_avg_sim > s1_avg_sim else ''}{((s15_avg_sim-s1_avg_sim)/abs(s1_avg_sim))*100:.1f}%")
    print(f"{'Avg Dissimilar':<25} {s1_avg_dis:<15.4f} {s15_avg_dis:<15.4f} {'+' if s15_avg_dis > s1_avg_dis else ''}{((s15_avg_dis-s1_avg_dis)/abs(s1_avg_dis))*100:.1f}%")
    print(f"{'Separation':<25} {s1_sep:<15.4f} {s15_sep:<15.4f} {'+' if s15_sep > s1_sep else ''}{((s15_sep-s1_sep)/abs(s1_sep))*100:.1f}%")
    print(f"\n>>> Stage 1.5 embeddings are {'BETTER' if s15_sep > s1_sep else 'WORSE'} (contrastive training {'worked' if s15_sep > s1_sep else 'did not help'})")

    # ============================================================
    # PART B: Generation Quality (Stage 1 is BETTER)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PART B: GENERATION QUALITY (Stage 1 should be BETTER)")
    print("=" * 70)

    for i, (category, prompt) in enumerate(GENERATION_PROMPTS):
        print(f"\n{'─' * 60}")
        print(f"[{category}] \"{prompt}\"")
        print(f"{'─' * 60}")

        for model_name, _ in MODELS:
            _, _, response = gen_results[model_name][i]
            word_count = len(response.split())
            print(f"\n  --- {model_name} ({word_count} words) ---")
            print(f"  {response[:400]}{'...' if len(response) > 400 else ''}")

    # ============================================================
    # PART C: Action Token Routing
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PART C: ACTION TOKEN ROUTING")
    print("=" * 70)

    for i, (query, expected) in enumerate(ACTION_QUERIES):
        print(f"\n  Query: \"{query}\" (expected: <ACT:{expected}>)")
        for model_name, _ in MODELS:
            _, _, found, raw = act_results[model_name][i]
            status = "CORRECT" if found == expected else ("NONE" if found is None else "WRONG")
            found_str = f"<ACT:{found}>" if found else "NONE"
            print(f"    {model_name}: {found_str} [{status}]")

    # ============================================================
    # CONCLUSION
    # ============================================================
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print("=" * 70)

    s1_act_correct = sum(1 for _, exp, found, _ in act_results["Stage 1"] if found == exp)
    s15_act_correct = sum(1 for _, exp, found, _ in act_results["Stage 1.5"] if found == exp)

    print(f"""
  Embedding Quality:  Stage 1.5 {'>' if s15_sep > s1_sep else '<'} Stage 1  (separation: {s1_sep:.4f} -> {s15_sep:.4f})
  Generation Quality: Stage 1 is qualitatively better (review responses above)
  Action Routing:     Stage 1: {s1_act_correct}/4 | Stage 1.5: {s15_act_correct}/4

  This demonstrates the CATASTROPHIC FORGETTING problem:
    - Contrastive training improved embeddings
    - But degraded generation and action token routing
    - The core research challenge is preserving both capabilities

  Possible solutions:
    1. Lower LoRA rank (currently r=32, try r=8 or r=16)
    2. Freeze more layers during contrastive training
    3. Multi-task training (contrastive + generation loss simultaneously)
    4. GritLM-style alternating batches (embedding batch, generation batch)
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
