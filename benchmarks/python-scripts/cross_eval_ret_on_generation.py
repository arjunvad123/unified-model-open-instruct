#!/usr/bin/env python3
"""
Cross-Task Eval: Retrieval/Embedding Models on Generation (lm-eval)

Tests whether retrieval-tuned models retain generation ability.
This is the flip side of catastrophic forgetting — if Qwen3-Embedding
models lose generation, it validates our Stage 1.5 findings.

Models tested:
  1. Qwen/Qwen3-Embedding-0.6B  (embedding-tuned, from Qwen3-0.6B)
  2. Qwen/Qwen3-Embedding-4B    (embedding-tuned, from Qwen3-4B)

Reference (generation models, for comparison):
  3. Qwen/Qwen3-4B              (base generation model — same size as Qwen3-Emb-4B)
  4. Qwen/Qwen2.5-3B-Instruct   (our base model — already benchmarked)

Compare against existing generation results:
  - Our Stage 1:   0.5616 avg
  - Our Stage 1.5: 0.4381 avg (-22% catastrophic forgetting)
  - Qwen2.5-3B-Instruct: 0.6630 avg

Benchmarks: ARC-Easy, ARC-Challenge, HellaSwag, Winogrande (0-shot), MMLU (5-shot)

Usage:
  pip install torch transformers accelerate lm-eval numpy
  python cross_eval_ret_on_generation.py

Note: Embedding models are loaded via AutoModelForCausalLM with trust_remote_code.
      If a model can't generate, lm-eval will fail gracefully and we report it.
"""
import json, os, subprocess, sys
from datetime import datetime

# Models to evaluate
# Format: (display_name, model_id, key, notes)
MODELS = [
    # Embedding models (the cross-task experiment)
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", "qwen3_emb_06",
     "Embedding-tuned from Qwen3-0.6B"),

    ("Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-4B", "qwen3_emb_4b",
     "Embedding-tuned from Qwen3-4B"),

    # Generation model reference (same architecture family as Qwen3-Embedding-4B)
    ("Qwen3-4B (gen)", "Qwen/Qwen3-4B", "qwen3_4b_gen",
     "Base generation model — same size, no embedding tuning"),
]

CORE_TASKS = "arc_easy,arc_challenge,hellaswag,winogrande"
RESULTS_DIR = "results/cross_eval"

# Known generation baselines (from previous benchmarks)
KNOWN_BASELINES = {
    "Ours Stage 1 (3B)": {
        "arc_easy": 0.6010, "arc_challenge": 0.3985, "hellaswag": 0.6110,
        "winogrande": 0.5706, "mmlu": 0.6269, "avg": 0.5616,
    },
    "Ours Stage 1.5 (3B)": {
        "arc_easy": 0.4253, "arc_challenge": 0.3029, "hellaswag": 0.4022,
        "winogrande": 0.5451, "mmlu": 0.5148, "avg": 0.4381,
    },
    "Qwen2.5-3B-Instruct": {
        "arc_easy": 0.7306, "arc_challenge": 0.4787, "hellaswag": 0.7289,
        "winogrande": 0.6935, "mmlu": 0.6835, "avg": 0.6630,
    },
}


def run_lm_eval(model_id, key, model_dir):
    """Run lm-eval for core tasks + MMLU. Returns parsed results dict."""
    os.makedirs(model_dir, exist_ok=True)
    model_results = {}

    # Core tasks (0-shot)
    print(f"\n  Running: {CORE_TASKS}")
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_id},trust_remote_code=True,dtype=bfloat16",
        "--tasks", CORE_TASKS,
        "--batch_size", "auto",
        "--output_path", model_dir,
    ]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  WARNING: Core tasks failed for {model_id}")
        # Don't return yet — try MMLU separately

    # MMLU (5-shot)
    print(f"\n  Running: MMLU (5-shot)")
    cmd_mmlu = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_id},trust_remote_code=True,dtype=bfloat16",
        "--tasks", "mmlu",
        "--num_fewshot", "5",
        "--batch_size", "auto",
        "--output_path", model_dir,
    ]
    result = subprocess.run(cmd_mmlu, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  WARNING: MMLU failed for {model_id}")

    # Parse results (lm-eval v0.4+ saves as results_TIMESTAMP.json)
    for root, dirs, files in os.walk(model_dir):
        for fname in sorted(files):
            if fname.startswith("results") and fname.endswith(".json"):
                fpath = os.path.join(root, fname)
                print(f"  Found results file: {fpath}")
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if "results" in data:
                        for task_name, task_data in data["results"].items():
                            flat = {}
                            for mk, mv in task_data.items():
                                clean_key = mk.split(",")[0] if "," in mk else mk
                                flat[clean_key] = mv
                            model_results[task_name] = flat
                except Exception as e:
                    print(f"  Error parsing {fpath}: {e}")

    return model_results


def main():
    all_results = {}

    for display_name, model_id, key, notes in MODELS:
        print("\n" + "=" * 70)
        print(f"EVALUATING: {display_name}")
        print(f"  Model: {model_id}")
        print(f"  Notes: {notes}")
        print("=" * 70)

        model_dir = os.path.join(RESULTS_DIR, key)
        model_results = run_lm_eval(model_id, key, model_dir)

        all_results[key] = {
            "display_name": display_name,
            "model_id": model_id,
            "notes": notes,
            "tasks": model_results,
        }
        print(f"\n  Parsed {len(model_results)} task results for {display_name}")

    # ============================================================
    # Print comparison table
    # ============================================================
    print("\n" + "=" * 140)
    print("CROSS-TASK EVALUATION: Retrieval Models on Generation")
    print("Question: Does embedding tuning destroy generation ability?")
    print("=" * 140)

    task_display = [
        ("arc_easy", "acc_norm", "ARC-Easy"),
        ("arc_challenge", "acc_norm", "ARC-Challenge"),
        ("hellaswag", "acc_norm", "HellaSwag"),
        ("winogrande", "acc", "Winogrande"),
    ]

    keys = [k for _, _, k, _ in MODELS]
    names = {k: dn for dn, _, k, _ in MODELS}
    baseline_keys = list(KNOWN_BASELINES.keys())

    # Build header with both new results and known baselines
    all_keys = keys + baseline_keys
    all_names = {**names, **{k: k for k in baseline_keys}}

    header = f"{'Task':<18}" + "".join(f"{all_names[k][:18]:<20}" for k in all_keys)
    print(f"\n{header}")
    print("-" * len(header))

    avg_scores = {k: [] for k in all_keys}

    for task_key, metric, task_label in task_display:
        row = f"{task_label:<18}"
        for k in all_keys:
            if k in all_results:
                tasks = all_results[k].get("tasks", {})
                found = False
                for tname, tdata in tasks.items():
                    if task_key in tname.lower():
                        val = tdata.get(metric, tdata.get("acc", None))
                        if val is not None:
                            row += f"{val:<20.4f}"
                            avg_scores[k].append(val)
                            found = True
                            break
                if not found:
                    row += f"{'FAILED':<20}"
            elif k in KNOWN_BASELINES:
                val = KNOWN_BASELINES[k].get(task_key, None)
                if val is not None:
                    row += f"{val:<20.4f}"
                    avg_scores[k].append(val)
                else:
                    row += f"{'N/A':<20}"
            else:
                row += f"{'N/A':<20}"
        print(row)

    # MMLU row
    row = f"{'MMLU (5-shot)':<18}"
    for k in all_keys:
        if k in all_results:
            tasks = all_results[k].get("tasks", {})
            found = False
            for tname, tdata in tasks.items():
                if tname.strip().lower() == "mmlu":
                    val = tdata.get("acc", tdata.get("acc_norm", None))
                    if val is not None:
                        row += f"{val:<20.4f}"
                        avg_scores[k].append(val)
                        found = True
                        break
            if not found:
                row += f"{'FAILED':<20}"
        elif k in KNOWN_BASELINES:
            val = KNOWN_BASELINES[k].get("mmlu", None)
            if val is not None:
                row += f"{val:<20.4f}"
                avg_scores[k].append(val)
            else:
                row += f"{'N/A':<20}"
        else:
            row += f"{'N/A':<20}"
    print(row)

    # Averages
    print("-" * len(header))
    row = f"{'AVERAGE':<18}"
    for k in all_keys:
        vals = avg_scores[k]
        if vals:
            row += f"{sum(vals)/len(vals):<20.4f}"
        else:
            row += f"{'N/A':<20}"
    print(row)

    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 140)
    print("KEY FINDINGS")
    print("=" * 140)

    qwen3_gen_scores = avg_scores.get("qwen3_4b_gen", [])
    qwen3_emb_4b_scores = avg_scores.get("qwen3_emb_4b", [])
    qwen3_emb_06_scores = avg_scores.get("qwen3_emb_06", [])

    if qwen3_gen_scores and qwen3_emb_4b_scores:
        gen_avg = sum(qwen3_gen_scores) / len(qwen3_gen_scores)
        emb_avg = sum(qwen3_emb_4b_scores) / len(qwen3_emb_4b_scores)
        delta = ((emb_avg - gen_avg) / gen_avg) * 100
        print(f"\n  Qwen3-4B (gen):          {gen_avg:.4f}")
        print(f"  Qwen3-Embedding-4B:      {emb_avg:.4f}")
        print(f"  Delta (embedding tuning): {'+' if delta > 0 else ''}{delta:.1f}%")
        if delta < -10:
            print("  >>> CONFIRMS: Embedding tuning degrades generation (catastrophic forgetting)")
        elif delta > -5:
            print("  >>> SURPRISING: Embedding tuning preserved generation ability")

    our_stage1_avg = KNOWN_BASELINES["Ours Stage 1 (3B)"]["avg"]
    our_stage15_avg = KNOWN_BASELINES["Ours Stage 1.5 (3B)"]["avg"]
    our_delta = ((our_stage15_avg - our_stage1_avg) / our_stage1_avg) * 100

    print(f"""
  COMPARISON TO OUR MODEL:
  - Our Stage 1 → 1.5: {our_delta:+.1f}% generation degradation from contrastive training
  - If Qwen3 shows similar degradation, our result is NORMAL (not a bug)
  - If Qwen3-Embedding retains generation, their training recipe may be better
  - Key difference: we used LoRA r=32, they likely used full fine-tuning with multitask
    """)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ret_on_generation.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiment": "retrieval_models_on_generation",
            "results": all_results,
            "known_baselines": KNOWN_BASELINES,
        }, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/ret_on_generation.json")
    print("=" * 140)


if __name__ == "__main__":
    main()
