#!/usr/bin/env python3
"""
Generation Benchmark: Stage 1.5 (Ours, 3B) vs Qwen2.5-3B-Instruct vs SmolLM3-3B
lm-evaluation-harness: ARC-Easy, ARC-Challenge, HellaSwag, Winogrande, MMLU

Usage:
  pip install torch transformers accelerate lm-eval numpy
  python eval_stage1_5_generation.py
"""
import json, os, subprocess, sys, re
from datetime import datetime

MODELS = [
    ("Stage 1.5 (Ours, 3B)", "Arjunvad/unified-model-stage1-5-embedding-v2", "stage1_5"),
    ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "qwen25_3b"),
    ("SmolLM3-3B", "HuggingFaceTB/SmolLM3-3B", "smollm3_3b"),
]

TASKS = "arc_easy,arc_challenge,hellaswag,winogrande"
MMLU_TASK = "mmlu"
RESULTS_DIR = "results/stage1_5_generation"

all_results = {}

for display_name, model_id, key in MODELS:
    print("\n" + "=" * 70)
    print(f"EVALUATING: {display_name} ({model_id})")
    print("=" * 70)

    model_dir = os.path.join(RESULTS_DIR, key)
    os.makedirs(model_dir, exist_ok=True)

    # Run core tasks (ARC, HellaSwag, Winogrande)
    print(f"\n  Running: {TASKS}")
    cmd_core = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_id},trust_remote_code=True,dtype=bfloat16",
        "--tasks", TASKS,
        "--batch_size", "8",
        "--output_path", model_dir,
        "--log_samples",
    ]
    result = subprocess.run(cmd_core, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  WARNING: Core tasks failed for {display_name}")

    # Run MMLU separately (memory intensive)
    print(f"\n  Running: {MMLU_TASK}")
    cmd_mmlu = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_id},trust_remote_code=True,dtype=bfloat16",
        "--tasks", MMLU_TASK,
        "--num_fewshot", "5",
        "--batch_size", "4",
        "--output_path", model_dir,
        "--log_samples",
    ]
    result = subprocess.run(cmd_mmlu, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  WARNING: MMLU failed for {display_name}")

    # Parse results from output files
    # lm-eval v0.4+ saves as results_TIMESTAMP.json with "acc_norm,none" keys
    model_results = {}
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
                            # Flatten metric keys: "acc_norm,none" -> "acc_norm"
                            flat = {}
                            for mk, mv in task_data.items():
                                clean_key = mk.split(",")[0] if "," in mk else mk
                                flat[clean_key] = mv
                            model_results[task_name] = flat
                except Exception as e:
                    print(f"  Error parsing {fpath}: {e}")

    all_results[key] = {
        "display_name": display_name,
        "model_id": model_id,
        "tasks": model_results,
    }
    print(f"\n  Parsed {len(model_results)} task results for {display_name}")

# ============================================================
# Print comparison table
# ============================================================
print("\n" + "=" * 100)
print("GENERATION BENCHMARK RESULTS")
print("=" * 100)

task_display = [
    ("arc_easy", "acc_norm", "ARC-Easy"),
    ("arc_challenge", "acc_norm", "ARC-Challenge"),
    ("hellaswag", "acc_norm", "HellaSwag"),
    ("winogrande", "acc", "Winogrande"),
]

keys = ["stage1_5", "qwen25_3b", "smollm3_3b"]
names = {"stage1_5": "Stage 1.5 (Ours)", "qwen25_3b": "Qwen2.5-3B", "smollm3_3b": "SmolLM3-3B"}

header = f"{'Task':<18}" + "".join(f"{names[k]:<22}" for k in keys)
print(f"\n{header}")
print("-" * len(header))

avg_scores = {k: [] for k in keys}

for task_key, metric, task_label in task_display:
    row = f"{task_label:<18}"
    for k in keys:
        found = False
        if k in all_results:
            tasks = all_results[k].get("tasks", {})
            for tname, tdata in tasks.items():
                if task_key in tname.lower():
                    val = tdata.get(metric, tdata.get("acc", None))
                    if val is not None:
                        row += f"{val:<22.4f}"
                        avg_scores[k].append(val)
                        found = True
                        break
        if not found:
            row += f"{'N/A':<22}"
    print(row)

# MMLU
row = f"{'MMLU (5-shot)':<18}"
for k in keys:
    found = False
    if k in all_results:
        tasks = all_results[k].get("tasks", {})
        for tname, tdata in tasks.items():
            if tname.strip().lower() == "mmlu":
                val = tdata.get("acc", tdata.get("acc_norm", None))
                if val is not None:
                    row += f"{val:<22.4f}"
                    avg_scores[k].append(val)
                    found = True
                    break
    if not found:
        row += f"{'N/A':<22}"
print(row)

print("-" * len(header))
row = f"{'AVERAGE':<18}"
for k in keys:
    vals = avg_scores[k]
    if vals:
        row += f"{sum(vals)/len(vals):<22.4f}"
    else:
        row += f"{'N/A':<22}"
print(row)

# Save full results
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "generation_benchmark.json"), "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "models": {k: v for k, v in all_results.items()},
    }, f, indent=2, default=str)

print(f"\nSaved to {RESULTS_DIR}/generation_benchmark.json")
print("=" * 100)
