#!/usr/bin/env python3
"""
SOTA Generation Baseline Benchmark (lm-evaluation-harness)
Our Model (3B) vs Top 5 SOTA ~3B Generation Models

Models:
  1. Ours: Arjunvad/unified-model-stage1-action-tokens-v2 (3B)
  2. Qwen/Qwen2.5-3B-Instruct (3B)
  3. Qwen/Qwen2.5-3B (3B)
  4. HuggingFaceTB/SmolLM3-3B (3B)
  5. microsoft/phi-2 (2.7B)
  6. HuggingFaceTB/SmolLM2-1.7B (1.7B)

Benchmarks: ARC-Easy, ARC-Challenge, HellaSwag, Winogrande (0-shot), MMLU (5-shot)

Usage:
  pip install torch transformers accelerate lm-eval numpy
  python baseline_generation_sota.py
"""
import json, os, subprocess, sys
from datetime import datetime

MODELS = [
    ("Ours (3B)",              "Arjunvad/unified-model-stage1-action-tokens-v2", "ours"),
    ("Qwen2.5-3B-Instruct",    "Qwen/Qwen2.5-3B-Instruct",                     "qwen25_3b_inst"),
    ("Qwen2.5-3B",             "Qwen/Qwen2.5-3B",                               "qwen25_3b_base"),
    ("SmolLM3-3B",             "HuggingFaceTB/SmolLM3-3B",                       "smollm3"),
    ("Phi-2 (2.7B)",           "microsoft/phi-2",                                "phi2"),
    ("SmolLM2-1.7B",           "HuggingFaceTB/SmolLM2-1.7B",                     "smollm2"),
]

CORE_TASKS = "arc_easy,arc_challenge,hellaswag,winogrande"
RESULTS_DIR = "results/generation_sota"

all_results = {}

for display_name, model_id, key in MODELS:
    print("\n" + "=" * 70)
    print(f"EVALUATING: {display_name} ({model_id})")
    print("=" * 70)

    model_dir = os.path.join(RESULTS_DIR, key)
    os.makedirs(model_dir, exist_ok=True)

    # Run core tasks (0-shot)
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
        print(f"  WARNING: Core tasks failed for {display_name}")

    # Run MMLU separately (5-shot)
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
        print(f"  WARNING: MMLU failed for {display_name}")

    # Parse results -- lm-eval v0.4+ saves as results_TIMESTAMP.json
    # Metric keys include filter: "acc_norm,none" not "acc_norm"
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

    all_results[key] = {"display_name": display_name, "model_id": model_id, "tasks": model_results}
    print(f"\n  Parsed {len(model_results)} task results for {display_name}")

# ============================================================
# Print comparison table
# ============================================================
print("\n" + "=" * 120)
print("SOTA GENERATION BASELINE BENCHMARK RESULTS")
print("=" * 120)

task_display = [
    ("arc_easy", "acc_norm", "ARC-Easy"),
    ("arc_challenge", "acc_norm", "ARC-Challenge"),
    ("hellaswag", "acc_norm", "HellaSwag"),
    ("winogrande", "acc", "Winogrande"),
]

keys = [k for _, _, k in MODELS]
names = {k: dn for dn, _, k in MODELS}

header = f"{'Task':<18}" + "".join(f"{names[k]:<20}" for k in keys)
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
                        row += f"{val:<20.4f}"
                        avg_scores[k].append(val)
                        found = True
                        break
        if not found:
            row += f"{'N/A':<20}"
    print(row)

# MMLU row
row = f"{'MMLU (5-shot)':<18}"
for k in keys:
    found = False
    if k in all_results:
        tasks = all_results[k].get("tasks", {})
        for tname, tdata in tasks.items():
            if tname.strip().lower() == "mmlu":
                val = tdata.get("acc", tdata.get("acc_norm", None))
                if val is not None:
                    row += f"{val:<20.4f}"
                    avg_scores[k].append(val)
                    found = True
                    break
    if not found:
        row += f"{'N/A':<20}"
print(row)

print("-" * len(header))
row = f"{'AVERAGE':<18}"
for k in keys:
    vals = avg_scores[k]
    if vals:
        row += f"{sum(vals)/len(vals):<20.4f}"
    else:
        row += f"{'N/A':<20}"
print(row)

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "baseline_generation_sota.json"), "w") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "models": all_results}, f, indent=2, default=str)
print(f"\nSaved to {RESULTS_DIR}/baseline_generation_sota.json")
print("=" * 120)
