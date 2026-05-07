#!/bin/bash
# Fetch results from a completed repro-stage1-base Nautilus job.
#
# CRITICAL: the YAML mounts no PVC for /workspace/results. Pod-ephemeral
# storage. If you delete the Job (or it auto-cleans via ttl), the data is
# GONE. Always run THIS SCRIPT FIRST, then delete. Lost a 32-minute A100
# run once because I deleted the job to re-apply an updated YAML without
# fetching first. Don't repeat.
#
# Run this once `kubectl get job repro-stage1-base -n svcl-self-improve`
# shows COMPLETIONS=1/1. Copies /workspace/results out of the pod into
# results/repro-stage1-base/ and prints the headline reproduction-delta
# table from reproduction_summary.json.
#
# The Job has ttlSecondsAfterFinished=86400 (24h) so the pod is
# garbage-collected a day after completion. Run this before then.
set -euo pipefail

NS=svcl-self-improve
JOB=repro-stage1-base
LOCAL_DIR=results/repro-stage1-base

POD=$(kubectl get pods -n "$NS" -l job-name="$JOB" -o jsonpath='{.items[0].metadata.name}')
if [ -z "$POD" ]; then
  echo "No pod found for job/$JOB in namespace $NS." >&2
  exit 1
fi

echo "pod=$POD"
echo "phase=$(kubectl get pod -n "$NS" "$POD" -o jsonpath='{.status.phase}')"

mkdir -p "$LOCAL_DIR"
echo "Copying /workspace/results -> $LOCAL_DIR ..."
kubectl cp -n "$NS" "$POD:/workspace/results" "$LOCAL_DIR"

SUMMARY="$LOCAL_DIR/reproduction_summary.json"
if [ ! -f "$SUMMARY" ]; then
  echo "Warning: $SUMMARY not found. Listing what we got:" >&2
  find "$LOCAL_DIR" -maxdepth 3 -type f | head
  exit 2
fi

echo
echo "=== reproduction_summary.json highlights ==="
python3 - "$SUMMARY" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
print(f"wall = {s.get('wall_seconds')}s")
cfg = s.get("config", {})
print(f"versions: lm_eval={cfg.get('lm_eval_version')} transformers={cfg.get('transformers_version')} torch={cfg.get('torch_version')}")
print(f"seed={cfg.get('seed')} dtype={cfg.get('dtype')} apply_chat_template={cfg.get('apply_chat_template')}")
print()
print(f"{'Task':<18}{'Metric':<10}{'Ours':<12}{'Report':<10}{'Delta':<10}{'BaseQwen3B-I':<14}")
print("-" * 74)
report_targets = s.get("report_targets", {})
ours = s["models"].get("ours_stage1", {}).get("tasks", {})
base = s["models"].get("base_qwen3b_instruct", {}).get("tasks", {})
def get(tasks, name_substr, metric):
    for k, v in tasks.items():
        if name_substr in k.lower():
            return v.get(metric)
    return None
for task, (metric, report_val) in report_targets.items():
    o = get(ours, task, metric)
    b = get(base, task, metric)
    o_s = f"{o:.4f}" if o is not None else "N/A"
    b_s = f"{b:.4f}" if b is not None else "N/A"
    d_s = f"{(o - report_val):+.4f}" if o is not None else "N/A"
    print(f"{task:<18}{metric:<10}{o_s:<12}{report_val:<10.4f}{d_s:<10}{b_s:<14}")
PY

echo
echo "Done. Inspect:    $LOCAL_DIR/"
echo "Cleanup later:    kubectl delete job/$JOB -n $NS"
