# Reproduction results — `BENCHMARK_REPORT.md` Sections 1 + 3

**Run date:** 2026-05-07 02:15–02:51 UTC
**Cluster:** Nautilus, single A100 on `gpu-13.nrp.mghpcc.org`
**Wall clock:** 35 min 42 s
**Job spec:** `scripts/nautilus/repro-stage1-with-base-baseline.yaml`
**Verdict:** ✅ **REPRODUCES.** All Stage 1 + Stage 1.5 cells within ±0.005 of the published numbers; one base-model outlier (hellaswag, +0.0210) flagged below.

## Reproduction-delta table

| Task          | Metric    | Stage 1 repro | S1 report | Δ        | Stage 1.5 repro | S1.5 report | Δ        | Base Qwen3B-I repro | Base report | Δ        |
|---------------|-----------|---------------|-----------|----------|-----------------|-------------|----------|----------------------|-------------|----------|
| arc_easy      | acc_norm  | 0.6023        | 0.6010    | +0.0013  | 0.4238          | 0.4253      | -0.0015  | 0.7285               | 0.7306      | -0.0021  |
| arc_challenge | acc_norm  | 0.3985        | 0.3985    | +0.0000  | 0.3003          | 0.3029      | -0.0026  | 0.4753               | 0.4787      | -0.0034  |
| hellaswag     | acc_norm  | 0.6102        | 0.6110    | -0.0008  | 0.4018          | 0.4022      | -0.0004  | **0.7499**           | **0.7289**  | **+0.0210** |
| winogrande    | acc       | 0.5659        | 0.5706    | -0.0047  | 0.5454          | 0.5451      | +0.0003  | 0.6914               | 0.6935      | -0.0021  |
| **4-task avg** |          | **0.5442**    | **0.5453**| -0.0011  | **0.4178**      | **0.4189**  | -0.0011  | **0.6613**           | **0.6579**  | +0.0034  |

## Catastrophic-forgetting headline (the report's primary finding)

- **Reproduction:** Stage 1 4-task avg `0.5442` → Stage 1.5 4-task avg `0.4178` = **-23.2%**
- **Report:** -22.0% across 5 tasks (the report's average includes MMLU which we omitted)

The headline finding holds. Slightly larger drop on this 4-task subset (-23.2% vs -22.0% over 5) is consistent with MMLU degrading less than the average — Stage 1.5 still has Qwen2.5-3B-Instruct's MMLU strength partially intact.

## Run config (pinned for future re-runs)

```
seed                 = 42
dtype                = bfloat16
batch_size           = auto (resolved to 64)
apply_chat_template  = false   (matches existing baseline-generation-lmeval.yaml)
lm_eval              = 0.4.11
transformers         = >=4.51.0
torch                = cu121 build
cluster              = Nautilus / gpu-13.nrp.mghpcc.org / NVIDIA-A100
```

## Open issues from this run

1. **hellaswag base-Qwen +0.0210 delta.** Only material discrepancy. Most likely lm-eval version drift (this run pinned `0.4.11`; the report says only "v0.4+"). Worth confirming with Arjun what version the report used. If different, re-running base-Qwen on the report's pinned version would close the gap.
2. **Chat template was NOT applied** — matches `baseline-generation-lmeval.yaml` methodology. Stage 1's base is `Qwen2.5-3B-Instruct` though, which has a chat template. Optional follow-up run with `--apply_chat_template` to see how much it shifts the Instruct numbers; not blocking.
3. **MMLU 5-shot omitted** to keep this run under 1 h. Adding MMLU would tighten the catastrophic-forgetting headline (-23.2% → ?% on the full 5-task set). ~6 GPU-hours (3 models × ~2 h MMLU each). Not blocking; nice-to-have.

## Postmortem note

The run's `/workspace/results/reproduction_summary.json` was unrecoverable
(pod-ephemeral storage; `kubectl exec` blocked on `Succeeded` pods; no PVC
mounted). Numbers above were extracted from `kubectl logs` (lm-eval prints
its own per-model results table at the end of each evaluation). Raw
per-sample JSONL not preserved.

**Fix for next time:** mount the project's existing `pvc-data.yaml` (CephFS,
200 GB) at `/workspace/results` in the YAML so results survive pod
deletion. Will propose to Arjun as part of the PR.
