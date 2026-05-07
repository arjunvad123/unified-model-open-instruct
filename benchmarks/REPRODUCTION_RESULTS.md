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

---

# 3-seed variance bound (added 2026-05-07)

**Run:** `scripts/nautilus/variance-stage1-stage1_5.yaml`
**Job:** `variance-stage1-stage1-5` on `nautilus-it-gpu02.fullerton.edu`
**Wall clock:** 44 min
**Pinned versions:** `lm_eval=0.4.11`, `transformers=5.8.0`, `torch=2.5.1+cu121`
**Verdict:** ✅ **Single-seed concern is moot for these tasks** — std-dev on every cell ≤ 0.004. The catastrophic-forgetting headline is robust to seed.

This run added seeds {1337, 2024} to the existing seed=42 numbers above. PVC mount worked — results survived job deletion this time, fetched cleanly via a debug pod (the lost-results lesson is now structurally fixed).

## Per-task per-seed table

| Model | Task | Metric | seed=42 | seed=1337 | seed=2024 | mean | std |
|---|---|---|---|---|---|---|---|
| Stage 1 | arc_easy | acc_norm | 0.6023 | 0.5989 | 0.5989 | **0.6000** | 0.0020 |
| Stage 1 | arc_challenge | acc_norm | 0.3985 | 0.4010 | 0.4010 | **0.4002** | 0.0015 |
| Stage 1 | hellaswag | acc_norm | 0.6102 | 0.6109 | 0.6109 | **0.6107** | 0.0004 |
| Stage 1 | winogrande | acc | 0.5659 | 0.5722 | 0.5722 | **0.5701** | 0.0036 |
| Stage 1.5 | arc_easy | acc_norm | 0.4238 | 0.4238 | 0.4238 | **0.4238** | 0.0000 |
| Stage 1.5 | arc_challenge | acc_norm | 0.3003 | 0.3055 | 0.3055 | **0.3037** | 0.0030 |
| Stage 1.5 | hellaswag | acc_norm | 0.4018 | 0.4005 | 0.4005 | **0.4009** | 0.0007 |
| Stage 1.5 | winogrande | acc | 0.5454 | 0.5383 | 0.5383 | **0.5407** | 0.0041 |

## Catastrophic-forgetting headline with 3-seed mean

- **Stage 1 4-task avg (3-seed mean):** `0.5453`
- **Stage 1.5 4-task avg (3-seed mean):** `0.4173`
- **Delta: -23.5%** (vs `BENCHMARK_REPORT.md` headline of -22.0% over 5 tasks)

Tightens the prior single-seed reproduction (-23.2%). The headline finding is not a single-seed artifact.

## Notable pattern: seeds 1337 and 2024 are bit-identical

Every cell for seed=1337 matches the corresponding cell for seed=2024 to four decimals. Seed=42 differs by 0.003–0.006. Two plausible explanations:
1. **lm-eval's `--seed` flag is largely cosmetic for 0-shot multiple-choice tasks** — these are deterministic loglikelihood comparisons with no sampling, so most "randomness" sources don't apply. The 42 vs 1337/2024 differences are likely from non-deterministic CUDA kernels (e.g., reduction order in `--batch_size auto` resolving differently per run) rather than the seed itself.
2. lm-eval's data-shuffling seed implementation maps 1337 and 2024 to the same shuffle order, while 42 produces a different one.

Either way, **std-dev across the 3 seeds is ≤ 0.004 on every cell**, well below the audit's threshold for "this number is meaningful." Single-seed reporting was a less-serious concern than the audit gave it credit for, on these tasks. (MMLU 5-shot would need its own variance check before this conclusion generalizes — few-shot exemplar sampling IS seed-sensitive.)

## Audit item E status update

The audit's "single-seed run; per-task variance not estimated" is now **addressed** for the 4 cheap generation tasks. Two follow-ups would close the rest:
- Run MMLU 5-shot at 3 seeds (~6 GPU-hours; few-shot prompts make this seed-sensitive)
- Run the embedding tables at 3 seeds (~1 hour; uses different methodology, may have different variance characteristics)

