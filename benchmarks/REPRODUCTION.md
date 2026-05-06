# Reproducing the baseline benchmark numbers

This file documents how to reproduce the numbers in
[`baseline_benchmark_report.md`](../baseline_benchmark_report.md) end-to-end.
The intent is that any collaborator (or future-you) can re-run a single
task in one command and get a number to compare against the report.

## Why this matters

The current baseline report does not pin: random seeds, lm-evaluation-harness
commit, dataset revisions, per-task batch size, or chat-template handling for
the Instruct-base model. Until one of the reported numbers is reproduced
from the documented description, every cell in the report carries
"reproducibility unknown" risk. This file gives the missing recipe.

## Pipeline validation (any laptop, ~3 min on MPS)

Confirms the lm-eval CLI and HF download path work locally:

```bash
pip install lm-eval transformers accelerate datasets
lm_eval run \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-0.5B,trust_remote_code=True \
  --tasks arc_easy --num_fewshot 0 --limit 5 \
  --device mps --batch_size 1 \
  --output_path results/repro/pipeline-validation
```

The `--limit 5` result is statistically meaningless (~0.4 ± 0.24) — this is
purely a smoke check that the harness loads, downloads, and writes JSON.
Do not cite the number.

## Single-number reproduction (cluster, ~10 min on one A100)

Reproduce the `Ours (3B) ARC-Easy = 0.6010` cell:

```bash
# from repo root
uv sync
uv run lm_eval run \
  --model hf \
  --model_args pretrained=Arjunvad/unified-model-stage1-action-tokens-v2,trust_remote_code=True,dtype=bfloat16 \
  --tasks arc_easy \
  --num_fewshot 0 \
  --device cuda \
  --batch_size auto \
  --seed 42 \
  --output_path results/repro/arc_easy_seed42 \
  --log_samples
```

Compare `acc_norm` in the output JSON against the report's `0.6010`.
A delta > ±0.01 with seed 42 is a finding worth surfacing before any
new training run starts.

## 3-seed variance bound (~30 min on one A100)

Bounds single-seed noise on 4 of the 5 reported generation tasks:

```bash
for SEED in 42 1337 2024; do
  uv run lm_eval run \
    --model hf \
    --model_args pretrained=Arjunvad/unified-model-stage1-action-tokens-v2,trust_remote_code=True,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande \
    --num_fewshot 0 \
    --device cuda \
    --batch_size auto \
    --seed $SEED \
    --output_path results/repro/seed${SEED} \
    --log_samples
done
```

MMLU 5-shot is a separate ~2h × 3 seed run.

## Base-model contrast (necessary for any Stage-1 contribution claim)

Run the same suite on the un-fine-tuned base so deltas can be attributed:

```bash
uv run lm_eval run \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,trust_remote_code=True,dtype=bfloat16 \
  --tasks arc_easy,arc_challenge,hellaswag,winogrande \
  --num_fewshot 0 \
  --device cuda \
  --batch_size auto \
  --seed 42 \
  --output_path results/repro/base_qwen3b_seed42 \
  --log_samples
```

## Knobs the original report does not pin — record these when running

| Knob | Suggested default | Why it matters |
|---|---|---|
| Random seed | `--seed 42` | lm-eval silent defaults are `random=0`, `numpy=torch=fewshot=1234`; report didn't specify |
| Model dtype | `dtype=bfloat16` in `--model_args` | Report says bf16 in appendix; confirm by inspecting harness logs |
| Chat template | `--apply_chat_template` (or omit, but document choice) | Stage 1 is built on Qwen2.5-3B-**Instruct**. Original report does not state whether the chat template was applied; `acc_norm` can shift several points either way |
| Harness commit | record `lm_eval --version` | Report says only "v0.4+"; metric implementations have changed within v0.4.x |
| Dataset revision | record from harness output | HF dataset revisions can change |
| Batch size | `--batch_size auto` | Affects nothing for `multiple_choice` tasks but record anyway |

## Expected runtime budget

| Task | Single seed, one A100 |
|---|---|
| arc_easy | ~5 min |
| arc_challenge | ~5 min |
| hellaswag | ~10 min |
| winogrande | ~5 min |
| mmlu (5-shot) | ~2 h |

Times for full sets, no `--limit`. MPS on a laptop is 5–10× slower and is
not recommended for anything beyond pipeline validation.
