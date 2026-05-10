# Reproducing the benchmark numbers

This file documents how to reproduce the numbers in
[`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) end-to-end. The intent is that
any collaborator (or future-you) can re-run a single task with one command
and get a number to compare against the report.

(The earlier [`baseline_benchmark_report.md`](../baseline_benchmark_report.md)
at the repo root is a partial subset — kept for history. Treat
`benchmarks/BENCHMARK_REPORT.md` as the canonical version.)

## Why this matters

The published reports do not pin: random seeds, lm-evaluation-harness commit,
dataset revisions, per-task batch size, or chat-template handling for the
Instruct-base model. Until one of the reported numbers is reproduced from the
documented description, every cell carries "reproducibility unknown" risk.
This file gives the missing recipe.

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

The `--limit 5` result is statistically meaningless — this is purely a smoke
check that the harness loads, downloads, and writes JSON. **Do not cite the
number.**

## Single-cell reproduction (cluster, ~10 min on one A100)

Reproduce the `Ours (Stage 1) ARC-Easy = 0.6010` cell from `BENCHMARK_REPORT.md`
Section 1:

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

Compare `acc_norm` in the output JSON against the report's `0.6010`. A delta
> ±0.01 with seed 42 is a finding worth surfacing before any new training run.

## Stage-1.5 catastrophic-forgetting reproduction (~20 min on one A100)

The headline finding of the report is **Stage 1 → Stage 1.5 generation
degradation of -22%** (BENCHMARK_REPORT.md Section 3). Verify both endpoints:

```bash
for MODEL in \
    Arjunvad/unified-model-stage1-action-tokens-v2 \
    Arjunvad/unified-model-stage1-5-embedding-v2; do
  uv run lm_eval run \
    --model hf \
    --model_args pretrained=$MODEL,trust_remote_code=True,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande \
    --num_fewshot 0 \
    --device cuda \
    --batch_size auto \
    --seed 42 \
    --output_path results/repro/forgetting/$(basename $MODEL) \
    --log_samples
done
```

The 4-task average for Stage 1 should be ~0.5453 (excluding MMLU), Stage 1.5
~0.4189 — i.e. roughly the same -22% drop the report claims. Differences
> ±0.01 per task suggest the published number was on a different seed or
config and we should re-derive the headline.

## Retrieval reproduction (BEIR via eval matrix)

**Canonical retrieval recipe** for any Stage 1.5 checkpoint
(`Arjunvad/unified-model-stage1-5` or
`Arjunvad/unified-model-stage1-5-embedding-v2`):

| Knob | Value |
|---|---|
| Query prefix | `<ACT:RET> ` (the trained routing token + a single space) |
| Document prefix | `""` (no prefix) |
| Pooling | mean over last hidden states, with attention mask |
| Normalization | L2-normalize before dot product |
| Max length | 512 tokens |
| Score | normalized embedding dot product (= cosine on unit vectors) |
| Dtype | bfloat16 |

**Why this recipe.** The 72-cell prefix × pooling × model matrix
(`scripts/nautilus/eval-matrix-stage1_5.yaml`) shows `act_ret + mean`
wins decisively:
- Wins for both Stage 1.5 v1 (full FT) and v2 (LoRA-merged)
- Wins ≥3/4 of NFCorpus / SciFact / ArguAna / SCIDOCS for each model
- Beats `qwen_instruction + mean` by +0.08 average nDCG@10 on v1
  (+32% relative) and +0.20 on v2 (+149% relative)
- Beats Stage 1 with the same recipe by +0.16 average nDCG@10 — the
  contrastive training genuinely teaches the routing-token protocol;
  this is not generic prefix geometry

**Headline numbers under the canonical recipe** (from the matrix run
that landed 2026-05-10 — `eval-matrix-stage1-5` job, eval_matrix.json):

| Model | NFCorpus | SciFact | ArguAna | SCIDOCS | Avg nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Stage 1 (action-token SFT) | 0.0329 | 0.3140 | 0.3340 | 0.0109 | 0.1730 |
| Stage 1.5 v1 (full FT) | **0.2635** | **0.6238** | 0.3163 | **0.1372** | **0.3352** |
| Stage 1.5 v2 (LoRA-merged) | **0.2738** | 0.5337 | **0.3191** | **0.1821** | **0.3272** |

**Caveats / what's not yet validated:**
- `hits@1` on ArguAna is ~0 across all v1/v2 recipes (persistent
  across the matrix). Likely a task structural quirk
  (counter-argument retrieval) rather than a model calibration issue,
  but worth a per-query rank dump before publishing.
- These absolute numbers are **not yet anchor-validated**. Job
  `eval-anchor-qwen3-emb-0-6b` runs Qwen3-Embedding-0.6B through the
  same loader/metric code; if Qwen3-Emb-0.6B lands within ~5 nDCG@10
  points of its published BEIR numbers under our pipeline, the
  numbers above are credible. If it lands much lower, the entire
  matrix is uninterpretable and these numbers should not be cited.
  Update this section with anchor results once the job lands.

**Reproduce on Nautilus:**

```bash
# Launch the matrix (3 models × 6 recipes × 4 tasks, ~3h on one A100)
kubectl apply -f scripts/nautilus/eval-matrix-stage1_5.yaml -n svcl-self-improve

# Watch state
kubectl get job eval-matrix-stage1-5 -n svcl-self-improve -w

# Once Succeeded, fetch the eval_matrix.json
bash scripts/nautilus/fetch_pvc_results.sh eval-matrix-stage1-5
# Results land in results/eval-matrix-stage1-5/

# To reproduce only the canonical recipe (skip the other 5),
# launch a smaller variant: edit QUERY_PREFIXES and POOLINGS in the
# inline Python to keep only "act_ret" + "mean".
```

**Reproduce a single (model, task) cell locally** (without the full
matrix infrastructure — useful for sanity-checking on an A10G or
laptop with an MPS device):

```bash
# After installing torch + transformers + mteb + numpy:
# Pseudo-snippet — full path is in scripts/nautilus/eval-matrix-stage1_5.yaml
# under the inline run_eval_matrix.py. The canonical recipe just calls
# encode_both_poolings() with prefix="<ACT:RET> " and reads .mean from
# the result, then compute_metrics() against MTEB BEIR test split qrels.
```

For the lower-level Python that powers each cell, see the inline
`run_eval_matrix.py` block in `scripts/nautilus/eval-matrix-stage1_5.yaml`
(lines ~71-460). It is the single source of truth for the loader,
pooling, normalization, and metric code. Any retrieval number cited
in `BENCHMARK_REPORT.md` MUST come from this code path or document
its deviation.

## Cross-task evaluation reproduction

Section 4 (retrieval models on generation) and Section 5 (generation models on
retrieval) are run by the dedicated python-scripts:

```bash
python benchmarks/python-scripts/cross_eval_ret_on_generation.py
python benchmarks/python-scripts/cross_eval_gen_on_retrieval.py
```

These already exist on the `benchmarking-suite` branch and run on Nautilus via
`benchmarks/kubernetes-jobs/cross-eval-*.yaml`. Their headline findings:
- Qwen3-4B → Qwen3-Embedding-4B loses **-44.3%** generation (worse than our -22%)
- Pure gen models score **0.13 retrieval avg** vs **0.41** for dedicated embeds
- Our Stage 1 (0.2220) already beats pure gen baselines on retrieval

If a future re-run produces materially different deltas, that's the most
surprising finding to surface immediately — the cross-task story is the
strongest claim in the report.

## 3-seed variance bound (~30 min on one A100)

Bounds single-seed noise on the 4 cheap generation tasks:

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

## Base-model contrast (necessary for Stage-1-contribution claims)

`BENCHMARK_REPORT.md` Section 1 already includes Qwen2.5-3B-Instruct in the
table — so the base-model row exists. To re-verify in your env:

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

Note: the **embedding** tables in the report do NOT include a base-Qwen-3B-Inst
row. To produce one (so we can isolate what action-token SFT contributes to
embedding quality, before any contrastive training), run Section 5's
gen-on-retrieval script with mean pooling on the base model and add it to the
table.

## Knobs the published reports do not pin — record these when running

| Knob | Canonical value | Why it matters |
|---|---|---|
| Random seed | `--seed 42` | lm-eval silent defaults are `random=0`, `numpy=torch=fewshot=1234`. 3-seed variance bound run (`REPRODUCTION_RESULTS.md`) shows std-dev ≤0.004 on all 4 cheap tasks, so this knob is essentially deterministic for 0-shot multiple-choice |
| Model dtype | `dtype=bfloat16` in `--model_args` | Matches `BENCHMARK_REPORT.md` Appendix |
| Chat template | **NOT applied** (do NOT pass `--apply_chat_template`) | Canonical per Arjun. Matches the methodology of `baseline-generation-lmeval.yaml` under which the original report was produced. The existence of a chat template on Qwen2.5-3B-Instruct does NOT mean it was used. |
| Harness version | `lm-eval==0.4.11` | Pinned. Original report wrote "v0.4+" but did not pin; `0.4.11` is the version under which independent reproduction succeeded (per-cell deltas <0.005 except one outlier consistent with metric drift across v0.4.x patch versions) |
| Dataset revision | record from harness output | HF dataset revisions can change |
| Pooling method (embedding) | mean pooling, last hidden state, L2 normalized | Matches Stage 1 / 1.5 training. Different pooling = different MTEB numbers |
| `<ACT:RET>` query prefix (embedding) | NOT applied in current eval scripts | Training prepends `<ACT:RET>` to queries (`open_instruct/contrastive_finetune.py`). Eval doesn't. Worth A/B testing whether the prefix improves MTEB |

## Expected runtime budget (single A100)

| Task | Single seed | All 4 seed×task combos |
|---|---|---|
| arc_easy | ~5 min | ~20 min |
| arc_challenge | ~5 min | ~20 min |
| hellaswag | ~10 min | ~40 min |
| winogrande | ~5 min | ~20 min |
| mmlu (5-shot) | ~2 h | ~8 h |

MPS on a laptop is 5–10× slower and is not recommended for anything beyond
pipeline validation.
