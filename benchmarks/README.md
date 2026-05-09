# Unified Agentic Model — Benchmarking Suite

## Overview

This directory contains all benchmarking and evaluation code for the **Unified Agentic Model** — a single 3B-parameter decoder (Qwen2.5-3B-Instruct base) that handles generation, embedding, and agentic routing in one architecture.

**Research question**: Can a single decoder-only LLM perform generation, produce useful embeddings, and route queries via action tokens — without catastrophic forgetting across training stages?

### Training Stages

| Stage | Model | What It Does |
|-------|-------|--------------|
| Base | `Qwen/Qwen2.5-3B-Instruct` | Pretrained instruction-following LLM |
| **Stage 1** | `Arjunvad/unified-model-stage1-action-tokens-v2` | + Action token SFT (`<ACT:GEN>`, `<ACT:RET>`, `<ACT:THINK>`, `<ACT:STOP>`) |
| **Stage 1.5** | `Arjunvad/unified-model-stage1-5-embedding-v2` | + Contrastive fine-tuning (MEDI2 500K, LoRA r=32, temp=0.02) |

### HuggingFace Models

| Name | HuggingFace ID | Params |
|------|----------------|--------|
| Stage 1 | [`Arjunvad/unified-model-stage1-action-tokens-v2`](https://huggingface.co/Arjunvad/unified-model-stage1-action-tokens-v2) | 3B |
| Stage 1.5 | [`Arjunvad/unified-model-stage1-5-embedding-v2`](https://huggingface.co/Arjunvad/unified-model-stage1-5-embedding-v2) | 3B |

---

## Results at a Glance

### Generation: Our 3B vs SOTA ~3B Models

Framework: `lm-evaluation-harness` | Benchmarks: ARC-Easy, ARC-Challenge, HellaSwag, Winogrande (0-shot), MMLU (5-shot)

| Task | Ours (Stage 1) | Qwen2.5-3B-Inst | SmolLM3-3B | Phi-2 (2.7B) |
|------|----------------|-----------------|------------|--------------|
| ARC-Easy | 0.6010 | 0.7306 | 0.7710 | 0.7803 |
| ARC-Challenge | 0.3985 | 0.4787 | 0.5393 | 0.5384 |
| HellaSwag | 0.6110 | 0.7289 | 0.7565 | 0.7361 |
| Winogrande | 0.5706 | 0.6935 | 0.6685 | 0.7522 |
| MMLU (5-shot) | 0.6269 | 0.6835 | 0.6025 | 0.5643 |
| **Average** | **0.5616** | **0.6630** | **0.6676** | **0.6743** |

### Embedding: Our 3B vs Dedicated Embedding Models (NDCG@10)

Framework: Custom MTEB Retrieval | Benchmarks: NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA2018

| Task | Ours (Stage 1) | Ours (Stage 1.5) | Qwen3-Emb-0.6B | Qwen3-Emb-4B |
|------|----------------|-------------------|-----------------|--------------|
| NFCorpus | 0.0782 | 0.1072 (+37%) | 0.2321 | 0.2676 |
| SciFact | 0.4778 | 0.4918 (+3%) | 0.5907 | 0.6614 |
| ArguAna | 0.3120 | 0.3408 (+9%) | 0.5515 | 0.5694 |
| SCIDOCS | 0.0139 | 0.0343 (+147%) | 0.1309 | 0.1453 |
| FiQA2018 | 0.0234 | 0.2109 (+801%) | 0.3294 | 0.4049 |
| **Average** | **0.1810** | **0.2370 (+31%)** | **0.3669** | **0.4097** |

### The Catastrophic Forgetting Problem (Stage 1 → Stage 1.5)

| Capability | Stage 1 | Stage 1.5 | Change |
|------------|---------|-----------|--------|
| Generation (avg 5 tasks) | 0.5616 | 0.4381 | **-22%** |
| Embedding (NDCG@10 avg) | 0.1810 | 0.2370 | **+31%** |
| Action Routing | eval pending | degraded | **harness needs revalidation** |

Contrastive training improved embeddings but degraded generation — the core research challenge.

### Cross-Task Evaluation: Is Catastrophic Forgetting Universal?

**Retrieval models on generation** — Does embedding tuning destroy generation in other models?

| Model | Gen Avg | Delta from Base |
|-------|---------|-----------------|
| Qwen3-4B (gen base) | 0.6731 | — |
| Qwen3-Embedding-4B | 0.3749 | **-44.3%** |
| Ours Stage 1 → Stage 1.5 | 0.5616 → 0.4381 | **-22.0%** |

**Answer: Yes.** Qwen3's forgetting (-44.3%) is even worse than ours (-22%). Our LoRA approach is actually more forgiving.

**Generation models on retrieval** — Can pure gen models do retrieval? (NDCG@10)

| Model | Retrieval Avg |
|-------|---------------|
| Qwen3-Emb-4B (dedicated) | 0.4109 |
| Qwen3-Emb-0.6B (dedicated) | 0.3763 |
| **Ours Stage 1.5** | **0.2836** |
| **Ours Stage 1** | **0.2220** |
| Qwen2.5-3B-Inst (mean pool) | 0.1443 |
| Qwen3-4B (mean pool) | 0.1310 |

**Answer: No.** Pure gen models score ~0.14 avg — far below dedicated embedding models (0.41). Our unified training already gives Stage 1 better retrieval (0.22) than pure gen baselines.

---

## Folder Structure

```
benchmarks/
├── README.md                         # This file
├── BENCHMARK_REPORT.md               # Full results report with all tables
│
├── inference-scripts/                # Qualitative behavior demos (see model outputs)
│   ├── README.md                     # Index of all demo scripts
│   ├── 01_generation_demo.py         # Stage 1 generation across 4 categories
│   ├── 02_embedding_demo.py          # Stage 1 vs 1.5 embedding quality
│   ├── 03_action_token_routing.py    # Action token routing with confusion matrix
│   ├── 04_retrieval_demo.py          # RAG pipeline: index + query + ranked results
│   ├── 05_stage1_vs_stage1_5.py      # Catastrophic forgetting comparison
│   └── inference-demos.yaml          # K8s Job to run all 5 on Nautilus
│
├── python-scripts/                   # Quantitative evaluation scripts
│   ├── baseline_generation_sota.py   # SOTA generation baseline (6 models)
│   ├── baseline_embedding_sota.py    # SOTA embedding baseline (6 models)
│   ├── eval_stage1_5_generation.py   # Stage 1.5 generation eval
│   ├── eval_stage1_5_embedding.py    # Stage 1.5 embedding eval
│   ├── cross_eval_gen_on_retrieval.py  # Gen models on retrieval (MTEB)
│   └── cross_eval_ret_on_generation.py # Retrieval models on generation (lm-eval)
│
├── kubernetes-jobs/                  # Nautilus K8s Job YAMLs
│   ├── baseline-generation-sota.yaml
│   ├── baseline-embedding-sota.yaml
│   ├── baseline-generation-lmeval.yaml
│   ├── baseline-embedding-mteb.yaml
│   ├── baseline-unified-gritlm.yaml
│   ├── eval-stage1.5-generation-comprehensive.yaml
│   ├── eval-stage1.5-embedding-comprehensive.yaml
│   ├── eval-stage1.5-mteb.yaml
│   ├── cross-eval-gen-on-retrieval.yaml   # Cross-task: gen → retrieval
│   └── cross-eval-ret-on-generation.yaml  # Cross-task: retrieval → generation
│
└── results/                          # (populated at runtime)
```

---

## Smoke tests vs. research benchmarks

Not every script in this directory is a research-grade benchmark. Treat
results carefully:

| Script | Kind | What it actually measures |
|---|---|---|
| `run_generation_eval.py` | **Smoke test** | ~20 hand-written prompts, substring/keyword matches. Use for regression detection only. |
| `run_lm_eval.py` | **Research** | Wraps `lm-evaluation-harness` (MMLU, GSM8K, HumanEval, …). Reportable. |
| `run_mteb.py` | **Research** | Standard MTEB tasks via the official `mteb` package. Reportable. |
| `run_mteb_comparison.py` | **Research** | MTEB retrieval against gte-Qwen2 / base-Qwen baselines. Reportable. |
| `run_embedding_eval.py` | **Smoke test** | Custom Banking77 / retrieval / STS-style eval without going through MTEB. |
| `run_ragas.py` | **Research (with caveat)** | RAGAS metrics; full numbers require an OpenAI judge key, otherwise falls back to manual eval. |

When publishing or comparing model versions, only cite the **research** rows.
The smoke tests exist to catch regressions during development — their numbers
should never appear in a report without that label.

### Action / control tokens

The trained tokens used by the unified model are defined in
`open_instruct/action_tokens.py` and consumed by both training
(`open_instruct/unified_finetune.py`) and every script in this directory.
**Do not hardcode token lists in eval scripts** — import from there so the
tokenizer and the eval harness can never drift apart. The current trained
set is `<ACT:GEN>`, `<ACT:RET>`, `<ACT:THINK>`, `<ACT:STOP>`. (Earlier versions of these scripts referenced
`<ACT:TOOL>` / `<ACT:CODE>` which were never trained — those tests have been
removed.)

## Quick Start

### Prerequisites

```bash
# HuggingFace token (needed for our private models)
export HF_TOKEN="your-token"

# Python dependencies
pip install torch transformers accelerate huggingface_hub

# For quantitative benchmarks only:
pip install lm-eval mteb numpy tqdm
```

### 1. See Model Behavior (Inference Demos)

These scripts show actual model outputs — generation quality, embedding scores, action token routing, retrieval rankings.

```bash
cd benchmarks/inference-scripts/

# Run any single demo (auto-detects MPS/CUDA/CPU)
python 01_generation_demo.py          # Stage 1 generation (16 prompts)
python 02_embedding_demo.py           # Embedding quality comparison
python 03_action_token_routing.py     # Action token routing accuracy
python 04_retrieval_demo.py           # RAG pipeline demo
python 05_stage1_vs_stage1_5.py       # Full catastrophic forgetting comparison
```

Each script is self-contained — no shared dependencies between scripts. Models are loaded sequentially to fit in memory.

### 2. Run Quantitative Benchmarks (Locally)

```bash
cd benchmarks/

# Generation benchmarks (lm-evaluation-harness)
python python-scripts/baseline_generation_sota.py      # Our model vs SOTA
python python-scripts/eval_stage1_5_generation.py      # Stage 1.5 generation

# Embedding benchmarks (MTEB Retrieval)
python python-scripts/baseline_embedding_sota.py       # Our model vs SOTA
python python-scripts/eval_stage1_5_embedding.py       # Stage 1.5 embedding
```

### 3. Deploy on Nautilus (Kubernetes)

```bash
# Inference demos (all 5 scripts in one job)
kubectl apply -f inference-scripts/inference-demos.yaml -n svcl-self-improve

# Quantitative baselines
kubectl apply -f kubernetes-jobs/baseline-generation-sota.yaml -n svcl-self-improve
kubectl apply -f kubernetes-jobs/baseline-embedding-sota.yaml -n svcl-self-improve

# Stage 1.5 evaluation
kubectl apply -f kubernetes-jobs/eval-stage1.5-generation-comprehensive.yaml -n svcl-self-improve
kubectl apply -f kubernetes-jobs/eval-stage1.5-embedding-comprehensive.yaml -n svcl-self-improve

# Check logs
kubectl logs job/inference-demos -n svcl-self-improve -f
kubectl logs job/baseline-gen-sota -n svcl-self-improve -f
```

All K8s jobs target A100/L40S/L40 GPUs via node affinity and use the `hf-token` secret.

---

## Benchmarks Used

### Generation (lm-evaluation-harness)

| Benchmark | Shots | Metric | What It Tests |
|-----------|-------|--------|---------------|
| ARC-Easy | 0 | acc_norm | Grade-school science questions |
| ARC-Challenge | 0 | acc_norm | Harder science reasoning |
| HellaSwag | 0 | acc_norm | Commonsense sentence completion |
| Winogrande | 0 | acc | Pronoun resolution / commonsense |
| MMLU | 5 | acc | 57-subject knowledge (our strongest) |

### Embedding (MTEB Retrieval)

| Benchmark | Metric | Domain |
|-----------|--------|--------|
| NFCorpus | NDCG@10 | Biomedical |
| SciFact | NDCG@10 | Scientific claims |
| ArguAna | NDCG@10 | Argument retrieval |
| SCIDOCS | NDCG@10 | Scientific documents |
| FiQA2018 | NDCG@10 | Financial QA |

Our embedding method: Mean pooling over `output_hidden_states[-1]`, L2 normalized.

---

## Key Findings

1. **MMLU is our strongest benchmark** (0.6269) — competitive with models 2-4x larger, beats Phi-2 and SmolLM2
2. **Unified architecture works** — single model handles generation, embedding, and action token routing
3. **Action token routing — eval pending.** Original harness tested untrained `<ACT:TOOL>`/`<ACT:CODE>` tokens; trained set is GEN/RET/THINK/STOP (see `open_instruct/action_tokens.py`). Reproducible number to be added after re-running the corrected `benchmarks/inference-scripts/03_action_token_routing.py`
4. **Contrastive training (Stage 1.5) improved embeddings +31%** — especially FiQA2018 (+801%)
5. **But degraded generation -22%** — catastrophic forgetting from contrastive fine-tuning
6. **Catastrophic forgetting is universal** — Qwen3-Embedding-4B loses -44.3% generation vs its base; our -22% is actually less severe
7. **Pure gen models cannot do retrieval** — PI's hypothesis disproven; gen models score 0.13 avg vs 0.41 for dedicated embedding models
8. **Our unified training adds retrieval ability** — Stage 1 (0.2220) already beats all pure gen baselines (0.13–0.14) on retrieval without contrastive training

## Next Steps

1. **Fix catastrophic forgetting**: Lower LoRA rank (r=8/16), freeze more layers, or GritLM-style alternating batches
2. **Run GritLM-7B baseline** in separate environment (needs transformers < 4.45)
3. **Instruction-tuned embedding** approach instead of naive mean pooling
4. **Multi-task training**: contrastive + generation loss simultaneously

---

## Technical Notes

- **lm-eval v0.4+ quirk**: Metric keys are `"acc_norm,none"` not `"acc_norm"` — scripts handle this
- **lm-eval v0.4+ filenames**: Results saved as `results_TIMESTAMP.json` not `results.json`
- **GTE models failed**: `tokenization_qwen2_fast` module missing in newer transformers
- **E5-Mistral-7B low scores**: Raw AutoModel + last-token pooling gives poor results; likely needs SentenceTransformer wrapper
- **Device detection**: All inference scripts auto-detect MPS > CUDA > CPU
- **Memory management**: Models loaded/freed sequentially with `gc.collect()` + `torch.cuda.empty_cache()`

For the full results with all tables, see [`BENCHMARK_REPORT.md`](./BENCHMARK_REPORT.md).
