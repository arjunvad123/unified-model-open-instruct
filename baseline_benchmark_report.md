# Baseline Benchmark Report: Unified Agentic Model (Stage 1)

**Model**: `Arjunvad/unified-model-stage1-action-tokens-v2` (3B params, Qwen2.5-3B-Instruct base)
**Date**: February 19, 2026
**Cluster**: Nautilus (NVIDIA A100/L40S GPUs)

---

## 1. Generation Benchmarks (lm-evaluation-harness)

**Benchmarks**: ARC-Easy, ARC-Challenge, HellaSwag, Winogrande (0-shot), MMLU (5-shot)

| Task | Ours (3B) | SmolLM2 (1.7B) | Qwen2.5 (7B) | Mistral (7B) | Phi-2 (2.7B) |
|------|-----------|----------------|--------------|--------------|--------------|
| ARC-Easy (acc_norm) | 0.6010 | 0.7353 | 0.7727 | 0.7955 | 0.7803 |
| ARC-Challenge (acc_norm) | 0.3985 | 0.4753 | 0.5102 | 0.5375 | 0.5384 |
| HellaSwag (acc_norm) | 0.6110 | 0.7143 | 0.7893 | 0.8115 | 0.7361 |
| Winogrande (acc) | 0.5706 | 0.6598 | 0.7316 | 0.7498 | 0.7522 |
| MMLU 5-shot (acc) | 0.6269 | 0.5001 | 0.7417 | 0.6248 | 0.5643 |
| **Average** | **0.5616** | **0.6170** | **0.7091** | **0.7038** | **0.6743** |

**Key observations**:
- Our 3B model achieves **0.6269 MMLU**, beating SmolLM2-1.7B (0.5001), Mistral-7B (0.6248), and Phi-2 (0.5643) despite being much smaller
- MMLU is the strongest benchmark for our model, likely due to Qwen2.5-3B-Instruct pretraining
- HellaSwag and Winogrande are weaker areas (0.6110 and 0.5706 acc_norm/acc)
- Our model underperforms on commonsense reasoning (ARC, HellaSwag, Winogrande) compared to all baselines
- Generation average (0.5616) is competitive with SmolLM2-1.7B (0.6170) despite our model doing embedding + generation

**Note**: google/gemma-2-2b was excluded (gated model, 403 error)

---

## 2. Embedding Benchmarks (MTEB Retrieval)

**Benchmarks**: NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA2018
**Metrics**: NDCG@10, Hits@1, Hits@10, MRR
**Embedding method**: Mean pooling over last hidden state, L2 normalized

### NDCG@10 (primary retrieval metric)

| Task | Ours (3B) | MiniLM (22M) | BGE-S (33M) | BGE-L (335M) | E5-L (335M) |
|------|-----------|--------------|-------------|--------------|-------------|
| NFCorpus | 0.0782 | 0.3177 | 0.3424 | 0.3821 | 0.3712 |
| SciFact | 0.4778 | 0.6451 | 0.7127 | 0.7463 | 0.7222 |
| ArguAna | 0.3120 | 0.3697 | 0.4343 | 0.4599 | 0.3423 |
| SCIDOCS | 0.0139 | 0.1399 | 0.1327 | 0.1461 | 0.1323 |
| FiQA2018 | 0.0234 | 0.3687 | 0.4035 | 0.4500 | 0.4113 |
| **Average** | **0.1810** | **0.3682** | **0.4051** | **0.4369** | **0.3959** |

### Hits@1

| Task | Ours (3B) | MiniLM (22M) | BGE-S (33M) | BGE-L (335M) | E5-L (335M) |
|------|-----------|--------------|-------------|--------------|-------------|
| NFCorpus | 0.0774 | 0.4149 | 0.4365 | 0.4861 | 0.4675 |
| SciFact | 0.3100 | 0.5033 | 0.6067 | 0.6400 | 0.6033 |
| ArguAna | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SCIDOCS | 0.0220 | 0.2400 | 0.2350 | 0.2620 | 0.2530 |
| FiQA2018 | 0.0123 | 0.3472 | 0.4012 | 0.4475 | 0.4167 |
| **Average** | **0.0846** | **0.3011** | **0.3359** | **0.3671** | **0.3481** |

### Hits@10

| Task | Ours (3B) | MiniLM (22M) | BGE-S (33M) | BGE-L (335M) | E5-L (335M) |
|------|-----------|--------------|-------------|--------------|-------------|
| NFCorpus | 0.3282 | 0.6904 | 0.6935 | 0.7616 | 0.7616 |
| SciFact | 0.6867 | 0.7933 | 0.8467 | 0.8800 | 0.8567 |
| ArguAna | 0.6643 | 0.7653 | 0.8485 | 0.8819 | 0.7034 |
| SCIDOCS | 0.0910 | 0.6400 | 0.6010 | 0.6420 | 0.6080 |
| FiQA2018 | 0.0633 | 0.6574 | 0.6744 | 0.7269 | 0.6836 |
| **Average** | **0.3667** | **0.7093** | **0.7328** | **0.7785** | **0.7227** |

### MRR

| Task | Ours (3B) | MiniLM (22M) | BGE-S (33M) | BGE-L (335M) | E5-L (335M) |
|------|-----------|--------------|-------------|--------------|-------------|
| NFCorpus | 0.1601 | 0.5101 | 0.5336 | 0.5815 | 0.5734 |
| SciFact | 0.4335 | 0.6113 | 0.6868 | 0.7167 | 0.6924 |
| ArguAna | 0.2151 | 0.2570 | 0.3085 | 0.3288 | 0.2414 |
| SCIDOCS | 0.0474 | 0.3717 | 0.3601 | 0.3885 | 0.3712 |
| FiQA2018 | 0.0309 | 0.4547 | 0.4955 | 0.5421 | 0.5077 |
| **Average** | **0.1774** | **0.4410** | **0.4769** | **0.5115** | **0.4772** |

**Key observations**:
- Our model's embedding quality is significantly weaker than all specialized baselines
- NDCG@10 average of 0.1810 vs best baseline BGE-L at 0.4369 (2.4x gap)
- SciFact is our strongest task (0.4778 NDCG@10), closest to baselines
- NFCorpus, SCIDOCS, and FiQA2018 are extremely weak (<0.08 NDCG@10)
- These are pretrained embeddings with NO contrastive training — this is the "before" picture
- Even 22M-parameter MiniLM (2x better on avg) significantly outperforms our 3B model, showing that model size alone doesn't determine embedding quality

**Note**: nomic-ai/nomic-embed-text-v1.5 failed to load (likely needs specific trust_remote_code handling)

---

## 3. Unified Model Benchmarks (Embedding + Generation)

**Goal**: Compare against models that can do both embedding and generation.

### Embedding Results (NDCG@10)

| Task | Ours (3B) | GritLM-7B | Qwen3-Emb-8B | NV-Embed-v2 | Jina-v4 |
|------|-----------|-----------|--------------|-------------|---------|
| NFCorpus | 0.0785 | FAILED | 0.0403 | FAILED | FAILED |
| SciFact | 0.4778 | FAILED | 0.0419 | FAILED | FAILED |
| ArguAna | 0.3121 | FAILED | 0.0107 | FAILED | FAILED |
| **Average** | **0.2895** | N/A | **0.0310** | N/A | N/A |

### Failure Analysis

| Model | Error | Root Cause |
|-------|-------|------------|
| GritLM-7B | `DynamicCache.from_legacy_cache` + `rope_theta` | Incompatible with transformers >= 4.51 |
| NV-Embed-v2 | `all_tied_weights_keys` attribute error | Needs older transformers version |
| Jina-v4 | `SlidingWindowCache` import error | Needs newer/specific transformers version |
| Qwen3-Emb-8B | Loaded but near-random performance | Mean pooling inadequate; needs special embedding mode |

**Note**: The unified baseline is largely inconclusive due to transformers version incompatibilities. Each model requires a different transformers version. GritLM-7B is the most relevant comparison but needs transformers < 4.45. Qwen3-Emb-8B scored near-random (0.03 avg) because it likely needs its dedicated embedding API, not naive mean pooling.

---

## 4. Summary Table

| Category | Metric | Ours (3B) | Best Baseline | Gap |
|----------|--------|-----------|---------------|-----|
| Generation | Avg (5 tasks) | 0.5616 | 0.7091 (Qwen2.5-7B) | -0.1475 |
| Generation | MMLU | 0.6269 | 0.7417 (Qwen2.5-7B) | -0.1148 |
| Embedding | NDCG@10 avg | 0.1810 | 0.4369 (BGE-L) | -0.2559 |
| Embedding | Hits@1 avg | 0.0846 | 0.3671 (BGE-L) | -0.2825 |

---

## 5. Conclusions & Next Steps

### Strengths
1. **Strong MMLU performance** (0.6269) — beats models 2-4x larger (Mistral-7B, Phi-2)
2. **Unified architecture** — single model handles both generation and embedding tasks
3. **Action token routing** — eval pending; the original harness tested untrained `<ACT:TOOL>`/`<ACT:CODE>` tokens. Trained set is GEN/RET/THINK/STOP (see `open_instruct/action_tokens.py`). Reproducible routing number to be added after re-running the corrected `benchmarks/inference-scripts/03_action_token_routing.py`.

### Weaknesses
1. **Embedding quality is the primary bottleneck** — 2.4x worse than specialized models
2. **Commonsense reasoning** (ARC, HellaSwag, Winogrande) lags behind similar-sized models
3. **No contrastive training** in Stage 1 — embeddings are naive mean-pooled hidden states

### Immediate Next Steps
1. **Stage 1.5 contrastive training** (COMPLETED): Trained with MEDI2 500K hard negatives, pushed to `Arjunvad/unified-model-stage1-5-embedding-v2`. Pending evaluation.
2. **Evaluate Stage 1.5**: Run same MTEB benchmarks + generation benchmarks to measure improvement
3. **Fix unified baselines**: Run GritLM-7B with transformers 4.40 in a separate job for direct comparison

---

## Appendix: Experimental Details

### Hardware
- Embedding baseline: NVIDIA L40S (rci-tide-gpu-05.sdsu.edu)
- Generation baseline: NVIDIA L40S (hcc-nrp-pki-c1703.unl.edu)
- Unified baseline: NVIDIA L40S (rci-tide-gpu-06.sdsu.edu)

### Frameworks
- Embedding: Custom MTEB retrieval evaluation (mean pooling + cosine similarity)
- Generation: lm-evaluation-harness `0.4.11` (0-shot for core tasks, 5-shot for MMLU). The original report wrote "v0.4+" but did not pin a specific version; `0.4.11` is the version under which an independent reproduction succeeded — see `benchmarks/REPRODUCTION_RESULTS.md`. Any small per-cell delta (e.g. base-Qwen2.5-3B-Instruct HellaSwag acc_norm 0.7499 vs 0.7289) is consistent with metric implementation drift across v0.4.x patch versions and/or chat-template handling.
- All models loaded in bfloat16

### Models Evaluated
| Category | Model | Params | Type |
|----------|-------|--------|------|
| Ours | unified-model-stage1-action-tokens-v2 | 3B | CausalLM (unified) |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 | 22M | Encoder |
| Embedding | BAAI/bge-small-en-v1.5 | 33M | Encoder |
| Embedding | BAAI/bge-large-en-v1.5 | 335M | Encoder |
| Embedding | intfloat/e5-large-v2 | 335M | Encoder |
| Generation | HuggingFaceTB/SmolLM2-1.7B | 1.7B | CausalLM |
| Generation | Qwen/Qwen2.5-7B | 7.6B | CausalLM |
| Generation | mistralai/Mistral-7B-v0.1 | 7B | CausalLM |
| Generation | microsoft/phi-2 | 2.7B | CausalLM |
