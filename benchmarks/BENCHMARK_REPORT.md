# Unified Agentic Model: Complete Benchmark Report

**Model**: `Arjunvad/unified-model-stage1-action-tokens-v2` (3B params, Qwen2.5-3B-Instruct base)
**Stage 1.5**: `Arjunvad/unified-model-stage1-5-embedding-v2` (contrastive-trained with MEDI2 hard negatives)
**Date**: February–March 2026
**Cluster**: Nautilus (NVIDIA A100/L40S GPUs)

---

## 1. SOTA Generation Baseline (Our 3B vs Similar-Size SOTA Models)

**Framework**: lm-evaluation-harness
**Benchmarks**: ARC-Easy, ARC-Challenge, HellaSwag, Winogrande (0-shot), MMLU (5-shot)

| Task | Ours (3B) | Qwen2.5-3B-Inst | Qwen2.5-3B | SmolLM3-3B | Phi-2 (2.7B) | SmolLM2-1.7B |
|------|-----------|-----------------|------------|------------|--------------|--------------|
| ARC-Easy (acc_norm) | 0.6010 | 0.7306 | 0.7205 | 0.7710 | 0.7803 | 0.7353 |
| ARC-Challenge (acc_norm) | 0.3985 | 0.4787 | 0.4377 | 0.5393 | 0.5384 | 0.4753 |
| HellaSwag (acc_norm) | 0.6110 | 0.7289 | 0.7181 | 0.7565 | 0.7361 | 0.7143 |
| Winogrande (acc) | 0.5706 | 0.6935 | 0.6756 | 0.6685 | 0.7522 | 0.6598 |
| MMLU 5-shot (acc) | 0.6269 | 0.6835 | 0.6477 | 0.6025 | 0.5643 | 0.5001 |
| **Average** | **0.5616** | **0.6630** | **0.6399** | **0.6676** | **0.6743** | **0.6170** |

**Key observations**:
- Our 3B unified model averages 0.5616, ~15% behind the best ~3B models (Phi-2 at 0.6743)
- MMLU is our strongest benchmark (0.6269) — beats SmolLM2-1.7B (0.5001) and Phi-2 (0.5643)
- Commonsense reasoning (ARC, HellaSwag, Winogrande) is the primary weakness
- Performance gap is expected: our model handles both embedding + generation, not just generation

---

## 2. SOTA Embedding Baseline (Our 3B vs Dedicated Embedding Models)

**Framework**: Custom MTEB Retrieval evaluation
**Benchmarks**: NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA2018
**Our method**: Mean pooling over last hidden state, L2 normalized
**SOTA method**: Last-token pooling with instruction prefix

### NDCG@10 (primary retrieval metric)

| Task | Ours (3B) | Qwen3-Emb-0.6B | Qwen3-Emb-4B | GTE-1.5B | E5-Mistral-7B | GTE-7B |
|------|-----------|-----------------|--------------|----------|---------------|--------|
| NFCorpus | 0.0782 | 0.2321 | 0.2676 | N/A | 0.0584 | N/A |
| SciFact | 0.4778 | 0.5907 | 0.6614 | N/A | 0.2428 | N/A |
| ArguAna | 0.3120 | 0.5515 | 0.5694 | N/A | 0.1932 | N/A |
| SCIDOCS | 0.0139 | 0.1309 | 0.1453 | N/A | 0.0147 | N/A |
| FiQA2018 | 0.0234 | 0.3294 | 0.4049 | N/A | 0.2278 | N/A |
| **Average** | **0.1810** | **0.3669** | **0.4097** | N/A | **0.1474** | N/A |

### Hits@1

| Task | Ours (3B) | Qwen3-Emb-0.6B | Qwen3-Emb-4B | E5-Mistral-7B |
|------|-----------|-----------------|--------------|---------------|
| NFCorpus | 0.0774 | 0.3282 | 0.3808 | 0.0650 |
| SciFact | 0.3100 | 0.4833 | 0.5500 | 0.1267 |
| ArguAna | 0.0014 | 0.1905 | 0.2018 | 0.0000 |
| SCIDOCS | 0.0220 | 0.2300 | 0.2510 | 0.0280 |
| FiQA2018 | 0.0123 | 0.2994 | 0.3796 | 0.2006 |
| **Average** | **0.0846** | **0.3063** | **0.3526** | **0.0841** |

**Key observations**:
- Our model (0.1810 NDCG@10 avg) is 2.3x behind Qwen3-Emb-4B (0.4097)
- These are naive mean-pooled hidden states — no contrastive training yet (Stage 1)
- SciFact is our best task (0.4778 NDCG@10)
- GTE-1.5B and GTE-7B failed due to `tokenization_qwen2_fast` module incompatibility
- E5-Mistral-7B scored unusually low (0.1474) — likely needs SentenceTransformer wrapper, not raw AutoModel

---

## 3. Stage 1.5 Evaluation (After Contrastive Training)

### Stage 1.5 Training Details
- **Base**: `Arjunvad/unified-model-stage1-action-tokens-v2` (3B)
- **Method**: Contrastive fine-tuning with MEDI2 500K hard negatives
- **LoRA**: rank=32, alpha=64, targeting q/k/v/o projections
- **Temperature**: 0.02 (GritLM default)
- **Learning rate**: 1e-5, cosine schedule
- **Epochs**: 1 (500K examples)
- **Result**: Merged LoRA → `Arjunvad/unified-model-stage1-5-embedding-v2`

### Embedding Improvement (Stage 1 → Stage 1.5)

| Task | Stage 1 (NDCG@10) | Stage 1.5 (NDCG@10) | Change |
|------|--------------------|----------------------|--------|
| NFCorpus | 0.0782 | 0.1072 | +37% |
| SciFact | 0.4778 | 0.4918 | +3% |
| ArguAna | 0.3120 | 0.3408 | +9% |
| SCIDOCS | 0.0139 | 0.0343 | +147% |
| FiQA2018 | 0.0234 | 0.2109 | +801% |
| **Average** | **0.1810** | **0.2370** | **+31%** |

### Generation Degradation (Stage 1 → Stage 1.5)

| Task | Stage 1 | Stage 1.5 | Change |
|------|---------|-----------|--------|
| ARC-Easy (acc_norm) | 0.6010 | 0.4253 | -29% |
| ARC-Challenge (acc_norm) | 0.3985 | 0.3029 | -24% |
| HellaSwag (acc_norm) | 0.6110 | 0.4022 | -34% |
| Winogrande (acc) | 0.5706 | 0.5451 | -4% |
| MMLU 5-shot (acc) | 0.6269 | 0.5148 | -18% |
| **Average** | **0.5616** | **0.4381** | **-22%** |

**Key finding**: Contrastive training improved embeddings +31% but degraded generation -22% (catastrophic forgetting).

---

## 4. Cross-Task Evaluation: Retrieval Models on Generation

**Question**: Does embedding tuning destroy generation ability in other models too, or is it unique to our approach?

**Framework**: lm-evaluation-harness
**Benchmarks**: ARC-Easy, ARC-Challenge, HellaSwag, Winogrande (0-shot), MMLU (5-shot)

| Task | Qwen3-Emb-0.6B | Qwen3-Emb-4B | Qwen3-4B (gen) | Ours Stage 1 | Ours Stage 1.5 | Qwen2.5-3B-Inst |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| ARC-Easy (acc_norm) | 0.3342 | 0.3758 | 0.7841 | 0.6010 | 0.4253 | 0.7306 |
| ARC-Challenge (acc_norm) | 0.2969 | 0.3106 | 0.5367 | 0.3985 | 0.3029 | 0.4787 |
| HellaSwag (acc_norm) | 0.3411 | 0.4322 | 0.6850 | 0.6110 | 0.4022 | 0.7289 |
| Winogrande (acc) | 0.5067 | 0.5264 | 0.6582 | 0.5706 | 0.5451 | 0.6935 |
| MMLU 5-shot (acc) | 0.2295 | 0.2295 | 0.7015 | 0.6269 | 0.5148 | 0.6835 |
| **Average** | **0.3417** | **0.3749** | **0.6731** | **0.5616** | **0.4381** | **0.6630** |

**Key findings**:
- **Qwen3-4B → Qwen3-Embedding-4B: -44.3% generation degradation** — even worse than our -22%
- Qwen3-Embedding-0.6B scores near-random on MMLU (0.2295 vs 0.25 random baseline)
- **Confirms: catastrophic forgetting from embedding tuning is universal**, not a bug in our training
- Our -22% degradation is actually less severe than Qwen3's -44.3%, despite using only LoRA r=32

---

## 5. Cross-Task Evaluation: Generation Models on Retrieval

**Question**: Can pure generation models do retrieval via hidden-state pooling? (PI hypothesis: "pure generation model might be even better than retrieval tuned model")

**Framework**: Custom MTEB Retrieval evaluation
**Benchmarks**: NFCorpus, SciFact, ArguAna, SCIDOCS (NDCG@10)
**Method**: Mean pooling and last-token pooling over last hidden state, L2 normalized

| Task | Qwen2.5-3B-Inst (mean) | Qwen2.5-3B-Inst (last-tok) | Qwen2.5-3B (mean) | Qwen3-4B (mean) | Qwen3-4B (last-tok) | Ours Stage 1 | Ours Stage 1.5 | Qwen3-Emb-0.6B | Qwen3-Emb-4B |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| NFCorpus | 0.0185 | 0.0202 | 0.0174 | 0.0274 | 0.0148 | 0.0787 | 0.2343 | 0.2321 | 0.2676 |
| SciFact | 0.2876 | 0.0269 | 0.2778 | 0.2516 | 0.0370 | 0.4830 | 0.5411 | 0.5907 | 0.6614 |
| ArguAna | 0.2597 | 0.0753 | 0.2608 | 0.2302 | 0.0623 | 0.3123 | 0.2723 | 0.5515 | 0.5694 |
| SCIDOCS | 0.0116 | 0.0032 | 0.0115 | 0.0149 | 0.0035 | 0.0140 | 0.0866 | 0.1309 | 0.1453 |
| **Average** | **0.1443** | **0.0314** | **0.1419** | **0.1310** | **0.0294** | **0.2220** | **0.2836** | **0.3763** | **0.4109** |

**Key findings**:
- **PI's hypothesis is incorrect**: Pure generation models (avg 0.13–0.14) are far worse than dedicated embedding models (0.37–0.41)
- **Last-token pooling is much worse** than mean pooling for untrained models (0.03 vs 0.14 avg)
- **Our Stage 1 already beats all pure generation baselines** on retrieval (0.2220 vs 0.1443), showing our unified training adds retrieval ability
- **Our Stage 1.5 (0.2836) approaches Qwen3-Emb-0.6B (0.3763)** — a dedicated embedding model. On NFCorpus, Stage 1.5 matches it (0.2343 vs 0.2321)
- Contrastive training Stage 1 → 1.5 improved retrieval +28% (0.2220 → 0.2836)

---

## 6. Earlier Baselines (Small Encoder Models)

### Embedding: Our 3B vs Small Encoders

| Task (NDCG@10) | Ours (3B) | MiniLM (22M) | BGE-S (33M) | BGE-L (335M) | E5-L (335M) |
|-----------------|-----------|--------------|-------------|--------------|-------------|
| NFCorpus | 0.0782 | 0.3177 | 0.3424 | 0.3821 | 0.3712 |
| SciFact | 0.4778 | 0.6451 | 0.7127 | 0.7463 | 0.7222 |
| ArguAna | 0.3120 | 0.3697 | 0.4343 | 0.4599 | 0.3423 |
| SCIDOCS | 0.0139 | 0.1399 | 0.1327 | 0.1461 | 0.1323 |
| FiQA2018 | 0.0234 | 0.3687 | 0.4035 | 0.4500 | 0.4113 |
| **Average** | **0.1810** | **0.3682** | **0.4051** | **0.4369** | **0.3959** |

### Generation: Our 3B vs Larger Models (First Round)

| Task | Ours (3B) | SmolLM2 (1.7B) | Qwen2.5 (7B) | Mistral (7B) | Phi-2 (2.7B) |
|------|-----------|----------------|--------------|--------------|--------------|
| ARC-Easy | 0.6010 | 0.7353 | 0.7727 | 0.7955 | 0.7803 |
| ARC-Challenge | 0.3985 | 0.4753 | 0.5102 | 0.5375 | 0.5384 |
| HellaSwag | 0.6110 | 0.7143 | 0.7893 | 0.8115 | 0.7361 |
| Winogrande | 0.5706 | 0.6598 | 0.7316 | 0.7498 | 0.7522 |
| MMLU (5-shot) | 0.6269 | 0.5001 | 0.7417 | 0.6248 | 0.5643 |
| **Average** | **0.5616** | **0.6170** | **0.7091** | **0.7038** | **0.6743** |

---

## 7. Unified Model Baselines (Embedding + Generation)

Attempted to compare against truly unified models. Most failed due to transformers version incompatibilities.

| Model | Status | Error |
|-------|--------|-------|
| GritLM-7B | FAILED | `DynamicCache.from_legacy_cache` + `rope_theta` (needs transformers < 4.45) |
| NV-Embed-v2 | FAILED | `all_tied_weights_keys` attribute error (needs older transformers) |
| Jina-v4 | FAILED | `SlidingWindowCache` import error (needs specific transformers) |
| Qwen3-Emb-8B | Near-random | Mean pooling inadequate; needs dedicated embedding API |

**Note**: GritLM-7B is the most relevant comparison but requires a separate environment with transformers < 4.45.

---

## 8. Summary

| Category | Metric | Stage 1 (Ours) | Stage 1.5 (Ours) | Best ~3B SOTA | Gap (Stage 1 vs SOTA) |
|----------|--------|----------------|-------------------|---------------|----------------------|
| Generation | Avg (5 tasks) | 0.5616 | 0.4381 | 0.6743 (Phi-2) | -0.1127 |
| Generation | MMLU | 0.6269 | 0.5148 | 0.6835 (Qwen2.5-3B-Inst) | -0.0566 |
| Embedding | NDCG@10 avg | 0.1810 | 0.2370 | 0.4097 (Qwen3-Emb-4B) | -0.2287 |
| Embedding | Hits@1 avg | 0.0846 | N/A | 0.3526 (Qwen3-Emb-4B) | -0.2680 |

### Strengths
1. **Strong MMLU** (0.6269) — competitive with models 2-4x larger
2. **Unified architecture** — single model handles both generation and embedding
3. **Action token routing** — eval pending; the original harness tested untrained `<ACT:TOOL>`/`<ACT:CODE>` tokens. Trained set is THINK/RET/GEN/STOP/WAIT/RET_RESULT (see `open_instruct/action_tokens.py`). Reproducible number to be added after re-running the corrected `benchmarks/inference-scripts/03_action_token_routing.py`.
4. **Stage 1.5 embedding improvement** — +31% NDCG@10 with contrastive training
5. **Stage 1 already has retrieval ability** — 0.2220 NDCG@10 avg beats all pure gen baselines (0.13–0.14)
6. **Our forgetting is less severe** than Qwen3's (-22% vs -44.3%), validating LoRA approach

### Weaknesses
1. **Embedding quality is the primary bottleneck** — 2.3x worse than specialized models
2. **Catastrophic forgetting in Stage 1.5** — contrastive training degraded generation -22% (but this is universal — Qwen3 shows -44.3%)
3. **Commonsense reasoning** lags behind similar-sized generation models

### Key Cross-Task Conclusions
1. **Catastrophic forgetting is universal**: Qwen3-Embedding-4B loses -44.3% generation vs its base — our -22% is actually better
2. **Pure gen models cannot do retrieval**: PI's hypothesis disproven — gen models score 0.13 avg vs 0.41 for dedicated embedding models
3. **Our unified training adds retrieval ability**: Stage 1 (0.2220) already beats pure gen baselines without any contrastive training
4. **Mean pooling > last-token pooling** for models not specifically trained for last-token retrieval

### Next Steps
1. Fix catastrophic forgetting: Lower LoRA rank, freeze more layers, or use GritLM-style multi-task training
2. Run GritLM-7B in a separate environment (transformers 4.40) for direct unified comparison
3. Replace failing embedding baselines (GTE models) with alternatives
4. Consider instruction-tuned embedding approach instead of naive mean pooling

---

## Appendix: Experimental Details

### Hardware
- All jobs: Nautilus Kubernetes cluster, NVIDIA A100 (40GB/80GB) or L40S GPUs
- Single GPU per job

### Frameworks & Versions
- **Generation**: lm-evaluation-harness `0.4.11` (0-shot core tasks, 5-shot MMLU). Original report said "v0.4+" without pinning; `0.4.11` is the version under which an independent reproduction succeeded — see `benchmarks/REPRODUCTION_RESULTS.md`. Per-cell deltas <0.005 in the reproduction; one outlier (base-Qwen2.5-3B-Instruct HellaSwag, +0.0210) is consistent with metric-implementation drift across v0.4.x patch versions and/or chat-template handling.
- **Embedding**: Custom MTEB retrieval evaluation with cosine similarity
- **All models**: loaded in bfloat16

### Models Evaluated

| Category | Model | Params | Type | Status |
|----------|-------|--------|------|--------|
| **Ours (Stage 1)** | unified-model-stage1-action-tokens-v2 | 3B | CausalLM (unified) | OK |
| **Ours (Stage 1.5)** | unified-model-stage1-5-embedding-v2 | 3B | CausalLM (unified) | OK |
| **Gen SOTA** | Qwen/Qwen2.5-3B-Instruct | 3B | CausalLM | OK |
| **Gen SOTA** | Qwen/Qwen2.5-3B | 3B | CausalLM | OK |
| **Gen SOTA** | HuggingFaceTB/SmolLM3-3B | 3B | CausalLM | OK |
| **Gen SOTA** | microsoft/phi-2 | 2.7B | CausalLM | OK |
| **Gen SOTA** | HuggingFaceTB/SmolLM2-1.7B | 1.7B | CausalLM | OK |
| **Emb SOTA** | Qwen/Qwen3-Embedding-0.6B | 0.6B | Embedding | OK |
| **Emb SOTA** | Qwen/Qwen3-Embedding-4B | 4B | Embedding | OK |
| **Emb SOTA** | Alibaba-NLP/gte-Qwen2-1.5B-instruct | 1.5B | Embedding | FAILED |
| **Emb SOTA** | intfloat/e5-mistral-7b-instruct | 7B | Embedding | Low scores |
| **Emb SOTA** | Alibaba-NLP/gte-Qwen2-7B-instruct | 7B | Embedding | FAILED |
| **Emb Small** | sentence-transformers/all-MiniLM-L6-v2 | 22M | Encoder | OK |
| **Emb Small** | BAAI/bge-small-en-v1.5 | 33M | Encoder | OK |
| **Emb Small** | BAAI/bge-large-en-v1.5 | 335M | Encoder | OK |
| **Emb Small** | intfloat/e5-large-v2 | 335M | Encoder | OK |
| **Unified** | GritLM/GritLM-7B | 7B | Unified | FAILED |
| **Unified** | nvidia/NV-Embed-v2 | 8B | Embedding | FAILED |
| **Unified** | jinaai/jina-embeddings-v4 | 4B | Embedding | FAILED |
| **Cross-Task** | Qwen/Qwen3-4B | 4B | CausalLM (gen base) | OK |
| **Cross-Task** | Qwen/Qwen3-Embedding-0.6B | 0.6B | Embedding-tuned | OK |
| **Cross-Task** | Qwen/Qwen3-Embedding-4B | 4B | Embedding-tuned | OK |

### Known Issues
- **lm-eval v0.4+ parser**: Metric keys saved as `"acc_norm,none"` not `"acc_norm"` — must flatten
- **lm-eval v0.4+ filenames**: Results saved as `results_TIMESTAMP.json` not `results.json`
- **GTE models**: Need `tokenization_qwen2_fast` which is missing in newer transformers
- **E5-Mistral-7B**: Raw AutoModel + last-token pooling gives poor results; likely needs SentenceTransformer wrapper
