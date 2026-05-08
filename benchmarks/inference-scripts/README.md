# Inference Demo Scripts

Qualitative behavior demos for the Unified Agentic Model. These scripts show **actual model outputs** so you can see what the model does — not just benchmark numbers.

## Models

| Name | HuggingFace ID | Params | Description |
|------|----------------|--------|-------------|
| Stage 1 | `Arjunvad/unified-model-stage1-action-tokens-v2` | 3B | Action token SFT (generation + routing) |
| Stage 1.5 | `Arjunvad/unified-model-stage1-5-embedding-v2` | 3B | + Contrastive fine-tuning (MEDI2 500K, LoRA r=32) |

## Scripts

| # | Script | What It Shows | Models Used |
|---|--------|---------------|-------------|
| 1 | `01_generation_demo.py` | Generation quality across 4 categories (16 prompts) | Stage 1 |
| 2 | `02_embedding_demo.py` | Embedding quality: similar/dissimilar pairs + topic clustering | Stage 1 + 1.5 |
| 3 | `03_action_token_routing.py` | Action token routing accuracy (GEN/RET/TOOL/CODE) with confusion matrix | Stage 1 |
| 4 | `04_retrieval_demo.py` | RAG pipeline: index 10 docs, query, show ranked results | Stage 1 + 1.5 |
| 5 | `05_stage1_vs_stage1_5.py` | Full comparison: embeddings improved, generation degraded (catastrophic forgetting) | Stage 1 + 1.5 |

## Quick Start

```bash
# Set your HuggingFace token
export HF_TOKEN="your-token"

# Run any script (auto-detects MPS/CUDA/CPU)
python 01_generation_demo.py
python 02_embedding_demo.py
python 03_action_token_routing.py
python 04_retrieval_demo.py
python 05_stage1_vs_stage1_5.py
```

### Requirements

```bash
pip install torch transformers huggingface_hub
```

No additional dependencies — all scripts use only PyTorch + Transformers.

## Deploy on Nautilus

```bash
kubectl apply -f inference-demos.yaml -n svcl-self-improve
kubectl logs job/inference-demos -n svcl-self-improve -f
```

## What Each Script Demonstrates

### 01 — Generation Demo
Loads Stage 1 and generates responses for 16 prompts across 4 categories:
- General Knowledge (4 prompts)
- Reasoning (4 prompts)
- Code Generation (4 prompts)
- Instruction Following (4 prompts)

Shows action tokens in raw output and prints per-category summary with average response lengths.

### 02 — Embedding Demo
Loads Stage 1 and Stage 1.5 sequentially. Computes cosine similarity for:
- 10 similar sentence pairs (should be high)
- 5 dissimilar sentence pairs (should be low)
- 3 topic clusters (intra-topic vs inter-topic similarity)

Summary table shows avg similar, avg dissimilar, separation, cluster margin. Stage 1.5 should show improvement from contrastive training.

### 03 — Action Token Routing
Loads Stage 1 and tests queries against the actually-trained routes (defined in
`open_instruct/action_tokens.py` and the trajectory templates in
`open_instruct/unified_finetune.py`):
- **direct**: `<ACT:THINK>` … `<ACT:GEN>` … `<ACT:STOP>` (open-ended generation)
- **retrieval**: `<ACT:THINK>` … `<ACT:RET>` … `<ACT:GEN>` … `<ACT:STOP>` (factual lookup)

Prints per-category accuracy and a full confusion matrix. Earlier versions of
this script tested `<ACT:TOOL>` and `<ACT:CODE>` — those tokens were never
trained and have been removed.

### 04 — Retrieval Demo
Indexes a 10-document corpus using both Stage 1 and Stage 1.5 embeddings. Runs 6 queries with known correct documents. Shows:
- Top-3 ranked results with similarity scores per model
- Hits@1, average top-1 score, average score gap (1st vs 2nd)

Demonstrates that Stage 1.5's contrastive training improves retrieval.

### 05 — Stage 1 vs Stage 1.5 (Catastrophic Forgetting)
The key narrative script. Runs both models through:
- **Part A**: Embedding pairs — shows Stage 1.5 is BETTER (contrastive training worked)
- **Part B**: Generation prompts — shows Stage 1 is BETTER (generation degraded)
- **Part C**: Action token routing — shows whether routing survived Stage 1.5 training

Prints a conclusion summarizing the catastrophic forgetting problem and possible solutions.

## Technical Notes

- **Embedding method**: Mean pooling over last hidden state (`output_hidden_states[-1]`), L2 normalized
- **Generation**: `do_sample=False` (greedy), chat template applied
- **Memory**: Models loaded sequentially, freed between loads (`gc.collect()`, `torch.cuda.empty_cache()`)
- **Device**: Auto-detected — MPS (Apple Silicon) > CUDA > CPU
- **Dtype**: float16 on GPU/MPS, float32 on CPU
