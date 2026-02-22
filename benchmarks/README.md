# Unified Agentic Model - Benchmarking Suite

Benchmarking code for evaluating the Unified Agentic Model (3B) against SOTA baselines on both generation and embedding tasks.

## Folder Structure

```
Unified-Model-Benchmarks/
├── README.md                          # This file
├── BENCHMARK_REPORT.md                # Complete results report with all tables
├── python-scripts/                    # Standalone Python evaluation scripts
│   ├── baseline_generation_sota.py    # SOTA generation baseline (6 models)
│   ├── baseline_embedding_sota.py     # SOTA embedding baseline (6 models)
│   ├── eval_stage1_5_generation.py    # Stage 1.5 generation eval
│   └── eval_stage1_5_embedding.py     # Stage 1.5 embedding eval
├── kubernetes-jobs/                   # Nautilus K8s job YAMLs (ready to deploy)
│   ├── baseline-generation-sota.yaml  # SOTA gen baseline job
│   ├── baseline-embedding-sota.yaml   # SOTA emb baseline job
│   ├── baseline-generation-lmeval.yaml    # Earlier gen baseline (larger models)
│   ├── baseline-embedding-mteb.yaml       # Earlier emb baseline (small encoders)
│   ├── baseline-unified-gritlm.yaml       # Unified model baseline (GritLM etc)
│   ├── eval-stage1.5-generation-comprehensive.yaml  # Stage 1.5 gen eval
│   ├── eval-stage1.5-embedding-comprehensive.yaml   # Stage 1.5 emb eval
│   └── eval-stage1.5-mteb.yaml            # Stage 1.5 vs Stage 1 comparison
└── results/                           # (empty, populated at runtime)
```

## Quick Start

### Run locally (requires GPU)

```bash
# Install dependencies
pip install torch transformers accelerate lm-eval mteb numpy tqdm

# Generation baseline (SOTA ~3B models)
python python-scripts/baseline_generation_sota.py

# Embedding baseline (SOTA embedding models)
python python-scripts/baseline_embedding_sota.py

# Stage 1.5 evaluation
python python-scripts/eval_stage1_5_generation.py
python python-scripts/eval_stage1_5_embedding.py
```

### Deploy on Nautilus (Kubernetes)

```bash
# SOTA baselines
kubectl apply -f kubernetes-jobs/baseline-generation-sota.yaml -n svcl-self-improve
kubectl apply -f kubernetes-jobs/baseline-embedding-sota.yaml -n svcl-self-improve

# Stage 1.5 evaluation
kubectl apply -f kubernetes-jobs/eval-stage1.5-generation-comprehensive.yaml -n svcl-self-improve
kubectl apply -f kubernetes-jobs/eval-stage1.5-embedding-comprehensive.yaml -n svcl-self-improve

# Check logs
kubectl logs job/baseline-gen-sota -n svcl-self-improve -f
kubectl logs job/baseline-emb-sota -n svcl-self-improve -f
```

## Models

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Ours (Stage 1) | 3B | `Arjunvad/unified-model-stage1-action-tokens-v2` |
| Ours (Stage 1.5) | 3B | `Arjunvad/unified-model-stage1-5-embedding-v2` |

## Benchmarks

**Generation** (lm-evaluation-harness):
- ARC-Easy, ARC-Challenge (0-shot, acc_norm)
- HellaSwag (0-shot, acc_norm)
- Winogrande (0-shot, acc)
- MMLU (5-shot, acc)

**Embedding** (MTEB Retrieval):
- NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA2018
- Metrics: NDCG@10, Hits@1, Hits@10, MRR

## Key Technical Notes

- **Embedding method for our model**: Mean pooling over `output_hidden_states[-1]`, L2 normalized
- **Embedding method for SOTA models**: Last-token pooling with instruction prefix
- **lm-eval v0.4+ quirk**: Metric keys are `"acc_norm,none"` not `"acc_norm"` — scripts handle this automatically
- **lm-eval v0.4+ filenames**: Results saved as `results_TIMESTAMP.json` not `results.json`
