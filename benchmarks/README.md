# Unified Agentic Model Benchmarks

This directory contains evaluation scripts for benchmarking the unified agentic model across multiple dimensions.

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
set is `<ACT:THINK>`, `<ACT:RET>`, `<ACT:GEN>`, `<ACT:STOP>`, `<WAIT>`,
`<RET_RESULT>`. (Earlier versions of these scripts referenced
`<ACT:TOOL>` / `<ACT:CODE>` which were never trained — those tests have been
removed.)

## Quick Start

```bash
# Install dependencies
python benchmarks/setup_benchmarks.py --install

# Check installations
python benchmarks/setup_benchmarks.py --check

# Show usage guide
python benchmarks/setup_benchmarks.py --guide
```

## Benchmarks Overview

| Benchmark | Purpose | Metrics |
|-----------|---------|---------|
| **MTEB** | Embedding/Retrieval quality | Retrieval@k, Classification acc, etc. |
| **LM-Eval** | LLM capabilities | MMLU, GSM8K, HumanEval, etc. |
| **RAGAS** | RAG pipeline quality | Faithfulness, Relevancy, Precision |

---

## 1. MTEB (Embedding Evaluation)

**Purpose**: Evaluate the quality of embeddings from our unified model.

### Quick Test
```bash
python benchmarks/run_mteb.py --tasks Banking77Classification
```

### Retrieval Tasks Only
```bash
python benchmarks/run_mteb.py --task_types Retrieval
```

### Compare with Base Qwen
```bash
python benchmarks/run_mteb.py --tasks NFCorpus --compare_base
```

### Full Benchmark
```bash
python benchmarks/run_mteb.py --benchmark "MTEB(eng, v2)"
```

### Key MTEB Tasks for Retrieval

| Task | Description | Size |
|------|-------------|------|
| NFCorpus | Medical/nutrition retrieval | Small |
| SciFact | Scientific claim verification | Small |
| FiQA | Financial QA retrieval | Medium |
| MSMARCO | Web search | Large |
| NQ | Natural Questions | Large |
| HotpotQA | Multi-hop QA | Large |

---

## 2. LM-Eval (LLM Benchmarks)

**Purpose**: Evaluate generation quality on standard LLM benchmarks.

### Standard Qwen Evaluation Setup

| Benchmark | Few-shot | Description |
|-----------|----------|-------------|
| MMLU | 5-shot | Knowledge/reasoning |
| GSM8K | 4-shot | Grade school math |
| HumanEval | 0-shot | Code generation |
| HellaSwag | 10-shot | Commonsense |
| ARC | 25-shot | Science reasoning |

### Quick Test
```bash
python benchmarks/run_lm_eval.py --tasks hellaswag --limit 100
```

### Full MMLU
```bash
python benchmarks/run_lm_eval.py --tasks mmlu
```

### All Benchmarks
```bash
python benchmarks/run_lm_eval.py --all
```

### CLI Alternative
```bash
lm_eval --model hf \
    --model_args pretrained=Arjunvad/unified-model-stage1-action-tokens-v2 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 4
```

---

## 3. RAGAS (RAG Evaluation)

**Purpose**: Evaluate end-to-end RAG pipeline quality.

### Metrics

| Metric | Description |
|--------|-------------|
| Context Precision | Is retrieved context relevant? |
| Context Recall | Does context cover ground truth? |
| Faithfulness | Is answer faithful to context? |
| Answer Relevancy | Is answer relevant to question? |

### Run Evaluation
```bash
python benchmarks/run_ragas.py
```

### With Custom Samples
```bash
python benchmarks/run_ragas.py --num_samples 10
```

**Note**: Full RAGAS evaluation requires an OpenAI API key (for LLM-as-judge). Without it, a manual evaluation fallback is used.

---

## Results Directory Structure

```
results/
├── mteb/
│   ├── unified_model/
│   │   ├── Banking77Classification/
│   │   ├── NFCorpus/
│   │   └── summary.json
│   └── base_qwen/
│       └── ...
├── lm_eval/
│   ├── mmlu_20260204_143022/
│   └── gsm8k_20260204_150033/
└── ragas/
    ├── ragas_results.json
    └── manual_eval_results.json
```

---

## Expected Results

### Embedding Quality (MTEB)

For our unified model trained with contrastive learning, we expect:
- **Improvement over base Qwen**: Since base Qwen isn't trained for embeddings
- **Competitive with E5/BGE**: If our contrastive training is effective

### Generation Quality (LM-Eval)

Since we only did selective training (embeddings + last 2 layers):
- **Preserved base capabilities**: Should match base Qwen on most tasks
- **Slight variations**: Due to embedding layer changes

### RAG Quality (RAGAS)

- **Context Precision**: Should be high if retrieval works
- **Faithfulness**: Depends on generation quality

---

## Visualization

After running benchmarks, create visualizations:

```bash
python benchmarks/visualize_results.py
```

This generates:
- Bar charts comparing models
- Radar plots for multi-dimensional comparison
- Training progress plots

---

## References

- [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [RAGAS Documentation](https://docs.ragas.io/)
- [OpenCompass](https://github.com/open-compass/opencompass)
