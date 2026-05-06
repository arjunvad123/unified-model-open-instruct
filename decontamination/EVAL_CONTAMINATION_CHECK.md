# Eval-set contamination check

This file documents how to check whether any eval prompt in `benchmarks/`
overlaps with the training corpus of the Stage 1 / Stage 1.5 models.
The contamination check should run before any benchmark number
appears in a published report.

## TL;DR

- **Stage 1.5-v3 has a structural leak**: the contrastive trainer reads
  `sentence-transformers/natural-questions` *train* split, and
  `benchmarks/run_embedding_eval.py::eval_retrieval` evaluates on the
  same source. Fix this regardless of any string-match outcome.
- **No empirical n-gram scan has been run yet** — the
  [`search.py`](search.py) tool requires an Elasticsearch instance and
  ~30 GB of disk for indexing the train splits. Cluster command below.
- The 51 hardcoded smoke prompts are dumped by
  [`benchmarks/dump_smoke_prompts.py`](../benchmarks/dump_smoke_prompts.py).
  When eval scripts gain new hardcoded prompts, add them there so the
  scan stays in sync.

## Training datasets (Stage 1 + Stage 1.5)

**Stage 1** (`open_instruct/unified_finetune.py`, launched via
`scripts/nautilus/unified-training-job.yaml`):

| Dataset | Split | Sample cap | Approx full-set size |
|---|---|---|---|
| `allenai/tulu-3-sft-mixture` | train | 50K | ~939K rows |
| `ms_marco` (config `v2.1`) | train | 50K | ~808K rows |

In-code defaults that the shipped Stage 1 job does NOT use but that
flags can re-enable: `Agent-Ark/Toucan-1.5M`, `rungalileo/ragbench`,
`hotpot_qa`. Synthetic agentic trajectories are generated locally from
the above generation rows; no extra dataset.

**Stage 1.5 contrastive** (`open_instruct/contrastive_finetune.py`):

| Variant | Dataset(s) | Cap |
|---|---|---|
| `contrastive-stage1.5-medi2-v1.yaml` (shipped v2 embedding model) | `GritLM/MEDI2` train | 500K triplets |
| `contrastive-stage1.5-train-v3.yaml` | `microsoft/ms_marco` v1.1 train, `sentence-transformers/natural-questions` train, `hotpotqa/hotpot_qa` fullwiki train, `rajpurkar/squad` train | 20K each |

## Eval prompts to scan

51 hardcoded smoke prompts across three eval scripts, dumped to
`benchmarks/smoke_prompts.jsonl` by `benchmarks/dump_smoke_prompts.py`:

| Source script | Counts |
|---|---|
| `run_generation_eval.py` | 8 routing + 5 QA + 5 math + 3 code = 21 |
| `run_ragas.py` | 5 RAG questions |
| `run_embedding_eval.py` | 10 query/doc pairs (20 strings) + 5 distractors = 25 |

This covers the smoke tests only. Standard eval suites (MTEB BEIR
tasks, lm-eval-harness ARC/HellaSwag/MMLU/Winogrande/GSM8K) are
themselves canonical benchmarks — contamination against those should
be checked separately by scanning the respective HF datasets against
the training corpus, not via this file.

## Running the empirical scan (cluster)

The local laptop cannot do this — it needs Elasticsearch, the train
splits materialized (~30 GB), and ideally a GPU for the vector path.

```bash
# 1) regenerate the prompts JSONL from source (it's gitignored)
python benchmarks/dump_smoke_prompts.py

# 2) build n-gram indices for each train split
for DATASET in \
    allenai/tulu-3-sft-mixture \
    ms_marco \
    GritLM/MEDI2 \
    sentence-transformers/natural-questions \
    hotpotqa/hotpot_qa \
    rajpurkar/squad; do
  python decontamination/index.py \
    --dataset_name "$DATASET" \
    --field_name text \
    --ngram_size 13
done

# 3) scan smoke prompts against the indices
python decontamination/search.py \
  --train_dataset_names allenai/tulu-3-sft-mixture ms_marco GritLM/MEDI2 \
                        sentence-transformers/natural-questions \
                        hotpotqa/hotpot_qa rajpurkar/squad \
  --dataset benchmarks/smoke_prompts.jsonl \
  --field prompt \
  --ngram_size 13 \
  --match_threshold 0.5 \
  --output_dir decontamination_results/
```

Estimated cost: ~30 GB disk, 1 CPU node for indexing, ~30 min
runtime per dataset. Vector mode (NV-Embed-v2 paraphrase detection)
needs a GPU and is much slower; skip unless string-match comes back
clean and you want a tighter bound.

## Caveats

- 13-gram substring/n-gram match catches verbatim and near-verbatim
  contamination but misses paraphrase, translation, and structural
  overlap (same QA pair, different phrasing).
- A negative result does not prove the model "didn't see" the answer
  at training time — it only bounds verbatim leakage.
- A positive result does not necessarily invalidate the eval — context
  matters (MS MARCO has the query but the gold answer may not be in
  the surrounding text). Each hit needs human review.

## Known-without-running

**Independent of the n-gram scan**, the Stage-1.5-v3 mix uses
`sentence-transformers/natural-questions` train as a contrastive source,
and `benchmarks/run_embedding_eval.py::eval_retrieval` calls
`load_dataset("sentence-transformers/natural-questions", split="train")`
for queries. This is contamination by construction. Either swap the
eval source (e.g. BEIR/nq dev or held-out NQ-dev) or stop reporting
Stage-1.5-v3 retrieval numbers from this code path.

A `# TODO(decontam)` flag has been added next to the offending load
in `run_embedding_eval.py`.
