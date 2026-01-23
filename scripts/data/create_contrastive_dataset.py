#!/usr/bin/env python3
"""
Create contrastive embedding dataset for Stage 1.5 training.

Uses 4 external retrieval datasets:
  1. MS MARCO (passage ranking)
  2. Natural Questions (Wikipedia passages)
  3. HotpotQA (multi-hop QA)
  4. SQuAD (reading comprehension)

Output format: {"query": "...", "positive": "...", "negative": "..."}

Usage:
    cd /path/to/unified-model-open-instruct
    python scripts/data/create_contrastive_dataset.py

    # Or limit samples per dataset:
    python scripts/data/create_contrastive_dataset.py --samples_per_dataset 10000
"""

import argparse
import json
import os
import random
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================
OUTPUT_PATH = "data/contrastive_train.jsonl"
SEED = 42
DEFAULT_SAMPLES_PER_DATASET = 20000  # 20K per dataset = 80K total


def process_ms_marco(num_samples: int) -> List[Dict[str, str]]:
    """
    MS MARCO passage ranking dataset.

    Format: query + 10 passages with is_selected labels.
    Positive = selected passage, Negative = non-selected passage.
    """
    print(f"\n{'='*60}")
    print("Processing MS MARCO...")
    print(f"{'='*60}")

    data = []
    dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)

    for item in tqdm(dataset, total=num_samples * 2, desc="MS MARCO"):
        query = item.get("query", "")
        passages = item.get("passages", {})

        if not query or not passages:
            continue

        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])

        # Find positive passage
        positive_idx = None
        for i, selected in enumerate(is_selected):
            if selected == 1:
                positive_idx = i
                break

        if positive_idx is None:
            continue

        positive = passage_texts[positive_idx]
        negatives = [p for i, p in enumerate(passage_texts) if i != positive_idx and p.strip()]

        if not positive.strip() or not negatives:
            continue

        negative = random.choice(negatives)

        data.append({
            "query": query,
            "positive": positive,
            "negative": negative,
        })

        if len(data) >= num_samples:
            break

    print(f"  Collected {len(data)} triplets from MS MARCO")
    return data


def process_natural_questions(num_samples: int) -> List[Dict[str, str]]:
    """
    Natural Questions with Wikipedia passage answers.

    Format: query + long Wikipedia passage answer.
    Positive = answer passage for this query.
    Negative = answer passage from a different query (in-batch style).
    """
    print(f"\n{'='*60}")
    print("Processing Natural Questions...")
    print(f"{'='*60}")

    # First collect all query-passage pairs
    pairs = []
    dataset = load_dataset("sentence-transformers/natural-questions", split="train", streaming=True)

    for item in tqdm(dataset, total=num_samples * 2, desc="NQ"):
        query = item.get("query", "")
        answer = item.get("answer", "")

        if not query.strip() or not answer.strip() or len(answer) < 50:
            continue

        pairs.append({"query": query, "answer": answer})

        if len(pairs) >= num_samples + 100:  # Extra buffer for negatives
            break

    # Create triplets with random negatives from other examples
    data = []
    for i in range(min(num_samples, len(pairs))):
        # Pick a random negative from a different example
        neg_idx = random.randint(0, len(pairs) - 1)
        while neg_idx == i:
            neg_idx = random.randint(0, len(pairs) - 1)

        data.append({
            "query": pairs[i]["query"],
            "positive": pairs[i]["answer"],
            "negative": pairs[neg_idx]["answer"],
        })

    print(f"  Collected {len(data)} triplets from Natural Questions")
    return data


def process_hotpotqa(num_samples: int) -> List[Dict[str, str]]:
    """
    HotpotQA multi-hop QA dataset.

    Format: question + context paragraphs + supporting_facts.
    Positive = sentences from supporting fact paragraphs.
    Negative = sentences from non-supporting paragraphs.
    """
    print(f"\n{'='*60}")
    print("Processing HotpotQA...")
    print(f"{'='*60}")

    data = []
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train", streaming=True)

    for item in tqdm(dataset, total=num_samples * 3, desc="HotpotQA"):
        question = item.get("question", "")
        context = item.get("context", {})
        supporting_facts = item.get("supporting_facts", {})

        if not question.strip():
            continue

        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        support_titles = set(supporting_facts.get("title", []))

        if not titles or not sentences_list or not support_titles:
            continue

        # Collect supporting and non-supporting paragraphs
        positive_sentences = []
        negative_sentences = []

        for title, sentences in zip(titles, sentences_list):
            paragraph = " ".join(sentences).strip()
            if not paragraph:
                continue
            if title in support_titles:
                positive_sentences.append(paragraph)
            else:
                negative_sentences.append(paragraph)

        if not positive_sentences or not negative_sentences:
            continue

        positive = " ".join(positive_sentences)
        negative = random.choice(negative_sentences)

        # Truncate if too long
        positive = positive[:1000]
        negative = negative[:1000]

        data.append({
            "query": question,
            "positive": positive,
            "negative": negative,
        })

        if len(data) >= num_samples:
            break

    print(f"  Collected {len(data)} triplets from HotpotQA")
    return data


def process_squad(num_samples: int) -> List[Dict[str, str]]:
    """
    SQuAD reading comprehension dataset.

    Format: question + context paragraph + answers.
    Positive = context paragraph containing the answer.
    Negative = context paragraph from a different question.
    """
    print(f"\n{'='*60}")
    print("Processing SQuAD...")
    print(f"{'='*60}")

    # Collect all question-context pairs
    pairs = []
    dataset = load_dataset("rajpurkar/squad", split="train", streaming=True)

    for item in tqdm(dataset, total=num_samples * 2, desc="SQuAD"):
        question = item.get("question", "")
        context = item.get("context", "")

        if not question.strip() or not context.strip() or len(context) < 50:
            continue

        pairs.append({"query": question, "context": context})

        if len(pairs) >= num_samples + 100:
            break

    # Create triplets with random negatives
    data = []
    for i in range(min(num_samples, len(pairs))):
        neg_idx = random.randint(0, len(pairs) - 1)
        while neg_idx == i:
            neg_idx = random.randint(0, len(pairs) - 1)

        data.append({
            "query": pairs[i]["query"],
            "positive": pairs[i]["context"],
            "negative": pairs[neg_idx]["context"],
        })

    print(f"  Collected {len(data)} triplets from SQuAD")
    return data


def main():
    parser = argparse.ArgumentParser(description="Create contrastive dataset from retrieval benchmarks")
    parser.add_argument("--samples_per_dataset", type=int, default=DEFAULT_SAMPLES_PER_DATASET,
                        help="Number of samples to collect per dataset")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Creating Contrastive Dataset from External RAG Sources")
    print("=" * 60)
    print(f"  Samples per dataset: {args.samples_per_dataset}")
    print(f"  Expected total: ~{args.samples_per_dataset * 4}")
    print(f"  Output: {args.output}")

    # Process all datasets
    all_data = []

    ms_marco_data = process_ms_marco(args.samples_per_dataset)
    all_data.extend(ms_marco_data)

    nq_data = process_natural_questions(args.samples_per_dataset)
    all_data.extend(nq_data)

    hotpotqa_data = process_hotpotqa(args.samples_per_dataset)
    all_data.extend(hotpotqa_data)

    squad_data = process_squad(args.samples_per_dataset)
    all_data.extend(squad_data)

    # Shuffle
    random.shuffle(all_data)

    # Summary
    print(f"\n{'='*60}")
    print("Dataset Summary:")
    print(f"{'='*60}")
    print(f"  MS MARCO:            {len(ms_marco_data):>6} triplets")
    print(f"  Natural Questions:   {len(nq_data):>6} triplets")
    print(f"  HotpotQA:            {len(hotpotqa_data):>6} triplets")
    print(f"  SQuAD:               {len(squad_data):>6} triplets")
    print(f"  {'─'*40}")
    print(f"  Total:               {len(all_data):>6} triplets")

    # Save as JSONL
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  Saved to: {args.output}")
    print(f"  File size: {file_size:.1f} MB")

    # Show samples
    print(f"\n{'='*60}")
    print("Sample Triplets:")
    print(f"{'='*60}")
    for i in range(4):
        idx = i * (len(all_data) // 4)
        print(f"\n--- Example {i+1} ---")
        print(f"Query:    {all_data[idx]['query'][:80]}")
        print(f"Positive: {all_data[idx]['positive'][:80]}...")
        print(f"Negative: {all_data[idx]['negative'][:80]}...")

    print(f"\n{'='*60}")
    print("Next: Train with contrastive_finetune.py")
    print(f"{'='*60}")
    print(f"""
    accelerate launch open_instruct/contrastive_finetune.py \\
        --model_name_or_path Qwen/Qwen2.5-3B-Instruct \\
        --lora_path output/action_token_stage1_large \\
        --dataset_mixer_list {args.output} 1.0 \\
        --temperature 0.05 \\
        --per_device_train_batch_size 8 \\
        --learning_rate 2e-5 \\
        --num_train_epochs 3 \\
        --output_dir output/unified_embedding_v1
""")


if __name__ == "__main__":
    main()
