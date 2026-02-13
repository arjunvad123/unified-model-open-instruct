#!/usr/bin/env python3
"""
Create contrastive embedding dataset from GritLM/MEDI2.

MEDI2 contains 1.23M training examples with hard negatives mined across
300+ datasets. Each example has instruction-prefixed query/pos/neg.

We convert to our simple triplet format: {"query", "positive", "negative"}
compatible with contrastive_tokenize_v1 (which adds <ACT:RET> prefix).

Usage:
    python scripts/data/create_medi2_contrastive_dataset.py
    python scripts/data/create_medi2_contrastive_dataset.py --max_samples 500000
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm


def extract_text(item):
    """Extract text from MEDI2 format (handles both string and [instruction, text] format)."""
    if isinstance(item, list):
        if len(item) == 0:
            return ""
        # Instruction-prefixed format: [instruction, text]
        return item[-1] if len(item) > 1 else item[0]
    return item


def main():
    parser = argparse.ArgumentParser(description="Convert GritLM/MEDI2 to contrastive triplet format")
    parser.add_argument("--max_samples", type=int, default=500000,
                        help="Maximum number of samples to include (default: 500K)")
    parser.add_argument("--output", type=str, default="data/medi2_contrastive_train.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_query_len", type=int, default=5,
                        help="Minimum query length in characters")
    parser.add_argument("--min_passage_len", type=int, default=10,
                        help="Minimum passage length in characters")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Converting GritLM/MEDI2 to Contrastive Triplet Format")
    print("=" * 60)
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output: {args.output}")

    # Load MEDI2 dataset
    print("\nDownloading GritLM/MEDI2 from HuggingFace...")
    dataset = load_dataset("GritLM/MEDI2", split="train")
    print(f"  Loaded {len(dataset)} examples")

    # Convert to our format
    print("\nConverting to triplet format...")
    data = []
    skipped = 0

    # Shuffle dataset for random sampling (much faster than random index access)
    dataset = dataset.shuffle(seed=args.seed)

    for item in tqdm(dataset, desc="Processing", total=len(dataset)):
        if len(data) >= args.max_samples:
            break

        # Extract query text
        query_raw = item["query"]
        query_text = extract_text(query_raw)

        # Extract positive text (take first positive)
        pos_raw = item["pos"]
        if not pos_raw or len(pos_raw) == 0:
            skipped += 1
            continue
        pos_text = extract_text(pos_raw[0])

        # Extract negative text (pick random hard negative)
        neg_raw = item["neg"]
        if not neg_raw or len(neg_raw) == 0:
            skipped += 1
            continue
        neg_choice = random.choice(neg_raw)
        neg_text = extract_text(neg_choice)

        # Validate lengths
        if not query_text or len(query_text.strip()) < args.min_query_len:
            skipped += 1
            continue
        if not pos_text or len(pos_text.strip()) < args.min_passage_len:
            skipped += 1
            continue
        if not neg_text or len(neg_text.strip()) < args.min_passage_len:
            skipped += 1
            continue

        data.append({
            "query": query_text.strip(),
            "positive": pos_text.strip(),
            "negative": neg_text.strip(),
        })

    print(f"\n  Converted: {len(data)} triplets")
    print(f"  Skipped: {skipped} (empty/short)")

    # Shuffle the output
    random.shuffle(data)

    # Save as JSONL
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  Saved to: {args.output}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Total triplets: {len(data)}")

    # Show samples
    print(f"\n{'=' * 60}")
    print("Sample Triplets:")
    print(f"{'=' * 60}")
    for i in range(min(5, len(data))):
        print(f"\n--- Example {i+1} ---")
        print(f"Query:    {data[i]['query'][:100]}")
        print(f"Positive: {data[i]['positive'][:100]}...")
        print(f"Negative: {data[i]['negative'][:100]}...")

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
