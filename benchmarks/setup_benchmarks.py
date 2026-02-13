#!/usr/bin/env python3
"""
Benchmark Setup for Unified Agentic Model Evaluation

This script sets up and runs benchmarks for:
1. MTEB - Embedding/Retrieval evaluation
2. OpenCompass/lm-eval - LLM evaluation (MMLU, GSM8K, HumanEval)
3. RAGAS - RAG pipeline evaluation

Usage:
    python benchmarks/setup_benchmarks.py --install    # Install dependencies
    python benchmarks/setup_benchmarks.py --check      # Check installations
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    print(f"  SUCCESS")
    return True

def install_mteb():
    """Install MTEB for embedding evaluation."""
    print("\n" + "="*60)
    print("INSTALLING MTEB (Massive Text Embedding Benchmark)")
    print("="*60)

    commands = [
        ("pip install mteb>=2.0.0", "Install MTEB"),
        ("pip install sentence-transformers", "Install SentenceTransformers"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)

def install_lm_eval():
    """Install lm-evaluation-harness for LLM benchmarks."""
    print("\n" + "="*60)
    print("INSTALLING LM-EVALUATION-HARNESS (MMLU, GSM8K, etc.)")
    print("="*60)

    commands = [
        ("pip install lm-eval", "Install lm-evaluation-harness"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)

def install_ragas():
    """Install RAGAS for RAG evaluation."""
    print("\n" + "="*60)
    print("INSTALLING RAGAS (RAG Assessment)")
    print("="*60)

    commands = [
        ("pip install ragas", "Install RAGAS"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)

def check_installations():
    """Check if all benchmarks are properly installed."""
    print("\n" + "="*60)
    print("CHECKING INSTALLATIONS")
    print("="*60)

    checks = [
        ("mteb", "import mteb; print(f'MTEB version: {mteb.__version__}')"),
        ("lm_eval", "import lm_eval; print('lm-eval installed')"),
        ("ragas", "import ragas; print(f'RAGAS version: {ragas.__version__}')"),
        ("sentence_transformers", "import sentence_transformers; print(f'SentenceTransformers installed')"),
    ]

    results = {}
    for name, check_code in checks:
        try:
            exec(check_code)
            results[name] = True
            print(f"  ✓ {name}: Installed")
        except ImportError as e:
            results[name] = False
            print(f"  ✗ {name}: NOT installed ({e})")

    return results

def print_usage_guide():
    """Print usage guide for each benchmark."""
    guide = """
================================================================================
                    BENCHMARK USAGE GUIDE
================================================================================

1. MTEB (Embedding Evaluation)
   ---------------------------
   Evaluates embedding quality on retrieval, classification, clustering, etc.

   Quick test:
   $ python benchmarks/run_mteb.py --model Arjunvad/unified-model-stage1-action-tokens-v2 --tasks Banking77Classification

   Full retrieval benchmark:
   $ python benchmarks/run_mteb.py --model YOUR_MODEL --benchmark "MTEB(eng, v2)"

2. LM-EVAL (LLM Benchmarks: MMLU, GSM8K, HumanEval)
   -------------------------------------------------
   Standard LLM evaluation benchmarks.

   MMLU (5-shot):
   $ lm_eval --model hf --model_args pretrained=Arjunvad/unified-model-stage1-action-tokens-v2 --tasks mmlu --num_fewshot 5 --batch_size 4

   GSM8K (4-shot):
   $ lm_eval --model hf --model_args pretrained=YOUR_MODEL --tasks gsm8k --num_fewshot 4 --batch_size 4

   HumanEval (0-shot):
   $ lm_eval --model hf --model_args pretrained=YOUR_MODEL --tasks humaneval --num_fewshot 0 --batch_size 1

3. RAGAS (RAG Evaluation)
   -----------------------
   Evaluates RAG pipeline quality (retrieval + generation).

   $ python benchmarks/run_ragas.py --model YOUR_MODEL

================================================================================
"""
    print(guide)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark setup for Unified Agentic Model")
    parser.add_argument("--install", action="store_true", help="Install all benchmark dependencies")
    parser.add_argument("--check", action="store_true", help="Check installations")
    parser.add_argument("--guide", action="store_true", help="Print usage guide")
    args = parser.parse_args()

    if args.install:
        install_mteb()
        install_lm_eval()
        install_ragas()
        print("\n" + "="*60)
        print("INSTALLATION COMPLETE")
        print("="*60)
        check_installations()

    if args.check:
        check_installations()

    if args.guide or (not args.install and not args.check):
        print_usage_guide()

if __name__ == "__main__":
    main()
