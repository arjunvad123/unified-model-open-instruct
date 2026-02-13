#!/bin/bash
# MTEB Retrieval Benchmark Script for GPU (Nautilus Cluster)
#
# Run this on a node with GPU access:
#   sbatch run_mteb_gpu.sh
#
# Or interactively:
#   python benchmarks/run_mteb_comparison.py --model all --tasks full --output-dir results/mteb

#SBATCH --job-name=mteb-benchmark
#SBATCH --output=mteb_%j.out
#SBATCH --error=mteb_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Activate your conda environment
# source ~/.bashrc
# conda activate your_env

# Set HuggingFace cache to persistent storage
export HF_HOME=/path/to/your/hf_cache
export TRANSFORMERS_CACHE=/path/to/your/hf_cache

cd /path/to/unified-model-open-instruct-new

# Run benchmarks on all models
echo "Starting MTEB Retrieval Benchmarks..."
echo "Models: unified_model, base_qwen_3b, gte_qwen2_1.5b"
echo "Tasks: NFCorpus, SciFact, ArguAna, SCIDOCS, FiQA2018, TRECCOVID"

python benchmarks/run_mteb_comparison.py \
    --model all \
    --tasks full \
    --output-dir results/mteb

echo "Benchmark complete. Results in results/mteb/"
