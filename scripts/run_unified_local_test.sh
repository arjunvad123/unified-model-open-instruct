#!/bin/bash
# Local test script for Unified Agentic Model training
# This runs a quick test with small model and minimal data

set -e

echo "=========================================="
echo "Unified Agentic Model - Local Test"
echo "=========================================="

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Working directory: $(pwd)"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q torch transformers datasets accelerate peft bitsandbytes tqdm rich

# Run the training
echo ""
echo "Starting local test training..."
echo "This will run 50 steps with a small model (Qwen2.5-0.5B)"
echo ""

python -m open_instruct.unified_finetune \
    --exp_name unified_local_test \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --use_flash_attn false \
    --use_qlora false \
    --use_lora true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --max_seq_length 512 \
    --embedding_max_length 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_train_steps 50 \
    --max_embedding_samples 100 \
    --max_generation_samples 100 \
    --max_agentic_samples 50 \
    --embedding_sources msmarco \
    --generation_sources hotpotqa \
    --output_dir output/unified_local_test/ \
    --logging_steps 5 \
    --checkpointing_steps 25 \
    --with_tracking false

echo ""
echo "=========================================="
echo "Local test complete!"
echo "Output saved to: output/unified_local_test/"
echo "=========================================="
