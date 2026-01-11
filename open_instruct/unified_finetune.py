#!/usr/bin/env python
# Copyright 2024 AllenAI. All rights reserved.
# Extended for Unified Agentic Model Training (GritLM-inspired)
#
# This script extends Open Instruct's finetune.py to support:
# 1. Combined embedding + generation loss (GritLM approach)
# 2. Action tokens for agentic behavior
# 3. Mixed dataset training (embedding pairs + generation data)

import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"
with contextlib.suppress(Exception):
    import deepspeed

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple

import datasets
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from datasets import load_dataset

from open_instruct import logger_utils, utils
from open_instruct.model_utils import push_folder_to_hub, save_with_accelerate
from open_instruct.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
)

logger = get_logger(__name__)


# =============================================================================
# ACTION TOKENS FOR AGENTIC BEHAVIOR (per Technical Report Section 3.2)
# =============================================================================
ACTION_TOKENS = [
    "<ACT:THINK>",   # Internal reasoning step
    "<ACT:RET>",     # Trigger retrieval action
    "<ACT:GEN>",     # Generate final response
    "<ACT:STOP>",    # Terminate generation
    "<WAIT>",        # Pause for external input
    "<RET_RESULT>",  # Marks retrieved content injection
]


# =============================================================================
# LOSS FUNCTIONS (GritLM-inspired)
# =============================================================================
def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for embedding extraction."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def contrastive_loss(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for embedding training.
    (per Technical Report Section 5.2)

    Args:
        q_emb: Query embeddings [batch, hidden_dim]
        p_emb: Positive embeddings [batch, hidden_dim]
        temperature: Softmax temperature (default 0.07)

    Returns:
        loss: Scalar loss value
    """
    # L2 normalize embeddings
    q_emb = F.normalize(q_emb.float(), p=2, dim=-1)
    p_emb = F.normalize(p_emb.float(), p=2, dim=-1)

    # Compute similarity matrix [batch, batch]
    similarity = torch.mm(q_emb, p_emb.T) / temperature

    # Clamp to prevent numerical overflow (per Technical Report)
    similarity = similarity.clamp(-100, 100)

    # Labels: diagonal entries should have highest similarity
    labels = torch.arange(q_emb.size(0), device=q_emb.device)

    return F.cross_entropy(similarity, labels)


# =============================================================================
# UNIFIED DATASET CLASSES
# =============================================================================
class EmbeddingDataset(Dataset):
    """Dataset for embedding training (query-passage pairs)."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        query = self.tokenizer(
            item["query"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        passage = self.tokenizer(
            item["passage"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "query_input_ids": query["input_ids"].squeeze(0),
            "query_attention_mask": query["attention_mask"].squeeze(0),
            "passage_input_ids": passage["input_ids"].squeeze(0),
            "passage_attention_mask": passage["attention_mask"].squeeze(0),
        }


class GenerationDataset(Dataset):
    """
    Dataset for generation training (instruction-response pairs).
    (per Technical Report Section 5.4)

    Only trains on predicting the response, not the instruction.
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract instruction and response
        if "instruction" in item and "response" in item:
            instruction = item['instruction']
            response = item['response']
        elif "prompt" in item and "completion" in item:
            instruction = item['prompt']
            response = item['completion']
        elif "question" in item and "answer" in item:
            instruction = item['question']
            response = item['answer']
        else:
            # Fallback: treat entire text as response
            instruction = ""
            response = item.get("text", str(item))

        # Format: instruction + response
        input_part = f"User: {instruction}\n\nAssistant: "
        full_text = input_part + response

        # Tokenize the full text
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask input portion (only predict response)
        labels = input_ids.clone()

        # Get length of input portion to mask
        input_len = len(self.tokenizer(input_part, add_special_tokens=False)["input_ids"])

        # Mask input tokens (set to -100 to ignore in loss)
        labels[:input_len] = -100
        labels[attention_mask == 0] = -100  # Ignore padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class AgenticDataset(Dataset):
    """
    Dataset for agentic behavior (trajectories with action tokens).
    (per Technical Report Section 5.4 and 7.2)

    Trajectories contain action tokens like <ACT:THINK>, <ACT:RET>, etc.
    Only trains on predicting the trajectory, not the user query.
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract query and trajectory
        if "query" in item and "trajectory" in item:
            query = item["query"]
            trajectory = item["trajectory"]
        elif "text" in item:
            # Parse text to find User: prefix
            text = item["text"]
            if "User:" in text:
                parts = text.split("\n", 1)
                query = parts[0].replace("User:", "").strip()
                trajectory = parts[1] if len(parts) > 1 else ""
            else:
                query = ""
                trajectory = text
        else:
            query = ""
            trajectory = str(item)

        # Format: query + trajectory
        input_part = f"User: {query}\n\n" if query else ""
        full_text = input_part + trajectory

        # Tokenize the full text
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask input portion (only predict trajectory)
        labels = input_ids.clone()

        if input_part:
            input_len = len(self.tokenizer(input_part, add_special_tokens=False)["input_ids"])
            labels[:input_len] = -100

        labels[attention_mask == 0] = -100  # Ignore padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_embedding_data(
    max_samples: int = 10000,
    sources: List[str] = ["toucan", "msmarco"]
) -> List[Dict]:
    """Load embedding training data from various sources."""
    data = []

    if "toucan" in sources:
        try:
            logger.info("Loading TOUCAN dataset (Agent-Ark/Toucan-1.5M, config=SFT)...")
            # TOUCAN contains tool-agentic trajectories - we extract query-response pairs
            # Available configs: 'Kimi-K2', 'OSS', 'Qwen3', 'SFT' - we use SFT for training
            toucan = load_dataset("Agent-Ark/Toucan-1.5M", "SFT", split="train", streaming=True)
            count = 0
            for item in toucan:
                if count >= max_samples // 2:
                    break
                # TOUCAN SFT keys: uuid, subset_name, question, target_tools, tools, messages
                if "question" in item and "messages" in item:
                    question = item["question"]
                    messages = item["messages"]
                    # Extract last assistant response as passage
                    if isinstance(messages, list) and len(messages) > 0:
                        # Find the last assistant message
                        for msg in reversed(messages):
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                content = msg.get("content", "")
                                if content:
                                    data.append({"query": question, "passage": content[:500]})
                                    count += 1
                                    break
        except Exception as e:
            logger.warning(f"Could not load TOUCAN: {e}")

    if "msmarco" in sources:
        try:
            logger.info("Loading MS MARCO dataset...")
            msmarco = load_dataset("ms_marco", "v2.1", split="train")
            for item in msmarco.select(range(min(len(msmarco), max_samples // 2))):
                if "query" in item and "passages" in item:
                    for p in item["passages"].get("passage_text", [])[:1]:
                        data.append({"query": item["query"], "passage": p})
        except Exception as e:
            logger.warning(f"Could not load MS MARCO: {e}")

    logger.info(f"Loaded {len(data)} embedding pairs")
    return data


def load_generation_data(
    max_samples: int = 10000,
    sources: List[str] = ["tulu3", "ragbench", "hotpotqa"]
) -> List[Dict]:
    """Load generation training data from various sources."""
    data = []

    if "tulu3" in sources:
        try:
            logger.info("Loading Tulu 3 dataset...")
            tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train")
            for item in tulu.select(range(min(len(tulu), max_samples // 3))):
                if "messages" in item:
                    messages = item["messages"]
                    if len(messages) >= 2:
                        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                        if user_msg and asst_msg:
                            data.append({"instruction": user_msg, "response": asst_msg})
        except Exception as e:
            logger.warning(f"Could not load Tulu 3: {e}")

    if "ragbench" in sources:
        try:
            logger.info("Loading RAGBench dataset...")
            ragbench = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")
            for item in ragbench.select(range(min(len(ragbench), max_samples // 3))):
                if "question" in item and "response" in item:
                    data.append({
                        "instruction": item["question"],
                        "response": item["response"]
                    })
        except Exception as e:
            logger.warning(f"Could not load RAGBench: {e}")

    if "hotpotqa" in sources:
        try:
            logger.info("Loading HotpotQA dataset...")
            hotpot = load_dataset("hotpot_qa", "fullwiki", split="train")
            for item in hotpot.select(range(min(len(hotpot), max_samples // 3))):
                if "question" in item and "answer" in item:
                    data.append({
                        "instruction": item["question"],
                        "response": item["answer"]
                    })
        except Exception as e:
            logger.warning(f"Could not load HotpotQA: {e}")

    logger.info(f"Loaded {len(data)} generation examples")
    return data


def create_agentic_trajectories(
    generation_data: List[Dict],
    max_samples: int = 5000
) -> List[Dict]:
    """
    Create synthetic agentic trajectories from generation data.
    (per Technical Report Section 7.2)

    Trajectories follow the iterative format with <WAIT> tokens
    to simulate real agentic execution.
    """
    trajectories = []

    # Trajectory templates (per Technical Report Section 4.5 and 7.2)
    templates = [
        # Template 1: Direct answer (simple queries)
        lambda q, a: f"""User: {q}

<ACT:THINK> This is a straightforward question that I can answer directly.<WAIT>

<ACT:GEN> {a}<ACT:STOP>""",

        # Template 2: Think then answer (moderate complexity)
        lambda q, a: f"""User: {q}

<ACT:THINK> Let me analyze this question and determine the best approach to answer it.<WAIT>

<ACT:THINK> I have enough knowledge to provide a comprehensive answer.<WAIT>

<ACT:GEN> {a}<ACT:STOP>""",

        # Template 3: Retrieval augmented (complex queries)
        lambda q, a: f"""User: {q}

<ACT:THINK> This requires factual information. Let me search for relevant information.<WAIT>

<ACT:RET> {' '.join(q.split()[:5]) if len(q.split()) > 5 else q}<WAIT>

<RET_RESULT>{a[:100]}...</RET_RESULT>

<ACT:THINK> I have the information needed to answer.<WAIT>

<ACT:GEN> {a}<ACT:STOP>""",

        # Template 4: Multi-step reasoning
        lambda q, a: f"""User: {q}

<ACT:THINK> This question requires careful analysis. Let me break it down.<WAIT>

<ACT:THINK> I'll consider the key aspects of this question.<WAIT>

<ACT:GEN> {a}<ACT:STOP>""",
    ]

    for i, item in enumerate(generation_data[:max_samples]):
        instruction = item.get("instruction", item.get("question", ""))
        response = item.get("response", item.get("answer", ""))

        if not instruction or not response:
            continue

        # Rotate through templates for variety
        template = templates[i % len(templates)]
        trajectory = template(instruction, response)

        trajectories.append({
            "query": instruction,
            "trajectory": trajectory.split("\n\n", 1)[1] if "\n\n" in trajectory else trajectory,
            "text": trajectory
        })

    logger.info(f"Created {len(trajectories)} agentic trajectories")
    return trajectories


# =============================================================================
# ARGUMENTS
# =============================================================================
@dataclass
class UnifiedFinetuneArguments:
    """Arguments for unified model fine-tuning."""

    exp_name: str = "unified_agentic_model"
    model_name_or_path: str = "Qwen/Qwen2.5-7B"
    model_revision: str = "main"

    # Training settings
    use_flash_attn: bool = True
    use_qlora: bool = True
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Sequence lengths
    max_seq_length: int = 1024
    embedding_max_length: int = 256

    # Training hyperparameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_train_steps: Optional[int] = None

    # Loss weighting
    contrastive_weight: float = 0.3
    temperature: float = 0.07

    # Dataset settings
    max_embedding_samples: int = 10000
    max_generation_samples: int = 10000
    max_agentic_samples: int = 5000
    embedding_sources: str = "toucan,msmarco"  # Comma-separated list
    generation_sources: str = "tulu3,ragbench,hotpotqa"  # Comma-separated list

    # Batch composition (per step)
    embedding_batch_ratio: float = 0.4
    generation_batch_ratio: float = 0.35
    agentic_batch_ratio: float = 0.25

    # Output and logging
    output_dir: str = "output/unified_model/"
    logging_steps: int = 10
    checkpointing_steps: int = 500
    keep_last_n_checkpoints: int = 3

    # Misc
    seed: int = 42
    gradient_checkpointing: bool = True
    with_tracking: bool = False
    report_to: str = "tensorboard"  # Comma-separated list (e.g., "tensorboard,wandb")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def main():
    parser = ArgumentParserPlus((UnifiedFinetuneArguments,))
    args = parser.parse()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to if args.with_tracking else None,
    )

    # Setup logging
    logger_utils.setup_logger()
    logger.info(accelerator.state, main_process_only=False)

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # ==========================================================================
    # LOAD TOKENIZER
    # ==========================================================================
    logger.info(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=True,
    )

    # Add action tokens
    logger.info(f"Adding {len(ACTION_TOKENS)} action tokens...")
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
    logger.info(f"Added {num_added} new tokens to tokenizer")

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ==========================================================================
    # LOAD MODEL
    # ==========================================================================
    logger.info(f"Loading model from {args.model_name_or_path}...")

    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_index = accelerator.local_process_index
        device_map = {"": device_index}

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        )

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Apply LoRA
    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing
            )

        logger.info("Initializing LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ==========================================================================
    # LOAD DATASETS
    # ==========================================================================
    logger.info("Loading datasets...")

    # Load embedding data
    embedding_sources_list = [s.strip() for s in args.embedding_sources.split(",")]
    embedding_data = load_embedding_data(
        max_samples=args.max_embedding_samples,
        sources=embedding_sources_list,
    )

    # Load generation data
    generation_sources_list = [s.strip() for s in args.generation_sources.split(",")]
    generation_data = load_generation_data(
        max_samples=args.max_generation_samples,
        sources=generation_sources_list,
    )

    # Create agentic trajectories
    agentic_data = create_agentic_trajectories(
        generation_data,
        max_samples=args.max_agentic_samples,
    )

    # Create datasets
    embedding_dataset = EmbeddingDataset(embedding_data, tokenizer, args.embedding_max_length)
    generation_dataset = GenerationDataset(generation_data, tokenizer, args.max_seq_length)
    agentic_dataset = AgenticDataset(agentic_data, tokenizer, args.max_seq_length)

    logger.info(f"Dataset sizes - Embedding: {len(embedding_dataset)}, "
                f"Generation: {len(generation_dataset)}, Agentic: {len(agentic_dataset)}")

    # Create dataloaders
    embedding_loader = DataLoader(
        embedding_dataset,
        batch_size=max(1, int(args.per_device_train_batch_size * args.embedding_batch_ratio)),
        shuffle=True,
    )
    generation_loader = DataLoader(
        generation_dataset,
        batch_size=max(1, int(args.per_device_train_batch_size * args.generation_batch_ratio)),
        shuffle=True,
    )
    agentic_loader = DataLoader(
        agentic_dataset,
        batch_size=max(1, int(args.per_device_train_batch_size * args.agentic_batch_ratio)),
        shuffle=True,
    )

    # Create iterators
    embedding_iter = iter(embedding_loader)
    generation_iter = iter(generation_loader)
    agentic_iter = iter(agentic_loader)

    # ==========================================================================
    # OPTIMIZER AND SCHEDULER
    # ==========================================================================
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate training steps
    total_batches = max(len(embedding_loader), len(generation_loader), len(agentic_loader))
    num_update_steps_per_epoch = math.ceil(total_batches / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    # Prepare with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    embedding_loader, generation_loader, agentic_loader = accelerator.prepare(
        embedding_loader, generation_loader, agentic_loader
    )

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    logger.info("***** Running Unified Model Training *****")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Contrastive weight = {args.contrastive_weight}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    global_step = 0

    model.train()

    for epoch in range(args.num_train_epochs):
        total_loss = 0
        total_lm_loss = 0
        total_contrastive_loss = 0

        # Reset iterators at epoch start
        embedding_iter = iter(embedding_loader)
        generation_iter = iter(generation_loader)
        agentic_iter = iter(agentic_loader)

        for step in range(num_update_steps_per_epoch):
            # Fetch batches
            try:
                emb_batch = next(embedding_iter)
            except StopIteration:
                embedding_iter = iter(embedding_loader)
                emb_batch = next(embedding_iter)

            try:
                gen_batch = next(generation_iter)
            except StopIteration:
                generation_iter = iter(generation_loader)
                gen_batch = next(generation_iter)

            try:
                agent_batch = next(agentic_iter)
            except StopIteration:
                agentic_iter = iter(agentic_loader)
                agent_batch = next(agentic_iter)

            # Single accumulation context for all forward passes + backward
            with accelerator.accumulate(model):
                # ====== EMBEDDING LOSS ======
                # Get query embeddings
                query_outputs = model(
                    input_ids=emb_batch["query_input_ids"],
                    attention_mask=emb_batch["query_attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                query_hidden = query_outputs.hidden_states[-1]
                query_emb = mean_pooling(query_hidden, emb_batch["query_attention_mask"])

                # Get passage embeddings
                passage_outputs = model(
                    input_ids=emb_batch["passage_input_ids"],
                    attention_mask=emb_batch["passage_attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                passage_hidden = passage_outputs.hidden_states[-1]
                passage_emb = mean_pooling(passage_hidden, emb_batch["passage_attention_mask"])

                # Contrastive loss
                emb_loss = contrastive_loss(query_emb, passage_emb, args.temperature)

                # ====== GENERATION LOSS ======
                gen_outputs = model(
                    input_ids=gen_batch["input_ids"],
                    attention_mask=gen_batch["attention_mask"],
                    labels=gen_batch["labels"],
                    use_cache=False,
                )
                gen_loss = gen_outputs.loss if gen_outputs.loss is not None else torch.tensor(0.0, device=accelerator.device)

                # ====== AGENTIC LOSS ======
                agent_outputs = model(
                    input_ids=agent_batch["input_ids"],
                    attention_mask=agent_batch["attention_mask"],
                    labels=agent_batch["labels"],
                    use_cache=False,
                )
                agent_loss = agent_outputs.loss if agent_outputs.loss is not None else torch.tensor(0.0, device=accelerator.device)

                # ====== COMBINED LOSS ======
                # Handle NaN values defensively
                if torch.isnan(gen_loss):
                    gen_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                if torch.isnan(agent_loss):
                    agent_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                if torch.isnan(emb_loss):
                    emb_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

                lm_loss = gen_loss + agent_loss
                combined_loss = lm_loss + args.contrastive_weight * emb_loss

                # Backward pass
                accelerator.backward(combined_loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Logging
            total_loss += combined_loss.detach().float()
            total_lm_loss += lm_loss.detach().float()
            total_contrastive_loss += emb_loss.detach().float()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                global_step += 1

                if completed_steps % args.logging_steps == 0:
                    avg_loss = total_loss.item() / args.logging_steps
                    avg_lm = total_lm_loss.item() / args.logging_steps
                    avg_contrastive = total_contrastive_loss.item() / args.logging_steps

                    logger.info(
                        f"Step {completed_steps} | Loss: {avg_loss:.4f} | "
                        f"LM: {avg_lm:.4f} | Contrastive: {avg_contrastive:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                    total_loss = 0
                    total_lm_loss = 0
                    total_contrastive_loss = 0

                # Checkpointing
                if completed_steps % args.checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                    accelerator.save_state(checkpoint_dir)

                    # Save tokenizer
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(checkpoint_dir)

                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

                    # Clean old checkpoints
                    if args.keep_last_n_checkpoints > 0:
                        clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)

            if completed_steps >= args.max_train_steps:
                break

        if completed_steps >= args.max_train_steps:
            break

    # ==========================================================================
    # SAVE FINAL MODEL
    # ==========================================================================
    logger.info("Saving final model...")
    final_dir = os.path.join(args.output_dir, "final")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    logger.info(f"Training complete! Model saved to {final_dir}")


if __name__ == "__main__":
    main()
