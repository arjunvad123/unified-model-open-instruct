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

# Imports: prefer the open_instruct package; fall back to standalone shims when
# the package isn't installed (e.g. running this script directly on Nautilus
# inside a fresh container with only its requirements installed).
try:
    from open_instruct import logger_utils, utils
    from open_instruct.action_tokens import ACTION_TOKENS
    from open_instruct.model_utils import push_folder_to_hub, save_with_accelerate
    from open_instruct.utils import (
        ArgumentParserPlus,
        clean_last_n_checkpoints,
        get_last_checkpoint_path,
    )
except ImportError:
    # Fallback for cluster runs where the full open_instruct package isn't on
    # PYTHONPATH. Shimming the few helpers we actually use plus inlining the
    # ACTION_TOKENS list as a last resort -- registry stays the source of truth
    # everywhere else.
    import logging as _logging
    import argparse
    import dataclasses

    class _LoggerUtils:
        @staticmethod
        def setup_logger(*args, **kwargs):
            _logging.basicConfig(
                format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d — %(message)s",
                level=_logging.INFO,
            )

    logger_utils = _LoggerUtils()

    class ArgumentParserPlus:
        """Minimal replacement that parses dataclass fields into argparse."""
        def __init__(self, dataclass_types):
            self.dc_types = dataclass_types if isinstance(dataclass_types, (list, tuple)) else [dataclass_types]
        def parse(self):
            parser = argparse.ArgumentParser()
            dc = self.dc_types[0]
            for f in dataclasses.fields(dc):
                name = f"--{f.name}"
                if f.type is bool or f.type == "bool":
                    parser.add_argument(name, type=lambda x: x.lower() in ("true", "1", "yes"), default=f.default)
                elif f.type == Optional[str] or "Optional" in str(f.type):
                    parser.add_argument(name, type=str, default=f.default)
                elif f.type == Optional[int]:
                    parser.add_argument(name, type=int, default=f.default)
                elif f.type is int or f.type == "int":
                    parser.add_argument(name, type=int, default=f.default)
                elif f.type is float or f.type == "float":
                    parser.add_argument(name, type=float, default=f.default)
                else:
                    parser.add_argument(name, type=str, default=f.default)
            args = parser.parse_args()
            return dc(**{f.name: getattr(args, f.name) for f in dataclasses.fields(dc)})

    def clean_last_n_checkpoints(output_dir, n):
        """Remove old checkpoints, keeping the last n."""
        import glob, shutil
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
        for ckpt in checkpoints[:-n]:
            shutil.rmtree(ckpt, ignore_errors=True)

    def get_last_checkpoint_path(output_dir):
        import glob
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
        return checkpoints[-1] if checkpoints else None

    # Registry not on path either -- fall back to the literal trained set.
    # MUST stay in lockstep with open_instruct/action_tokens.py.
    ACTION_TOKENS = [
        "<ACT:THINK>",
        "<ACT:RET>",
        "<ACT:GEN>",
        "<ACT:STOP>",
        "<WAIT>",
        "<RET_RESULT>",
    ]

logger = get_logger(__name__)

# ACTION_TOKENS is re-exported via `open_instruct.action_tokens` (per Technical
# Report Section 3.2). Keeping a single source of truth there means the
# tokenizer the model was trained with and any downstream eval harness stay in
# lockstep.


# =============================================================================
# LOSS FUNCTIONS (GritLM-inspired)
# =============================================================================
def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for embedding extraction with numerical stability."""
    # Ensure float32 for stability
    hidden_states = hidden_states.float()
    mask = attention_mask.unsqueeze(-1).float()

    # Apply mask and sum
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)

    result = summed / counts

    # Check for NaN/Inf and handle gracefully
    if torch.isnan(result).any() or torch.isinf(result).any():
        # Fallback: use last token embedding instead
        result = hidden_states[:, -1, :]

    return result


def make_bidirectional_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert an attention mask into a 4D bidirectional mask that allows all
    tokens to attend to all other tokens (overriding the default causal mask).

    This is the key GritLM innovation: use bidirectional attention for embedding
    but causal attention for generation. Bidirectional attention gives +5.5 MTEB
    points per GritLM ablation (Table 4) with zero generation degradation.

    Accepts 2D [batch, seq_len] or already-expanded masks.
    """
    if attention_mask.dim() == 2:
        bsz, seq_len = attention_mask.shape
        # Expand to [batch, 1, seq_len, seq_len]
        # Row-wise: each query position can attend to all key positions where mask=1
        expanded = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(dtype)
    elif attention_mask.dim() == 4:
        # Already 4D (e.g., from accelerate) — fill with bidirectional pattern
        # Use the last dim to determine padding positions
        bsz, _, seq_len, _ = attention_mask.shape
        # Extract 2D mask: a position is valid if any attention is allowed from it
        mask_2d = (attention_mask[:, 0, 0, :] > torch.finfo(dtype).min * 0.5).float()
        expanded = mask_2d[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(dtype)
    else:
        # 3D or other — just return as-is, let the model handle it
        return attention_mask

    # Invert: 1→0.0 (attend), 0→large_neg (mask out padding)
    inverted = (1.0 - expanded) * torch.finfo(dtype).min
    return inverted


def contrastive_loss(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    temperature: float = 0.07,
    debug: bool = False
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for embedding training.
    (per Technical Report Section 5.2)

    Args:
        q_emb: Query embeddings [batch, hidden_dim]
        p_emb: Positive embeddings [batch, hidden_dim]
        temperature: Softmax temperature (default 0.07)
        debug: If True, print debug info

    Returns:
        loss: Scalar loss value
    """
    batch_size = q_emb.size(0)

    # Ensure we have enough samples for contrastive learning
    if batch_size < 2:
        if debug:
            logger.warning(f"Contrastive batch size too small: {batch_size}, returning zero loss")
        return torch.tensor(0.0, device=q_emb.device, requires_grad=True)

    # Convert to float32 for numerical stability
    q_emb = q_emb.float()
    p_emb = p_emb.float()

    # Check for NaN/Inf in inputs
    if torch.isnan(q_emb).any() or torch.isinf(q_emb).any():
        if debug:
            logger.warning("NaN/Inf detected in query embeddings")
        return torch.tensor(0.0, device=q_emb.device, requires_grad=True)

    if torch.isnan(p_emb).any() or torch.isinf(p_emb).any():
        if debug:
            logger.warning("NaN/Inf detected in passage embeddings")
        return torch.tensor(0.0, device=q_emb.device, requires_grad=True)

    # L2 normalize embeddings (add small epsilon for numerical stability)
    q_norm = q_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    p_norm = p_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q_emb = q_emb / q_norm
    p_emb = p_emb / p_norm

    # Compute similarity matrix [batch, batch]
    similarity = torch.mm(q_emb, p_emb.T) / temperature

    # Clamp to prevent numerical overflow (per Technical Report)
    similarity = similarity.clamp(-100, 100)

    # Labels: diagonal entries should have highest similarity
    labels = torch.arange(batch_size, device=q_emb.device)

    loss = F.cross_entropy(similarity, labels)

    if debug and torch.isnan(loss):
        logger.warning(f"NaN loss detected! Similarity stats: min={similarity.min():.4f}, max={similarity.max():.4f}")

    return loss


# =============================================================================
# UNIFIED DATASET CLASSES
# =============================================================================
class EmbeddingDataset(Dataset):
    """Dataset for embedding training (query-passage pairs)."""

    QUERY_PREFIX = "<ACT:RET> "

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        query = self.tokenizer(
            self.QUERY_PREFIX + item["query"],
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

        negative = None
        if item.get("negative"):
            negative = self.tokenizer(
                item["negative"],
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
            "negative_input_ids": negative["input_ids"].squeeze(0) if negative else None,
            "negative_attention_mask": negative["attention_mask"].squeeze(0) if negative else None,
        }


def embedding_collate_fn(batch):
    """Collate fn that handles optional MEDI2 hard negatives."""
    collated = {
        "query_input_ids": torch.stack([item["query_input_ids"] for item in batch]),
        "query_attention_mask": torch.stack([item["query_attention_mask"] for item in batch]),
        "passage_input_ids": torch.stack([item["passage_input_ids"] for item in batch]),
        "passage_attention_mask": torch.stack([item["passage_attention_mask"] for item in batch]),
    }

    negatives = [item for item in batch if item["negative_input_ids"] is not None]
    if negatives:
        collated["negative_input_ids"] = torch.stack([item["negative_input_ids"] for item in negatives])
        collated["negative_attention_mask"] = torch.stack([item["negative_attention_mask"] for item in negatives])
    else:
        collated["negative_input_ids"] = None
        collated["negative_attention_mask"] = None

    return collated


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
    sources: List[str] = ["medi2", "toucan", "msmarco"]
) -> List[Dict]:
    """Load embedding training data from various sources."""
    data = []

    if "medi2" in sources:
        try:
            logger.info("Loading MEDI2 dataset (GritLM/MEDI2)...")
            medi2 = load_dataset("GritLM/MEDI2", split="train", streaming=True)
            count = 0
            per_source_max = max_samples // max(len(sources), 1)
            for item in medi2:
                if count >= per_source_max:
                    break
                # MEDI2 format: query, pos, neg, task_name
                if "query" in item and "pos" in item:
                    pos_text = item["pos"]
                    if isinstance(pos_text, list):
                        pos_text = pos_text[0] if pos_text else ""
                    neg_text = item.get("neg")
                    if isinstance(neg_text, list):
                        neg_text = neg_text[0] if neg_text else None
                    if pos_text:
                        row = {"query": item["query"], "passage": str(pos_text)[:1000]}
                        if neg_text:
                            row["negative"] = str(neg_text)[:1000]
                        data.append(row)
                        count += 1
            logger.info(f"Loaded {count} MEDI2 pairs")
        except Exception as e:
            logger.warning(f"Could not load MEDI2: {e}")

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
    model_name_or_path: str = "Arjunvad/unified-model-stage1-action-tokens-v2"
    model_revision: str = "main"

    # Training settings
    use_flash_attn: bool = True
    use_qlora: bool = False
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Bidirectional attention for embedding (GritLM key innovation)
    # When True, embedding forward passes use bidirectional (non-causal) attention.
    # GritLM Table 4: bidirectional gives +5.5 MTEB points with zero gen degradation.
    use_bidirectional_embedding: bool = True

    # Sequence lengths
    max_seq_length: int = 1024
    embedding_max_length: int = 512

    # Training hyperparameters
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None

    # Loss weighting (GritLM uses 1.0 — equal weight for emb and gen)
    contrastive_weight: float = 1.0
    temperature: float = 0.02

    # Embedding-specific settings (CRITICAL for good retrieval)
    embedding_batch_size: int = 32  # Larger batch = more negatives = better embeddings
    gather_embeddings_across_gpus: bool = True  # Gather across GPUs for even larger effective batch

    # Dataset settings
    max_embedding_samples: int = 500000
    max_generation_samples: int = 100000
    max_agentic_samples: int = 25000
    embedding_sources: str = "medi2,toucan,msmarco"  # Comma-separated list
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

    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


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
    # IMPORTANT: Contrastive learning needs large batch size for meaningful negatives
    # State-of-the-art embedding models use 256-2048 batch size
    # We use a dedicated embedding_batch_size argument (default 32)
    effective_embedding_batch = max(args.embedding_batch_size, 16)  # Minimum 16 for decent negatives
    logger.info(f"Embedding batch size: {effective_embedding_batch} "
                f"(gather_across_gpus: {args.gather_embeddings_across_gpus})")
    if args.gather_embeddings_across_gpus:
        logger.info(f"Effective batch with {accelerator.num_processes} GPUs: "
                   f"{effective_embedding_batch * accelerator.num_processes}")

    embedding_loader = DataLoader(
        embedding_dataset,
        batch_size=effective_embedding_batch,
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch for consistent contrastive learning
        collate_fn=embedding_collate_fn,
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

    # Prepare with accelerator FIRST (before calculating steps)
    # This is important because accelerator.prepare() shards the dataloaders
    model, optimizer = accelerator.prepare(model, optimizer)
    embedding_loader, generation_loader, agentic_loader = accelerator.prepare(
        embedding_loader, generation_loader, agentic_loader
    )

    # Calculate training steps AFTER prepare (so we get the correct sharded lengths)
    # Each GPU sees len(dataloader) batches per epoch
    # With accelerator.accumulate(), we iterate over ALL batches and it handles accumulation
    total_batches_per_epoch = max(len(embedding_loader), len(generation_loader), len(agentic_loader))
    # Number of optimizer updates per epoch = total_batches / gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(total_batches_per_epoch / args.gradient_accumulation_steps)

    logger.info(f"Batches per epoch (after DDP sharding): {total_batches_per_epoch}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Update steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"Loop will iterate {total_batches_per_epoch} times per epoch (micro-batches)")

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    # Prepare scheduler after creating it
    lr_scheduler = accelerator.prepare(lr_scheduler)

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

    logger.info("=" * 70)
    logger.info("JOINT TRAINING CONFIGURATION")
    logger.info(f"  Bidirectional embedding:  {args.use_bidirectional_embedding}")
    logger.info(f"  Contrastive weight:       {args.contrastive_weight}")
    logger.info(f"  Temperature:              {args.temperature}")
    logger.info(f"  LoRA rank:                {args.lora_rank}")
    logger.info(f"  Learning rate:            {args.learning_rate}")
    logger.info(f"  Embedding samples:        {len(embedding_dataset)}")
    logger.info(f"  Generation samples:       {len(generation_dataset)}")
    logger.info(f"  Agentic samples:          {len(agentic_dataset)}")
    logger.info(f"  Max train steps:          {args.max_train_steps}")
    logger.info("=" * 70)

    for epoch in range(args.num_train_epochs):
        total_loss = 0
        total_lm_loss = 0
        total_contrastive_loss = 0

        # Reset iterators at epoch start
        embedding_iter = iter(embedding_loader)
        generation_iter = iter(generation_loader)
        agentic_iter = iter(agentic_loader)

        for step in range(total_batches_per_epoch):
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

            # Debug mode for first 5 steps
            debug_mode = (completed_steps < 5)

            # Single accumulation context for all forward passes + backward
            with accelerator.accumulate(model):
                # ====== EMBEDDING LOSS (bidirectional attention per GritLM) ======
                # Build attention kwargs for embedding forward passes.
                # If use_bidirectional_embedding is True, we construct a 4D
                # attention mask that allows full bidirectional attention
                # (all tokens attend to all tokens). This is the key GritLM
                # innovation: +5.5 MTEB points with zero generation impact.
                query_attn_kwargs = {}
                passage_attn_kwargs = {}
                negative_attn_kwargs = {}
                has_negatives = emb_batch.get("negative_input_ids") is not None
                if args.use_bidirectional_embedding:
                    query_attn_kwargs["attention_mask"] = make_bidirectional_attention_mask(
                        emb_batch["query_attention_mask"], dtype=torch.bfloat16
                    )
                    passage_attn_kwargs["attention_mask"] = make_bidirectional_attention_mask(
                        emb_batch["passage_attention_mask"], dtype=torch.bfloat16
                    )
                    if has_negatives:
                        negative_attn_kwargs["attention_mask"] = make_bidirectional_attention_mask(
                            emb_batch["negative_attention_mask"], dtype=torch.bfloat16
                        )
                else:
                    query_attn_kwargs["attention_mask"] = emb_batch["query_attention_mask"]
                    passage_attn_kwargs["attention_mask"] = emb_batch["passage_attention_mask"]
                    if has_negatives:
                        negative_attn_kwargs["attention_mask"] = emb_batch["negative_attention_mask"]

                # Get query embeddings
                query_outputs = model(
                    input_ids=emb_batch["query_input_ids"],
                    output_hidden_states=True,
                    use_cache=False,
                    **query_attn_kwargs,
                )
                query_hidden = query_outputs.hidden_states[-1]
                # Pool using the original 2D mask (not the 4D bidirectional one)
                query_emb = mean_pooling(query_hidden, emb_batch["query_attention_mask"])

                # Get passage embeddings
                passage_outputs = model(
                    input_ids=emb_batch["passage_input_ids"],
                    output_hidden_states=True,
                    use_cache=False,
                    **passage_attn_kwargs,
                )
                passage_hidden = passage_outputs.hidden_states[-1]
                passage_emb = mean_pooling(passage_hidden, emb_batch["passage_attention_mask"])

                negative_emb = None
                if has_negatives:
                    negative_outputs = model(
                        input_ids=emb_batch["negative_input_ids"],
                        output_hidden_states=True,
                        use_cache=False,
                        **negative_attn_kwargs,
                    )
                    negative_hidden = negative_outputs.hidden_states[-1]
                    negative_emb = mean_pooling(negative_hidden, emb_batch["negative_attention_mask"])

                # Debug: log embedding statistics for first few steps
                if debug_mode and accelerator.is_main_process:
                    logger.info(f"[DEBUG Step {completed_steps}] Query emb shape: {query_emb.shape}, "
                               f"norm: {query_emb.norm(dim=-1).mean():.4f}, "
                               f"has_nan: {torch.isnan(query_emb).any()}")
                    logger.info(f"[DEBUG Step {completed_steps}] Passage emb shape: {passage_emb.shape}, "
                               f"norm: {passage_emb.norm(dim=-1).mean():.4f}, "
                               f"has_nan: {torch.isnan(passage_emb).any()}")

                # Gather embeddings across GPUs for larger effective batch (CRITICAL for good embeddings)
                # This increases the number of negatives dramatically
                if args.gather_embeddings_across_gpus and accelerator.num_processes > 1:
                    # Gather all embeddings across GPUs
                    all_query_emb = accelerator.gather(query_emb)
                    all_passage_emb = accelerator.gather(passage_emb)

                    if debug_mode and accelerator.is_main_process:
                        logger.info(f"[DEBUG Step {completed_steps}] After gathering: "
                                   f"query shape {all_query_emb.shape}, passage shape {all_passage_emb.shape}")

                    if negative_emb is not None:
                        all_passage_emb = torch.cat([all_passage_emb, negative_emb], dim=0)

                    # Compute contrastive loss on gathered embeddings (all processes compute same loss)
                    emb_loss = contrastive_loss(all_query_emb, all_passage_emb, args.temperature, debug=debug_mode)
                else:
                    # Single GPU: use local batch only
                    candidate_emb = passage_emb
                    if negative_emb is not None:
                        candidate_emb = torch.cat([candidate_emb, negative_emb], dim=0)
                    emb_loss = contrastive_loss(query_emb, candidate_emb, args.temperature, debug=debug_mode)

                # Debug: log contrastive loss
                if debug_mode and accelerator.is_main_process:
                    logger.info(f"[DEBUG Step {completed_steps}] Contrastive loss: {emb_loss.item():.4f}, "
                               f"is_nan: {torch.isnan(emb_loss).item()}")

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
        # Merge LoRA weights into base model for clean deployment
        from peft import PeftModel
        if isinstance(unwrapped_model, PeftModel):
            logger.info("Merging LoRA adapters into base model...")
            unwrapped_model = unwrapped_model.merge_and_unload()

        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Push to HuggingFace Hub
        if args.push_to_hub and args.hub_model_id:
            logger.info(f"Pushing model to HuggingFace Hub: {args.hub_model_id}")
            try:
                api = HfApi()
                api.upload_folder(
                    folder_path=final_dir,
                    repo_id=args.hub_model_id,
                    repo_type="model",
                    commit_message=f"Joint training: {args.exp_name} (bidirectional={args.use_bidirectional_embedding}, "
                                   f"lr={args.learning_rate}, rank={args.lora_rank}, temp={args.temperature})",
                )
                logger.info(f"Successfully pushed to {args.hub_model_id}")
            except Exception as e:
                logger.warning(f"Failed to push to hub: {e}")

    logger.info(f"Training complete! Model saved to {final_dir}")


if __name__ == "__main__":
    main()
