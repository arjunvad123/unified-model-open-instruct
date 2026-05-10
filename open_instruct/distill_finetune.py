#!/usr/bin/env python
# Copyright 2026 SVCL/UCSD. All rights reserved.
#
# Distill embedding/retrieval geometry from Qwen3-Embedding-0.6B into
# our unified Stage 1.5 model. Generation behavior is preserved by
# construction: only LoRA adapters are trained; the un-adapted forward
# pass is bit-identical to the Stage 1.5 base.
#
# Loss: similarity-matrix KL distillation, with a small InfoNCE
# anchor for stability. See `.agents/distillation_design.md` for the
# full design rationale (loss choice, hyperparameters,
# reviewer-defense evidence plan).

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

import datasets
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

# Resilient ACT_RET import (matches the eval matrix / anchor scripts).
try:
    from open_instruct.action_tokens import ACT_RET, ACTION_TOKENS
except ImportError:
    try:
        from open_instruct.action_tokens import ACTION_TOKENS
        ACT_RET = next(t for t in ACTION_TOKENS if t == "<ACT:RET>")
    except ImportError:
        ACT_RET = "<ACT:RET>"
        ACTION_TOKENS = ["<ACT:THINK>", "<ACT:RET>", "<ACT:GEN>", "<ACT:STOP>", "<WAIT>", "<RET_RESULT>"]

logger = get_logger(__name__)


# =============================================================================
# DATA — reuse the EmbeddingDataset pattern from unified_finetune.py but
# inlined so this script doesn't depend on the full open_instruct package.
# =============================================================================
class EmbeddingDataset(Dataset):
    """Query + passage (+ optional hard negative) for distillation.

    Mirrors unified_finetune.EmbeddingDataset so checkpoints trained here
    eval cleanly through the same code path.
    """

    QUERY_PREFIX = f"{ACT_RET} "

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
            return_tensors="pt",
        )
        passage = self.tokenizer(
            item["passage"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        negative = None
        if item.get("negative"):
            negative = self.tokenizer(
                item["negative"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
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
    collated = {
        "query_input_ids": torch.stack([item["query_input_ids"] for item in batch]),
        "query_attention_mask": torch.stack([item["query_attention_mask"] for item in batch]),
        "passage_input_ids": torch.stack([item["passage_input_ids"] for item in batch]),
        "passage_attention_mask": torch.stack([item["passage_attention_mask"] for item in batch]),
    }
    negs = [it for it in batch if it["negative_input_ids"] is not None]
    if negs:
        collated["negative_input_ids"] = torch.stack([it["negative_input_ids"] for it in negs])
        collated["negative_attention_mask"] = torch.stack([it["negative_attention_mask"] for it in negs])
    else:
        collated["negative_input_ids"] = None
        collated["negative_attention_mask"] = None
    return collated


def load_medi2_examples(num_examples: int, seed: int = 42) -> List[Dict]:
    """Load MEDI2 query-positive-(negative) triples, same source Stage 1.5 used."""
    logger.info(f"Loading {num_examples} MEDI2 examples (seed={seed})")
    ds = load_dataset("GritLM/MEDI2", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    out = []
    for item in ds:
        # MEDI2 uses pos / neg lists; take the first of each.
        pos_list = item.get("pos", [])
        neg_list = item.get("neg", [])
        if not pos_list:
            continue
        record = {
            "query": item.get("query", ""),
            "passage": pos_list[0],
        }
        if neg_list:
            record["negative"] = neg_list[0]
        if not record["query"].strip() or not record["passage"].strip():
            continue
        out.append(record)
        if len(out) >= num_examples:
            break
    logger.info(f"Loaded {len(out)} MEDI2 examples ({sum(1 for r in out if 'negative' in r)} with hard negatives)")
    return out


# =============================================================================
# POOLING + ENCODE
# =============================================================================
def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the last hidden state. Matches the canonical
    `act_ret + mean` recipe established by the eval-matrix H1 result."""
    hidden = hidden.float()
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_student(model, input_ids, attention_mask) -> torch.Tensor:
    """Forward through student (causal LM, returns last hidden state)."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    pooled = mean_pool(outputs.hidden_states[-1], attention_mask)
    return F.normalize(pooled, p=2, dim=-1)


@torch.no_grad()
def encode_teacher(model, input_ids, attention_mask) -> torch.Tensor:
    """Forward through teacher (frozen). Same mean-pool over last
    hidden state. Teacher returns AutoModel-style outputs."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    pooled = mean_pool(outputs.hidden_states[-1], attention_mask)
    return F.normalize(pooled, p=2, dim=-1)


# =============================================================================
# LOSSES
# =============================================================================
def similarity_matrix_kl_loss(
    q_t: torch.Tensor, p_t: torch.Tensor,
    q_s: torch.Tensor, p_s: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    KL(P_teacher || P_student) where P_x = softmax(q_x @ p_x.T / temperature).

    Both q_t and p_t are L2-normalized teacher embeddings; same for student.
    Each row of the similarity matrix corresponds to one query's
    distribution over candidate passages. We push the student's
    distribution toward the teacher's.
    """
    sim_t = q_t @ p_t.T / temperature   # [bsz, bsz]
    sim_s = q_s @ p_s.T / temperature

    # log_softmax for student (numerator of KL), softmax for teacher (target)
    log_p_s = F.log_softmax(sim_s, dim=-1)
    p_t = F.softmax(sim_t, dim=-1)

    # KL divergence: sum_j p_t[i,j] * (log p_t[i,j] - log_p_s[i,j])
    # Use F.kl_div with reduction='batchmean' (averages over the batch dim).
    return F.kl_div(log_p_s, p_t, reduction="batchmean")


def info_nce_loss(
    q: torch.Tensor, p: torch.Tensor,
    negs: Optional[torch.Tensor] = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Standard contrastive loss on the student. Anchors training when
    distillation alone might collapse on noisy teacher signal."""
    candidates = p
    if negs is not None and negs.size(0) > 0:
        candidates = torch.cat([p, negs], dim=0)
    sim = q @ candidates.T / temperature   # [bsz, bsz + n_negs]
    labels = torch.arange(q.size(0), device=q.device)   # diagonal = positive
    return F.cross_entropy(sim, labels)


# =============================================================================
# MAIN
# =============================================================================
@dataclass
class DistillArgs:
    student_model: str = "Arjunvad/unified-model-stage1-5"
    teacher_model: str = "Qwen/Qwen3-Embedding-0.6B"
    output_dir: str = "/workspace/results/distill_v1"

    num_train_examples: int = 100_000
    max_length: int = 256
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    distill_temperature: float = 0.05
    distill_weight: float = 1.0
    info_nce_weight: float = 0.1

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    seed: int = 42
    log_every: int = 10
    save_every: int = 500
    push_to_hub_repo: Optional[str] = None
    hub_token: Optional[str] = None


def parse_args() -> DistillArgs:
    parser = argparse.ArgumentParser()
    for f in DistillArgs.__dataclass_fields__.values():
        kwargs = {"default": f.default}
        if f.type is bool or f.type == "bool":
            parser.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("true", "1", "yes"), **kwargs)
        elif f.type is int or "int" in str(f.type):
            parser.add_argument(f"--{f.name}", type=int, **kwargs)
        elif f.type is float or "float" in str(f.type):
            parser.add_argument(f"--{f.name}", type=float, **kwargs)
        else:
            parser.add_argument(f"--{f.name}", type=str, **kwargs)
    args = parser.parse_args()
    return DistillArgs(**{f.name: getattr(args, f.name) for f in DistillArgs.__dataclass_fields__.values()})


# Fields that must NEVER be serialized to disk -- they contain live
# credentials. Any persisted artifact (training_log.json, distill_summary.json,
# wandb config, anything else) MUST go through `_args_for_log()` first.
# Fix per Codex review on PR #9 (P1).
_SENSITIVE_ARG_FIELDS = frozenset({"hub_token"})


def _args_for_log(args: DistillArgs) -> Dict:
    """Return a dict-of-args with sensitive fields masked. Use this
    instead of `vars(args)` for anything that lands in a file."""
    safe = {}
    for k, v in vars(args).items():
        if k in _SENSITIVE_ARG_FIELDS:
            safe[k] = "<redacted>" if v else None
        else:
            safe[k] = v
    return safe


def main():
    args = parse_args()

    # Accelerator setup. Distillation is single-GPU friendly but we
    # configure for DDP so a future multi-GPU spot AWS run drops in
    # cleanly without rewrites.
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=120))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[init_kwargs],
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
    )

    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Pretty-print the safe view -- never leak hub_token to stdout
        # (logs may be captured by CI, sent to monitoring, etc).
        pprint(_args_for_log(args))

    # ----- Tokenizers -----
    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True, token=args.hub_token)
    if student_tok.pad_token is None:
        student_tok.pad_token = student_tok.eos_token
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True, token=args.hub_token)
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    # ----- Student (LoRA on causal LM) -----
    logger.info(f"Loading student: {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=args.hub_token,
    )
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    student = get_peft_model(student, lora_config)
    if accelerator.is_main_process:
        student.print_trainable_parameters()

    # ----- Teacher (frozen embedding model) -----
    logger.info(f"Loading teacher: {args.teacher_model}")
    try:
        teacher = AutoModel.from_pretrained(
            args.teacher_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            token=args.hub_token,
        )
    except Exception as e:
        logger.warning(f"AutoModel failed ({e!r}); falling back to AutoModelForCausalLM for teacher")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            token=args.hub_token,
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Tokenizer compatibility check, ONCE before training. Doing this
    # per-batch (as the v1 draft did) is O(vocab_size) per step and
    # delays the failure until after expensive model+data setup. Hoist
    # to startup so we fail fast with a clear error. Fix per Codex
    # review on PR #9 (P2).
    if teacher_tok.get_vocab() != student_tok.get_vocab():
        raise RuntimeError(
            f"teacher tokenizer ({args.teacher_model}) and student tokenizer "
            f"({args.student_model}) have different vocabularies. v1 of this "
            f"script reuses student-tokenized batches for teacher encoding, "
            f"which is only valid when the vocabs are identical (Qwen-family "
            f"teacher with Qwen-family student). Re-tokenization for cross-"
            f"vocab teachers is not implemented."
        )
    logger.info("Tokenizer compatibility verified (student/teacher vocabs match)")

    # ----- Data -----
    raw = load_medi2_examples(args.num_train_examples, seed=args.seed)
    dataset = EmbeddingDataset(raw, student_tok, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=embedding_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ----- Optimizer + scheduler -----
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # Use ceil so a partial accumulation cycle at the end of an epoch
    # still gets counted as one optimizer step. Floor division here
    # underestimates total_steps and causes the LR scheduler's
    # warmup/decay to terminate before the actual last step, drifting
    # the LR off-schedule for the final partial cycle. Fix per Codex
    # review on PR #9 (P2).
    if args.max_train_steps:
        total_steps = args.max_train_steps
    else:
        steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    student, teacher, optimizer, loader, scheduler = accelerator.prepare(
        student, teacher, optimizer, loader, scheduler
    )

    # ----- Training loop -----
    logger.info(f"Starting distillation ({total_steps} optimization steps)")
    completed_steps = 0
    losses = {"distill": [], "info_nce": [], "total": []}
    t0 = time.time()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(loader):
            with accelerator.accumulate(student):
                # Student encodes
                q_s = encode_student(student, batch["query_input_ids"], batch["query_attention_mask"])
                p_s = encode_student(student, batch["passage_input_ids"], batch["passage_attention_mask"])

                # Teacher encodes the SAME tokenized batch (vocab compat
                # was verified once at startup, so we can reuse the
                # student's tokenized IDs without re-encoding).
                q_t = encode_teacher(teacher, batch["query_input_ids"], batch["query_attention_mask"])
                p_t = encode_teacher(teacher, batch["passage_input_ids"], batch["passage_attention_mask"])

                # Distillation loss (similarity-matrix KL)
                distill_loss = similarity_matrix_kl_loss(
                    q_t, p_t, q_s, p_s, temperature=args.distill_temperature
                )

                # InfoNCE anchor on the student
                neg_emb = None
                if batch["negative_input_ids"] is not None:
                    neg_emb = encode_student(
                        student, batch["negative_input_ids"], batch["negative_attention_mask"]
                    )
                info_loss = info_nce_loss(q_s, p_s, neg_emb, temperature=args.distill_temperature)

                total_loss = args.distill_weight * distill_loss + args.info_nce_weight * info_loss

                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                losses["distill"].append(distill_loss.item())
                losses["info_nce"].append(info_loss.item())
                losses["total"].append(total_loss.item())

                if completed_steps % args.log_every == 0 and accelerator.is_main_process:
                    avg_d = sum(losses["distill"][-args.log_every :]) / args.log_every
                    avg_i = sum(losses["info_nce"][-args.log_every :]) / args.log_every
                    avg_t = sum(losses["total"][-args.log_every :]) / args.log_every
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {completed_steps}/{total_steps} | "
                        f"distill={avg_d:.4f} info_nce={avg_i:.4f} total={avg_t:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"elapsed={elapsed:.0f}s"
                    )

                if completed_steps % args.save_every == 0 and accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                    accelerator.unwrap_model(student).save_pretrained(ckpt_dir)
                    student_tok.save_pretrained(ckpt_dir)
                    logger.info(f"Saved checkpoint to {ckpt_dir}")

                if args.max_train_steps and completed_steps >= args.max_train_steps:
                    break

        if args.max_train_steps and completed_steps >= args.max_train_steps:
            break

    # ----- Final save -----
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        accelerator.unwrap_model(student).save_pretrained(final_dir)
        student_tok.save_pretrained(final_dir)
        with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
            # Use the redacted view so hub_token never lands on disk.
            json.dump(
                {"args": _args_for_log(args), "losses": losses, "elapsed_s": time.time() - t0},
                f, indent=2,
            )
        logger.info(f"Final checkpoint saved to {final_dir}")

        if args.push_to_hub_repo:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=final_dir,
                repo_id=args.push_to_hub_repo,
                repo_type="model",
                token=args.hub_token,
            )
            logger.info(f"Pushed final adapter to https://huggingface.co/{args.push_to_hub_repo}")


if __name__ == "__main__":
    main()
