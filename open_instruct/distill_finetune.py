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
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, get_scheduler

from open_instruct.action_tokens import ACT_RET

logger = get_logger(__name__)


# =============================================================================
# DATA — reuse the EmbeddingDataset pattern from unified_finetune.py but
# inlined so this script doesn't depend on the full open_instruct package.
# =============================================================================
class EmbeddingDataset(Dataset):
    """Query + passage (+ optional hard negative) for distillation.

    Tokenizes separately for the student and teacher because their
    vocabularies differ: Stage 1.5 students add the project's action
    tokens (`<ACT:RET>`, etc.) which the published Qwen3-Embedding
    teacher tokenizer doesn't have. Both also use different canonical
    query prefixes per the H1 / anchor results:
      - student: `<ACT:RET> ` (the trained routing token)
      - teacher: `Instruct: Given...\nQuery: ` (Qwen-style, the
        teacher's best recipe at 0.4427 avg vs only 0.2573 for raw)
    Documents are passed without a prefix on either side (raw doc was
    the doc-prefix ablation winner for v1, and the teacher's published
    recipe is also raw doc).

    Fix per Codex review on PR #9 (round 2, P1).
    """

    def __init__(
        self,
        data: list[dict],
        student_tok,
        teacher_tok,
        student_query_prefix: str,
        teacher_query_prefix: str,
        max_length: int = 256,
    ):
        self.data = data
        self.student_tok = student_tok
        self.teacher_tok = teacher_tok
        self.student_query_prefix = student_query_prefix
        self.teacher_query_prefix = teacher_query_prefix
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _tokenize(self, tok, text):
        return tok(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        item = self.data[idx]
        s_q = self._tokenize(self.student_tok, self.student_query_prefix + item["query"])
        s_p = self._tokenize(self.student_tok, item["passage"])
        t_q = self._tokenize(self.teacher_tok, self.teacher_query_prefix + item["query"])
        t_p = self._tokenize(self.teacher_tok, item["passage"])
        s_n = None
        if item.get("negative"):
            # Hard negatives only feed the student-side InfoNCE anchor,
            # so we tokenize them only with the student tokenizer.
            s_n = self._tokenize(self.student_tok, item["negative"])
        return {
            "s_query_ids": s_q["input_ids"].squeeze(0),
            "s_query_mask": s_q["attention_mask"].squeeze(0),
            "s_passage_ids": s_p["input_ids"].squeeze(0),
            "s_passage_mask": s_p["attention_mask"].squeeze(0),
            "t_query_ids": t_q["input_ids"].squeeze(0),
            "t_query_mask": t_q["attention_mask"].squeeze(0),
            "t_passage_ids": t_p["input_ids"].squeeze(0),
            "t_passage_mask": t_p["attention_mask"].squeeze(0),
            "s_negative_ids": s_n["input_ids"].squeeze(0) if s_n else None,
            "s_negative_mask": s_n["attention_mask"].squeeze(0) if s_n else None,
        }


def embedding_collate_fn(batch):
    collated = {
        "s_query_ids": torch.stack([it["s_query_ids"] for it in batch]),
        "s_query_mask": torch.stack([it["s_query_mask"] for it in batch]),
        "s_passage_ids": torch.stack([it["s_passage_ids"] for it in batch]),
        "s_passage_mask": torch.stack([it["s_passage_mask"] for it in batch]),
        "t_query_ids": torch.stack([it["t_query_ids"] for it in batch]),
        "t_query_mask": torch.stack([it["t_query_mask"] for it in batch]),
        "t_passage_ids": torch.stack([it["t_passage_ids"] for it in batch]),
        "t_passage_mask": torch.stack([it["t_passage_mask"] for it in batch]),
    }
    negs = [it for it in batch if it.get("s_negative_ids") is not None]
    if negs:
        collated["s_negative_ids"] = torch.stack([it["s_negative_ids"] for it in negs])
        collated["s_negative_mask"] = torch.stack([it["s_negative_mask"] for it in negs])
    else:
        collated["s_negative_ids"] = None
        collated["s_negative_mask"] = None
    return collated


def _extract_medi2_text(value) -> str:
    """Extract the actual text from MEDI2's instruction/text pair schema.

    MEDI2 rows commonly store text as [instruction, text]. Positives and
    negatives are lists of those pairs, i.e. [[instruction, text], ...].
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], str):
            return value[1].strip()
        for item in value:
            text = _extract_medi2_text(item)
            if text:
                return text
        return ""
    return str(value).strip()


def load_medi2_examples(num_examples: int, seed: int = 42) -> list[dict]:
    """Load MEDI2 query-positive-(negative) triples, same source Stage 1.5 used."""
    logger.info(f"Loading {num_examples} MEDI2 examples (seed={seed})")
    ds = load_dataset("GritLM/MEDI2", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    out = []
    for item in ds:
        # MEDI2 stores query/pos/neg as instruction-text pairs. The model
        # already supplies its own query prefixes, so use only the text side.
        query = _extract_medi2_text(item.get("query"))
        passage = _extract_medi2_text(item.get("pos"))
        if not query or not passage:
            continue
        record = {"query": query, "passage": passage}
        negative = _extract_medi2_text(item.get("neg"))
        if negative:
            record["negative"] = negative
        out.append(record)
        if len(out) >= num_examples:
            break
    logger.info(f"Loaded {len(out)} MEDI2 examples ({sum(1 for r in out if 'negative' in r)} with hard negatives)")
    return out


# =============================================================================
# POOLING + ENCODE
# =============================================================================
def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the last hidden state."""
    hidden = hidden.float()
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the final non-pad token, robust to padding side. Matches
    Qwen3-Embedding's published recipe (the teacher hits 0.4427 avg
    under this; mean-pool collapses it to 0.2573 -- using mean for the
    teacher would distill from a badly-degraded signal).

    The naive `sum(mask) - 1` only works for right-padded batches; for
    left-padded inputs (e.g., mask [0,0,1,1] -> sum=2 -> index 1) it
    picks a pad position. Instead, find the LAST 1 in the mask by
    argmax-ing on the reversed mask -- works for any padding side.
    Fix per Codex review on PR #9 (round 3, P1).
    """
    hidden = hidden.float()
    seq_len = attention_mask.size(1)
    # Reverse the mask along the sequence axis, then argmax to find the
    # first 1 in the reversed view = the last 1 in the original view.
    reversed_mask = attention_mask.flip(dims=[1])
    last_from_end = reversed_mask.argmax(dim=1)
    last_valid = (seq_len - 1) - last_from_end
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_valid]


def _pool(name: str, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if name == "mean":
        return mean_pool(hidden, attention_mask)
    if name == "last_token":
        return last_token_pool(hidden, attention_mask)
    raise ValueError(f"unknown pooling: {name!r}")


def encode_student(model, input_ids, attention_mask, pooling: str) -> torch.Tensor:
    """Forward through student (causal LM, returns last hidden state).
    Pooling is configurable -- canonical recipe per H1 result is
    `mean`, but we keep it parameterized for future ablations."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    pooled = _pool(pooling, outputs.hidden_states[-1], attention_mask)
    return F.normalize(pooled, p=2, dim=-1)


@torch.no_grad()
def encode_teacher(model, input_ids, attention_mask, pooling: str) -> torch.Tensor:
    """Forward through teacher (frozen). Pooling is whatever the
    teacher's canonical recipe is -- for Qwen3-Embedding-0.6B that's
    `last_token` per the anchor result (best 0.4427 avg vs mean's
    0.2573)."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    pooled = _pool(pooling, outputs.hidden_states[-1], attention_mask)
    return F.normalize(pooled, p=2, dim=-1)


# =============================================================================
# LOSSES
# =============================================================================
def similarity_matrix_kl_loss(
    q_t: torch.Tensor, p_t: torch.Tensor, q_s: torch.Tensor, p_s: torch.Tensor, temperature: float = 0.05
) -> torch.Tensor:
    """
    KL(P_teacher || P_student) where P_x = softmax(q_x @ p_x.T / temperature).

    Both q_t and p_t are L2-normalized teacher embeddings; same for student.
    Each row of the similarity matrix corresponds to one query's
    distribution over candidate passages. We push the student's
    distribution toward the teacher's.
    """
    sim_t = q_t @ p_t.T / temperature  # [bsz, bsz]
    sim_s = q_s @ p_s.T / temperature

    # log_softmax for student (numerator of KL), softmax for teacher (target)
    log_p_s = F.log_softmax(sim_s, dim=-1)
    p_t = F.softmax(sim_t, dim=-1)

    # KL divergence: sum_j p_t[i,j] * (log p_t[i,j] - log_p_s[i,j])
    # Use F.kl_div with reduction='batchmean' (averages over the batch dim).
    return F.kl_div(log_p_s, p_t, reduction="batchmean")


def info_nce_loss(
    q: torch.Tensor, p: torch.Tensor, negs: torch.Tensor | None = None, temperature: float = 0.05
) -> torch.Tensor:
    """Standard contrastive loss on the student. Anchors training when
    distillation alone might collapse on noisy teacher signal."""
    candidates = p
    if negs is not None and negs.size(0) > 0:
        candidates = torch.cat([p, negs], dim=0)
    sim = q @ candidates.T / temperature  # [bsz, bsz + n_negs]
    labels = torch.arange(q.size(0), device=q.device)  # diagonal = positive
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
    max_train_steps: int | None = None

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    distill_temperature: float = 0.05
    distill_weight: float = 1.0
    info_nce_weight: float = 0.1

    # Per-model encoding recipes. Defaults match each model's anchor-
    # winning recipe so we distill from the best teacher signal into the
    # student under its own canonical eval recipe.
    # `student_query_prefix` is derived from `ACT_RET` at module load so
    # any future renaming of the routing token in the action_tokens
    # registry propagates here automatically. Fix per Codex review on
    # PR #9 (round 3, P2).
    student_query_prefix: str = f"{ACT_RET} "
    student_pooling: str = "mean"  # H1 winner for Stage 1.5
    teacher_query_prefix: str = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    )
    teacher_pooling: str = "last_token"  # Qwen3-Emb anchor winner

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    seed: int = 42
    log_every: int = 10
    save_every: int = 500
    push_to_hub_repo: str | None = None
    hub_token: str | None = None


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


def _args_for_log(args: DistillArgs) -> dict:
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
        args.student_model, trust_remote_code=True, torch_dtype=torch.bfloat16, token=args.hub_token
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
            args.teacher_model, trust_remote_code=True, torch_dtype=torch.bfloat16, token=args.hub_token
        )
    except Exception as e:
        logger.warning(f"AutoModel failed ({e!r}); falling back to AutoModelForCausalLM for teacher")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, trust_remote_code=True, torch_dtype=torch.bfloat16, token=args.hub_token
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Note: we deliberately do NOT require student/teacher tokenizers to
    # share a vocabulary. Stage 1.5 students add Stage 1's action tokens
    # (`<ACT:RET>`, etc.) which the published Qwen3-Embedding teacher
    # tokenizer doesn't have, so a vocab-equality check would always
    # fail with the advertised defaults. Instead, the dataset tokenizes
    # each side with its own tokenizer at __getitem__ time, with each
    # side's own canonical query prefix. Fix per Codex review on PR #9
    # (round 2, P1).
    logger.info(
        f"Student tokenizer vocab={len(student_tok)}, teacher tokenizer "
        f"vocab={len(teacher_tok)} (separate tokenization paths)"
    )

    # ----- Data -----
    raw = load_medi2_examples(args.num_train_examples, seed=args.seed)
    dataset = EmbeddingDataset(
        raw,
        student_tok=student_tok,
        teacher_tok=teacher_tok,
        student_query_prefix=args.student_query_prefix,
        teacher_query_prefix=args.teacher_query_prefix,
        max_length=args.max_length,
    )
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
        [p for p in student.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay
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

    # `from_pretrained` returns models in eval() mode by default, so
    # without this call dropout (including LoRA dropout=0.05) stays
    # disabled for the entire run. Fix per Codex review on PR #9
    # (round 2, P2).
    student.train()
    # Teacher stays in eval() mode -- dropout disabled for stable
    # distillation targets.

    # ----- Training loop -----
    logger.info(f"Starting distillation ({total_steps} optimization steps)")
    completed_steps = 0
    losses = {"distill": [], "info_nce": [], "total": []}
    t0 = time.time()

    for _epoch in range(args.num_train_epochs):
        for _step, batch in enumerate(loader):
            with accelerator.accumulate(student):
                # Student encodes its OWN tokenized inputs (with the
                # student's canonical query prefix `<ACT:RET>`).
                q_s = encode_student(student, batch["s_query_ids"], batch["s_query_mask"], args.student_pooling)
                p_s = encode_student(student, batch["s_passage_ids"], batch["s_passage_mask"], args.student_pooling)

                # Teacher encodes its OWN tokenized inputs (with the
                # teacher's Qwen-style instruction prefix). This is
                # the recipe that scored 0.4427 avg in the anchor; using
                # the student's tokenization here would feed the teacher
                # OOV action-token IDs and degrade the signal.
                q_t = encode_teacher(teacher, batch["t_query_ids"], batch["t_query_mask"], args.teacher_pooling)
                p_t = encode_teacher(teacher, batch["t_passage_ids"], batch["t_passage_mask"], args.teacher_pooling)

                # Distillation loss (similarity-matrix KL).
                # Note: q_s and q_t come from different models with
                # different hidden dims (3B Qwen2.5 = 2048, Qwen3-Emb-
                # 0.6B = 1024), but the loss is over scalar similarity
                # distributions so the dim difference is invisible.
                distill_loss = similarity_matrix_kl_loss(q_t, p_t, q_s, p_s, temperature=args.distill_temperature)

                # InfoNCE anchor on the student (uses student-tokenized
                # negatives only).
                neg_emb = None
                if batch["s_negative_ids"] is not None:
                    neg_emb = encode_student(
                        student, batch["s_negative_ids"], batch["s_negative_mask"], args.student_pooling
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
            json.dump({"args": _args_for_log(args), "losses": losses, "elapsed_s": time.time() - t0}, f, indent=2)
        logger.info(f"Final checkpoint saved to {final_dir}")

        if args.push_to_hub_repo:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_folder(
                folder_path=final_dir, repo_id=args.push_to_hub_repo, repo_type="model", token=args.hub_token
            )
            logger.info(f"Pushed final adapter to https://huggingface.co/{args.push_to_hub_repo}")


if __name__ == "__main__":
    main()
