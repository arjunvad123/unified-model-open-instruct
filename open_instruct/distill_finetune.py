#!/usr/bin/env python
# Copyright 2026 SVCL/UCSD. All rights reserved.
#
# Distill embedding/retrieval geometry from Qwen3-Embedding-0.6B into
# our unified Stage 1.5 model. Only LoRA adapters are trained, so the
# un-adapted Stage 1.5 base remains bit-identical. Generation with the
# adapter enabled is not preserved by construction; use generation replay
# or disable the adapter on generation paths if that behavior matters.
#
# Loss: similarity-matrix KL distillation, with a small InfoNCE anchor for
# stability. Optional generation replay adds KL(base logits || active-adapter
# logits) on assistant tokens to target a single adapter that preserves
# generation while improving retrieval. See `internal-log/distillation_design.md`
# for the full design rationale.

import argparse
import ast
import copy
import json
import math
import os
import re
import time
from collections import Counter
from contextlib import nullcontext
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


class GenerationReplayDataset(Dataset):
    """Chat examples for preserving generation behavior during retrieval distillation.

    The replay objective uses the Stage 1.5 base model itself as the teacher:
    the active LoRA adapter is trained to match base-model next-token logits
    on assistant tokens, while the retrieval losses train the embedding
    geometry. This directly targets the stronger "one active adapter" claim.
    """

    def __init__(self, data: list[dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        prompt_messages = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages

        input_ids = _apply_chat_template_ids(
            self.tokenizer, messages, add_generation_prompt=False, max_length=self.max_length
        )
        prompt_ids = _apply_chat_template_ids(
            self.tokenizer, prompt_messages, add_generation_prompt=True, max_length=self.max_length
        )

        labels = copy.deepcopy(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def generation_replay_collate_fn(batch, pad_token_id: int):
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - item["input_ids"].numel()
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        attention_mask.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        labels.append(F.pad(item["labels"], (0, pad_len), value=-100))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


class GenerationReplayCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        return generation_replay_collate_fn(batch, self.pad_token_id)


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


def _parse_messages(value) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return []
    if not isinstance(value, list):
        return []

    messages = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value")
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    return messages


def _messages_from_generation_row(item: dict) -> list[dict]:
    messages = _parse_messages(item.get("messages"))
    if len(messages) >= 2 and any(msg["role"] == "assistant" for msg in messages):
        return messages

    conversions = [
        ("instruction", "output"),
        ("instruction", "response"),
        ("prompt", "completion"),
        ("question", "response"),
        ("question", "answer"),
        ("query", "answer"),
        ("query", "response"),
    ]
    for prompt_key, response_key in conversions:
        prompt = item.get(prompt_key)
        response = item.get(response_key)
        if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
            context = item.get("context") or item.get("input")
            if isinstance(context, str) and context.strip():
                prompt = f"{prompt.strip()}\n\n{context.strip()}"
            return [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": response.strip()}]
    return []


def _apply_chat_template_ids(
    tokenizer, messages: list[dict], add_generation_prompt: bool, max_length: int
) -> list[int]:
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
        ids = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids)


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


GENERATION_REPLAY_CATEGORY_ORDER = ("summary", "rewrite", "code", "classification", "math", "qa", "general")
GENERATION_REPLAY_CATEGORY_PATTERNS = (
    ("summary", re.compile(r"\b(summarize|summary|tl;dr|one sentence|briefly summarize)\b", re.IGNORECASE)),
    ("rewrite", re.compile(r"\b(rewrite|rephrase|paraphrase|polish|edit|grammar|politely)\b", re.IGNORECASE)),
    (
        "code",
        re.compile(
            r"```|\b(python|javascript|typescript|java|c\+\+|function|class|program|algorithm|debug|sql|code)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "classification",
        re.compile(
            r"\b(classify|classification|categorize|category|label|sentiment|answer with one word)\b", re.IGNORECASE
        ),
    ),
    (
        "math",
        re.compile(
            r"\\\(|\\\[|\$|\b(prove|solve|equation|integer|digits?|polygon|triangle|geometry|olympiad|log_|sin\^|frac\{)\b",
            re.IGNORECASE,
        ),
    ),
    ("qa", re.compile(r"\?$|\b(what|why|how|when|where|who|explain)\b", re.IGNORECASE)),
)


def _generation_replay_category(messages: list[dict]) -> str:
    user_text = " ".join(msg["content"] for msg in messages if msg.get("role") == "user")
    for category, pattern in GENERATION_REPLAY_CATEGORY_PATTERNS:
        if pattern.search(user_text):
            return category
    return "general"


def summarize_generation_replay_examples(examples: list[dict]) -> dict:
    return {
        "num_examples": len(examples),
        "source_counts": dict(Counter(example.get("source", "unknown") for example in examples)),
        "upstream_source_counts": dict(Counter(example.get("upstream_source", "unknown") for example in examples)),
        "category_counts": dict(Counter(example.get("category", "unknown") for example in examples)),
    }


def _load_generation_replay_dataset(source: str):
    if source == "tulu3":
        logger.info("Loading generation replay data from allenai/tulu-3-sft-mixture")
        return load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    if source == "dolly15k":
        logger.info("Loading generation replay data from databricks/databricks-dolly-15k")
        return load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    if source == "alpaca":
        logger.info("Loading generation replay data from tatsu-lab/alpaca")
        return load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    if source == "code_alpaca":
        logger.info("Loading generation replay data from sahil2801/CodeAlpaca-20k")
        return load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
    if source == "ragbench":
        logger.info("Loading generation replay data from rungalileo/ragbench/hotpotqa")
        return load_dataset("rungalileo/ragbench", "hotpotqa", split="test", streaming=True)
    if source == "hotpotqa":
        logger.info("Loading generation replay data from hotpot_qa/fullwiki")
        return load_dataset("hotpot_qa", "fullwiki", split="train", streaming=True)

    logger.info(f"Loading generation replay data from {source}")
    return load_dataset(source, split="train", streaming=True)


def _balance_generation_replay_examples(examples: list[dict], num_examples: int) -> list[dict]:
    buckets = {category: [] for category in GENERATION_REPLAY_CATEGORY_ORDER}
    buckets["unknown"] = []
    for example in examples:
        buckets.setdefault(example.get("category", "unknown"), []).append(example)

    selected = []
    selected_ids = set()
    while len(selected) < num_examples:
        added = False
        for category in GENERATION_REPLAY_CATEGORY_ORDER:
            while buckets.get(category) and id(buckets[category][0]) in selected_ids:
                buckets[category].pop(0)
            if buckets.get(category):
                example = buckets[category].pop(0)
                selected.append(example)
                selected_ids.add(id(example))
                added = True
                if len(selected) >= num_examples:
                    break
        if not added:
            break

    if len(selected) < num_examples:
        for example in examples:
            if id(example) in selected_ids:
                continue
            selected.append(example)
            selected_ids.add(id(example))
            if len(selected) >= num_examples:
                break

    return selected[:num_examples]


def validate_generation_replay_coverage(
    examples: list[dict], required_categories: str, min_category_examples: int
) -> dict:
    summary = summarize_generation_replay_examples(examples)
    required = [category.strip() for category in required_categories.split(",") if category.strip()]
    if required and min_category_examples > 0:
        category_counts = summary["category_counts"]
        missing = [category for category in required if category_counts.get(category, 0) < min_category_examples]
        if missing:
            raise RuntimeError(
                "generation replay coverage check failed: "
                f"required >= {min_category_examples} examples for {missing}; "
                f"category_counts={category_counts}"
            )
    return summary


def load_generation_replay_examples(
    num_examples: int,
    seed: int = 42,
    sources: str = "tulu3",
    balance_categories: bool = True,
    scan_multiplier: int = 4,
) -> list[dict]:
    """Load chat examples for base-logit generation replay.

    Defaults to the same Tulu 3 SFT mixture used elsewhere in this repo's
    generation data path. Additional source aliases can be enabled as a
    comma-separated list: `dolly15k,alpaca,code_alpaca,tulu3,ragbench,hotpotqa`.
    """
    source_names = [s.strip().lower() for s in sources.split(",") if s.strip()]
    if num_examples <= 0:
        raise RuntimeError("generation replay requires generation_replay_num_examples > 0")
    if not source_names:
        raise RuntimeError("generation replay requires at least one generation_replay_sources entry")
    if scan_multiplier < 1:
        raise RuntimeError("generation replay scan_multiplier must be >= 1")

    candidates = []
    per_source_target = max(1, math.ceil(num_examples / len(source_names)))
    per_source_scan_limit = per_source_target * scan_multiplier
    for source in source_names:
        try:
            ds = _load_generation_replay_dataset(source)
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            source_count = 0
            for item in ds:
                messages = _messages_from_generation_row(item)
                if not messages:
                    continue
                candidates.append(
                    {
                        "messages": messages,
                        "source": source,
                        "upstream_source": item.get("source", source),
                        "category": _generation_replay_category(messages),
                    }
                )
                source_count += 1
                if source_count >= per_source_scan_limit:
                    break
            logger.info(f"Loaded {source_count} generation replay examples from {source}")
        except Exception as e:
            logger.warning(f"Could not load generation replay source {source!r}: {e}")

    if not candidates:
        raise RuntimeError(
            f"generation replay was requested but no examples loaded from sources={sources!r}; "
            "set generation_replay_weight=0 to disable it"
        )
    out = (
        _balance_generation_replay_examples(candidates, num_examples)
        if balance_categories
        else candidates[:num_examples]
    )
    summary = summarize_generation_replay_examples(out)
    logger.info(f"Loaded {len(out)} total generation replay examples: {summary}")
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


def _causal_lm_backbone(causal_lm):
    """Return a causal LM's backbone module so embedding training skips logits.

    `AutoModelForCausalLM.forward()` always materializes vocab-sized logits.
    That is unnecessary for this distillation objective and can OOM before the
    first training step. LoRA adapters are injected into the wrapped backbone
    modules, so calling that backbone directly still trains the adapter weights.
    """
    peft_base = getattr(causal_lm, "base_model", None)
    peft_wrapped = getattr(peft_base, "model", None) if peft_base is not None else None
    peft_backbone = getattr(peft_wrapped, "model", None) if peft_wrapped is not None else None
    if peft_backbone is not None:
        return peft_backbone

    direct_backbone = getattr(causal_lm, "model", None)
    if direct_backbone is not None and direct_backbone is not causal_lm:
        nested_backbone = getattr(direct_backbone, "model", None)
        return nested_backbone if nested_backbone is not None else direct_backbone

    return causal_lm


class StudentEmbeddingModel(torch.nn.Module):
    """Accelerate/DDP-wrappable student forward for embedding distillation.

    The wrapped PEFT causal LM remains the saved artifact and owner of trainable
    LoRA parameters, but the forward path returns hidden states from the
    backbone instead of materializing vocab logits.
    """

    def __init__(self, causal_lm):
        super().__init__()
        self.causal_lm = causal_lm

    def forward(self, input_ids, attention_mask, return_lm_logits: bool = False, disable_adapter: bool = False):
        if return_lm_logits:
            adapter_context = self.causal_lm.disable_adapter() if disable_adapter else nullcontext()
            was_training = self.causal_lm.training
            with adapter_context:
                try:
                    if disable_adapter:
                        self.causal_lm.eval()
                    outputs = self.causal_lm(
                        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
                    )
                finally:
                    if disable_adapter and was_training:
                        self.causal_lm.train()
            return outputs.logits

        outputs = _causal_lm_backbone(self.causal_lm)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state

    def print_trainable_parameters(self):
        return self.causal_lm.print_trainable_parameters()

    def save_pretrained(self, *args, **kwargs):
        return self.causal_lm.save_pretrained(*args, **kwargs)


def encode_student(model, input_ids, attention_mask, pooling: str) -> torch.Tensor:
    """Forward through the student backbone and pool hidden states.

    Pooling is configurable -- canonical recipe per H1 result is `mean`,
    but we keep it parameterized for future ablations.
    """
    hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = _pool(pooling, hidden_states, attention_mask)
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


def masked_next_token_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """KL(base || active adapter) on next-token assistant positions."""
    shifted_student = student_logits[:, :-1, :].float() / temperature
    shifted_teacher = teacher_logits[:, :-1, :].float() / temperature
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    if not mask.any():
        return shifted_student.sum() * 0.0

    student_log_probs = F.log_softmax(shifted_student, dim=-1)
    teacher_probs = F.softmax(shifted_teacher, dim=-1)
    per_token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return (per_token_kl * mask.float()).sum() / mask.float().sum() * (temperature**2)


def generation_replay_kl_loss(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Match active-adapter generation logits to the Stage 1.5 base.

    This uses the same PEFT model twice: first with the adapter disabled as the
    frozen base teacher, then with the adapter active as the trainable student.
    It avoids loading a second 3B base model just to provide replay targets.
    """
    with torch.no_grad():
        base_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, return_lm_logits=True, disable_adapter=True
        )
    active_logits = model(
        input_ids=input_ids, attention_mask=attention_mask, return_lm_logits=True, disable_adapter=False
    )
    return masked_next_token_kl_loss(active_logits, base_logits, labels, temperature=temperature)


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
    generation_replay_weight: float = 0.0
    generation_replay_temperature: float = 1.0
    generation_replay_num_examples: int = 0
    generation_replay_sources: str = "tulu3"
    generation_replay_max_length: int = 512
    generation_replay_batch_size: int = 1
    generation_replay_balance_categories: bool = True
    generation_replay_scan_multiplier: int = 4
    generation_replay_required_categories: str = ""
    generation_replay_min_category_examples: int = 0

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
    if args.hub_token is None:
        args.hub_token = os.environ.get("HF_TOKEN")

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
    student = StudentEmbeddingModel(get_peft_model(student, lora_config))
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

    replay_loader = None
    replay_preflight = None
    if args.generation_replay_weight > 0:
        replay_raw = load_generation_replay_examples(
            args.generation_replay_num_examples,
            seed=args.seed,
            sources=args.generation_replay_sources,
            balance_categories=args.generation_replay_balance_categories,
            scan_multiplier=args.generation_replay_scan_multiplier,
        )
        replay_preflight = validate_generation_replay_coverage(
            replay_raw,
            required_categories=args.generation_replay_required_categories,
            min_category_examples=args.generation_replay_min_category_examples,
        )
        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "generation_replay_preflight.json"), "w") as f:
                json.dump(replay_preflight, f, indent=2, sort_keys=True)
        replay_dataset = GenerationReplayDataset(
            replay_raw, tokenizer=student_tok, max_length=args.generation_replay_max_length
        )
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=args.generation_replay_batch_size,
            shuffle=True,
            collate_fn=GenerationReplayCollator(student_tok.pad_token_id),
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

    if replay_loader is not None:
        student, teacher, optimizer, loader, replay_loader, scheduler = accelerator.prepare(
            student, teacher, optimizer, loader, replay_loader, scheduler
        )
    else:
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
    losses = {"distill": [], "info_nce": [], "generation_kl": [], "total": []}
    t0 = time.time()
    replay_iter = iter(replay_loader) if replay_loader is not None else None

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

                gen_loss = None
                if replay_iter is not None:
                    try:
                        replay_batch = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        replay_batch = next(replay_iter)
                    gen_loss = generation_replay_kl_loss(
                        student,
                        replay_batch["input_ids"],
                        replay_batch["attention_mask"],
                        replay_batch["labels"],
                        temperature=args.generation_replay_temperature,
                    )

                total_loss = args.distill_weight * distill_loss + args.info_nce_weight * info_loss
                if gen_loss is not None:
                    total_loss = total_loss + args.generation_replay_weight * gen_loss

                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                losses["distill"].append(distill_loss.item())
                losses["info_nce"].append(info_loss.item())
                losses["generation_kl"].append(gen_loss.item() if gen_loss is not None else 0.0)
                losses["total"].append(total_loss.item())

                if completed_steps % args.log_every == 0 and accelerator.is_main_process:
                    avg_d = sum(losses["distill"][-args.log_every :]) / args.log_every
                    avg_i = sum(losses["info_nce"][-args.log_every :]) / args.log_every
                    avg_g = sum(losses["generation_kl"][-args.log_every :]) / args.log_every
                    avg_t = sum(losses["total"][-args.log_every :]) / args.log_every
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {completed_steps}/{total_steps} | "
                        f"distill={avg_d:.4f} info_nce={avg_i:.4f} gen_kl={avg_g:.4f} "
                        f"total={avg_t:.4f} | "
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
                {
                    "args": _args_for_log(args),
                    "losses": losses,
                    "elapsed_s": time.time() - t0,
                    "generation_replay_preflight": replay_preflight,
                },
                f,
                indent=2,
            )
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
