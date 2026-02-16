# !/usr/bin/env python
# Stage 1.5: Contrastive Embedding Training for Action Token Model
#
# Self-contained script — no dependency on open_instruct.dataset_transformation
# or open_instruct.utils, so it can run in any environment with torch/transformers.
#
# When the model outputs <ACT:RET>, it should produce high-quality embeddings
# via mean pooling over hidden states, trained with InfoNCE contrastive loss.
#
# Usage:
#   python open_instruct/contrastive_finetune.py \
#       --model_name_or_path Arjunvad/unified-model-stage1-action-tokens-v2 \
#       --use_lora true \
#       --dataset_mixer_list data/medi2_contrastive_train.jsonl 1.0 \
#       --temperature 0.02 \
#       --per_device_train_batch_size 16 \
#       --learning_rate 1e-5 \
#       --num_train_epochs 1 \
#       --output_dir output/unified_embedding_v1

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
)

logger = get_logger(__name__)

# Dataset column keys
QUERY_INPUT_IDS_KEY = "query_input_ids"
QUERY_ATTENTION_MASK_KEY = "query_attention_mask"
POS_INPUT_IDS_KEY = "pos_input_ids"
POS_ATTENTION_MASK_KEY = "pos_attention_mask"
NEG_INPUT_IDS_KEY = "neg_input_ids"
NEG_ATTENTION_MASK_KEY = "neg_attention_mask"


@dataclass
class ContrastiveFinetuneArguments:
    """Arguments for contrastive embedding training."""

    # Model
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-3B-Instruct",
        metadata={"help": "Path to base model."},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Stage 1 LoRA adapter to load before training."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer. If None, uses lora_path or model_name_or_path."},
    )

    # Dataset
    dataset_mixer_list: list[str] = field(
        default_factory=lambda: ["data/contrastive_train.jsonl", "1.0"],
        metadata={"help": "Dataset path and fraction, e.g. 'data/train.jsonl 1.0'."},
    )
    dataset_mixer_list_splits: list[str] = field(
        default_factory=lambda: ["train"],
        metadata={"help": "Dataset splits to use."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization."},
    )

    # Training
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    num_train_epochs: int = field(default=3)
    max_train_steps: Optional[int] = field(default=None)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.05)
    seed: int = field(default=42)
    gradient_checkpointing: bool = field(default=True)

    # Contrastive loss
    temperature: float = field(
        default=0.02,
        metadata={"help": "Temperature for InfoNCE contrastive loss."},
    )
    use_in_batch_negatives: bool = field(
        default=True,
        metadata={"help": "Use other positives in the batch as additional negatives."},
    )

    # Output
    output_dir: str = field(default="output/unified_embedding_v1")
    logging_steps: int = field(default=10)
    checkpointing_steps: Optional[int] = field(default=500)
    clip_grad_norm: float = field(default=1.0)

    # LoRA (create new LoRA on a full model, e.g. Stage 1 model)
    use_lora: bool = field(
        default=False,
        metadata={"help": "Create a new LoRA adapter on the base model for training."},
    )
    lora_rank: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)

    # Misc
    with_tracking: bool = field(default=False)
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace repo ID to push model (e.g. Arjunvad/unified-model-stage1.5-embedding-v2)."},
    )


def load_contrastive_dataset(data_path: str, fraction: float, tokenizer, max_seq_length: int, seed: int):
    """Load JSONL triplet dataset and tokenize for contrastive training.

    Input format: {"query": "...", "positive": "...", "negative": "..."}
    Query is prefixed with <ACT:RET> to teach embedding mode.
    """
    print(f"Loading dataset from {data_path} (fraction={fraction})...")
    records = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Apply fraction
    n = int(len(records) * fraction)
    records = records[:n]
    print(f"  Loaded {len(records)} examples")

    # Tokenize
    tokenized = {
        QUERY_INPUT_IDS_KEY: [],
        QUERY_ATTENTION_MASK_KEY: [],
        POS_INPUT_IDS_KEY: [],
        POS_ATTENTION_MASK_KEY: [],
        NEG_INPUT_IDS_KEY: [],
        NEG_ATTENTION_MASK_KEY: [],
    }

    skipped = 0
    for row in tqdm(records, desc="Tokenizing"):
        query_text = "<ACT:RET> " + row["query"]
        pos_text = row["positive"]
        neg_text = row["negative"]

        query_enc = tokenizer(query_text, truncation=True, max_length=max_seq_length, padding=False)
        pos_enc = tokenizer(pos_text, truncation=True, max_length=max_seq_length, padding=False)
        neg_enc = tokenizer(neg_text, truncation=True, max_length=max_seq_length, padding=False)

        # Filter empty
        if not query_enc["input_ids"] or not pos_enc["input_ids"] or not neg_enc["input_ids"]:
            skipped += 1
            continue

        tokenized[QUERY_INPUT_IDS_KEY].append(query_enc["input_ids"])
        tokenized[QUERY_ATTENTION_MASK_KEY].append(query_enc["attention_mask"])
        tokenized[POS_INPUT_IDS_KEY].append(pos_enc["input_ids"])
        tokenized[POS_ATTENTION_MASK_KEY].append(pos_enc["attention_mask"])
        tokenized[NEG_INPUT_IDS_KEY].append(neg_enc["input_ids"])
        tokenized[NEG_ATTENTION_MASK_KEY].append(neg_enc["attention_mask"])

    print(f"  Tokenized: {len(tokenized[QUERY_INPUT_IDS_KEY])} examples (skipped {skipped})")
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.shuffle(seed=seed)
    return dataset


class ContrastiveDataCollator:
    """Pads query, positive, and negative sequences separately."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _pad_sequences(self, sequences: List[List[int]], attention_masks: List[List[int]]):
        """Pad a list of sequences to the same length."""
        max_length = max(len(seq) for seq in sequences)
        padded_ids = []
        padded_masks = []
        for seq, mask in zip(sequences, attention_masks):
            pad_length = max_length - len(seq)
            padded_ids.append(seq + [self.pad_token_id] * pad_length)
            padded_masks.append(mask + [0] * pad_length)
        return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(padded_masks, dtype=torch.long)

    def __call__(self, batch: List[Dict[str, Any]]):
        query_ids, query_mask = self._pad_sequences(
            [x[QUERY_INPUT_IDS_KEY] for x in batch],
            [x[QUERY_ATTENTION_MASK_KEY] for x in batch],
        )
        pos_ids, pos_mask = self._pad_sequences(
            [x[POS_INPUT_IDS_KEY] for x in batch],
            [x[POS_ATTENTION_MASK_KEY] for x in batch],
        )
        neg_ids, neg_mask = self._pad_sequences(
            [x[NEG_INPUT_IDS_KEY] for x in batch],
            [x[NEG_ATTENTION_MASK_KEY] for x in batch],
        )
        return {
            QUERY_INPUT_IDS_KEY: query_ids,
            QUERY_ATTENTION_MASK_KEY: query_mask,
            POS_INPUT_IDS_KEY: pos_ids,
            POS_ATTENTION_MASK_KEY: pos_mask,
            NEG_INPUT_IDS_KEY: neg_ids,
            NEG_ATTENTION_MASK_KEY: neg_mask,
        }


def get_embedding(model, input_ids, attention_mask):
    """
    Get normalized embeddings using mean pooling over the last hidden state.

    This follows the GritLM approach: use the full sequence hidden states
    and mean pool with the attention mask to get a fixed-size embedding.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    # Use last hidden state
    hidden_states = outputs.hidden_states[-1].float()

    # Mean pooling with attention mask
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embedding = summed / counts

    # L2 normalize
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding


def contrastive_loss(query_emb, pos_emb, neg_emb, temperature, use_in_batch_negatives=True):
    """
    InfoNCE contrastive loss.

    For each query, the positive should be more similar than all negatives.
    Uses in-batch negatives (other positives in the batch) for efficiency.
    """
    batch_size = query_emb.shape[0]

    if use_in_batch_negatives:
        # All docs = positives + explicit negatives
        # In-batch negatives: other positives serve as negatives for each query
        all_docs = torch.cat([pos_emb, neg_emb], dim=0)  # [2*batch, dim]
        all_sim = torch.mm(query_emb, all_docs.t()) / temperature  # [batch, 2*batch]
        # Labels: positive for query i is at position i
        labels = torch.arange(batch_size, device=query_emb.device)
    else:
        # Only use explicit positive/negative pairs
        pos_sim = (query_emb * pos_emb).sum(dim=1, keepdim=True) / temperature  # [batch, 1]
        neg_sim = (query_emb * neg_emb).sum(dim=1, keepdim=True) / temperature  # [batch, 1]
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [batch, 2]
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)

    loss = F.cross_entropy(all_sim, labels)
    return loss


def main():
    parser = HfArgumentParser((ContrastiveFinetuneArguments,))
    (args,) = parser.parse_args_into_dataclasses()

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.with_tracking else None,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Load tokenizer
    tokenizer_path = args.tokenizer_name_or_path or args.lora_path or args.model_name_or_path
    accelerator.print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    accelerator.print("Loading dataset...")
    # Parse dataset_mixer_list: [path, fraction, path2, fraction2, ...]
    data_path = args.dataset_mixer_list[0]
    fraction = float(args.dataset_mixer_list[1]) if len(args.dataset_mixer_list) > 1 else 1.0

    with accelerator.main_process_first():
        train_dataset = load_contrastive_dataset(
            data_path, fraction, tokenizer, args.max_seq_length, args.seed
        )

    accelerator.print(f"Dataset loaded: {len(train_dataset)} examples")

    # Load model
    accelerator.print(f"Loading base model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Resize embeddings for action tokens
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        accelerator.print(f"Resized embeddings to {model.get_input_embeddings().weight.shape[0]}")

    # Load Stage 1 LoRA if provided
    if args.lora_path:
        accelerator.print(f"Loading Stage 1 LoRA from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
        model.print_trainable_parameters()
    elif args.use_lora:
        # Create a new LoRA adapter on the full model (e.g. Stage 1 model loaded as base)
        from peft import LoraConfig, get_peft_model
        accelerator.print(f"Creating new LoRA adapter (r={args.lora_rank}, alpha={args.lora_alpha})...")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        # Keep embedding layer trainable for action tokens
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        model.print_trainable_parameters()
    else:
        accelerator.print("No LoRA path provided, training base model directly.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # DataLoader
    collate_fn = ContrastiveDataCollator(pad_token_id=tokenizer.pad_token_id)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    num_warmup_steps = int(args.warmup_ratio * args.max_train_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Setup HuggingFace Hub for checkpoint saving
    hf_api = None
    hub_repo_id = None
    if args.push_to_hub and accelerator.is_main_process:
        try:
            from huggingface_hub import HfApi, login
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
            hub_repo_id = args.hub_model_id or "Arjunvad/unified-model-stage1.5-embedding-v2"
            hf_api = HfApi()
            hf_api.create_repo(repo_id=hub_repo_id, private=True, exist_ok=True)
            accelerator.print(f"HuggingFace Hub ready: {hub_repo_id}")
        except Exception as e:
            accelerator.print(f"Warning: Could not setup HF Hub: {e}")
            hf_api = None

    # Training info
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("=" * 70)
    accelerator.print("***** Running Contrastive Embedding Training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {args.num_train_epochs}")
    accelerator.print(f"  Per-device batch size = {args.per_device_train_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total train batch size = {total_batch_size}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    accelerator.print(f"  Temperature = {args.temperature}")
    accelerator.print(f"  In-batch negatives = {args.use_in_batch_negatives}")
    accelerator.print(f"  Learning rate = {args.learning_rate}")
    accelerator.print("=" * 70)

    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    start_time = time.perf_counter()

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                # Get embeddings
                query_emb = get_embedding(
                    model, batch[QUERY_INPUT_IDS_KEY], batch[QUERY_ATTENTION_MASK_KEY]
                )
                pos_emb = get_embedding(
                    model, batch[POS_INPUT_IDS_KEY], batch[POS_ATTENTION_MASK_KEY]
                )
                neg_emb = get_embedding(
                    model, batch[NEG_INPUT_IDS_KEY], batch[NEG_ATTENTION_MASK_KEY]
                )

                # Compute contrastive loss
                loss = contrastive_loss(
                    query_emb, pos_emb, neg_emb,
                    temperature=args.temperature,
                    use_in_batch_negatives=args.use_in_batch_negatives,
                )

                total_loss += loss.detach().float()
                num_batches += 1

                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Only step LR scheduler on actual optimizer steps (not every micro-batch)
            if accelerator.sync_gradients:
                lr_scheduler.step()

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / num_batches
                    elapsed = time.perf_counter() - start_time
                    steps_per_sec = completed_steps / elapsed
                    accelerator.print(
                        f"  Step {completed_steps}/{args.max_train_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e} | "
                        f"Speed: {steps_per_sec:.2f} steps/s"
                    )

                # Checkpointing — save locally and push to HuggingFace
                if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        # Save LoRA adapters (doesn't modify model, safe for continued training)
                        unwrapped.save_pretrained(checkpoint_dir)
                        # Also save embedding weights separately (includes action tokens)
                        emb_path = os.path.join(checkpoint_dir, "embedding_weights.pt")
                        torch.save(unwrapped.get_input_embeddings().state_dict(), emb_path)
                        tokenizer.save_pretrained(checkpoint_dir)
                        accelerator.print(f"  Saved checkpoint at step {completed_steps}")

                        # Push checkpoint to HuggingFace
                        if hf_api and hub_repo_id:
                            try:
                                hf_api.upload_folder(
                                    folder_path=checkpoint_dir,
                                    repo_id=hub_repo_id,
                                    commit_message=f"Checkpoint at step {completed_steps}",
                                )
                                accelerator.print(f"  Pushed checkpoint to {hub_repo_id}")
                            except Exception as e:
                                accelerator.print(f"  Warning: Failed to push checkpoint: {e}")

                if completed_steps >= args.max_train_steps:
                    break

        # End of epoch
        avg_epoch_loss = accelerator.gather(total_loss).mean().item() / max(num_batches, 1)
        accelerator.print(f"\nEpoch {epoch + 1}/{args.num_train_epochs} - Avg Loss: {avg_epoch_loss:.4f}")

    # Save final model
    accelerator.print(f"\nSaving model to {args.output_dir}...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        if isinstance(unwrapped_model, PeftModel):
            # Merge LoRA into base model so we save a complete model
            # (PeftModel.save_pretrained only saves adapter weights,
            #  which would lose trained embedding weights for action tokens)
            accelerator.print("Merging LoRA weights into base model...")
            unwrapped_model = unwrapped_model.merge_and_unload()
            accelerator.print("LoRA merged. Saving full model...")
        unwrapped_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Push final model to HuggingFace Hub
    if args.push_to_hub and accelerator.is_main_process and hf_api and hub_repo_id:
        try:
            accelerator.print(f"Pushing final model to {hub_repo_id}...")
            hf_api.upload_folder(
                folder_path=args.output_dir,
                repo_id=hub_repo_id,
                commit_message="Final merged model",
            )
            accelerator.print(f"Pushed to https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            accelerator.print(f"Failed to push to hub: {e}")

    # Quick validation
    if accelerator.is_main_process:
        accelerator.print("\n" + "=" * 70)
        accelerator.print("Quick Embedding Validation")
        accelerator.print("=" * 70)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        device = next(unwrapped_model.parameters()).device

        test_cases = [
            ("What is machine learning?", "ML enables systems to learn from data.", "The weather is sunny."),
            ("Explain quantum computing", "Quantum computers use qubits and superposition.", "Cooking requires heat."),
            ("Tell me about climate change", "Climate change shifts global temperatures.", "Football is a popular sport."),
        ]

        correct = 0
        for query, pos, neg in test_cases:
            q_text = "<ACT:RET> " + query
            q_enc = tokenizer(q_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            p_enc = tokenizer(pos, return_tensors="pt", truncation=True, max_length=512).to(device)
            n_enc = tokenizer(neg, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                q_emb = get_embedding(unwrapped_model, q_enc["input_ids"], q_enc["attention_mask"])
                p_emb = get_embedding(unwrapped_model, p_enc["input_ids"], p_enc["attention_mask"])
                n_emb = get_embedding(unwrapped_model, n_enc["input_ids"], n_enc["attention_mask"])

            pos_sim = (q_emb * p_emb).sum().item()
            neg_sim = (q_emb * n_emb).sum().item()
            is_correct = pos_sim > neg_sim
            if is_correct:
                correct += 1

            accelerator.print(f"  Query: {query}")
            accelerator.print(f"    Pos sim: {pos_sim:.4f} | Neg sim: {neg_sim:.4f} | Correct: {is_correct}")

        accelerator.print(f"\n  Validation: {correct}/{len(test_cases)} correct rankings")

    accelerator.print("\n" + "=" * 70)
    accelerator.print("Contrastive Embedding Training Complete!")
    accelerator.print(f"Model saved to: {args.output_dir}")
    accelerator.print("=" * 70)


if __name__ == "__main__":
    main()
