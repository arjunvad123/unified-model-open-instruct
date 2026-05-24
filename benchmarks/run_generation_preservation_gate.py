from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from open_instruct.action_tokens import ACTION_TOKENS


@dataclass(frozen=True)
class PromptSpec:
    id: str
    user: str
    max_new_tokens: int
    min_chars: int
    max_chars: int | None = None
    required_regex: str | None = None
    max_non_ascii_ratio: float = 0.08
    category: str = "general"


SMOKE_PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec(
        id="plain_explanation",
        user="Explain in two sentences why clean water matters for public health.",
        max_new_tokens=96,
        min_chars=70,
        category="explanation",
    ),
    PromptSpec(
        id="simple_math",
        user=("Solve: if a train travels 120 miles in 2 hours, what is its average speed? Give the number and units."),
        max_new_tokens=48,
        min_chars=8,
        max_chars=160,
        required_regex=r"\b60\b|sixty",
        category="math",
    ),
    PromptSpec(
        id="small_code",
        user="Write a Python function called is_palindrome(s) that ignores case and spaces.",
        max_new_tokens=128,
        min_chars=60,
        required_regex=r"def\s+is_palindrome|return",
        category="code",
    ),
    PromptSpec(
        id="concise_classification",
        user=(
            "Classify this request as generate, retrieve, or stop: "
            "Find papers about LoRA adapters. Answer with one word."
        ),
        max_new_tokens=16,
        min_chars=3,
        max_chars=40,
        required_regex=r"^\s*retrieve\b",
        category="classification",
    ),
    PromptSpec(
        id="short_summary",
        user="Summarize this in one sentence: LoRA adapters update a small number of weights during fine-tuning.",
        max_new_tokens=64,
        min_chars=35,
        max_chars=220,
        category="summary",
    ),
    PromptSpec(
        id="rewrite",
        user="Rewrite this politely: send me the file now.",
        max_new_tokens=48,
        min_chars=25,
        max_chars=180,
        category="rewrite",
    ),
    PromptSpec(
        id="factual_qa",
        user="In one sentence, what does a tokenizer do in a language model pipeline?",
        max_new_tokens=72,
        min_chars=45,
        max_chars=260,
        category="qa",
    ),
    PromptSpec(
        id="ordered_steps",
        user="List three short steps for checking whether a machine-learning experiment finished successfully.",
        max_new_tokens=96,
        min_chars=55,
        max_chars=360,
        category="steps",
    ),
)

EXPANDED_PROMPTS: tuple[PromptSpec, ...] = SMOKE_PROMPTS + (
    PromptSpec(
        id="exact_hello",
        user="Reply with exactly this word: hello",
        max_new_tokens=8,
        min_chars=5,
        max_chars=24,
        required_regex=r"^\s*hello\b",
        category="exact",
    ),
    PromptSpec(
        id="exact_ok",
        user="Reply with exactly these two letters: OK",
        max_new_tokens=8,
        min_chars=2,
        max_chars=24,
        required_regex=r"^\s*ok\b",
        category="exact",
    ),
    PromptSpec(
        id="yes_no_fact",
        user="Answer yes or no: is clean drinking water important for health?",
        max_new_tokens=8,
        min_chars=2,
        max_chars=24,
        required_regex=r"^\s*yes\b",
        category="classification",
    ),
    PromptSpec(
        id="extract_city",
        user="Extract the city from this sentence and answer only with the city: I moved to San Diego in 2024.",
        max_new_tokens=16,
        min_chars=8,
        max_chars=48,
        required_regex=r"san\s+diego",
        category="extraction",
    ),
    PromptSpec(
        id="capital_completion",
        user="Complete the sentence with one word: The capital of France is",
        max_new_tokens=12,
        min_chars=5,
        max_chars=32,
        required_regex=r"\bparis\b",
        category="qa",
    ),
    PromptSpec(
        id="json_yes",
        user='Return JSON with one key "answer" and value "yes".',
        max_new_tokens=32,
        min_chars=15,
        max_chars=96,
        required_regex=r'"answer"\s*:\s*"yes"',
        category="format",
    ),
    PromptSpec(
        id="two_item_list",
        user="List exactly two common sources of renewable energy.",
        max_new_tokens=40,
        min_chars=12,
        max_chars=160,
        required_regex=r"solar|wind|hydro|geothermal",
        category="list",
    ),
    PromptSpec(
        id="one_sentence_definition",
        user="Define photosynthesis in one sentence.",
        max_new_tokens=48,
        min_chars=35,
        max_chars=220,
        required_regex=r"plants|light|energy|sunlight",
        category="definition",
    ),
    PromptSpec(
        id="polite_email_sentence",
        user="Write one polite sentence asking a teammate to review a pull request.",
        max_new_tokens=48,
        min_chars=35,
        max_chars=220,
        required_regex=r"review|pull request|pr",
        category="rewrite",
    ),
    PromptSpec(
        id="unit_conversion",
        user="How many centimeters are in 2 meters? Answer with the number and unit.",
        max_new_tokens=24,
        min_chars=5,
        max_chars=80,
        required_regex=r"\b200\b|two hundred",
        category="math",
    ),
    PromptSpec(
        id="short_summary_2",
        user="Summarize in one sentence: A GPU can accelerate matrix multiplication for neural network training.",
        max_new_tokens=48,
        min_chars=35,
        max_chars=220,
        required_regex=r"gpu|matrix|neural|training",
        category="summary",
    ),
    PromptSpec(
        id="tiny_python",
        user="Write a Python function add_one(x) that returns x plus one.",
        max_new_tokens=64,
        min_chars=35,
        max_chars=260,
        required_regex=r"def\s+add_one|return\s+x\s*\+\s*1",
        category="code",
    ),
)

PROMPT_SUITES: dict[str, tuple[PromptSpec, ...]] = {"smoke": SMOKE_PROMPTS, "expanded": EXPANDED_PROMPTS}
PROMPTS = SMOKE_PROMPTS

CHAT_ARTIFACT_RE = re.compile(r"<\|im_(?:start|end)\|>|<s>|</s>")
ACTION_TOKEN_RE = re.compile("|".join(re.escape(token) for token in ACTION_TOKENS))
WORD_RE = re.compile(r"\w+")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an active LoRA adapter against the same model with the adapter disabled. "
            "The gate is intentionally stricter than the older qualitative sanity checks: it reports "
            "deterministic generations, text-quality checks, and next-token KL against the disabled adapter."
        )
    )
    parser.add_argument("--base-model-id", default="Arjunvad/unified-model-stage1-5")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--prompt-suite", choices=tuple(PROMPT_SUITES), default="smoke")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--mean-kl-threshold", type=float, default=0.12)
    parser.add_argument("--p95-kl-threshold", type=float, default=0.75)
    parser.add_argument("--min-quality-pass-rate", type=float, default=0.875)
    parser.add_argument("--min-quality-preservation-pass-rate", type=float, default=0.875)
    parser.add_argument("--min-quality-preservation-evaluable-prompts", type=int, default=1)
    parser.add_argument("--fail-on-gate-fail", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def max_repeated_ngram(words: list[str], n: int = 4) -> int:
    if len(words) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - n + 1):
        gram = tuple(words[index : index + n])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values(), default=0)


def max_token_run(words: list[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for word in words:
        if word == previous:
            current += 1
        else:
            current = 1
            previous = word
        longest = max(longest, current)
    return longest


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if ord(char) > 127) / len(text)


def analyze_text(text: str, prompt: PromptSpec) -> dict[str, Any]:
    stripped = text.strip()
    words = WORD_RE.findall(stripped.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    repeated_4gram = max_repeated_ngram(words, 4)
    repeated_run = max_token_run(words)
    non_ascii = non_ascii_ratio(stripped)
    required_regex_passed = True
    if prompt.required_regex:
        required_regex_passed = bool(re.search(prompt.required_regex, stripped, flags=re.IGNORECASE | re.DOTALL))

    checks = {
        "non_empty": bool(stripped),
        "long_enough": len(stripped) >= prompt.min_chars,
        "not_too_long": prompt.max_chars is None or len(stripped) <= prompt.max_chars,
        "no_action_tokens": not ACTION_TOKEN_RE.search(stripped),
        "no_chat_artifacts": not CHAT_ARTIFACT_RE.search(stripped),
        "not_low_diversity": not (len(words) >= 24 and unique_ratio < 0.27),
        "no_repeated_4gram_spam": repeated_4gram <= 2,
        "no_same_token_run_spam": repeated_run <= 4,
        "mostly_expected_script": non_ascii <= prompt.max_non_ascii_ratio,
        "required_pattern": required_regex_passed,
    }
    return {
        "char_count": len(stripped),
        "word_count": len(words),
        "unique_word_ratio": unique_ratio,
        "max_repeated_4gram_count": repeated_4gram,
        "max_same_token_run": repeated_run,
        "non_ascii_ratio": non_ascii,
        "checks": checks,
        "passed": all(checks.values()),
    }


def text_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left.strip(), right.strip()).ratio()


def format_prompt(tokenizer: Any, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def generate_one(model: Any, tokenizer: Any, prompt: PromptSpec, disable_adapter: bool) -> dict[str, Any]:
    prompt_text = format_prompt(tokenizer, prompt.user)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = encoded["input_ids"][0].tolist()
    encoded = {key: value.to(first_device(model)) for key, value in encoded.items()}
    context = model.disable_adapter() if disable_adapter else nullcontext()
    with context, torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=prompt.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = generated[0, len(prompt_ids) :].detach().cpu().tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "prompt": prompt.user,
        "prompt_token_count": len(prompt_ids),
        "generated_tokens": len(new_tokens),
        "new_token_ids": new_tokens,
        "output": text,
        "analysis": analyze_text(text, prompt),
    }


def build_teacher_forcing_batch(
    prompt_token_ids: list[int], reference_token_ids: list[int], device: torch.device
) -> dict[str, torch.Tensor]:
    if not reference_token_ids:
        raise ValueError("reference output has no generated tokens")
    input_ids = torch.tensor([prompt_token_ids + reference_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, len(prompt_token_ids) :] = input_ids[:, len(prompt_token_ids) :]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def masked_next_token_kl(
    active_logits: torch.Tensor, disabled_logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    if not mask.any():
        return {"mean": 0.0, "max": 0.0, "p95": 0.0, "tokens": 0}

    active_log_probs = F.log_softmax(active_logits[:, :-1, :].float(), dim=-1)
    disabled_log_probs = F.log_softmax(disabled_logits[:, :-1, :].float(), dim=-1)
    disabled_probs = disabled_log_probs.exp()
    per_token = F.kl_div(active_log_probs, disabled_probs, reduction="none").sum(dim=-1)
    selected = per_token[mask].detach().float().clamp_min(0.0).cpu()
    return {
        "mean": float(selected.mean().item()),
        "max": float(selected.max().item()),
        "p95": float(torch.quantile(selected, 0.95).item()),
        "tokens": int(selected.numel()),
    }


def compute_kl_against_disabled(
    model: Any, prompt_token_ids: list[int], reference_token_ids: list[int]
) -> dict[str, float]:
    batch = build_teacher_forcing_batch(prompt_token_ids, reference_token_ids, first_device(model))
    with torch.no_grad():
        active = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, return_dict=True
        )
        with model.disable_adapter():
            disabled = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, return_dict=True
            )
    return masked_next_token_kl(active.logits, disabled.logits, batch["labels"])


def summarize_prompt(
    prompt: PromptSpec, active_result: dict[str, Any], disabled_result: dict[str, Any], kl: dict[str, float]
) -> dict[str, Any]:
    active_quality_passed = active_result["analysis"]["passed"]
    disabled_quality_passed = disabled_result["analysis"]["passed"]
    return {
        "id": prompt.id,
        "prompt": asdict(prompt),
        "adapter_active": active_result,
        "adapter_disabled": disabled_result,
        "comparisons": {
            "active_matches_disabled": active_result["output"] == disabled_result["output"],
            "active_disabled_similarity": text_similarity(active_result["output"], disabled_result["output"]),
            "active_char_ratio_vs_disabled": (
                len(active_result["output"].strip()) / max(1, len(disabled_result["output"].strip()))
            ),
            "kl_active_vs_disabled_on_disabled_tokens": kl,
            "quality": {
                "active_passed": active_quality_passed,
                "disabled_passed": disabled_quality_passed,
                "base_already_failed": not disabled_quality_passed,
                "quality_degraded": disabled_quality_passed and not active_quality_passed,
                "quality_improved": active_quality_passed and not disabled_quality_passed,
            },
        },
    }


def summarize_gate(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(results)
    active_quality_passes = sum(1 for item in results if item["adapter_active"]["analysis"]["passed"])
    disabled_quality_passes = sum(1 for item in results if item["adapter_disabled"]["analysis"]["passed"])
    quality_degradations = sum(1 for item in results if item["comparisons"]["quality"]["quality_degraded"])
    quality_improvements = sum(1 for item in results if item["comparisons"]["quality"]["quality_improved"])
    base_already_failed = sum(1 for item in results if item["comparisons"]["quality"]["base_already_failed"])
    exact_matches = sum(1 for item in results if item["comparisons"]["active_matches_disabled"])
    mean_kls = [item["comparisons"]["kl_active_vs_disabled_on_disabled_tokens"]["mean"] for item in results]
    p95_kls = [item["comparisons"]["kl_active_vs_disabled_on_disabled_tokens"]["p95"] for item in results]
    similarities = [item["comparisons"]["active_disabled_similarity"] for item in results]

    active_quality_pass_rate = active_quality_passes / max(1, total)
    disabled_quality_pass_rate = disabled_quality_passes / max(1, total)
    quality_preservation_evaluable_prompts = disabled_quality_passes
    if quality_preservation_evaluable_prompts:
        quality_preservation_pass_rate = (
            quality_preservation_evaluable_prompts - quality_degradations
        ) / quality_preservation_evaluable_prompts
    else:
        quality_preservation_pass_rate = 1.0
    active_disabled_exact_match_rate = exact_matches / max(1, total)
    aggregate_mean_kl = sum(mean_kls) / max(1, len(mean_kls))
    aggregate_p95_kl = max(p95_kls) if p95_kls else 0.0
    average_similarity = sum(similarities) / max(1, len(similarities))

    checks = {
        "quality_preservation_evaluable_prompts": (
            quality_preservation_evaluable_prompts >= args.min_quality_preservation_evaluable_prompts
        ),
        "quality_preservation_pass_rate": (quality_preservation_pass_rate >= args.min_quality_preservation_pass_rate),
        "mean_kl": aggregate_mean_kl <= args.mean_kl_threshold,
        "p95_kl": aggregate_p95_kl <= args.p95_kl_threshold,
    }
    return {
        "finished_utc": utc_now(),
        "num_prompts": total,
        "active_quality_pass_rate": active_quality_pass_rate,
        "disabled_quality_pass_rate": disabled_quality_pass_rate,
        "quality_preservation_pass_rate": quality_preservation_pass_rate,
        "quality_preservation_evaluable_prompts": quality_preservation_evaluable_prompts,
        "quality_degradation_count": quality_degradations,
        "quality_improvement_count": quality_improvements,
        "base_already_failed_count": base_already_failed,
        "active_disabled_exact_match_rate": active_disabled_exact_match_rate,
        "average_active_disabled_similarity": average_similarity,
        "mean_kl_active_vs_disabled": aggregate_mean_kl,
        "max_prompt_p95_kl_active_vs_disabled": aggregate_p95_kl,
        "thresholds": {
            "min_quality_pass_rate": args.min_quality_pass_rate,
            "min_quality_preservation_pass_rate": args.min_quality_preservation_pass_rate,
            "min_quality_preservation_evaluable_prompts": args.min_quality_preservation_evaluable_prompts,
            "mean_kl_threshold": args.mean_kl_threshold,
            "p95_kl_threshold": args.p95_kl_threshold,
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(args.adapter_path)
    if not adapter_path.joinpath("adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = load_model_and_tokenizer(args)
    prompts = list(PROMPT_SUITES[args.prompt_suite])
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    prompt_results = []
    for prompt in prompts:
        print(f"Running prompt: {prompt.id}", flush=True)
        disabled_result = generate_one(model, tokenizer, prompt, disable_adapter=True)
        active_result = generate_one(model, tokenizer, prompt, disable_adapter=False)
        prompt_text = format_prompt(tokenizer, prompt.user)
        prompt_token_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
        kl = compute_kl_against_disabled(model, prompt_token_ids, disabled_result["new_token_ids"])
        result = summarize_prompt(prompt, active_result, disabled_result, kl)
        prompt_results.append(result)
        print(
            json.dumps(
                {
                    "prompt": prompt.id,
                    "active_passed": active_result["analysis"]["passed"],
                    "similarity": result["comparisons"]["active_disabled_similarity"],
                    "kl_mean": kl["mean"],
                    "kl_p95": kl["p95"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "config": {
            "created_utc": utc_now(),
            "base_model_id": args.base_model_id,
            "adapter_path": args.adapter_path,
            "dtype": args.dtype,
            "prompt_suite": args.prompt_suite,
            "max_prompts": args.max_prompts,
            "argv": sys.argv,
        },
        "summary": summarize_gate(prompt_results, args),
        "results": prompt_results,
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = run_gate(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output_path}", flush=True)
    if args.fail_on_gate_fail and not payload["summary"]["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
