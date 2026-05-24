from types import SimpleNamespace

import torch
from benchmarks.run_generation_preservation_gate import (
    PROMPT_SUITES,
    PromptSpec,
    analyze_text,
    masked_next_token_kl,
    summarize_gate,
)


def test_analyze_text_catches_repetition_and_unexpected_script():
    prompt = PromptSpec(
        id="repetition", user="Explain clean water.", max_new_tokens=64, min_chars=10, max_non_ascii_ratio=0.05
    )
    text = "clean water matters " * 10 + "生成" * 20

    analysis = analyze_text(text, prompt)

    assert not analysis["passed"]
    assert not analysis["checks"]["no_repeated_4gram_spam"]
    assert not analysis["checks"]["mostly_expected_script"]


def test_masked_next_token_kl_is_zero_for_identical_logits():
    logits = torch.randn(1, 5, 8)
    labels = torch.tensor([[-100, -100, 2, 3, 4]])

    metrics = masked_next_token_kl(logits, logits.clone(), labels)

    assert metrics["tokens"] == 3
    assert metrics["mean"] >= 0.0
    assert metrics["mean"] < 1e-6
    assert metrics["p95"] < 1e-6


def test_summarize_gate_requires_preserved_quality_and_kl_thresholds():
    args = SimpleNamespace(
        min_quality_pass_rate=1.0,
        min_quality_preservation_pass_rate=1.0,
        min_quality_preservation_evaluable_prompts=1,
        mean_kl_threshold=0.1,
        p95_kl_threshold=0.5,
    )
    result = {
        "adapter_active": {"analysis": {"passed": False}},
        "adapter_disabled": {"analysis": {"passed": True}},
        "comparisons": {
            "active_matches_disabled": True,
            "active_disabled_similarity": 0.9,
            "quality": {
                "active_passed": False,
                "disabled_passed": True,
                "base_already_failed": False,
                "quality_degraded": True,
                "quality_improved": False,
            },
            "kl_active_vs_disabled_on_disabled_tokens": {"mean": 0.2, "p95": 0.6},
        },
    }

    summary = summarize_gate([result], args)

    assert summary["active_quality_pass_rate"] == 0.0
    assert summary["quality_preservation_pass_rate"] == 0.0
    assert not summary["checks"]["quality_preservation_pass_rate"]
    assert not summary["checks"]["mean_kl"]
    assert not summary["checks"]["p95_kl"]
    assert not summary["gate_passed"]


def test_summarize_gate_does_not_treat_base_failure_as_adapter_degradation():
    args = SimpleNamespace(
        min_quality_pass_rate=1.0,
        min_quality_preservation_pass_rate=1.0,
        min_quality_preservation_evaluable_prompts=1,
        mean_kl_threshold=0.1,
        p95_kl_threshold=0.5,
    )
    result = {
        "adapter_active": {"analysis": {"passed": False}},
        "adapter_disabled": {"analysis": {"passed": False}},
        "comparisons": {
            "active_matches_disabled": False,
            "active_disabled_similarity": 0.2,
            "quality": {
                "active_passed": False,
                "disabled_passed": False,
                "base_already_failed": True,
                "quality_degraded": False,
                "quality_improved": False,
            },
            "kl_active_vs_disabled_on_disabled_tokens": {"mean": 0.01, "p95": 0.02},
        },
    }

    summary = summarize_gate([result], args)

    assert summary["active_quality_pass_rate"] == 0.0
    assert summary["disabled_quality_pass_rate"] == 0.0
    assert summary["quality_preservation_evaluable_prompts"] == 0
    assert summary["quality_preservation_pass_rate"] == 1.0
    assert not summary["checks"]["quality_preservation_evaluable_prompts"]
    assert summary["checks"]["quality_preservation_pass_rate"]
    assert not summary["gate_passed"]


def test_expanded_prompt_suite_adds_easy_evaluable_anchors():
    smoke_ids = {prompt.id for prompt in PROMPT_SUITES["smoke"]}
    expanded_ids = {prompt.id for prompt in PROMPT_SUITES["expanded"]}

    assert smoke_ids < expanded_ids
    assert {"exact_hello", "exact_ok", "json_yes", "tiny_python"} <= expanded_ids
    assert len(expanded_ids) == len(PROMPT_SUITES["expanded"])
