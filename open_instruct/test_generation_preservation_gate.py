from types import SimpleNamespace

import torch
from benchmarks.run_generation_preservation_gate import PromptSpec, analyze_text, masked_next_token_kl, summarize_gate


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


def test_summarize_gate_requires_quality_and_kl_thresholds():
    args = SimpleNamespace(min_quality_pass_rate=1.0, mean_kl_threshold=0.1, p95_kl_threshold=0.5)
    result = {
        "adapter_active": {"analysis": {"passed": False}},
        "adapter_disabled": {"analysis": {"passed": True}},
        "comparisons": {
            "active_matches_disabled": True,
            "active_disabled_similarity": 0.9,
            "kl_active_vs_disabled_on_disabled_tokens": {"mean": 0.2, "p95": 0.6},
        },
    }

    summary = summarize_gate([result], args)

    assert summary["active_quality_pass_rate"] == 0.0
    assert not summary["checks"]["active_quality_pass_rate"]
    assert not summary["checks"]["mean_kl"]
    assert not summary["checks"]["p95_kl"]
    assert not summary["gate_passed"]
