from types import SimpleNamespace

import torch

from open_instruct.distill_finetune import (
    GenerationReplayDataset,
    StudentEmbeddingModel,
    _balance_generation_replay_examples,
    _extract_medi2_text,
    _generation_replay_category,
    _synthetic_exact_format_replay_examples,
    encode_student,
    load_generation_replay_examples,
    load_medi2_examples,
    masked_next_token_kl_loss,
    validate_generation_replay_coverage,
)


class _FakeStreamingDataset:
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed: int, buffer_size: int):
        return self

    def __iter__(self):
        return iter(self.rows)


def test_extract_medi2_text_handles_instruction_pairs():
    assert _extract_medi2_text(["Represent this sentence.", "Roman Atwood is a content creator."]) == (
        "Roman Atwood is a content creator."
    )
    assert _extract_medi2_text([["Represent the article.", "Roman Atwood\nRoman Bernard Atwood..."]]) == (
        "Roman Atwood\nRoman Bernard Atwood..."
    )
    assert _extract_medi2_text("") == ""
    assert _extract_medi2_text([]) == ""


def test_load_medi2_examples_normalizes_nested_text(monkeypatch):
    rows = [
        {
            "query": [
                "Represent this sentence to retrieve a Wikipedia article all about it.",
                "Roman Atwood is a content creator.",
            ],
            "pos": [
                [
                    "Represent the article for finding a claim of about one sentence that the article confirms",
                    "Roman Atwood\nRoman Bernard Atwood is an American YouTube personality.",
                ]
            ],
            "neg": [["Represent this", "Casey Neistat and Jesse Wellens, PrankvsPrank."]],
        },
        {"query": ["instruction", ""], "pos": [["instruction", "passage"]], "neg": []},
    ]

    def fake_load_dataset(*args, **kwargs):
        return _FakeStreamingDataset(rows)

    monkeypatch.setattr("open_instruct.distill_finetune.load_dataset", fake_load_dataset)
    monkeypatch.setattr("open_instruct.distill_finetune.logger.info", lambda *args, **kwargs: None)

    examples = load_medi2_examples(2)

    assert examples == [
        {
            "query": "Roman Atwood is a content creator.",
            "passage": "Roman Atwood\nRoman Bernard Atwood is an American YouTube personality.",
            "negative": "Casey Neistat and Jesse Wellens, PrankvsPrank.",
        }
    ]


def test_encode_student_skips_causal_lm_logits_path():
    class FakeBackbone:
        def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache, return_dict):
            assert output_hidden_states is True
            assert use_cache is False
            assert return_dict is True
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
            return SimpleNamespace(hidden_states=(hidden,))

    class FakeCausalLm:
        def __init__(self):
            self.model = FakeBackbone()

        def __call__(self, *args, **kwargs):
            raise AssertionError("causal LM forward should not be used for embedding hidden states")

    class FakePeftModel:
        def __init__(self):
            self.base_model = SimpleNamespace(model=FakeCausalLm())

    input_ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    encoded = encode_student(StudentEmbeddingModel(FakePeftModel()), input_ids, attention_mask, "mean")

    assert encoded.shape == (2, 2)
    assert torch.isfinite(encoded).all()


def test_generation_replay_dataset_masks_prompt_tokens():
    class FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, add_generation_prompt, tokenize, truncation, max_length):
            ids = []
            for message in messages:
                role_token = {"user": 10, "assistant": 20, "system": 30}[message["role"]]
                ids.append(role_token)
                ids.extend(ord(char) % 50 + 40 for char in message["content"])
            if add_generation_prompt:
                ids.append(20)
            return ids[:max_length]

    dataset = GenerationReplayDataset(
        [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}],
        FakeTokenizer(),
        max_length=64,
    )

    item = dataset[0]
    first_label = next(index for index, token in enumerate(item["labels"].tolist()) if token != -100)

    assert first_label > 0
    assert all(token == -100 for token in item["labels"][:first_label])
    assert item["labels"][first_label:].ne(-100).any()


def test_generation_replay_category_covers_gate_prompt_shapes():
    assert (
        _generation_replay_category(
            [
                {"role": "user", "content": "Reply with exactly this word: hello"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        == "exact"
    )
    assert (
        _generation_replay_category(
            [{"role": "user", "content": 'Return JSON with one key "answer".'}, {"role": "assistant", "content": "{}"}]
        )
        == "format"
    )
    assert (
        _generation_replay_category(
            [
                {"role": "user", "content": "Extract the city from this sentence."},
                {"role": "assistant", "content": "Ok."},
            ]
        )
        == "extraction"
    )
    assert (
        _generation_replay_category(
            [{"role": "user", "content": "Summarize this in one sentence."}, {"role": "assistant", "content": "Ok."}]
        )
        == "summary"
    )
    assert (
        _generation_replay_category(
            [{"role": "user", "content": "Rewrite this politely."}, {"role": "assistant", "content": "Ok."}]
        )
        == "rewrite"
    )
    assert (
        _generation_replay_category(
            [{"role": "user", "content": "Write a Python function."}, {"role": "assistant", "content": "Ok."}]
        )
        == "code"
    )
    assert (
        _generation_replay_category(
            [{"role": "user", "content": "Classify this request."}, {"role": "assistant", "content": "Ok."}]
        )
        == "classification"
    )
    assert (
        _generation_replay_category(
            [{"role": "user", "content": "Solve the equation x + 1 = 2."}, {"role": "assistant", "content": "Ok."}]
        )
        == "math"
    )
    assert (
        _generation_replay_category(
            [{"role": "user", "content": "What does a tokenizer do?"}, {"role": "assistant", "content": "Ok."}]
        )
        == "qa"
    )


def test_balance_generation_replay_examples_round_robins_categories():
    examples = [
        {"messages": [], "category": "general", "source": "s"},
        {"messages": [], "category": "general", "source": "s"},
        {"messages": [], "category": "summary", "source": "s"},
        {"messages": [], "category": "rewrite", "source": "s"},
        {"messages": [], "category": "code", "source": "s"},
    ]

    balanced = _balance_generation_replay_examples(examples, 4)

    assert [item["category"] for item in balanced] == ["summary", "rewrite", "code", "general"]


def test_synthetic_exact_format_replay_has_required_target_categories():
    examples = _synthetic_exact_format_replay_examples(seed=123)
    counts = {}
    for example in examples:
        counts[example["category"]] = counts.get(example["category"], 0) + 1

    assert counts["exact"] >= 900
    assert counts["format"] >= 900
    assert counts["extraction"] >= 400
    assert (
        sum(
            1
            for example in examples
            if example["messages"][0]["content"] == "Reply with exactly this word: hello"
            and example["messages"][1]["content"] == "hello"
        )
        >= 500
    )
    assert (
        sum(
            1
            for example in examples
            if example["messages"][0]["content"] == 'Return JSON with one key "answer" and value "yes".'
            and example["messages"][1]["content"] == '{"answer": "yes"}'
        )
        >= 500
    )
    assert any(
        example["messages"][0]["content"] == "Reply with exactly this word: hello"
        and example["messages"][1]["content"] == "hello"
        for example in examples
    )


def test_load_generation_replay_examples_supports_synthetic_exact_format_only(monkeypatch):
    monkeypatch.setattr("open_instruct.distill_finetune.logger.info", lambda *args, **kwargs: None)
    monkeypatch.setattr("open_instruct.distill_finetune.logger.warning", lambda *args, **kwargs: None)

    examples = load_generation_replay_examples(
        120, sources="exact_format", balance_categories=True, scan_multiplier=20
    )
    summary = validate_generation_replay_coverage(examples, "exact,format,extraction", 30)

    assert summary["source_counts"] == {"exact_format": 120}
    assert set(summary["category_counts"]) == {"exact", "format", "extraction"}


def test_validate_generation_replay_coverage_rejects_missing_required_category():
    examples = [
        {"messages": [], "category": "summary", "source": "s"},
        {"messages": [], "category": "qa", "source": "s"},
    ]

    try:
        validate_generation_replay_coverage(examples, "summary,rewrite", 1)
    except RuntimeError as exc:
        assert "rewrite" in str(exc)
    else:
        raise AssertionError("expected missing replay category to fail validation")


def test_masked_next_token_kl_loss_is_zero_for_identical_logits():
    logits = torch.randn(2, 4, 8)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])

    loss = masked_next_token_kl_loss(logits, logits.clone(), labels)

    assert loss.item() > -1e-6
    assert loss.item() < 1e-6
