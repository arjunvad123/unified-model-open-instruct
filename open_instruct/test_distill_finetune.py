from types import SimpleNamespace

import torch

from open_instruct.distill_finetune import (
    StudentEmbeddingModel,
    _extract_medi2_text,
    encode_student,
    load_medi2_examples,
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
