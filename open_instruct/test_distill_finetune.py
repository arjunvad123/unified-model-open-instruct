from open_instruct.distill_finetune import _extract_medi2_text, load_medi2_examples


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
