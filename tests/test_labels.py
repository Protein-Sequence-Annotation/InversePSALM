from inverse_psalm.data.labels import generate_fine_labels, tokenize_and_label


class DummyTokenizer:
    cls_token_id = 0
    eos_token_id = 2

    def __call__(self, sequence, add_special_tokens=True, truncation=False, padding=False, return_attention_mask=True):
        ids = [ord(ch) % 20 + 4 for ch in sequence]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.eos_token_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_generate_fine_labels_for_domain_span():
    token_output = {"input_ids": [0, 10, 11, 12, 2], "attention_mask": [1, 1, 1, 1, 1]}
    mapping = {"None": 0, "PF00001": {"start": 1, "middle": 2, "stop": 3}}
    labels = generate_fine_labels(
        "ACD",
        token_output,
        [("PF00001", 1, 3)],
        mapping,
        "seq1",
        -100,
    )
    assert labels == [-100, 1, 2, 3, -100]


def test_generate_fine_labels_sets_background_to_none():
    token_output = {
        "input_ids": [0, 10, 11, 12, 13, 14, 2],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1],
    }
    mapping = {"None": 0, "PF00001": {"start": 1, "middle": 2, "stop": 3}}
    labels = generate_fine_labels(
        "ABCDE",
        token_output,
        [("PF00001", 2, 3)],
        mapping,
        "seq1",
        -100,
    )
    assert labels == [-100, 0, 1, 3, 0, 0, -100]


def test_tokenize_and_label_middle_crop():
    mapping = {"None": 0, "PF00001": {"start": 1, "middle": 2, "stop": 3}}
    out = tokenize_and_label(
        seq_id="seq1_M",
        sequence="ACD",
        domain_info=[("PF00001", 1, 3)],
        label_mapping=mapping,
        tokenizer=DummyTokenizer(),
        max_aa_length=10,
        ignore_label=-100,
    )
    assert out["annotation_ids"] == [1, 2, 3]
    assert len(out["input_ids"]) == 3
    assert out["residue_mask"] == [1, 1, 1]
    assert out["domain_mask"] == [1, 1, 1]


def test_example_type_and_masks_for_original_background():
    mapping = {"None": 0, "PF00001": {"start": 1, "middle": 2, "stop": 3}}
    out = tokenize_and_label(
        seq_id="seq1",
        sequence="ABCDE",
        domain_info=[("PF00001", 2, 3)],
        label_mapping=mapping,
        tokenizer=DummyTokenizer(),
        max_aa_length=10,
        ignore_label=-100,
    )
    assert out["example_type"] == "original"
    assert out["annotation_ids"] == [-100, 0, 1, 3, 0, 0, -100]
    assert out["residue_mask"] == [0, 1, 1, 1, 1, 1, 0]
    assert out["domain_mask"] == [0, 0, 1, 1, 0, 0, 0]
