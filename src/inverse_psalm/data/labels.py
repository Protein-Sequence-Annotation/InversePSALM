from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and "label_mapping" in data and isinstance(data["label_mapping"], dict):
        return data["label_mapping"]
    return data


def load_domain_dict(path: str | Path) -> dict[str, list[tuple[str, int, int]]]:
    with Path(path).expanduser().open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and isinstance(data.get("domain_dict"), dict):
        data = data["domain_dict"]
    out = {}
    for key, domains in data.items():
        out[str(key)] = [(entry[0], int(entry[1]), int(entry[2])) for entry in domains]
    return out


def infer_num_labels(label_mapping: dict[str, Any]) -> int:
    max_idx = -1
    for key, value in label_mapping.items():
        if key == "None":
            max_idx = max(max_idx, int(value))
        elif isinstance(value, dict):
            max_idx = max(max_idx, int(value["start"]), int(value["middle"]), int(value["stop"]))
    if max_idx < 0:
        raise ValueError("Could not infer annotation label count from label mapping.")
    return max_idx + 1


EXAMPLE_TYPE_TO_ID = {
    "original": 0,
    "shuffled": 1,
    "negative": 2,
    "domain_slice": 3,
}


ID_TO_EXAMPLE_TYPE = {value: key for key, value in EXAMPLE_TYPE_TO_ID.items()}


def infer_example_type(seq_id: str) -> str:
    if seq_id.startswith("negative_"):
        return "negative"
    if "/" in seq_id:
        return "domain_slice"
    if seq_id.startswith("shuffled_"):
        return "shuffled"
    return "original"


def generate_fine_labels(
    sequence: str,
    token_output: dict[str, list[int]],
    domain_info: list[tuple[str, int, int]],
    label_mapping: dict[str, Any],
    seq_id: str,
    ignore_label: int,
) -> list[int]:
    n_tokens = len(token_output["input_ids"])
    none_val = int(label_mapping["None"])
    labels = [none_val] * n_tokens
    labels[0] = ignore_label
    labels[-1] = ignore_label

    if seq_id.startswith("negative_"):
        pass
    else:
        if not domain_info:
            raise ValueError(f"Sequence {seq_id} not found in domain dictionary.")

        for pfam, start, stop in sorted(domain_info, key=lambda x: x[1]):
            if start < 1 or stop > len(sequence):
                raise ValueError(
                    f"Domain {(pfam, start, stop)} out of range for sequence {seq_id} length {len(sequence)}."
                )
            state_map = label_mapping.get(pfam, {})
            start_label = int(state_map.get("start", none_val)) if isinstance(state_map, dict) else none_val
            middle_label = int(state_map.get("middle", none_val)) if isinstance(state_map, dict) else none_val
            stop_label = int(state_map.get("stop", none_val)) if isinstance(state_map, dict) else none_val

            labels[start] = start_label
            if stop >= n_tokens:
                raise IndexError(f"Stop token index {stop} out of range for {seq_id}.")
            if start != stop:
                labels[stop] = stop_label
                for idx in range(start + 1, stop):
                    labels[idx] = middle_label

    if seq_id.endswith("_B"):
        labels = labels[:-1]
        token_output["input_ids"] = token_output["input_ids"][:-1]
        token_output["attention_mask"] = token_output["attention_mask"][:-1]
    elif seq_id.endswith("_M"):
        labels = labels[1:-1]
        token_output["input_ids"] = token_output["input_ids"][1:-1]
        token_output["attention_mask"] = token_output["attention_mask"][1:-1]
    elif seq_id.endswith("_E"):
        labels = labels[1:]
        token_output["input_ids"] = token_output["input_ids"][1:]
        token_output["attention_mask"] = token_output["attention_mask"][1:]

    if len(token_output["input_ids"]) != len(labels):
        raise AssertionError(f"Token/label length mismatch for {seq_id}.")
    return labels


def tokenize_and_label(
    seq_id: str,
    sequence: str,
    domain_info: list[tuple[str, int, int]],
    label_mapping: dict[str, Any],
    tokenizer,
    max_aa_length: int,
    ignore_label: int,
) -> dict[str, list[int] | int | str]:
    if len(sequence) > max_aa_length:
        raise ValueError(f"Sequence {seq_id} length {len(sequence)} exceeds max_aa_length={max_aa_length}.")
    token_output = tokenizer(
        sequence,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=True,
    )
    token_output["input_ids"] = list(token_output["input_ids"])
    token_output["attention_mask"] = list(token_output["attention_mask"])
    annotation_ids = generate_fine_labels(
        sequence=sequence,
        token_output=token_output,
        domain_info=domain_info,
        label_mapping=label_mapping,
        seq_id=seq_id,
        ignore_label=ignore_label,
    )
    residue_mask = [1 if label != ignore_label else 0 for label in annotation_ids]
    none_val = int(label_mapping["None"])
    domain_mask = [1 if label not in (ignore_label, none_val) else 0 for label in annotation_ids]
    example_type = infer_example_type(seq_id)
    return {
        "id": seq_id,
        "example_type": example_type,
        "example_type_id": EXAMPLE_TYPE_TO_ID[example_type],
        "input_ids": token_output["input_ids"],
        "attention_mask": token_output["attention_mask"],
        "labels": token_output["input_ids"].copy(),
        "annotation_ids": annotation_ids,
        "residue_mask": residue_mask,
        "domain_mask": domain_mask,
        "sequence_length": len(token_output["input_ids"]),
    }
