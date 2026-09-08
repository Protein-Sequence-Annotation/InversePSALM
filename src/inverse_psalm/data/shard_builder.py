from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from Bio import SeqIO
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from inverse_psalm.config import deep_get
from inverse_psalm.data.labels import load_domain_dict, load_mapping, tokenize_and_label


def pad_batch(batch: list[dict[str, Any]], max_length: int, pad_token_id: int, ignore_label: int) -> None:
    for example in batch:
        pad_len = max_length - len(example["input_ids"])
        if pad_len < 0:
            raise ValueError("Cannot pad an example longer than max_length.")
        example["input_ids"] += [pad_token_id] * pad_len
        example["attention_mask"] += [0] * pad_len
        example["labels"] += [pad_token_id] * pad_len
        example["annotation_ids"] += [ignore_label] * pad_len
        example["residue_mask"] += [0] * pad_len
        example["domain_mask"] += [0] * pad_len
        assert len(example["input_ids"]) == max_length
        assert len(example["annotation_ids"]) == max_length
        assert len(example["residue_mask"]) == max_length
        assert len(example["domain_mask"]) == max_length


def _flush_shard(
    shard_batches: list[list[dict[str, Any]]],
    output_dir: str,
    shard_index: int,
    rng: random.Random,
) -> int:
    if not shard_batches:
        return shard_index
    rng.shuffle(shard_batches)
    shard_examples = [example for batch in shard_batches for example in batch]
    shard_dir = os.path.join(output_dir, f"shard-{shard_index:05d}")
    Dataset.from_list(shard_examples).save_to_disk(shard_dir)
    print(
        f"Saved shard {shard_index} with {len(shard_batches)} batches "
        f"and {len(shard_examples)} examples to {shard_dir}"
    )
    return shard_index + 1


def build_shards_from_augmented(
    *,
    config: dict[str, Any],
    fasta: str,
    domain_dict_path: str,
    label_mapping_path: str,
    output_dir: str,
    chunk_size: int | None = None,
    shard_size: int | None = None,
    max_aa_length: int | None = None,
    max_tokens_per_batch: int | None = None,
    seed: int | None = None,
) -> None:
    data_cfg = config.get("data", {})
    model_name = deep_get(config, "model.generator_model_name", "facebook/esm2_t33_650M_UR50D")
    chunk_size = int(chunk_size or data_cfg.get("chunk_size", 100000))
    shard_size = int(shard_size or data_cfg.get("default_shard_size", 25000))
    max_aa_length = int(max_aa_length or data_cfg.get("max_aa_length", 4096))
    max_tokens_per_batch = int(max_tokens_per_batch or data_cfg.get("max_tokens_per_batch", 8192))
    ignore_label = int(data_cfg.get("ignore_label", -100))
    seed = int(seed if seed is not None else data_cfg.get("seed", 100))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer {model_name} does not define pad_token_id.")
    label_mapping = load_mapping(label_mapping_path)
    domain_dict = load_domain_dict(domain_dict_path)
    rng = random.Random(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shard_batches: list[list[dict[str, Any]]] = []
    shard_index = 0
    batch_id = 0
    window: list[tuple[str, str]] = []

    def process_window(window_records: list[tuple[str, str]]) -> None:
        nonlocal batch_id, shard_index, shard_batches
        if not window_records:
            return
        window_records.sort(key=lambda item: len(item[1]))
        current_batch: list[dict[str, Any]] = []
        max_padded_length = 0

        for seq_id, sequence in window_records:
            if seq_id not in domain_dict:
                raise ValueError(f"Sequence {seq_id} missing from domain dictionary.")
            example = tokenize_and_label(
                seq_id=seq_id,
                sequence=sequence,
                domain_info=domain_dict[seq_id],
                label_mapping=label_mapping,
                tokenizer=tokenizer,
                max_aa_length=max_aa_length,
                ignore_label=ignore_label,
            )
            seq_len = int(example["sequence_length"])
            prospective_max = max(max_padded_length, seq_len)
            prospective_tokens = prospective_max * (len(current_batch) + 1)
            if current_batch and prospective_tokens > max_tokens_per_batch:
                pad_batch(current_batch, max_padded_length, tokenizer.pad_token_id, ignore_label)
                for ex in current_batch:
                    ex["batch_id"] = batch_id
                shard_batches.append(current_batch)
                batch_id += 1
                if len(shard_batches) >= shard_size:
                    shard_index = _flush_shard(shard_batches, output_dir, shard_index, rng)
                    shard_batches = []
                current_batch = []
                max_padded_length = 0
                prospective_max = seq_len
                prospective_tokens = seq_len
            if prospective_tokens > max_tokens_per_batch:
                raise ValueError(
                    f"Single sequence {seq_id} tokenized length {seq_len} exceeds token budget "
                    f"{max_tokens_per_batch}."
                )
            current_batch.append(example)
            max_padded_length = max(max_padded_length, seq_len)

        if current_batch:
            pad_batch(current_batch, max_padded_length, tokenizer.pad_token_id, ignore_label)
            for ex in current_batch:
                ex["batch_id"] = batch_id
            shard_batches.append(current_batch)
            batch_id += 1
            if len(shard_batches) >= shard_size:
                shard_index = _flush_shard(shard_batches, output_dir, shard_index, rng)
                shard_batches = []

    records = SeqIO.parse(str(fasta), "fasta")
    for rec in tqdm(records, desc="Tokenizing augmented FASTA"):
        window.append((rec.id, str(rec.seq)))
        if len(window) >= chunk_size:
            process_window(window)
            window = []
    process_window(window)
    if shard_batches:
        _flush_shard(shard_batches, output_dir, shard_index, rng)
