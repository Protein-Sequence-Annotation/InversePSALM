#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_psalm.config import load_config
from inverse_psalm.models.inverse_psalm import build_inverse_psalm
from inverse_psalm.training.trainer import make_data_collator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny forward pass on one InversePSALM batch."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    shard = args.shard_dir
    if not os.path.isfile(os.path.join(shard, "dataset_info.json")):
        shards = sorted(glob.glob(os.path.join(shard, "shard-*")))
        if not shards:
            raise ValueError(f"No HF shard found at {shard}")
        shard = shards[0]

    ds = load_from_disk(shard)
    first_batch_id = ds[0]["batch_id"]
    examples = [ex for ex in ds if ex["batch_id"] == first_batch_id]

    model, tokenizer = build_inverse_psalm(cfg)
    collator = make_data_collator(tokenizer)
    batch = collator(examples)
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    model.to(args.device)
    model.train()
    outputs = model(**batch)
    print({k: float(v.detach().cpu()) for k, v in outputs.items() if k.endswith("loss")})


if __name__ == "__main__":
    main()
