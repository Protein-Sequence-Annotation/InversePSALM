#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_psalm.config import deep_get, load_config


SHARD_RE = re.compile(r"Saved shard\s+(\d+)\s+with\s+(\d+)\s+batches\s+and\s+(\d+)\s+examples")


def parse_build_log(path: str | Path) -> tuple[int, int, int]:
    shards = 0
    batches = 0
    examples = 0
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            match = SHARD_RE.search(line)
            if not match:
                continue
            shards += 1
            batches += int(match.group(2))
            examples += int(match.group(3))
    if shards == 0:
        raise ValueError(f"No shard summary lines found in {path}.")
    return shards, batches, examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate InversePSALM steps per epoch from a shard build log.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-log", default="logs/build_shards.out")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    grad_accum = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else deep_get(cfg, "training.gradient_accumulation_steps", 1)
    )
    world_size = max(1, int(args.world_size))
    shards, total_batches, total_examples = parse_build_log(args.build_log)
    micro_steps = math.ceil(total_batches / world_size)
    optimizer_steps = math.ceil(micro_steps / max(1, grad_accum))

    print(f"shards: {shards}")
    print(f"examples: {total_examples}")
    print(f"precomputed_batches: {total_batches}")
    print(f"world_size: {world_size}")
    print(f"gradient_accumulation_steps: {grad_accum}")
    print(f"micro_steps_per_epoch_estimate: {micro_steps}")
    print(f"optimizer_steps_per_epoch_estimate: {optimizer_steps}")


if __name__ == "__main__":
    main()
