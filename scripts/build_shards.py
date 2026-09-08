#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_psalm.config import load_config
from inverse_psalm.data.shard_builder import build_shards_from_augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build batch-major InversePSALM training shards."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--domain-dict", required=True)
    parser.add_argument("--label-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--max-aa-length", type=int, default=None)
    parser.add_argument("--max-tokens-per-batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_shards_from_augmented(
        config=cfg,
        fasta=args.fasta,
        domain_dict_path=args.domain_dict,
        label_mapping_path=args.label_mapping,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        shard_size=args.shard_size,
        max_aa_length=args.max_aa_length,
        max_tokens_per_batch=args.max_tokens_per_batch,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
