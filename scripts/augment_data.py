#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_psalm.data.augmentation import run_augmentation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PSALM-style augmentation for InversePSALM training data."
    )
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--domain-dict", required=True)
    parser.add_argument("--output-fasta", required=True)
    parser.add_argument("--output-dict", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--negative-prob", type=float, default=0.05)
    parser.add_argument("--include-domain-slices", action="store_true")
    parser.add_argument("--domain-slices-only", action="store_true")
    parser.add_argument("--shuffle-only", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--large-data", action="store_true")
    parser.add_argument("--p-shuffled", type=float, default=0.5)
    parser.add_argument("--domain-counts-tsv", default=None)
    parser.add_argument("--domain-slice-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report_path = args.report_json
    if report_path is None:
        report_path = str(Path(args.output_fasta).with_suffix(".report.json"))

    run_augmentation(
        fasta=args.fasta,
        domain_dict=args.domain_dict,
        output_fasta=args.output_fasta,
        output_dict=args.output_dict,
        report_json=report_path,
        max_length=args.max_length,
        negative_prob=args.negative_prob,
        include_domain_slices=args.include_domain_slices,
        domain_slices_only=args.domain_slices_only,
        shuffle_only=args.shuffle_only,
        no_shuffle=args.no_shuffle,
        large_data=args.large_data,
        p_shuffled=args.p_shuffled,
        domain_counts_tsv=args.domain_counts_tsv,
        domain_slice_frac=args.domain_slice_frac,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
