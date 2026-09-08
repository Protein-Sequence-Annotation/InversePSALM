#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from Bio.SeqIO.FastaIO import SimpleFastaParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_psalm.data.augmentation import simplify_id


REPORT_KEYS = (
    "loaded_fasta_records",
    "pruned_domain_records",
    "emitted_records",
    "originals",
    "shuffled",
    "negatives",
    "domain_slices",
    "skipped",
)


def _clean_domains(domains: list[Any]) -> list[tuple[str, int, int]]:
    clean_domains = []
    for entry in domains or []:
        if entry is None or len(entry) < 3:
            continue
        pfam, start, stop = entry[:3]
        start = int(start)
        stop = int(stop)
        if stop - start + 1 > 5:
            clean_domains.append((str(pfam), start, stop))
    return sorted(clean_domains, key=lambda x: x[1])


def _load_domain_dicts(
    path: str | Path,
) -> tuple[dict[str, list[tuple[str, int, int]]], dict[str, list[tuple[str, int, int]]]]:
    with Path(path).expanduser().open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and isinstance(data.get("domain_dict"), dict):
        data = data["domain_dict"]

    raw_out: dict[str, list[tuple[str, int, int]]] = {}
    simple_out: dict[str, list[tuple[str, int, int]]] = {}
    for key, domains in data.items():
        raw_key = str(key)
        simple_key = simplify_id(str(key))
        clean_domains = _clean_domains(domains)
        if clean_domains:
            raw_out[raw_key] = clean_domains
            simple_out.setdefault(simple_key, []).extend(clean_domains)
    simple_out = {key: sorted(value, key=lambda x: x[1]) for key, value in simple_out.items()}
    return raw_out, simple_out


def shard_inputs(
    *,
    fasta: str | Path,
    domain_dict_path: str | Path,
    workers: int,
    shard_dir: Path,
) -> int:
    shard_dir.mkdir(parents=True, exist_ok=True)
    raw_domain_dict, simple_domain_dict = _load_domain_dicts(domain_dict_path)
    shard_fastas = [(shard_dir / f"shard_{idx}.fasta").open("w", encoding="utf-8") for idx in range(workers)]
    shard_dicts: list[dict[str, list[tuple[str, int, int]]]] = [dict() for _ in range(workers)]
    candidates: dict[str, tuple[str, list[tuple[str, int, int]]]] = {}
    duplicate_ids = 0

    try:
        with Path(fasta).expanduser().open("r", encoding="utf-8") as handle:
            for title, sequence in SimpleFastaParser(handle):
                raw_id = title.split()[0]
                seq_id = simplify_id(raw_id)
                domains = raw_domain_dict.get(raw_id) or simple_domain_dict.get(seq_id)
                if not domains:
                    continue
                previous = candidates.get(seq_id)
                if previous is None:
                    candidates[seq_id] = (sequence, domains)
                    continue
                duplicate_ids += 1
                _, previous_domains = previous
                if len(domains) > len(previous_domains):
                    candidates[seq_id] = (sequence, domains)

        for seq_id, (sequence, domains) in candidates.items():
            shard_idx = int(hashlib.md5(seq_id.encode("utf-8")).hexdigest(), 16) % workers
            shard_fastas[shard_idx].write(f">{seq_id}\n{sequence}\n")
            shard_dicts[shard_idx][seq_id] = domains
    finally:
        for handle in shard_fastas:
            handle.close()

    for idx, shard_dict in enumerate(shard_dicts):
        with (shard_dir / f"shard_{idx}.pkl").open("wb") as handle:
            pickle.dump(shard_dict, handle)
    if duplicate_ids:
        print(
            f"Resolved {duplicate_ids} duplicate FASTA records after ID simplification "
            "by keeping the record with the most domains, ties kept first.",
            file=sys.stderr,
        )
    return len(candidates)


def _forwarded_args(args: argparse.Namespace, shard_idx: int) -> list[str]:
    out = [
        "--max-length",
        str(args.max_length),
        "--negative-prob",
        str(args.negative_prob),
        "--seed",
        str(int(args.seed) + shard_idx),
    ]
    flag_names = [
        "include_domain_slices",
        "domain_slices_only",
        "shuffle_only",
        "no_shuffle",
        "large_data",
        "verbose",
    ]
    for name in flag_names:
        if getattr(args, name):
            out.append("--" + name.replace("_", "-"))
    if args.domain_counts_tsv:
        out.extend(["--domain-counts-tsv", str(args.domain_counts_tsv)])
    if args.p_shuffled is not None:
        out.extend(["--p-shuffled", str(args.p_shuffled)])
    if args.domain_slice_frac is not None:
        out.extend(["--domain-slice-frac", str(args.domain_slice_frac)])
    return out


def run_shard(shard_idx: int, shard_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    shard_fasta = shard_dir / f"shard_{shard_idx}.fasta"
    shard_dict = shard_dir / f"shard_{shard_idx}.pkl"
    output_fasta = shard_dir / f"out_{shard_idx}.fasta"
    output_dict = shard_dir / f"out_{shard_idx}.pkl"
    report_json = shard_dir / f"out_{shard_idx}.report.json"
    if not shard_fasta.exists() or shard_fasta.stat().st_size == 0:
        return None

    command = [
        args.python_bin,
        args.augment_script,
        "--fasta",
        str(shard_fasta),
        "--domain-dict",
        str(shard_dict),
        "--output-fasta",
        str(output_fasta),
        "--output-dict",
        str(output_dict),
        "--report-json",
        str(report_json),
        *_forwarded_args(args, shard_idx),
    ]
    subprocess.run(command, check=True)

    with report_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_outputs(shard_dir: Path, output_fasta: str | Path, output_dict: str | Path) -> dict[str, list[Any]]:
    output_fasta = Path(output_fasta).expanduser()
    output_dict = Path(output_dict).expanduser()
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    output_dict.parent.mkdir(parents=True, exist_ok=True)

    merged_dict: dict[str, list[Any]] = {}
    with output_fasta.open("w", encoding="utf-8") as output_handle:
        for shard_fasta in sorted(shard_dir.glob("out_*.fasta"), key=lambda p: int(p.stem.split("_")[1])):
            with shard_fasta.open("r", encoding="utf-8") as input_handle:
                shutil.copyfileobj(input_handle, output_handle)

    for shard_pickle in sorted(shard_dir.glob("out_*.pkl"), key=lambda p: int(p.stem.split("_")[1])):
        with shard_pickle.open("rb") as handle:
            shard_dict = pickle.load(handle)
        overlap = set(merged_dict).intersection(shard_dict)
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Duplicate augmented IDs while merging: {sample}")
        merged_dict.update(shard_dict)

    with output_dict.open("wb") as handle:
        pickle.dump(merged_dict, handle)
    return merged_dict


def combine_reports(reports: list[dict[str, Any]], output_report: str | Path, merged_dict: dict[str, list[Any]]) -> None:
    combined = {key: sum(int(report.get(key, 0)) for report in reports) for key in REPORT_KEYS}
    combined["emitted_records"] = len(merged_dict)
    combined["workers_with_output"] = len(reports)
    combined["example_type_counts"] = dict(
        Counter(
            "negative"
            if seq_id.startswith("negative_")
            else "domain_slice"
            if "/" in seq_id
            else "shuffled"
            if seq_id.startswith("shuffled_")
            else "original"
            for seq_id in merged_dict
        )
    )
    total = max(1, combined["emitted_records"])
    combined["negative_fraction_actual"] = float(combined["negatives"]) / float(total)

    output_report = Path(output_report).expanduser()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel augmentation wrapper for InversePSALM.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--domain-dict", required=True)
    parser.add_argument("--output-fasta", required=True)
    parser.add_argument("--output-dict", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--augment-script", default=str(Path(__file__).resolve().parent / "augment_data.py"))
    parser.add_argument("--python-bin", default=sys.executable)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    report_json = args.report_json or str(Path(args.output_fasta).with_suffix(".report.json"))
    shard_dir = Path(args.temp_dir).expanduser() if args.temp_dir else Path(tempfile.mkdtemp(prefix="inverse_psalm_aug_"))
    if args.temp_dir and shard_dir.exists():
        shutil.rmtree(shard_dir)

    try:
        kept = shard_inputs(
            fasta=args.fasta,
            domain_dict_path=args.domain_dict,
            workers=args.workers,
            shard_dir=shard_dir,
        )
        print(f"Sharded {kept} input sequences into {args.workers} shards at {shard_dir}", file=sys.stderr)
        if kept == 0:
            raise SystemExit("No sequences with domains remained after pruning.")

        reports: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_shard, idx, shard_dir, args) for idx in range(args.workers)]
            for future in concurrent.futures.as_completed(futures):
                report = future.result()
                if report is not None:
                    reports.append(report)

        merged_dict = merge_outputs(shard_dir, args.output_fasta, args.output_dict)
        combine_reports(reports, report_json, merged_dict)
        print(f"Done. FASTA: {args.output_fasta}")
        print(f"Done. domain dict: {args.output_dict}")
        print(f"Done. report: {report_json}")
    finally:
        if args.keep_temp:
            print(f"Kept temporary shards at {shard_dir}")
        else:
            shutil.rmtree(shard_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
