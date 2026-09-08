from __future__ import annotations

import json
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Container, Iterable

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


_NEG_SUFFIXES = ("", "_B", "_E", "_M")


@dataclass
class AugmentationReport:
    loaded_fasta_records: int = 0
    pruned_domain_records: int = 0
    emitted_records: int = 0
    originals: int = 0
    shuffled: int = 0
    negatives: int = 0
    domain_slices: int = 0
    skipped: int = 0


def simplify_id(seq_id: str) -> str:
    parts = seq_id.split("|")
    if len(parts) >= 3:
        return parts[1]
    return seq_id


def _rand_neg_suffix() -> str:
    return random.choice(_NEG_SUFFIXES)


def _make_record(seq_str: str, rec_id: str) -> SeqRecord:
    return SeqRecord(Seq(seq_str), id=rec_id, name=rec_id, description="")


def _safe_id_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("_")
    return safe or "domain"


def _unique_id_with_suffix(stem: str, suffix: str, existing_ids: Container[str]) -> str:
    candidate = f"{stem}{suffix}"
    if candidate not in existing_ids:
        return candidate
    idx = 2
    while True:
        candidate = f"{stem}__dup{idx}{suffix}"
        if candidate not in existing_ids:
            return candidate
        idx += 1


def _ensure_unique_ids(records: Iterable[SeqRecord], label: str) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for rec in records:
        if rec.id in seen:
            dupes.add(rec.id)
        seen.add(rec.id)
    if dupes:
        sample = ", ".join(sorted(dupes)[:5])
        raise ValueError(f"{label} has duplicate IDs after normalization: {sample}")


def _prune_domain_dict_to_fasta(domain_dict: dict, fasta_id_set: set[str]) -> dict:
    pruned = {}
    for key, doms in domain_dict.items():
        simple_key = simplify_id(str(key))
        if simple_key not in fasta_id_set:
            continue
        filtered = []
        for entry in doms or []:
            if len(entry) < 3:
                continue
            pfam, start, stop = entry[:3]
            if int(stop) - int(start) + 1 > 5:
                filtered.append((pfam, int(start), int(stop)))
        if filtered:
            pruned.setdefault(simple_key, []).extend(filtered)
    return {key: sorted(value, key=lambda x: x[1]) for key, value in pruned.items()}


def choose_slice(group_min: int, group_max: int, length: int, max_length: int) -> tuple[int, int, str]:
    group_span = group_max - group_min + 1
    available_extra = max_length - group_span
    offset = available_extra // 2
    candidate_start = group_min - offset
    if candidate_start < 1:
        candidate_start = 1
    if candidate_start > length - max_length + 1:
        candidate_start = length - max_length + 1
    slice_start = candidate_start
    slice_end = candidate_start + max_length - 1
    if slice_start == 1:
        label = "B"
    elif slice_end == length:
        label = "E"
    else:
        label = "M"
    return slice_start, slice_end, label


def process_sequence(
    seq_record: SeqRecord,
    domains: list[tuple[str, int, int]],
    max_length: int,
) -> list[tuple[str, int, int, Seq, list[tuple[str, int, int]]]]:
    length = len(seq_record.seq)
    sorted_domains = sorted(domains, key=lambda x: x[1])
    max_domain_stop = max(d[2] for d in sorted_domains)
    if max_domain_stop > length:
        raise ValueError(
            f"Sequence {seq_record.id} has domain stop {max_domain_stop} beyond length {length}."
        )
    overlong_domains = [d for d in sorted_domains if d[2] - d[1] + 1 > max_length]
    if overlong_domains:
        sample = ", ".join(f"{pfam}:{start}-{stop}" for pfam, start, stop in overlong_domains[:3])
        raise ValueError(
            f"Sequence {seq_record.id} has domains longer than max_length={max_length}: {sample}."
        )

    def update_domains(domain_list, slice_start):
        return [(pfam, start - slice_start + 1, stop - slice_start + 1) for pfam, start, stop in domain_list]

    slices_info = []
    if len(sorted_domains) == 1:
        pfam, start, stop = sorted_domains[0]
        if stop <= max_length:
            slice_start, slice_end, label = 1, max_length, "B"
        elif start >= length - max_length + 1:
            slice_start, slice_end, label = length - max_length + 1, length, "E"
        else:
            slice_start, slice_end, label = choose_slice(start, stop, length, max_length)
        slices_info.append((None, slice_start, slice_end, seq_record.seq[slice_start - 1:slice_end], [(pfam, start, stop)], label))
    else:
        overall_min = min(d[1] for d in sorted_domains)
        overall_max = max(d[2] for d in sorted_domains)
        if overall_max - overall_min + 1 <= max_length:
            slice_start, slice_end, label = choose_slice(overall_min, overall_max, length, max_length)
            slices_info.append((None, slice_start, slice_end, seq_record.seq[slice_start - 1:slice_end], sorted_domains, label))
        else:
            groups = []
            current = [sorted_domains[0]]
            for dom in sorted_domains[1:]:
                new_min = min([d[1] for d in current] + [dom[1]])
                new_max = max([d[2] for d in current] + [dom[2]])
                if new_max - new_min + 1 <= max_length:
                    current.append(dom)
                else:
                    groups.append(current)
                    current = [dom]
            groups.append(current)
            for idx, group in enumerate(groups, start=1):
                group_min = min(d[1] for d in group)
                group_max = max(d[2] for d in group)
                slice_start, slice_end, _ = choose_slice(group_min, group_max, length, max_length)
                label = "B" if idx == 1 and slice_start == 1 else "E" if idx == len(groups) and slice_end == length else "M"
                slices_info.append((idx, slice_start, slice_end, seq_record.seq[slice_start - 1:slice_end], group, label))

    out = []
    for slice_id, slice_start, slice_end, seq, doms, label in slices_info:
        if slice_id is None:
            name = f"{seq_record.id}_{label}"
        else:
            name = f"{seq_record.id}_{slice_id}_{label}"
        out.append((name, slice_start, slice_end, seq, update_domains(doms, slice_start)))
    return out


def shuffle_entire(seq_str: str) -> str:
    chars = list(seq_str)
    random.shuffle(chars)
    return "".join(chars)


def shuffle_non_domain(seq_str: str, domains: list[tuple[str, int, int]]) -> str:
    if not domains:
        return shuffle_entire(seq_str)
    seq_list = list(seq_str)
    current = 0
    for _, start, stop in sorted(domains, key=lambda x: x[1]):
        if start - 1 > current:
            region = seq_list[current:start - 1]
            random.shuffle(region)
            seq_list[current:start - 1] = region
        current = stop
    if current < len(seq_list):
        region = seq_list[current:]
        random.shuffle(region)
        seq_list[current:] = region
    return "".join(seq_list)


def _emit_record(records: list[SeqRecord], domain_dict: dict, record: SeqRecord, domains: list[tuple[str, int, int]]) -> None:
    if record.id in domain_dict:
        raise ValueError(f"Duplicate output record id: {record.id}")
    records.append(record)
    domain_dict[record.id] = domains


def _emit_negative(records, domain_dict, base_id: str, seq_str: str, report: AugmentationReport) -> None:
    neg_name = f"negative_{base_id}{_rand_neg_suffix()}"
    _emit_record(records, domain_dict, _make_record(shuffle_entire(seq_str), neg_name), [("None", 1, len(seq_str))])
    report.negatives += 1


def _neg_emit_probability(target_frac: float) -> float:
    if target_frac < 0 or target_frac >= 1:
        raise ValueError("negative_prob must satisfy 0 <= p < 1.")
    if target_frac == 0:
        return 0.0
    return min(1.0, target_frac / (1.0 - target_frac))


def _load_domain_dict(path: str | Path) -> dict:
    with Path(path).expanduser().open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and isinstance(data.get("domain_dict"), dict):
        return data["domain_dict"]
    return data


def _write_outputs(output_fasta: str, output_dict: str, records: list[SeqRecord], domain_dict: dict) -> None:
    record_ids = {rec.id for rec in records}
    dict_ids = set(domain_dict)
    if record_ids != dict_ids:
        raise ValueError("Output FASTA IDs do not match output domain dictionary keys.")
    with Path(output_fasta).expanduser().open("w", encoding="utf-8") as handle:
        SeqIO.write(records, handle, "fasta")
    with Path(output_dict).expanduser().open("wb") as handle:
        pickle.dump(domain_dict, handle)


def run_augmentation(**kwargs) -> AugmentationReport:
    args = SimpleNamespace(**kwargs)
    random.seed(int(args.seed))
    report = AugmentationReport()

    fasta_records = list(SeqIO.parse(str(args.fasta), "fasta"))
    report.loaded_fasta_records = len(fasta_records)
    for rec in fasta_records:
        simple_id = simplify_id(rec.id)
        rec.id = simple_id
        rec.name = simple_id
        rec.description = simple_id
    _ensure_unique_ids(fasta_records, "FASTA")

    raw_domain_dict = _load_domain_dict(args.domain_dict)
    domain_dict = _prune_domain_dict_to_fasta(raw_domain_dict, {rec.id for rec in fasta_records})
    report.pruned_domain_records = len(domain_dict)
    fasta_records = [rec for rec in fasta_records if rec.id in domain_dict]

    output_records: list[SeqRecord] = []
    output_domain_dict: dict[str, list[tuple[str, int, int]]] = {}
    p_neg = _neg_emit_probability(float(args.negative_prob))

    for rec in tqdm(fasta_records, desc="Augmenting"):
        doms = domain_dict.get(rec.id, [])
        if not doms:
            report.skipped += 1
            continue
        seq_str = str(rec.seq)
        units = []
        if len(rec.seq) > args.max_length:
            try:
                units.extend(process_sequence(rec, doms, args.max_length))
            except Exception:
                report.skipped += 1
                continue
        else:
            units.append((rec.id, 1, len(rec.seq), rec.seq, doms))

        if args.domain_slices_only:
            units = []

        for name, _start, _end, seq, updated_domains in units:
            seq_unit = str(seq)
            if not args.shuffle_only:
                _emit_record(output_records, output_domain_dict, _make_record(seq_unit, name), updated_domains)
                report.originals += 1
                if random.random() < p_neg:
                    _emit_negative(output_records, output_domain_dict, name, seq_unit, report)

            if not args.no_shuffle:
                shuffled = shuffle_non_domain(seq_unit, updated_domains)
                shuffled_name = f"shuffled_{name}"
                _emit_record(output_records, output_domain_dict, _make_record(shuffled, shuffled_name), updated_domains)
                report.shuffled += 1
                if random.random() < p_neg:
                    _emit_negative(output_records, output_domain_dict, shuffled_name, shuffled, report)

        if args.include_domain_slices or args.domain_slices_only:
            for pfam, start, stop in doms:
                if pfam == "None":
                    continue
                if stop - start + 1 > args.max_length:
                    report.skipped += 1
                    continue
                dom_seq = seq_str[start - 1:stop]
                if start == 1 and stop == len(seq_str):
                    suffix = ""
                elif start == 1:
                    suffix = "_B"
                elif stop == len(seq_str):
                    suffix = "_E"
                else:
                    suffix = "_M"
                dom_stem = f"{rec.id}/{start}-{stop}/{_safe_id_part(pfam)}"
                dom_name = _unique_id_with_suffix(dom_stem, suffix, output_domain_dict)
                _emit_record(output_records, output_domain_dict, _make_record(dom_seq, dom_name), [(pfam, 1, stop - start + 1)])
                report.domain_slices += 1
                if random.random() < p_neg:
                    _emit_negative(output_records, output_domain_dict, dom_name, dom_seq, report)

    report.emitted_records = len(output_records)
    _write_outputs(args.output_fasta, args.output_dict, output_records, output_domain_dict)
    if args.report_json:
        with Path(args.report_json).expanduser().open("w", encoding="utf-8") as handle:
            json.dump(asdict(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
    return report
