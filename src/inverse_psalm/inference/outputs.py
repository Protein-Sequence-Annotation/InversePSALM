from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from inverse_psalm.inference.design import DesignResult


def write_design_result(result: "DesignResult", output_dir: str | Path) -> None:
    """Write final sequence, trajectory FASTA, JSONL steps, and run metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_fasta(out / "final.fasta", [(result.output_id, result.final_sequence)])
    _write_fasta(
        out / "trajectory.fasta",
        [
            (f"{result.output_id}_step{step.step:04d}_masked{step.remaining_masked}", step.sequence)
            for step in result.steps
        ],
    )
    _write_fasta(
        out / "trajectory_foldable.fasta",
        [
            (f"{result.output_id}_step{step.step:04d}_masked{step.remaining_masked}", step.foldable_sequence)
            for step in result.steps
        ],
    )
    with (out / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for step in result.steps:
            handle.write(json.dumps(asdict(step), sort_keys=True) + "\n")
    with (out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            handle.write(sequence + "\n")
