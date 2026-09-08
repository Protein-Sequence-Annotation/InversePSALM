#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold a design trajectory FASTA with ESMFold.")
    parser.add_argument("--trajectory-fasta", required=True, help="FASTA written by scripts/design_sequence.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for PDB files and manifest.json.")
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Optional ESMFold axial attention chunk size.")
    parser.add_argument("--max-records", type=int, default=None, help="Fold only the first N FASTA records.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repo root used for local model/cache directories.",
    )
    args = parser.parse_args()

    _configure_repo_caches(Path(args.repo_root))

    import torch
    import esm

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    if args.chunk_size is not None:
        model.set_chunk_size(int(args.chunk_size))

    manifest = []
    for index, (record_id, sequence) in enumerate(_read_fasta(Path(args.trajectory_fasta))):
        if args.max_records is not None and index >= args.max_records:
            break
        pdb = model.infer_pdb(sequence)
        pdb_name = f"{index:04d}_{_safe_name(record_id)}.pdb"
        pdb_path = output_dir / pdb_name
        pdb_path.write_text(pdb, encoding="utf-8")
        manifest.append(
            {
                "index": index,
                "record_id": record_id,
                "sequence_length": len(sequence),
                "pdb": pdb_name,
            }
        )
        print(f"[esmfold] wrote {pdb_path}", flush=True)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": manifest}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _configure_repo_caches(repo_root: Path) -> None:
    cache_root = Path(os.environ.get("INVERSE_PSALM_CACHE_ROOT", repo_root / ".cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
    os.environ.setdefault("CONDA_PKGS_DIRS", str(cache_root / "conda" / "pkgs"))
    for key in ("XDG_CACHE_HOME", "HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _read_fasta(path: Path):
    record_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record_id is not None:
                    yield record_id, "".join(chunks)
                record_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if record_id is not None:
        yield record_id, "".join(chunks)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160]


if __name__ == "__main__":
    main()
