from __future__ import annotations

import math
from typing import Any

import torch


def schedule_value(spec: Any, progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    if isinstance(spec, (float, int)):
        return float(spec)
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported schedule spec: {spec!r}")
    kind = str(spec.get("type", "constant")).lower()
    if kind == "constant":
        return float(spec.get("value", spec.get("start", 0.0)))
    start = float(spec.get("start", spec.get("value", 0.0)))
    end = float(spec.get("end", start))
    if kind == "linear":
        return start + (end - start) * progress
    if kind == "cosine":
        mix = 0.5 * (1.0 - math.cos(math.pi * progress))
        return start + (end - start) * mix
    if kind == "piecewise":
        points = spec.get("points", [])
        if not points:
            return start
        last_p, last_v = 0.0, float(points[0][1])
        for p, v in points:
            p = float(p)
            v = float(v)
            if progress < p:
                denom = max(1e-12, p - last_p)
                return last_v + (v - last_v) * ((progress - last_p) / denom)
            last_p, last_v = p, v
        return last_v
    raise ValueError(f"Unknown schedule type: {kind}")


def sample_schedule_value(
    spec: Any,
    progress: float,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    """Resolve a deterministic schedule or sample a stochastic mask rate."""
    if not isinstance(spec, dict):
        return schedule_value(spec, progress)

    kind = str(spec.get("type", "constant")).lower()
    if kind == "dsm_mixture":
        return _sample_dsm_mixture_branch(spec, generator, device)["mask_rate"]
    return schedule_value(spec, progress)


def _sample_dsm_mixture_branch(
    spec: dict[str, Any],
    generator: torch.Generator,
    device: torch.device,
    default_mask_type: str = "random",
) -> dict[str, float | str]:
    branches = spec.get("branches", [])
    if not isinstance(branches, list) or not branches:
        raise ValueError("dsm_mixture schedule requires a non-empty branches list.")

    parsed: list[tuple[float, float, float, str]] = []
    total_probability = 0.0
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError(f"DSM mixture branch must be a mapping, got {branch!r}.")
        probability = float(branch.get("probability", 0.0))
        min_rate = min(max(float(branch.get("min_rate", 0.0)), 0.0), 1.0)
        max_rate = min(max(float(branch.get("max_rate", 1.0)), 0.0), 1.0)
        mask_type = _normalize_mask_type(branch.get("mask_type", default_mask_type))
        if probability < 0.0:
            raise ValueError(f"DSM mixture branch probability must be non-negative, got {probability}.")
        if max_rate < min_rate:
            raise ValueError(f"DSM mixture branch max_rate {max_rate} is below min_rate {min_rate}.")
        if probability == 0.0:
            continue
        parsed.append((probability, min_rate, max_rate, mask_type))
        total_probability += probability

    if total_probability <= 0.0 or not parsed:
        raise ValueError("dsm_mixture schedule requires positive total branch probability.")

    threshold = torch.rand((), generator=generator, device=device).item() * total_probability
    cumulative = 0.0
    selected = parsed[-1]
    for branch in parsed:
        cumulative += branch[0]
        if threshold <= cumulative:
            selected = branch
            break

    _, min_rate, max_rate, mask_type = selected
    u = torch.rand((), generator=generator, device=device).item()
    return {"mask_rate": min_rate + (max_rate - min_rate) * u, "mask_type": mask_type}


def _normalize_mask_type(mask_type: Any) -> str:
    normalized = str(mask_type).lower()
    if normalized not in {"random", "contiguous"}:
        raise ValueError(f"Unknown mask_type: {normalized!r}")
    return normalized


def _resolve_masking_plan(
    masking_cfg: dict[str, Any],
    progress: float,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, float | str]:
    schedule = masking_cfg.get("schedule", 0.15)
    default_mask_type = _normalize_mask_type(masking_cfg.get("mask_type", "random"))
    if isinstance(schedule, dict) and str(schedule.get("type", "constant")).lower() == "dsm_mixture":
        return _sample_dsm_mixture_branch(schedule, generator, device, default_mask_type)
    return {
        "mask_rate": sample_schedule_value(schedule, progress, generator, device),
        "mask_type": default_mask_type,
    }


def _sample_contiguous_span_mask(
    residue_mask: torch.Tensor,
    target_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = torch.zeros_like(residue_mask, dtype=torch.bool)
    valid_positions = residue_mask.nonzero(as_tuple=False).flatten()
    if target_count <= 0 or valid_positions.numel() == 0:
        return mask

    runs: list[tuple[int, int]] = []
    run_start = int(valid_positions[0].item())
    prev = run_start
    for pos_tensor in valid_positions[1:]:
        pos = int(pos_tensor.item())
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((run_start, prev + 1))
        run_start = pos
        prev = pos
    runs.append((run_start, prev + 1))

    longest_run = max(end - start for start, end in runs)
    span_len = min(int(target_count), int(longest_run))
    candidate_starts: list[int] = []
    for start, end in runs:
        run_len = end - start
        if run_len >= span_len:
            candidate_starts.extend(range(start, end - span_len + 1))
    if not candidate_starts:
        return mask

    start_idx = int(
        candidate_starts[
            torch.randint(
                len(candidate_starts),
                (1,),
                generator=generator,
                device=residue_mask.device,
            ).item()
        ]
    )
    mask[start_idx : start_idx + span_len] = True
    return mask


def apply_masking(
    *,
    input_ids: torch.Tensor,
    residue_mask: torch.Tensor,
    mask_token_id: int,
    config: dict[str, Any],
    progress: float,
    seed: int,
) -> dict[str, torch.Tensor | float]:
    masking_cfg = config.get("masking", {})
    device = input_ids.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    masking_plan = _resolve_masking_plan(masking_cfg, progress, generator, device)
    mask_rate = float(masking_plan["mask_rate"])
    mask_type = str(masking_plan["mask_type"])

    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    final_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    residue_bool = residue_mask.bool()

    for row in range(input_ids.size(0)):
        row_residues = residue_bool[row]
        n_residues = int(row_residues.sum().item())
        if n_residues == 0:
            continue
        target_count = min(n_residues, max(1, int(round(mask_rate * n_residues))))
        if mask_type == "contiguous":
            row_mask = _sample_contiguous_span_mask(row_residues, target_count, generator)
        elif mask_type == "random":
            valid = row_residues.nonzero(as_tuple=False).flatten()
            order = torch.randperm(valid.numel(), generator=generator, device=device)
            chosen = valid[order[:target_count]]
            row_mask = torch.zeros_like(row_residues, dtype=torch.bool)
            row_mask[chosen] = True
        else:
            raise ValueError(f"Unknown mask_type: {mask_type!r}")
        final_mask[row] = row_mask

    labels[final_mask] = input_ids[final_mask]
    masked_input_ids[final_mask] = int(mask_token_id)
    masked_tokens = int(final_mask.sum().item())
    total_residues = int(residue_bool.sum().item())
    return {
        "masked_input_ids": masked_input_ids,
        "mlm_labels": labels,
        "mask_positions": final_mask,
        "mask_rate_actual": float(masked_tokens / max(1, total_residues)),
        "mask_rate_target": float(mask_rate),
        "mask_type": mask_type,
    }
