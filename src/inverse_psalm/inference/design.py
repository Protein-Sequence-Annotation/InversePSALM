from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from inverse_psalm.inference.annotations import BackgroundMode, DomainSpan, build_annotation_track
from inverse_psalm.inference.sampling import (
    AaPolicy,
    PositionPolicy,
    allowed_amino_acid_token_ids,
    resolve_step_size,
    sample_amino_acids,
    select_positions,
    token_probabilities,
    token_ids_to_sequence,
)
from inverse_psalm.models.generator import InversePSALM


@dataclass(frozen=True)
class DesignRequest:
    checkpoint: str | Path
    length: int
    domains: list[DomainSpan]
    label_mapping: dict[str, Any]
    output_id: str = "design"
    step_aa: int | None = None
    step_frac: float | None = None
    position_policy: PositionPolicy = "confidence"
    aa_policy: AaPolicy = "top-p"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = 0.95
    min_p: float | None = 0.05
    refine_steps: int = 0
    refine_step_aa: int | None = None
    refine_step_frac: float | None = None
    refine_position_policy: PositionPolicy = "confidence"
    refine_aa_policy: AaPolicy = "top-p"
    refine_temperature: float = 1.0
    refine_top_k: int | None = None
    refine_top_p: float | None = 0.95
    refine_min_p: float | None = 0.05
    background_mode: BackgroundMode = "ignore"
    canonical_only: bool = True
    masked_char: str = "?"
    masked_fill_aa: str = "A"
    seed: int = 100
    device: str = "cuda"
    dtype: str = "auto"
    max_aa_length: int = 4096
    ignore_label: int = -100
    use_fa: bool | None = None


@dataclass(frozen=True)
class DesignStep:
    phase: str
    step: int
    sequence: str
    foldable_sequence: str
    committed_positions: list[int]
    remaining_masked: int


@dataclass(frozen=True)
class DesignResult:
    output_id: str
    final_sequence: str
    steps: list[DesignStep]
    metadata: dict[str, Any]


def design_sequence(request: DesignRequest) -> DesignResult:
    """Iteratively unmask a sequence conditioned on a user-specified annotation track."""
    _validate_request(request)
    device = _resolve_device(request.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(request.seed))
    use_fa = False if device.type == "cpu" and request.use_fa is None else request.use_fa
    dtype = _resolve_dtype(request.dtype, device)

    model = InversePSALM.from_pretrained(
        request.checkpoint,
        use_fa=use_fa,
        map_location="cpu",
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = model.tokenizer
    allowed_ids = allowed_amino_acid_token_ids(tokenizer, canonical_only=request.canonical_only)
    step_size = resolve_step_size(length=request.length, step_aa=request.step_aa, step_frac=request.step_frac)

    annotation_track = build_annotation_track(
        length=request.length,
        domains=request.domains,
        label_mapping=request.label_mapping,
        ignore_label=request.ignore_label,
        background_mode=request.background_mode,
    )
    input_ids = _initial_input_ids(tokenizer, request.length, device)
    attention_mask = torch.ones_like(input_ids)
    annotation_ids = torch.tensor([annotation_track.annotation_ids], dtype=torch.long, device=device)
    remaining_mask = torch.tensor([[False] + [True] * request.length + [False]], dtype=torch.bool, device=device)

    steps = [
        _snapshot_step(
            step=0,
            tokenizer=tokenizer,
            input_ids=input_ids,
            remaining_mask=remaining_mask,
            committed_positions=[],
            masked_char=request.masked_char,
            masked_fill_aa=request.masked_fill_aa,
        )
    ]
    step_idx = 0
    with torch.no_grad():
        while bool(remaining_mask.any()):
            step_idx += 1
            outputs = model(input_ids=input_ids, annotation_ids=annotation_ids, attention_mask=attention_mask)
            logits = outputs["logits"][0]
            selected = select_positions(
                logits,
                remaining_mask[0],
                count=step_size,
                allowed_token_ids=allowed_ids,
                policy=request.position_policy,
                temperature=request.temperature,
                generator=generator,
            )
            sampled_ids = sample_amino_acids(
                logits.index_select(dim=0, index=selected),
                allowed_token_ids=allowed_ids,
                policy=request.aa_policy,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                min_p=request.min_p,
                generator=generator,
            )
            input_ids[0, selected] = sampled_ids
            remaining_mask[0, selected] = False
            steps.append(
                _snapshot_step(
                    step=step_idx,
                    phase="generation",
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    remaining_mask=remaining_mask,
                    committed_positions=[int(pos.item()) for pos in selected],
                    masked_char=request.masked_char,
                    masked_fill_aa=request.masked_fill_aa,
                )
            )

        if request.refine_steps > 0:
            refine_step_size = resolve_step_size(
                length=request.length,
                step_aa=request.refine_step_aa,
                step_frac=request.refine_step_frac,
            )
            residue_positions = torch.tensor(
                [[False] + [True] * request.length + [False]],
                dtype=torch.bool,
                device=device,
            )
            for _ in range(int(request.refine_steps)):
                step_idx += 1
                full_logits = model(
                    input_ids=input_ids,
                    annotation_ids=annotation_ids,
                    attention_mask=attention_mask,
                )["logits"][0]
                selected = select_positions(
                    full_logits,
                    residue_positions[0],
                    count=refine_step_size,
                    allowed_token_ids=allowed_ids,
                    policy=request.refine_position_policy,
                    temperature=request.refine_temperature,
                    generator=generator,
                    uncertain=True,
                )
                old_ids = input_ids[0, selected].clone()
                masked_input_ids = input_ids.clone()
                masked_input_ids[0, selected] = int(tokenizer.mask_token_id)
                masked_logits = model(
                    input_ids=masked_input_ids,
                    annotation_ids=annotation_ids,
                    attention_mask=attention_mask,
                )["logits"][0].index_select(dim=0, index=selected)
                sampled_ids = sample_amino_acids(
                    masked_logits,
                    allowed_token_ids=allowed_ids,
                    policy=request.refine_aa_policy,
                    temperature=request.refine_temperature,
                    top_k=request.refine_top_k,
                    top_p=request.refine_top_p,
                    min_p=request.refine_min_p,
                    generator=generator,
                )
                old_probs = token_probabilities(
                    masked_logits,
                    allowed_token_ids=allowed_ids,
                    token_ids=old_ids,
                    temperature=request.refine_temperature,
                )
                new_probs = token_probabilities(
                    masked_logits,
                    allowed_token_ids=allowed_ids,
                    token_ids=sampled_ids,
                    temperature=request.refine_temperature,
                )
                accepted = new_probs >= old_probs
                accepted_positions = selected[accepted]
                input_ids[0, accepted_positions] = sampled_ids[accepted]
                steps.append(
                    _snapshot_step(
                        step=step_idx,
                        phase="refinement",
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        remaining_mask=remaining_mask,
                        committed_positions=[int(pos.item()) for pos in accepted_positions],
                        masked_char=request.masked_char,
                        masked_fill_aa=request.masked_fill_aa,
                    )
                )

    metadata = {
        "request": _request_metadata(request),
        "step_size": step_size,
        "num_steps": len(steps) - 1,
        "generation_steps": step_idx - int(request.refine_steps),
        "refinement_steps": int(request.refine_steps),
        "annotation_track": {
            "residue_count": request.length,
            "domain_count": len(request.domains),
            "background_mode": request.background_mode,
        },
    }
    return DesignResult(
        output_id=request.output_id,
        final_sequence=steps[-1].sequence,
        steps=steps,
        metadata=metadata,
    )


def _initial_input_ids(tokenizer, length: int, device: torch.device) -> torch.Tensor:
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id.")
    if tokenizer.cls_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define cls_token_id and eos_token_id.")
    input_ids = torch.full((1, length + 2), int(tokenizer.mask_token_id), dtype=torch.long, device=device)
    input_ids[0, 0] = int(tokenizer.cls_token_id)
    input_ids[0, -1] = int(tokenizer.eos_token_id)
    return input_ids


def _resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for design generation, but torch.cuda.is_available() is false.")
    return resolved


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    normalized = str(dtype).lower()
    if normalized == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        if device.type == "cuda":
            raise ValueError("CUDA generation with FlashAttention requires --dtype bf16 or fp16.")
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype!r}; use auto, bf16, fp16, or fp32.")


def _snapshot_step(
    *,
    step: int,
    phase: str = "generation",
    tokenizer,
    input_ids: torch.Tensor,
    remaining_mask: torch.Tensor,
    committed_positions: list[int],
    masked_char: str,
    masked_fill_aa: str,
) -> DesignStep:
    residue_ids = [int(value) for value in input_ids[0, 1:-1].detach().cpu().tolist()]
    residue_masked = [bool(value) for value in remaining_mask[0, 1:-1].detach().cpu().tolist()]
    sequence = token_ids_to_sequence(tokenizer, residue_ids, residue_masked, masked_char=masked_char)
    foldable = "".join(masked_fill_aa if is_masked else aa for aa, is_masked in zip(sequence, residue_masked))
    return DesignStep(
        phase=phase,
        step=int(step),
        sequence=sequence,
        foldable_sequence=foldable,
        committed_positions=[_token_pos_to_residue_pos(pos) for pos in committed_positions],
        remaining_masked=int(sum(residue_masked)),
    )


def _token_pos_to_residue_pos(token_pos: int) -> int:
    return int(token_pos)


def _validate_request(request: DesignRequest) -> None:
    if request.length < 1:
        raise ValueError(f"length must be positive, got {request.length}.")
    if request.length > request.max_aa_length:
        raise ValueError(f"length {request.length} exceeds max_aa_length={request.max_aa_length}.")
    if len(request.masked_char) != 1:
        raise ValueError(f"masked_char must be a single character, got {request.masked_char!r}.")
    if len(request.masked_fill_aa) != 1 or not request.masked_fill_aa.isalpha():
        raise ValueError(f"masked_fill_aa must be a single amino-acid character, got {request.masked_fill_aa!r}.")
    resolve_step_size(length=request.length, step_aa=request.step_aa, step_frac=request.step_frac)
    if request.refine_steps < 0:
        raise ValueError(f"refine_steps must be >= 0, got {request.refine_steps}.")
    if request.refine_steps > 0:
        resolve_step_size(
            length=request.length,
            step_aa=request.refine_step_aa,
            step_frac=request.refine_step_frac,
        )


def _request_metadata(request: DesignRequest) -> dict[str, Any]:
    data = asdict(request)
    data["checkpoint"] = str(request.checkpoint)
    data["domains"] = [asdict(domain) for domain in request.domains]
    data.pop("label_mapping", None)
    return data
