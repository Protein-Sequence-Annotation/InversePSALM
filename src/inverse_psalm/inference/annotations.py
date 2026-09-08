from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


BackgroundMode = Literal["none", "ignore"]


@dataclass(frozen=True)
class DomainSpan:
    """A 1-indexed inclusive annotation span over residue coordinates."""

    pfam: str | None
    start: int
    stop: int


@dataclass(frozen=True)
class AnnotationTrack:
    """Token-aligned labels and masks for a synthetic design target."""

    annotation_ids: list[int]
    residue_mask: list[int]
    domain_mask: list[int]
    domains: list[DomainSpan]


def parse_domain_spec(spec: str) -> DomainSpan:
    """Parse a CLI domain spec like `PF00042:25-175` or `None:176-250`."""
    try:
        label, coords = spec.split(":", 1)
        start_text, stop_text = coords.split("-", 1)
    except ValueError as exc:
        raise ValueError(f"Domain spec must look like PF00042:25-175, got {spec!r}.") from exc

    label = label.strip()
    pfam = None if label.lower() in {"none", "null", "na", ""} else label
    try:
        start = int(start_text)
        stop = int(stop_text)
    except ValueError as exc:
        raise ValueError(f"Domain coordinates must be integers in {spec!r}.") from exc
    return DomainSpan(pfam=pfam, start=start, stop=stop)


def build_annotation_track(
    *,
    length: int,
    domains: list[DomainSpan],
    label_mapping: dict[str, Any],
    ignore_label: int = -100,
    background_mode: BackgroundMode = "ignore",
) -> AnnotationTrack:
    """Build token-aligned PSALM fine labels for a requested design length.

    Coordinates are 1-indexed and inclusive, matching the existing training data
    convention in `inverse_psalm.data.labels`.
    """
    if length < 1:
        raise ValueError(f"length must be positive, got {length}.")
    if background_mode not in {"none", "ignore"}:
        raise ValueError(f"background_mode must be 'none' or 'ignore', got {background_mode!r}.")
    if "None" not in label_mapping:
        raise ValueError("label_mapping must contain a 'None' label.")

    none_label = int(label_mapping["None"])
    background_label = none_label if background_mode == "none" else int(ignore_label)
    annotation_ids = [int(ignore_label)] + [background_label] * length + [int(ignore_label)]
    occupied: list[DomainSpan | None] = [None] * (length + 1)

    for domain in sorted(domains, key=lambda item: (item.start, item.stop)):
        _validate_domain(domain, length)
        for residue_idx in range(domain.start, domain.stop + 1):
            previous = occupied[residue_idx]
            if previous is not None:
                raise ValueError(
                    "Domain spans overlap at residue "
                    f"{residue_idx}: {previous!r} conflicts with {domain!r}."
                )
            occupied[residue_idx] = domain
        if domain.pfam is None:
            fill = none_label
            for token_idx in range(domain.start, domain.stop + 1):
                annotation_ids[token_idx] = fill
            continue

        state_map = label_mapping.get(domain.pfam)
        if not isinstance(state_map, dict):
            raise ValueError(f"Pfam family {domain.pfam!r} is missing from the label mapping.")
        start_label = int(state_map["start"])
        middle_label = int(state_map["middle"])
        stop_label = int(state_map["stop"])
        annotation_ids[domain.start] = start_label
        if domain.start == domain.stop:
            continue
        annotation_ids[domain.stop] = stop_label
        for token_idx in range(domain.start + 1, domain.stop):
            annotation_ids[token_idx] = middle_label

    residue_mask = [0] + [1] * length + [0]
    domain_mask = [
        1 if label not in (int(ignore_label), none_label) else 0
        for label in annotation_ids
    ]
    return AnnotationTrack(
        annotation_ids=annotation_ids,
        residue_mask=residue_mask,
        domain_mask=domain_mask,
        domains=list(domains),
    )


def _validate_domain(domain: DomainSpan, length: int) -> None:
    if domain.start < 1:
        raise ValueError(f"Domain start must be >= 1, got {domain.start}.")
    if domain.stop < domain.start:
        raise ValueError(f"Domain stop {domain.stop} is before start {domain.start}.")
    if domain.stop > length:
        raise ValueError(f"Domain {domain!r} exceeds requested length {length}.")
