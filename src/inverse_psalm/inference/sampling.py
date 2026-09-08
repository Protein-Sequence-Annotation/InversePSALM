from __future__ import annotations

from typing import Literal

import torch


PositionPolicy = Literal["random", "confidence", "entropy", "margin"]
AaPolicy = Literal["argmax", "sample", "top-k", "top-p", "min-p"]

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


def resolve_step_size(*, length: int, step_aa: int | None = None, step_frac: float | None = None) -> int:
    """Resolve a design step size from either residues or fraction of full length."""
    if length < 1:
        raise ValueError(f"length must be positive, got {length}.")
    if (step_aa is None) == (step_frac is None):
        raise ValueError("Specify exactly one of step_aa or step_frac.")
    if step_aa is not None:
        if step_aa < 1:
            raise ValueError(f"step_aa must be positive, got {step_aa}.")
        return min(int(step_aa), int(length))
    assert step_frac is not None
    if step_frac <= 0.0 or step_frac > 1.0:
        raise ValueError(f"step_frac must be in (0, 1], got {step_frac}.")
    return min(int(length), max(1, int(round(float(step_frac) * int(length)))))


def allowed_amino_acid_token_ids(tokenizer, *, canonical_only: bool = True) -> list[int]:
    """Return tokenizer IDs allowed during design."""
    amino_acids = CANONICAL_AA if canonical_only else _single_letter_tokens(tokenizer)
    token_ids: list[int] = []
    for aa in amino_acids:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is None or token_id == tokenizer.unk_token_id:
            continue
        token_ids.append(int(token_id))
    if not token_ids:
        raise ValueError("Could not resolve any amino-acid token IDs from the tokenizer.")
    return sorted(set(token_ids))


def select_positions(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    *,
    count: int,
    allowed_token_ids: list[int],
    policy: PositionPolicy,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    uncertain: bool = False,
) -> torch.Tensor:
    """Select token positions to commit from the currently masked coordinates."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count}.")
    candidates = masked_positions.bool().nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        return candidates
    count = min(int(count), int(candidates.numel()))
    if policy == "random":
        order = torch.randperm(candidates.numel(), generator=generator, device=candidates.device)
        return candidates[order[:count]]

    probs = _allowed_probs(logits[candidates], allowed_token_ids, temperature)
    if policy == "confidence":
        scores = probs.max(dim=-1).values
        descending = not uncertain
    elif policy == "entropy":
        scores = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        descending = uncertain
    elif policy == "margin":
        top2 = probs.topk(k=min(2, probs.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            scores = top2[:, 0]
        else:
            scores = top2[:, 0] - top2[:, 1]
        descending = not uncertain
    else:
        raise ValueError(f"Unknown position policy: {policy!r}.")

    order = torch.argsort(scores, descending=descending)
    return candidates[order[:count]]


def sample_amino_acids(
    logits: torch.Tensor,
    *,
    allowed_token_ids: list[int],
    policy: AaPolicy,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose amino-acid token IDs from selected-position logits."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be rank 2 [positions, vocab], got shape {tuple(logits.shape)}.")
    allowed_ids = torch.tensor(allowed_token_ids, dtype=torch.long, device=logits.device)
    local_logits = logits.index_select(dim=-1, index=allowed_ids)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    local_logits = local_logits / float(temperature)

    if policy == "argmax":
        local_choice = local_logits.argmax(dim=-1)
    elif policy == "sample":
        probs = torch.softmax(local_logits, dim=-1)
        local_choice = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    elif policy == "top-k":
        k = int(top_k or 0)
        if k < 1:
            raise ValueError("top-k amino-acid policy requires --top-k >= 1.")
        local_choice = _sample_top_k(local_logits, k=min(k, local_logits.size(-1)), generator=generator)
    elif policy == "top-p":
        p = float(1.0 if top_p is None else top_p)
        if p <= 0.0 or p > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {p}.")
        local_choice = _sample_top_p(local_logits, top_p=p, generator=generator)
    elif policy == "min-p":
        p = float(0.05 if min_p is None else min_p)
        if p < 0.0 or p > 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {p}.")
        local_choice = _sample_min_p(local_logits, min_p=p, generator=generator)
    else:
        raise ValueError(f"Unknown amino-acid policy: {policy!r}.")
    return allowed_ids[local_choice]


def token_probabilities(
    logits: torch.Tensor,
    *,
    allowed_token_ids: list[int],
    token_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return probabilities assigned to token IDs under allowed amino-acid logits."""
    probs = _allowed_probs(logits, allowed_token_ids, temperature)
    allowed_ids = torch.tensor(allowed_token_ids, dtype=torch.long, device=logits.device)
    out = torch.zeros(token_ids.shape, dtype=probs.dtype, device=logits.device)
    for idx, token_id in enumerate(token_ids):
        matches = (allowed_ids == token_id).nonzero(as_tuple=False).flatten()
        if matches.numel() > 0:
            out[idx] = probs[idx, int(matches[0].item())]
    return out


def token_ids_to_sequence(tokenizer, token_ids: list[int], masked_positions: list[bool], masked_char: str = "X") -> str:
    """Decode residue token IDs to a plain sequence, marking masked positions."""
    residues: list[str] = []
    for token_id, is_masked in zip(token_ids, masked_positions):
        if is_masked:
            residues.append(masked_char)
            continue
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        residues.append(token if isinstance(token, str) and len(token) == 1 else masked_char)
    return "".join(residues)


def _allowed_probs(logits: torch.Tensor, allowed_token_ids: list[int], temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    allowed_ids = torch.tensor(allowed_token_ids, dtype=torch.long, device=logits.device)
    return torch.softmax(logits.index_select(dim=-1, index=allowed_ids) / float(temperature), dim=-1)


def _sample_top_k(logits: torch.Tensor, *, k: int, generator: torch.Generator | None) -> torch.Tensor:
    values, indices = logits.topk(k=k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return indices.gather(dim=-1, index=sampled[:, None]).squeeze(-1)


def _sample_top_p(logits: torch.Tensor, *, top_p: float, generator: torch.Generator | None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    keep = cumulative <= top_p
    keep[:, 0] = True
    next_after_cutoff = (cumulative - sorted_probs) < top_p
    keep = keep | next_after_cutoff
    filtered = sorted_probs.masked_fill(~keep, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(filtered, num_samples=1, generator=generator).squeeze(-1)
    return sorted_indices.gather(dim=-1, index=sampled[:, None]).squeeze(-1)


def _sample_min_p(logits: torch.Tensor, *, min_p: float, generator: torch.Generator | None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    top_probs = probs.max(dim=-1, keepdim=True).values
    keep = probs >= (float(min_p) * top_probs)
    keep.scatter_(dim=-1, index=probs.argmax(dim=-1, keepdim=True), value=True)
    filtered = probs.masked_fill(~keep, 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(filtered, num_samples=1, generator=generator).squeeze(-1)


def _single_letter_tokens(tokenizer) -> str:
    vocab = tokenizer.get_vocab()
    tokens = sorted(token for token in vocab if len(token) == 1 and token.isalpha() and token.isupper())
    return "".join(tokens)
