from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if not bool((labels != ignore_index).any()):
        zero_source = torch.nan_to_num(
            logits.reshape(-1)[:1].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return zero_source.sum().to(dtype=logits.dtype) * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    valid = labels != ignore_index
    if not bool(valid.any()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    pred = logits.argmax(dim=-1)
    return (pred[valid] == labels[valid]).to(dtype=logits.dtype).mean()
