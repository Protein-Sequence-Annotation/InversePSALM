from __future__ import annotations

import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5, ignore_index: int = -100) -> torch.Tensor:
    valid = labels != ignore_index
    if not bool(valid.any()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    topk = logits.topk(k, dim=-1).indices
    hits = (topk[valid] == labels[valid].unsqueeze(-1)).any(dim=-1)
    return hits.to(dtype=logits.dtype).mean()


def mean_loss(loss_sum: float, count: int) -> float:
    return float(loss_sum) / float(max(1, count))
