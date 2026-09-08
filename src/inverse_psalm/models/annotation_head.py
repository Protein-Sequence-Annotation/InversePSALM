from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    kind = str(name).lower()
    if kind == "relu":
        return nn.ReLU()
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown annotation encoder activation: {name!r}")


def _make_mlp(
    input_dim: int,
    mlp_dim: int,
    hidden_dim: int,
    activation: str,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, mlp_dim),
        _activation(activation),
        nn.Linear(mlp_dim, hidden_dim),
    )


def decode_psalm_labels(ann_ids: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Decode PSALM fine-label IDs into family IDs and state types.

    `0` maps to `(family=0, state=0)` for `None`. Positive IDs follow contiguous
    family triples: start, middle, stop.
    """
    safe_ids = ann_ids.clamp_min(0)
    family_ids = torch.zeros_like(safe_ids)
    state_types = torch.zeros_like(safe_ids)
    mask = safe_ids > 0
    k = safe_ids[mask] - 1
    family_ids[mask] = k // 3 + 1
    state_types[mask] = k % 3 + 1
    return family_ids, state_types


class AnnotationEncoderA(nn.Module):
    """Full-token annotation embedding over PSALM fine-label IDs."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        embedding_dim: int = 640,
        mlp_dim: int = 1280,
        dropout: float = 0.0,
        ignore_label: int = -100,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_labels = int(vocab_size)
        self.ignore_label = int(ignore_label)
        self.embedding = nn.Embedding(self.num_labels, int(embedding_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.proj = _make_mlp(
            input_dim=int(embedding_dim),
            mlp_dim=int(mlp_dim),
            hidden_dim=int(hidden_size),
            activation=activation,
        )

    def forward(self, annotation_ids: torch.Tensor) -> torch.Tensor:
        valid = annotation_ids != self.ignore_label
        safe_ids = annotation_ids.clone()
        safe_ids = safe_ids.masked_fill(safe_ids == self.ignore_label, 0)
        safe_ids = safe_ids.clamp(min=0, max=self.num_labels - 1)
        projected = self.proj(self.dropout(self.embedding(safe_ids)))
        return projected * valid.unsqueeze(-1).to(dtype=projected.dtype)


class AnnotationEncoderB(nn.Module):
    """Factorized family/state annotation embedding for PSALM fine labels."""

    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        family_embedding_dim: int = 576,
        state_embedding_dim: int = 64,
        mlp_dim: int = 1280,
        dropout: float = 0.0,
        ignore_label: int = -100,
        activation: str = "gelu",
    ):
        super().__init__()
        if (int(num_labels) - 1) % 3 != 0:
            raise ValueError(f"Factorized annotation encoder requires 1 + 3N labels, got {num_labels}.")
        self.num_labels = int(num_labels)
        self.num_families = (self.num_labels - 1) // 3
        self.ignore_label = int(ignore_label)
        self.family_embedding = nn.Embedding(self.num_families + 1, int(family_embedding_dim))
        self.state_embedding = nn.Embedding(4, int(state_embedding_dim))
        self.dropout = nn.Dropout(float(dropout))
        input_dim = int(family_embedding_dim) + int(state_embedding_dim)
        self.proj = _make_mlp(
            input_dim=input_dim,
            mlp_dim=int(mlp_dim),
            hidden_dim=int(hidden_size),
            activation=activation,
        )

    def forward(self, annotation_ids: torch.Tensor) -> torch.Tensor:
        valid = annotation_ids != self.ignore_label
        safe_ids = annotation_ids.masked_fill(~valid, 0).clamp(min=0, max=self.num_labels - 1)
        family_ids, state_types = decode_psalm_labels(safe_ids)
        family = self.family_embedding(family_ids)
        state = self.state_embedding(state_types)
        projected = self.proj(self.dropout(torch.cat([family, state], dim=-1)))
        return projected * valid.unsqueeze(-1).to(dtype=projected.dtype)


def build_annotation_encoder(
    *,
    config: dict[str, Any],
    num_labels: int,
    hidden_size: int,
    dropout: float,
    ignore_label: int,
) -> nn.Module:
    model_cfg = config.get("model", {})
    encoder_cfg = model_cfg.get("annotation_encoder", {})
    encoder_type = str(encoder_cfg.get("type", "full_token")).lower()
    if encoder_type in {"full_token", "a"}:
        cfg = encoder_cfg.get("full_token", {})
        return AnnotationEncoderA(
            vocab_size=int(num_labels),
            hidden_size=int(hidden_size),
            embedding_dim=int(cfg.get("embedding_dim", 640)),
            mlp_dim=int(cfg.get("mlp_dim", 1280)),
            dropout=float(dropout),
            ignore_label=int(ignore_label),
            activation=str(cfg.get("activation", "gelu")),
        )
    if encoder_type in {"factorized", "b"}:
        cfg = encoder_cfg.get("factorized", {})
        return AnnotationEncoderB(
            num_labels=int(num_labels),
            hidden_size=int(hidden_size),
            family_embedding_dim=int(cfg.get("family_embedding_dim", 576)),
            state_embedding_dim=int(cfg.get("state_embedding_dim", 64)),
            mlp_dim=int(cfg.get("mlp_dim", 1280)),
            dropout=float(dropout),
            ignore_label=int(ignore_label),
            activation=str(cfg.get("activation", "gelu")),
        )
    raise ValueError(f"Unknown annotation encoder type: {encoder_type!r}")
