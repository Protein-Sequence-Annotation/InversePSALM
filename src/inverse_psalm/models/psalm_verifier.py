from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


CANONICAL_AAS = tuple("ACDEFGHIKLMNPQRSTVWY")


class FrozenPSALMVerifier(nn.Module):
    """Differentiable frozen PSALM verifier fed by soft generator distributions."""

    def __init__(
        self,
        model_name: str = "ProteinSequenceAnnotation/PSALM-2",
        device: str | None = None,
        use_fa: bool | None = None,
        warmup: bool = False,
        canonical_aa_only: bool = True,
        input_mode: str = "st_top1",
        psalm_repo_path: str | None = None,
    ):
        super().__init__()
        if psalm_repo_path:
            repo_path = str(Path(psalm_repo_path).expanduser())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        try:
            from psalm.psalm_model import PSALM
        except ImportError as exc:
            raise ImportError(
                "The PSALM package is required for the frozen verifier. Use the "
                "PSALM-github environment or install protein-sequence-annotation."
            ) from exc
        self.psalm = PSALM(model_name=model_name, device=device, use_fa=use_fa, warmup=warmup)
        self._disable_verifier_token_dropout()
        self.psalm.eval()
        for parameter in self.psalm.parameters():
            parameter.requires_grad = False
        self.canonical_aa_only = bool(canonical_aa_only)
        self.input_mode = str(input_mode).lower()
        self._bridge_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _disable_verifier_token_dropout(self) -> None:
        """Disable ESM token-dropout for verifier soft-embedding calls.

        The verifier consumes `inputs_embeds` (expected embeddings from generator logits),
        not discrete `input_ids`. ESM token-dropout logic assumes `input_ids` are present
        and can crash on embed-only calls. Turning it off is the correct behavior here.
        """
        esm_model = self.psalm.esm_model
        if hasattr(esm_model, "config") and hasattr(esm_model.config, "token_dropout"):
            esm_model.config.token_dropout = False
        embeddings = getattr(esm_model, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "token_dropout"):
            embeddings.token_dropout = False

    def train(self, mode: bool = True):
        """Keep the verifier in eval mode even when the parent model trains."""
        super().train(False)
        self.psalm.eval()
        return self

    @property
    def num_labels(self) -> int:
        return int(self.psalm.classes)

    @property
    def tokenizer(self):
        return self.psalm.tokenizer

    def _verifier_embedding_weight(self) -> torch.Tensor:
        if hasattr(self.psalm.esm_model, "get_input_embeddings"):
            return self.psalm.esm_model.get_input_embeddings().weight
        return self.psalm.esm_model.embeddings.word_embeddings.weight

    def _build_bridge(self, generator_tokenizer, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        gen_vocab = int(len(generator_tokenizer))
        verifier_vocab = int(len(self.tokenizer))
        cache_key = (gen_vocab, verifier_vocab, str(device))
        cached = self._bridge_cache.get(cache_key)
        if cached is not None:
            return cached

        bridge = torch.zeros(gen_vocab, verifier_vocab, device=device)
        hard_map = torch.full((gen_vocab,), -1, dtype=torch.long, device=device)

        token_set = set(CANONICAL_AAS) if self.canonical_aa_only else set(generator_tokenizer.get_vocab())
        special_tokens = [
            generator_tokenizer.cls_token,
            generator_tokenizer.eos_token,
            generator_tokenizer.pad_token,
            generator_tokenizer.mask_token,
        ]
        token_set.update(tok for tok in special_tokens if tok is not None)

        for token in sorted(token_set):
            gen_id = generator_tokenizer.convert_tokens_to_ids(token)
            ver_id = self.tokenizer.convert_tokens_to_ids(token)
            if gen_id is None or ver_id is None or gen_id < 0 or ver_id < 0:
                continue
            if gen_id >= gen_vocab or ver_id >= verifier_vocab:
                continue
            if token in CANONICAL_AAS:
                bridge[gen_id, ver_id] = 1.0
            hard_map[gen_id] = ver_id

        if bridge.sum() == 0:
            raise ValueError("No generator-to-verifier amino-acid token overlap found.")
        self._bridge_cache[cache_key] = (bridge, hard_map)
        return bridge, hard_map

    def _head(self, reps: torch.Tensor) -> torch.Tensor:
        x = self.psalm.fc1(reps)
        x = self.psalm.relu(x)
        x = self.psalm.ln1(x)
        x = self.psalm.fc2(x)
        x = self.psalm.relu(x)
        x = self.psalm.ln2(x)
        return self.psalm.fc3(x)

    def forward(
        self,
        generator_logits: torch.Tensor,
        generator_tokenizer,
        attention_mask: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        hard_input_ids: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        device = generator_logits.device
        bridge, hard_map = self._build_bridge(generator_tokenizer, device)
        temp = max(float(temperature), 1e-6)
        probs = torch.softmax(generator_logits / temp, dim=-1)

        if self.canonical_aa_only:
            aa_support = bridge.sum(dim=-1) > 0
            probs = probs * aa_support.to(dtype=probs.dtype).view(1, 1, -1)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if self.input_mode in {"st_top1", "straight_through_top1"}:
            # Straight-through top-1:
            # - forward pass uses a hard argmax one-hot distribution (PSALM-like input),
            # - backward pass uses the soft distribution gradients for stable learning.
            top1 = probs.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(probs).scatter_(-1, top1, 1.0)
            probs = hard + probs - probs.detach()
        elif self.input_mode in {"soft", "softmax"}:
            pass
        else:
            raise ValueError(
                f"Unknown verifier input_mode {self.input_mode!r}. Use 'st_top1' or 'soft'."
            )

        verifier_probs = torch.matmul(probs, bridge.to(dtype=probs.dtype))
        embed_weight = self._verifier_embedding_weight().to(device=device, dtype=verifier_probs.dtype)
        expected_embeds = torch.matmul(verifier_probs, embed_weight)

        if residue_mask is not None and hard_input_ids is not None:
            mapped = hard_map[hard_input_ids.clamp_min(0)]
            safe_mapped = mapped.clamp_min(0)
            hard_embeds = embed_weight.index_select(0, safe_mapped.reshape(-1)).view(
                *hard_input_ids.shape, embed_weight.size(-1)
            )
            use_hard = (~residue_mask.bool()).unsqueeze(-1) | (mapped < 0).unsqueeze(-1)
            expected_embeds = torch.where(use_hard, hard_embeds, expected_embeds)

        out = self.psalm.esm_model(
            input_ids=None,
            inputs_embeds=expected_embeds,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            return_dict=True,
        )
        logits = self._head(out.last_hidden_state)
        return {"logits": logits}


def build_verifier_from_config(config: dict[str, Any]) -> FrozenPSALMVerifier:
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    return FrozenPSALMVerifier(
        model_name=model_cfg.get("verifier_model_name", "ProteinSequenceAnnotation/PSALM-2"),
        device=model_cfg.get("verifier_device", "cpu"),
        use_fa=bool(model_cfg.get("use_fa", True)),
        warmup=False,
        canonical_aa_only=bool(model_cfg.get("canonical_aa_only_for_verifier", True)),
        input_mode=str(loss_cfg.get("verifier_input_mode", "st_top1")),
        psalm_repo_path=model_cfg.get("psalm_repo_path"),
    )
