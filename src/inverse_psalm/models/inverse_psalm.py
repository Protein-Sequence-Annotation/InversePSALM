from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from inverse_psalm.config import annotation_loss_enabled, deep_get, validate_annotation_settings
from inverse_psalm.models.generator import InversePSALM
from inverse_psalm.models.psalm_verifier import FrozenPSALMVerifier, build_verifier_from_config
from inverse_psalm.training.losses import masked_cross_entropy, token_accuracy


class InversePSALMTrainingModule(nn.Module):
    """Training module combining the annotated generator and frozen verifier."""

    def __init__(
        self,
        generator: InversePSALM,
        verifier: FrozenPSALMVerifier | None,
        config: dict[str, Any],
    ):
        super().__init__()
        self.generator = generator
        self.verifier = verifier
        self.config = config
        self.ignore_label = int(deep_get(config, "data.ignore_label", -100))
        self.annotation_loss_enabled = annotation_loss_enabled(config)

    @property
    def tokenizer(self):
        return self.generator.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        annotation_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        mlm_labels: torch.Tensor | None = None,
        mask_positions: torch.Tensor | None = None,
        annotation_loss_mask: torch.Tensor | None = None,
        conditioning_annotation_ids: torch.Tensor | None = None,
        compute_verifier: bool = True,
        mask_rate_target: float | torch.Tensor | None = None,
        verifier_warmup: float | torch.Tensor = 1.0,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        generator_annotation_ids = annotation_ids if conditioning_annotation_ids is None else conditioning_annotation_ids
        gen = self.generator(
            input_ids=input_ids,
            annotation_ids=generator_annotation_ids,
            attention_mask=attention_mask,
        )
        generator_logits = gen["logits"]

        losses = {}
        if mlm_labels is not None:
            mlm_loss = masked_cross_entropy(generator_logits, mlm_labels, ignore_index=self.ignore_label)
            losses["mlm_loss"] = mlm_loss
            losses["mlm_accuracy"] = token_accuracy(generator_logits, mlm_labels, ignore_index=self.ignore_label)
        else:
            mlm_loss = torch.zeros((), device=generator_logits.device, dtype=generator_logits.dtype)

        run_verifier = bool(compute_verifier and self.annotation_loss_enabled)
        if run_verifier:
            if self.verifier is None:
                raise ValueError("Verifier was not built, but annotation loss was requested.")
            verifier_temperature = float(deep_get(self.config, "loss.verifier_temperature", 1.0))
            verifier_out = self.verifier(
                generator_logits=generator_logits,
                generator_tokenizer=self.generator.tokenizer,
                attention_mask=attention_mask,
                residue_mask=residue_mask,
                hard_input_ids=input_ids,
                temperature=verifier_temperature,
            )
            verifier_logits = verifier_out["logits"]
            annotation_targets = annotation_ids
            if annotation_loss_mask is not None:
                annotation_targets = annotation_targets.masked_fill(~annotation_loss_mask.bool(), self.ignore_label)
            verifier_loss = masked_cross_entropy(
                verifier_logits,
                annotation_targets,
                ignore_index=self.ignore_label,
            )
            losses["verifier_loss"] = verifier_loss
            losses["verifier_accuracy"] = token_accuracy(
                verifier_logits,
                annotation_targets,
                ignore_index=self.ignore_label,
            )
            losses["verifier_logits"] = verifier_logits
        else:
            verifier_loss = torch.zeros((), device=generator_logits.device, dtype=generator_logits.dtype)
            if self.annotation_loss_enabled:
                losses["verifier_loss"] = verifier_loss
                losses["verifier_accuracy"] = verifier_loss

        mlm_weight = float(deep_get(self.config, "loss.mlm_weight", 1.0))
        if self.annotation_loss_enabled:
            verifier_weight = deep_get(self.config, "loss.verifier_weight.value", 0.1)
            verifier_scale = self._verifier_mask_scale(mask_rate_target, generator_logits.device, generator_logits.dtype)
            verifier_warmup_tensor = torch.as_tensor(
                verifier_warmup,
                device=generator_logits.device,
                dtype=generator_logits.dtype,
            )
            effective_verifier_weight = float(verifier_weight) * verifier_warmup_tensor * verifier_scale
            losses["verifier_mask_scale"] = verifier_scale
            losses["verifier_warmup"] = verifier_warmup_tensor
            losses["effective_verifier_weight"] = effective_verifier_weight
        else:
            effective_verifier_weight = torch.zeros((), device=generator_logits.device, dtype=generator_logits.dtype)
        gate_l2_weight = float(deep_get(self.config, "model.annotation_gating.gate_l2", 0.0))
        gate_l2_raw = self.generator.annotation_gate_l2_loss().to(
            device=generator_logits.device,
            dtype=generator_logits.dtype,
        )
        gate_l2_loss = gate_l2_weight * gate_l2_raw
        total_loss = mlm_weight * mlm_loss + effective_verifier_weight * verifier_loss + gate_l2_loss
        losses["loss"] = total_loss
        losses["gate_l2_loss"] = gate_l2_loss
        losses["annotation_gate_l2"] = gate_l2_raw
        losses["logits"] = generator_logits
        return losses

    def _verifier_mask_scale(
        self,
        mask_rate_target: float | torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cfg = deep_get(self.config, "loss.verifier_mask_scale", {})
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)) or mask_rate_target is None:
            return torch.ones((), device=device, dtype=dtype)
        full_strength_until = float(cfg.get("full_strength_until", 0.30))
        min_scale = float(cfg.get("min_scale", 0.10))
        if full_strength_until >= 1.0:
            raise ValueError("loss.verifier_mask_scale.full_strength_until must be < 1.0.")
        if min_scale < 0.0 or min_scale > 1.0:
            raise ValueError("loss.verifier_mask_scale.min_scale must be within [0, 1].")
        t = torch.as_tensor(mask_rate_target, device=device, dtype=dtype)
        scale = (1.0 - t) / max(1e-12, 1.0 - full_strength_until)
        return scale.clamp(min=min_scale, max=1.0)


def build_inverse_psalm(config: dict[str, Any]) -> tuple[InversePSALMTrainingModule, Any]:
    validate_annotation_settings(config)
    use_annotation_loss = annotation_loss_enabled(config)
    verifier = build_verifier_from_config(config) if use_annotation_loss else None
    num_labels = deep_get(config, "model.annotation_vocab_size", None)
    if num_labels is None:
        if verifier is None:
            raise ValueError("model.annotation_vocab_size must be set when model.annotation_loss=false.")
        num_labels = verifier.num_labels
    generator = InversePSALM(
        model_name=deep_get(config, "model.generator_model_name", "facebook/esm2_t33_650M_UR50D"),
        num_annotation_labels=int(num_labels),
        annotation_dropout=float(deep_get(config, "model.annotation_dropout", 0.0)),
        ignore_label=int(deep_get(config, "data.ignore_label", -100)),
        inverse_psalm_config=config,
        freeze_generator_trunk=bool(deep_get(config, "model.freeze_generator_trunk", True)),
        freeze_lm_head=bool(deep_get(config, "model.freeze_lm_head", True)),
        gradient_checkpointing=bool(deep_get(config, "model.gradient_checkpointing", False)),
        use_fa=bool(deep_get(config, "model.use_fa", True)),
    )
    model = InversePSALMTrainingModule(generator=generator, verifier=verifier, config=config)
    return model, generator.tokenizer
