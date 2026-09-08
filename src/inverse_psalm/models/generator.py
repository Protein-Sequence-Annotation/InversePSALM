from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_model as save_safetensors_model
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import cached_file

from inverse_psalm.config import annotation_track_enabled
from inverse_psalm.models.annotation_head import build_annotation_encoder

GENERATOR_CONFIG_NAME = "inverse_psalm_generator_config.json"
GENERATOR_WEIGHTS_NAME = "model.safetensors"


def _require_faesm():
    original_get_device_capability = None
    if not torch.cuda.is_available():
        # faesm checks CUDA capability during import. Allow CPU-only inference
        # with use_fa=False on login nodes or other machines without drivers.
        original_get_device_capability = torch.cuda.get_device_capability
        torch.cuda.get_device_capability = lambda *args, **kwargs: (0, 0)
    try:
        from faesm.esm import FAEsmForMaskedLM
    except ImportError as exc:
        raise ImportError(
            "FAESM is required for InversePSALM v1. Activate the PSALM-github "
            "environment that provides faesm before running training."
        ) from exc
    finally:
        if original_get_device_capability is not None:
            torch.cuda.get_device_capability = original_get_device_capability
    return FAEsmForMaskedLM


def parse_annotation_injection_blocks(spec: Any, num_blocks: int) -> set[int]:
    """Parse 1-indexed annotation injection block spec.

    Accepts formats like:
    - 1
    - "1-33"
    - "1,17,26-33"
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}.")

    def _validate_block(value: int) -> int:
        if value < 1 or value > num_blocks:
            raise ValueError(f"Block index {value} out of range [1, {num_blocks}].")
        return value

    if isinstance(spec, int):
        return {_validate_block(int(spec))}
    if isinstance(spec, str):
        parsed: set[int] = set()
        for chunk in [part.strip() for part in spec.split(",") if part.strip()]:
            if "-" in chunk:
                lo_txt, hi_txt = chunk.split("-", 1)
                lo = _validate_block(int(lo_txt.strip()))
                hi = _validate_block(int(hi_txt.strip()))
                if hi < lo:
                    raise ValueError(f"Invalid block range {chunk!r}: upper bound < lower bound.")
                parsed.update(range(lo, hi + 1))
            else:
                parsed.add(_validate_block(int(chunk)))
        if not parsed:
            raise ValueError("annotation_injection_blocks parsed to empty set.")
        return parsed
    raise ValueError(
        "annotation_injection_blocks must be an int or str, "
        f"got {type(spec).__name__}."
    )


def _logit_probability(value: float) -> float:
    if value <= 0.0:
        return -20.0
    if value >= 1.0:
        return 20.0
    return torch.logit(torch.tensor(float(value), dtype=torch.float32)).item()


def _portable_model_config(config: dict[str, Any] | None) -> dict[str, Any]:
    model_cfg = dict((config or {}).get("model", {}))
    out: dict[str, Any] = {}
    for key in (
        "max_position_embeddings",
        "generator_logits",
        "annotation_track",
        "annotation_encoder",
        "annotation_gating",
        "annotation_injection_blocks",
    ):
        if key in model_cfg:
            out[key] = model_cfg[key]
    return out


class InversePSALM(nn.Module):
    """Loadable InversePSALM generator with additive annotation conditioning."""

    def __init__(
        self,
        model_name: str,
        num_annotation_labels: int,
        annotation_dropout: float = 0.0,
        ignore_label: int = -100,
        inverse_psalm_config: dict[str, Any] | None = None,
        freeze_generator_trunk: bool = True,
        freeze_lm_head: bool = True,
        gradient_checkpointing: bool = False,
        use_fa: bool = True,
        *,
        base_config_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        load_pretrained_base: bool = True,
    ):
        super().__init__()
        FAEsmForMaskedLM = _require_faesm()
        self.model_name = model_name
        self.num_annotation_labels = int(num_annotation_labels)
        self.annotation_dropout = float(annotation_dropout)
        self.generator_model_config = _portable_model_config(inverse_psalm_config)
        self.annotation_track_enabled = annotation_track_enabled({"model": self.generator_model_config})
        self.use_fa = bool(use_fa)
        self.ignore_label = int(ignore_label)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or model_name))

        self.fae = self._build_fae(
            FAEsmForMaskedLM,
            model_name=model_name,
            use_fa=self.use_fa,
            base_config_path=base_config_path,
            load_pretrained_base=load_pretrained_base,
        )
        self._drop_unused_contact_head()
        self._freeze_unused_position_embeddings()

        hf_cfg = self.fae.config
        hf_cfg.is_decoder = False
        hf_cfg.add_cross_attention = False
        model_cfg = self.generator_model_config
        mpe = model_cfg.get("max_position_embeddings")
        if mpe is not None:
            hf_cfg.max_position_embeddings = int(mpe)
        logit_cfg = model_cfg.get("generator_logits", {})
        soft_cap = logit_cfg.get("soft_cap", None) if isinstance(logit_cfg, dict) else None
        self.generator_logit_soft_cap = None if soft_cap is None else float(soft_cap)
        if self.generator_logit_soft_cap is not None and self.generator_logit_soft_cap <= 0.0:
            raise ValueError("model.generator_logits.soft_cap must be positive when set.")

        self._gradient_checkpointing_enabled = bool(gradient_checkpointing)
        if self._gradient_checkpointing_enabled and hasattr(self.fae, "gradient_checkpointing_enable"):
            # FAESM encoder checkpoint path passes layer kwargs (hidden_states, attention_mask, ...).
            # On modern torch, reentrant checkpointing rejects kwargs and crashes.
            # Non-reentrant mode supports kwargs and is the recommended long-term path.
            try:
                self.fae.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.fae.gradient_checkpointing_enable()

        hidden_size = int(hf_cfg.hidden_size)
        self.annotation = build_annotation_encoder(
            config={"model": model_cfg},
            num_labels=int(num_annotation_labels),
            hidden_size=hidden_size,
            dropout=float(annotation_dropout),
            ignore_label=int(ignore_label),
        )
        gating_cfg = model_cfg.get("annotation_gating", {})
        self.annotation_gating_enabled = bool(gating_cfg.get("enabled", True))
        self.annotation_gate_min = float(gating_cfg.get("min_gate", 0.0))
        self.annotation_gate_max = float(gating_cfg.get("max_gate", 1.0))
        init_gate = float(gating_cfg.get("init_gate", 0.005))
        if self.annotation_gate_min < 0.0 or self.annotation_gate_max > 1.0:
            raise ValueError("Annotation gate bounds must stay within [0, 1].")
        if self.annotation_gate_max < self.annotation_gate_min:
            raise ValueError("annotation_gating.max_gate must be >= min_gate.")
        if init_gate < self.annotation_gate_min or init_gate > self.annotation_gate_max:
            raise ValueError("annotation_gating.init_gate must lie within [min_gate, max_gate].")
        normalize_delta = bool(gating_cfg.get("normalize_delta", True))
        self.annotation_delta_norm = (
            nn.LayerNorm(hidden_size, eps=float(getattr(hf_cfg, "layer_norm_eps", 1e-5)))
            if normalize_delta
            else nn.Identity()
        )
        encoder_layers = getattr(getattr(self.base_model, "encoder", None), "layer", None)
        if encoder_layers is None:
            raise AttributeError("Could not find base_model.encoder.layer for annotation block injection.")
        self._num_blocks = int(len(encoder_layers))
        block_spec = model_cfg.get("annotation_injection_blocks", f"1-{self._num_blocks}")
        self.annotation_injection_blocks = (
            parse_annotation_injection_blocks(block_spec, self._num_blocks)
            if self.annotation_gating_enabled
            else set()
        )
        gate_span = self.annotation_gate_max - self.annotation_gate_min
        normalized_init = 0.0 if gate_span == 0.0 else (init_gate - self.annotation_gate_min) / gate_span
        raw_init = _logit_probability(normalized_init)
        self.annotation_gate_logits = nn.ParameterDict(
            {
                self._gate_key(block_idx): nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
                for block_idx in sorted(self.annotation_injection_blocks)
            }
        )

        self._active_annotation_embeds_padded: torch.Tensor | None = None
        self._active_annotation_embeds_packed: torch.Tensor | None = None
        self._active_annotation_valid_padded: torch.Tensor | None = None
        self._active_annotation_valid_packed: torch.Tensor | None = None
        self._last_effective_injection_norm_ratios: dict[int, float] = {}
        self._hook_handles: list[Any] = []
        for block_idx in sorted(self.annotation_injection_blocks):
            block_module = encoder_layers[block_idx - 1]
            self._hook_handles.append(
                block_module.register_forward_pre_hook(
                    self._make_block_injection_pre_hook(block_idx),
                    with_kwargs=True,
                )
            )
        self.set_freeze_state(
            freeze_generator_trunk=freeze_generator_trunk,
            freeze_lm_head=freeze_lm_head,
        )
        if not self.annotation_track_enabled:
            self._freeze_annotation_track_parameters()

    @staticmethod
    def _build_fae(
        FAEsmForMaskedLM,
        *,
        model_name: str,
        use_fa: bool,
        base_config_path: str | Path | None,
        load_pretrained_base: bool,
    ):
        if load_pretrained_base:
            # FAESM.from_pretrained internally does cls(config, **kwargs); passing config= here duplicates it.
            return FAEsmForMaskedLM.from_pretrained(model_name, use_fa=use_fa)
        config = AutoConfig.from_pretrained(str(base_config_path or model_name))
        try:
            return FAEsmForMaskedLM(config, use_fa=use_fa)
        except TypeError:
            return FAEsmForMaskedLM(config)

    @property
    def vocab_size(self) -> int:
        return int(self.fae.config.vocab_size)

    @property
    def base_model(self):
        if hasattr(self.fae, "esm"):
            return self.fae.esm
        if hasattr(self.fae, "esm_model"):
            return self.fae.esm_model
        raise AttributeError("Could not find FAESM base model attribute (`esm` or `esm_model`).")

    @property
    def lm_head(self):
        if hasattr(self.fae, "lm_head"):
            return self.fae.lm_head
        if hasattr(self.fae, "cls"):
            return self.fae.cls
        raise AttributeError("Could not find FAESM LM head attribute (`lm_head` or `cls`).")

    def _drop_unused_contact_head(self) -> None:
        base_model = self.base_model
        if hasattr(base_model, "contact_head"):
            base_model.contact_head = None
        if hasattr(self.fae, "contact_head"):
            self.fae.contact_head = None

    def _freeze_unused_position_embeddings(self) -> None:
        embeddings = getattr(self.base_model, "embeddings", None)
        position_embeddings = getattr(embeddings, "position_embeddings", None)
        if position_embeddings is None:
            return
        position_type = str(
            getattr(
                embeddings,
                "position_embedding_type",
                getattr(self.fae.config, "position_embedding_type", "absolute"),
            )
        ).lower()
        if position_type == "absolute":
            return
        for parameter in position_embeddings.parameters():
            parameter.requires_grad = False

    def _apply_logit_soft_cap(self, logits: torch.Tensor) -> torch.Tensor:
        if self.generator_logit_soft_cap is None:
            return logits
        cap = torch.as_tensor(self.generator_logit_soft_cap, device=logits.device, dtype=logits.dtype)
        return cap * torch.tanh(logits / cap)

    def set_freeze_state(self, *, freeze_generator_trunk: bool, freeze_lm_head: bool) -> None:
        self.freeze_generator_trunk = bool(freeze_generator_trunk)
        self.freeze_lm_head = bool(freeze_lm_head)
        for parameter in self.base_model.parameters():
            parameter.requires_grad = not self.freeze_generator_trunk
        self._freeze_unused_position_embeddings()
        for parameter in self.lm_head.parameters():
            parameter.requires_grad = not self.freeze_lm_head

    def _freeze_annotation_track_parameters(self) -> None:
        for module in (self.annotation, self.annotation_delta_norm):
            for parameter in module.parameters():
                parameter.requires_grad = False
        for parameter in self.annotation_gate_logits.parameters():
            parameter.requires_grad = False

    def _artifact_config(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "base_model_name": str(self.model_name),
            "num_annotation_labels": int(self.num_annotation_labels),
            "annotation_dropout": float(self.annotation_dropout),
            "ignore_label": int(self.ignore_label),
            "use_fa": bool(self.use_fa),
            "model": self.generator_model_config,
        }

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.fae.config, "save_pretrained"):
            self.fae.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # save_model records tied tensor metadata, which plain save_file cannot represent safely.
        save_safetensors_model(self, str(save_path / GENERATOR_WEIGHTS_NAME), metadata={"format": "pt"})
        artifact_config = self._artifact_config()
        with (save_path / GENERATOR_CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(artifact_config, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _resolve_pretrained_file(pretrained_model_name_or_path: str | Path, filename: str) -> Path:
        local_path = Path(pretrained_model_name_or_path) / filename
        if local_path.exists():
            return local_path
        resolved = cached_file(str(pretrained_model_name_or_path), filename)
        if resolved is None:
            raise FileNotFoundError(f"Could not find {filename!r} in {pretrained_model_name_or_path!r}.")
        return Path(resolved)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        use_fa: bool | None = None,
        gradient_checkpointing: bool = False,
        freeze_generator_trunk: bool | None = None,
        freeze_lm_head: bool | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "InversePSALM":
        config_path = cls._resolve_pretrained_file(pretrained_model_name_or_path, GENERATOR_CONFIG_NAME)
        with config_path.open("r", encoding="utf-8") as handle:
            artifact_config = json.load(handle)

        model_name = str(artifact_config["base_model_name"])
        saved_use_fa = bool(artifact_config.get("use_fa", True))
        generator = cls(
            model_name=model_name,
            num_annotation_labels=int(artifact_config["num_annotation_labels"]),
            annotation_dropout=float(artifact_config.get("annotation_dropout", 0.0)),
            ignore_label=int(artifact_config.get("ignore_label", -100)),
            inverse_psalm_config={"model": artifact_config.get("model", {})},
            freeze_generator_trunk=True if freeze_generator_trunk is None else bool(freeze_generator_trunk),
            freeze_lm_head=True if freeze_lm_head is None else bool(freeze_lm_head),
            gradient_checkpointing=bool(gradient_checkpointing),
            use_fa=saved_use_fa if use_fa is None else bool(use_fa),
            base_config_path=pretrained_model_name_or_path,
            tokenizer_path=pretrained_model_name_or_path,
            load_pretrained_base=False,
        )
        weights_path = cls._resolve_pretrained_file(pretrained_model_name_or_path, GENERATOR_WEIGHTS_NAME)
        cls._load_pretrained_weights(generator, weights_path, strict=strict, map_location=map_location)
        return generator

    @staticmethod
    def _load_pretrained_weights(
        generator: "InversePSALM",
        weights_path: Path,
        *,
        strict: bool,
        map_location: str | torch.device,
    ) -> None:
        state_dict = load_safetensors_file(str(weights_path), device=str(map_location))
        ignored_keys = InversePSALM._drop_legacy_rotary_position_embedding_mismatch(generator, state_dict)
        missing, unexpected = generator.load_state_dict(state_dict, strict=False)
        ignored_keys.update(InversePSALM._expected_missing_tied_weight_keys(generator, missing))
        missing = [key for key in missing if key not in ignored_keys]
        unexpected = [key for key in unexpected if key not in ignored_keys]
        if strict and (missing or unexpected):
            problems: list[str] = []
            if missing:
                problems.append(f"missing keys: {missing}")
            if unexpected:
                problems.append(f"unexpected keys: {unexpected}")
            raise RuntimeError("Error(s) in loading state_dict for InversePSALM:\n" + "\n".join(problems))


    @staticmethod
    def _drop_legacy_rotary_position_embedding_mismatch(
        generator: "InversePSALM",
        state_dict: dict[str, torch.Tensor],
    ) -> set[str]:
        """Ignore legacy rotary checkpoint position-embedding shape mismatches.

        FAESM/ESM-2 rotary models may still carry an absolute position embedding
        module for compatibility, but it is not used by the rotary forward path.
        Some older artifacts saved the original unused 1026-row table while the
        config records a larger max position count. Keep the runtime model exactly
        as the config builds it and drop only this unused mismatched tensor.
        """
        embeddings = getattr(generator.base_model, "embeddings", None)
        position_embeddings = getattr(embeddings, "position_embeddings", None)
        if position_embeddings is None:
            return set()
        position_type = str(
            getattr(
                embeddings,
                "position_embedding_type",
                getattr(generator.fae.config, "position_embedding_type", "absolute"),
            )
        ).lower()
        if position_type == "absolute":
            return set()
        key = "fae.esm.embeddings.position_embeddings.weight"
        saved = state_dict.get(key)
        if saved is None or tuple(position_embeddings.weight.shape) == tuple(saved.shape):
            return set()
        if int(saved.shape[1]) != int(position_embeddings.weight.shape[1]):
            raise ValueError(
                "Cannot ignore rotary position embedding mismatch with different hidden size: "
                f"checkpoint={tuple(saved.shape)} model={tuple(position_embeddings.weight.shape)}."
            )
        del state_dict[key]
        return {key}

    @staticmethod
    def _expected_missing_tied_weight_keys(generator: "InversePSALM", missing: list[str]) -> set[str]:
        ignored: set[str] = set()
        key = "fae.lm_head.decoder.weight"
        if key not in missing:
            return ignored
        decoder = getattr(generator.lm_head, "decoder", None)
        word_embeddings = getattr(getattr(generator.base_model, "embeddings", None), "word_embeddings", None)
        if decoder is None or word_embeddings is None:
            return ignored
        decoder_weight = getattr(decoder, "weight", None)
        embedding_weight = getattr(word_embeddings, "weight", None)
        if decoder_weight is None or embedding_weight is None:
            return ignored
        if decoder_weight.data_ptr() == embedding_weight.data_ptr():
            ignored.add(key)
        return ignored

    @staticmethod
    def _gate_key(block_idx: int) -> str:
        return f"block_{int(block_idx):02d}"

    def _gate_for_block(self, block_idx: int) -> torch.Tensor:
        raw_gate = self.annotation_gate_logits[self._gate_key(block_idx)]
        return self.annotation_gate_min + (self.annotation_gate_max - self.annotation_gate_min) * torch.sigmoid(raw_gate)

    def annotation_gate_values(self) -> dict[str, float]:
        if not getattr(self, "annotation_track_enabled", True):
            return {}
        return {
            self._gate_key(block_idx): float(self._gate_for_block(block_idx).detach().cpu().item())
            for block_idx in sorted(self.annotation_injection_blocks)
        }

    def annotation_gate_l2_loss(self) -> torch.Tensor:
        if not getattr(self, "annotation_track_enabled", True):
            device = next(self.parameters()).device
            return torch.zeros((), dtype=torch.float32, device=device)
        gates = [self._gate_for_block(block_idx) for block_idx in sorted(self.annotation_injection_blocks)]
        if not gates:
            device = next(self.parameters()).device
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack([gate.float().square() for gate in gates]).mean()

    def annotation_effective_injection_norm_ratios(self) -> dict[str, float]:
        if not getattr(self, "annotation_track_enabled", True):
            return {}
        return {
            self._gate_key(block_idx): ratio
            for block_idx, ratio in sorted(self._last_effective_injection_norm_ratios.items())
        }

    def _clear_active_annotation_cache(self) -> None:
        self._active_annotation_embeds_padded = None
        self._active_annotation_embeds_packed = None
        self._active_annotation_valid_padded = None
        self._active_annotation_valid_packed = None

    def _add_annotation_before_block(self, block_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "annotation_track_enabled", True):
            return hidden_states
        if self._active_annotation_embeds_padded is None:
            return hidden_states
        if hidden_states.dim() == 3:
            ann = self._active_annotation_embeds_padded.to(dtype=hidden_states.dtype, device=hidden_states.device)
            valid = self._active_annotation_valid_padded
        elif hidden_states.dim() == 2:
            if self._active_annotation_embeds_packed is None:
                return hidden_states
            ann = self._active_annotation_embeds_packed.to(dtype=hidden_states.dtype, device=hidden_states.device)
            valid = self._active_annotation_valid_packed
        else:
            return hidden_states

        delta = self.annotation_delta_norm(ann)
        if valid is not None:
            delta = delta * valid.to(device=hidden_states.device).unsqueeze(-1).to(dtype=delta.dtype)
        gate = self._gate_for_block(block_idx).to(dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            hidden_norm = hidden_states.detach().float().norm().clamp_min(1e-12)
            injected_delta_norm = (gate.detach().float() * delta.detach().float()).norm()
            self._last_effective_injection_norm_ratios[int(block_idx)] = float(
                (injected_delta_norm / hidden_norm).cpu().item()
            )
        return hidden_states + gate * delta

    def _make_block_injection_pre_hook(self, block_idx: int):
        def _hook(_module, args, kwargs):
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = self._add_annotation_before_block(block_idx, kwargs["hidden_states"])
                return args, kwargs
            if args:
                mutable = list(args)
                mutable[0] = self._add_annotation_before_block(block_idx, mutable[0])
                return tuple(mutable), kwargs
            return args, kwargs

        return _hook

    def forward(
        self,
        input_ids: torch.Tensor,
        annotation_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        self._clear_active_annotation_cache()
        keep_cache_for_checkpoint_backward = (
            self._gradient_checkpointing_enabled
            and self.training
            and torch.is_grad_enabled()
        )
        forward_completed = False
        self._last_effective_injection_norm_ratios = {}
        if self.annotation_track_enabled:
            attention_bool = (
                attention_mask.bool()
                if attention_mask is not None
                else torch.ones_like(annotation_ids, dtype=torch.bool)
            )
            ann = self.annotation(annotation_ids)
            annotation_valid = annotation_ids != self.ignore_label
            self._active_annotation_embeds_padded = ann
            self._active_annotation_embeds_packed = ann[attention_bool]
            self._active_annotation_valid_padded = annotation_valid
            self._active_annotation_valid_packed = annotation_valid[attention_bool]
        try:
            try:
                outputs = self.fae(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    **kwargs,
                )
            except TypeError:
                outputs = self.fae(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            logits = self._apply_logit_soft_cap(logits)
            hidden = getattr(outputs, "hidden_states", None) if not isinstance(outputs, dict) else outputs.get("hidden_states")
            forward_completed = True
            return {"logits": logits, "hidden_states": hidden}
        finally:
            # Checkpointed FAESM blocks are recomputed during backward, so their pre-hooks
            # still need the same annotation tensors after the original forward returns.
            if not (keep_cache_for_checkpoint_backward and forward_completed):
                self._clear_active_annotation_cache()

    def annotation_ids_like(self, input_ids: torch.Tensor, fill_value: int | None = None) -> torch.Tensor:
        """Return annotation IDs shaped like `input_ids`; default is no annotation injection."""
        fill = self.ignore_label if fill_value is None else int(fill_value)
        return torch.full_like(input_ids, fill)


def build_generator_from_config(config: dict[str, Any], num_annotation_labels: int) -> InversePSALM:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    return InversePSALM(
        model_name=model_cfg.get("generator_model_name", "facebook/esm2_t33_650M_UR50D"),
        num_annotation_labels=num_annotation_labels,
        annotation_dropout=float(model_cfg.get("annotation_dropout", 0.0)),
        ignore_label=int(data_cfg.get("ignore_label", -100)),
        inverse_psalm_config=config,
        freeze_generator_trunk=bool(model_cfg.get("freeze_generator_trunk", True)),
        freeze_lm_head=bool(model_cfg.get("freeze_lm_head", True)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        use_fa=bool(model_cfg.get("use_fa", True)),
    )
