from __future__ import annotations

import inspect
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from safetensors.torch import save_model as save_safetensors_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.trainer import SAFE_WEIGHTS_NAME, TRAINING_ARGS_NAME

from inverse_psalm.config import annotation_loss_enabled, annotation_track_enabled
from inverse_psalm.data.masking import apply_masking
from inverse_psalm.data.streaming import ShardBatchDataset, group_by_batch


def make_data_collator(tokenizer):
    tensor_keys = {
        "input_ids",
        "attention_mask",
        "labels",
        "annotation_ids",
        "residue_mask",
        "domain_mask",
        "example_type_id",
        "sequence_length",
        "batch_id",
    }

    def collate(batch_list):
        if len(batch_list) == 1 and isinstance(batch_list[0], list):
            examples = batch_list[0]
        else:
            examples = batch_list
        batch: dict[str, Any] = {}
        for key in tensor_keys:
            if key in examples[0]:
                batch[key] = torch.tensor([ex[key] for ex in examples], dtype=torch.long)
        batch["ids"] = [ex.get("id", "") for ex in examples]
        batch["example_types"] = [ex.get("example_type", "original") for ex in examples]
        return batch

    return collate


class InversePSALMTrainer(Trainer):
    def __init__(self, *, config: dict[str, Any], tokenizer, **kwargs):
        # HF Trainer deprecates `tokenizer` in favor of `processing_class`.
        kwargs.setdefault("processing_class", tokenizer)
        super().__init__(**kwargs)
        self.config = config
        # InversePSALM returns mean per-microbatch losses. Transformers' newer
        # num_items_in_batch path is for summed token losses over an accumulated
        # batch; enabling it would make gradients scale with grad accumulation.
        self.model_accepts_loss_kwargs = False
        self.true_global_step = 0
        self.optimizer_steps_per_epoch_estimate = self._read_optimizer_steps_per_epoch_estimate()

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        save_dir = Path(output_dir or self.args.output_dir)
        if self.args.save_safetensors:
            save_dir.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
            model = getattr(model, "module", model)
            save_safetensors_model(model, str(save_dir / SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
            if self.processing_class is not None:
                self.processing_class.save_pretrained(save_dir)
            torch.save(self.args, save_dir / TRAINING_ARGS_NAME)
        else:
            super()._save(output_dir=output_dir, state_dict=state_dict)
        if not self.is_world_process_zero():
            return
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        model = getattr(model, "module", model)
        generator = getattr(model, "generator", None)
        if generator is not None and hasattr(generator, "save_pretrained"):
            generator.save_pretrained(save_dir / "generator")

    def _read_optimizer_steps_per_epoch_estimate(self) -> int | None:
        train_cfg = self.config.get("training", {})
        value = train_cfg.get("optimizer_steps_per_epoch_estimate", None)
        if value is None:
            return None
        steps = int(value)
        if steps <= 0:
            return None
        return steps

    def _estimated_epoch(self) -> float | None:
        if not self.optimizer_steps_per_epoch_estimate:
            return None
        return float(self.state.global_step) / float(self.optimizer_steps_per_epoch_estimate)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs = dict(logs)
        est_epoch = self._estimated_epoch()
        if est_epoch is not None:
            logs["epoch"] = est_epoch
            logs["epoch_estimated"] = est_epoch
            logs["optimizer_steps_per_epoch_estimate"] = float(self.optimizer_steps_per_epoch_estimate)
        super().log(logs, start_time)

    @staticmethod
    def _strip_non_model_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in inputs.items()
            if k not in {"ids", "labels", "sequence_length", "batch_id", "example_types", "example_type_id", "domain_mask"}
        }

    @staticmethod
    def _zero_loss_with_grad(model) -> torch.Tensor:
        """Return a graph-connected zero scalar touching all trainable parameters.

        This keeps DDP gradient buckets aligned even on skipped steps; connecting to
        only one parameter can make the rest appear unused and trigger collectives
        mismatches/timeouts across ranks.
        """
        pieces: list[torch.Tensor] = []
        for parameter in model.parameters():
            if parameter.requires_grad:
                pieces.append(parameter.float().sum() * 0.0)
        if not pieces:
            return torch.zeros((), dtype=torch.float32, requires_grad=True)
        return torch.stack(pieces).sum()

    def _example_class_config(self, example_type: str) -> dict[str, Any]:
        defaults = {
            "original": {
                "mlm_loss": "all_residues",
                "annotation_loss": "domain_only",
                "conditioning": "domain_only_or_unconditional",
                "unconditional_prob": 0.25,
            },
            "shuffled": {
                "mlm_loss": "domain_only",
                "annotation_loss": "all_residues",
                "conditioning": "all_residues",
            },
            "negative": {
                "mlm_loss": "none",
                "annotation_loss": "all_residues",
                "conditioning": "all_residues",
            },
            "domain_slice": {
                "mlm_loss": "all_residues",
                "annotation_loss": "all_residues",
                "conditioning": "all_residues",
            },
        }
        configured = self.config.get("loss", {}).get("example_classes", {})
        out = dict(defaults.get(example_type, defaults["original"]))
        out.update(configured.get(example_type, {}))
        return out

    def _conditioning_mode_for_example(
        self,
        *,
        example_type: str,
        class_cfg: dict[str, Any],
        rng: random.Random,
        original_conditioning_override: str | None,
    ) -> str:
        if example_type == "original" and original_conditioning_override is not None:
            return str(original_conditioning_override).lower()
        mode = str(class_cfg.get("conditioning", "all_residues")).lower()
        if mode == "domain_only_or_unconditional":
            unconditional_prob = float(class_cfg.get("unconditional_prob", 0.25))
            if unconditional_prob < 0.0 or unconditional_prob > 1.0:
                raise ValueError("unconditional_prob must be within [0, 1].")
            return "unconditional" if rng.random() < unconditional_prob else "domain_only"
        return mode

    def _apply_example_class_loss_masks(
        self,
        inputs: dict[str, Any],
        *,
        conditioning_seed: int | None = None,
        original_conditioning_override: str | None = None,
    ) -> dict[str, Any]:
        out = dict(inputs)
        ignore_label = int(self.config.get("data", {}).get("ignore_label", -100))
        mlm_labels = out["mlm_labels"].clone()
        conditioning_annotation_ids = out["annotation_ids"].clone()
        annotation_loss_mask = torch.zeros_like(out["residue_mask"], dtype=torch.bool)
        example_types = out.get("example_types")
        if example_types is None:
            example_types = ["original"] * int(out["input_ids"].size(0))
        rng = random.Random(0 if conditioning_seed is None else int(conditioning_seed))
        use_annotation_loss = annotation_loss_enabled(self.config)
        use_annotation_track = annotation_track_enabled(self.config)

        for row, example_type in enumerate(example_types):
            class_cfg = self._example_class_config(str(example_type))
            mlm_mode = class_cfg.get("mlm_loss", "all_residues")
            if isinstance(mlm_mode, bool):
                mlm_mode = "all_residues" if mlm_mode else "none"
            mlm_mode = str(mlm_mode).lower()
            if mlm_mode in {"none", "off", "false"}:
                mlm_labels[row] = ignore_label
            elif mlm_mode == "domain_only":
                mlm_labels[row] = mlm_labels[row].masked_fill(~out["domain_mask"][row].bool(), ignore_label)
            elif mlm_mode in {"all_residues", "all", "true"}:
                pass
            else:
                raise ValueError(
                    f"Unknown mlm_loss mode {mlm_mode!r} for example class {example_type!r}."
                )
            if use_annotation_loss:
                ann_mode = str(class_cfg.get("annotation_loss", "domain_only")).lower()
                if ann_mode == "domain_only":
                    annotation_loss_mask[row] = out["domain_mask"][row].bool()
                elif ann_mode == "all_residues":
                    annotation_loss_mask[row] = out["residue_mask"][row].bool()
                elif ann_mode in {"none", "off", "false"}:
                    annotation_loss_mask[row] = False
                else:
                    raise ValueError(
                        f"Unknown annotation_loss mode {ann_mode!r} for example class {example_type!r}."
                    )
            if use_annotation_track:
                conditioning_mode = self._conditioning_mode_for_example(
                    example_type=str(example_type),
                    class_cfg=class_cfg,
                    rng=rng,
                    original_conditioning_override=original_conditioning_override,
                )
                if conditioning_mode == "domain_only":
                    conditioning_annotation_ids[row] = conditioning_annotation_ids[row].masked_fill(
                        ~out["domain_mask"][row].bool(),
                        ignore_label,
                    )
                elif conditioning_mode in {"unconditional", "none", "off", "false"}:
                    conditioning_annotation_ids[row] = ignore_label
                    if str(example_type) == "original":
                        annotation_loss_mask[row] = False
                elif conditioning_mode in {"all_residues", "all", "true"}:
                    pass
                else:
                    raise ValueError(
                        f"Unknown conditioning mode {conditioning_mode!r} for example class {example_type!r}."
                    )
            else:
                conditioning_annotation_ids[row] = ignore_label
        out["mlm_labels"] = mlm_labels
        out["conditioning_annotation_ids"] = conditioning_annotation_ids
        out["annotation_loss_mask"] = annotation_loss_mask
        return out

    def get_train_dataloader(self):
        if not isinstance(self.train_dataset, list):
            return super().get_train_dataloader()
        shard_iterable = ShardBatchDataset(
            shard_paths=self.train_dataset,
            seed=int(self.args.seed),
            rank=self.args.local_rank,
            world_size=self.args.world_size,
        )
        return DataLoader(
            shard_iterable,
            batch_size=1,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2)
            if self.args.dataloader_num_workers > 0
            else None,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.eval_dataset
        if isinstance(eval_dataset, list):
            examples = []
            for shard_path in eval_dataset:
                examples.extend(list(load_from_disk(shard_path)))
        else:
            examples = list(eval_dataset)
        batches = group_by_batch(examples)
        batches.sort(key=lambda batch: len(batch[-1]["input_ids"]))
        return DataLoader(
            batches,
            batch_size=1,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def _mask_inputs_with_config(
        self,
        inputs: dict[str, Any],
        masking_config: dict[str, Any],
        seed: int,
        progress: float,
        original_conditioning_override: str | None = None,
    ) -> dict[str, Any]:
        config = dict(self.config)
        config["masking"] = masking_config
        masked = apply_masking(
            input_ids=inputs["input_ids"],
            residue_mask=inputs["residue_mask"],
            mask_token_id=int(self.processing_class.mask_token_id),
            config=config,
            progress=progress,
            seed=seed,
        )
        out = dict(inputs)
        out["input_ids"] = masked["masked_input_ids"]
        out["mlm_labels"] = masked["mlm_labels"]
        out["mask_positions"] = masked["mask_positions"]
        out["_mask_stats"] = masked
        return self._apply_example_class_loss_masks(
            out,
            conditioning_seed=seed,
            original_conditioning_override=original_conditioning_override,
        )

    def _mask_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        progress = float(self.state.global_step) / float(max(1, self.args.max_steps))
        local_rank = 0 if self.args.local_rank is None or self.args.local_rank < 0 else int(self.args.local_rank)
        seed = int(self.args.seed) + int(self.state.global_step) * 1009 + local_rank
        return self._mask_inputs_with_config(
            inputs=inputs,
            masking_config=self.config.get("masking", {}),
            seed=seed,
            progress=progress,
        )

    def _verifier_warmup(self) -> float:
        verifier_cfg = self.config.get("loss", {}).get("verifier_weight", {})
        warmup_steps = int(verifier_cfg.get("warmup_steps", 0)) if isinstance(verifier_cfg, dict) else 0
        if warmup_steps <= 0:
            return 1.0
        return min(float(self.state.global_step) / float(warmup_steps), 1.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        masked_inputs = self._mask_inputs(inputs)
        stats = masked_inputs.pop("_mask_stats")
        masked_inputs["mask_rate_target"] = float(stats["mask_rate_target"])
        if annotation_loss_enabled(self.config):
            masked_inputs["verifier_warmup"] = self._verifier_warmup()
        else:
            masked_inputs["compute_verifier"] = False
        model_inputs = self._strip_non_model_inputs(masked_inputs)
        outputs = model(**model_inputs)
        loss = outputs["loss"]

        if not torch.isfinite(loss):
            mlm_loss = outputs.get("mlm_loss", torch.zeros((), device=loss.device))
            verifier_loss = outputs.get("verifier_loss", torch.zeros((), device=loss.device))
            local_rank = getattr(self.args, "local_rank", -1)
            local_rank = -1 if local_rank is None else int(local_rank)
            marker = (
                "[NONFINITE_LOSS_SKIP] "
                f"step={int(self.state.global_step)} "
                f"rank={local_rank} "
                f"loss={float(loss.detach().float().cpu().item()) if torch.isfinite(loss.detach()).item() else 'nonfinite'} "
                f"mlm_loss={float(mlm_loss.detach().float().cpu().item()) if torch.isfinite(mlm_loss.detach()).item() else 'nonfinite'} "
                f"verifier_loss={float(verifier_loss.detach().float().cpu().item()) if torch.isfinite(verifier_loss.detach()).item() else 'nonfinite'} "
                f"mask_rate_target={float(stats['mask_rate_target'])} "
                f"mask_rate_actual={float(stats['mask_rate_actual'])}"
            )
            print(marker)
            if local_rank in (-1, 0):
                self.log(
                    {
                        "train_nonfinite_loss_skip": 1.0,
                        "train_mask_rate_actual": float(stats["mask_rate_actual"]),
                        "train_mask_rate_target": float(stats["mask_rate_target"]),
                    }
                )
            safe_loss = self._zero_loss_with_grad(model)
            outputs["loss"] = safe_loss
            if return_outputs:
                return safe_loss, outputs
            return safe_loss

        if self.args.local_rank in (-1, 0) and self.state.global_step % max(1, self.args.logging_steps) == 0:
            logs = {
                "train_mlm_loss_component": outputs["mlm_loss"].detach().float().item(),
                "train_gate_l2_loss_component": outputs["gate_l2_loss"].detach().float().item(),
                "train_annotation_gate_l2": outputs["annotation_gate_l2"].detach().float().item(),
                "train_mlm_accuracy": outputs["mlm_accuracy"].detach().float().item(),
                "train_mask_rate_actual": float(stats["mask_rate_actual"]),
                "train_mask_rate_target": float(stats["mask_rate_target"]),
            }
            if annotation_loss_enabled(self.config):
                logs.update(
                    {
                        "train_verifier_loss_component": outputs["verifier_loss"].detach().float().item(),
                        "train_verifier_warmup": outputs["verifier_warmup"].detach().float().item(),
                        "train_verifier_mask_scale": outputs["verifier_mask_scale"].detach().float().item(),
                        "train_effective_verifier_weight": outputs["effective_verifier_weight"].detach().float().item(),
                        "train_verifier_accuracy": outputs["verifier_accuracy"].detach().float().item(),
                    }
                )
            generator = getattr(getattr(model, "module", model), "generator", None)
            if generator is not None:
                for name, value in generator.annotation_gate_values().items():
                    logs[f"train/annotation_gate_{name}"] = value
                for name, value in generator.annotation_effective_injection_norm_ratios().items():
                    logs[f"train/annotation_effective_delta_hidden_norm_ratio_{name}"] = value
            self.log(logs)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        train_cfg = self.config.get("training", {})
        lr_cfg = train_cfg.get("learning_rate", {})
        opt_cfg = train_cfg.get("optimizer", {})
        annotation_gate_lr = lr_cfg.get("annotation_gate", None)
        grouped = defaultdict(list)
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("generator.annotation_gate_logits") and annotation_gate_lr is not None:
                grouped["annotation_gate"].append(parameter)
            elif (
                name.startswith("generator.annotation")
                or name.startswith("generator.annotation_delta_norm")
                or name.startswith("generator.annotation_gate_logits")
            ):
                grouped["annotation"].append(parameter)
            elif ".lm_head" in name or ".cls" in name:
                grouped["lm_head"].append(parameter)
            elif name.startswith("generator."):
                grouped["generator"].append(parameter)
            elif name.startswith("verifier."):
                raise ValueError("Frozen verifier parameter unexpectedly requires grad: " + name)
            else:
                grouped["annotation"].append(parameter)

        beta1 = float(opt_cfg.get("beta_1", 0.9))
        beta2 = float(opt_cfg.get("beta_2", 0.95))
        eps = float(opt_cfg.get("epsilon", 1e-8))
        weight_decay_cfg = opt_cfg.get("weight_decay", 0.01)

        def weight_decay_for(group_name: str) -> float:
            if isinstance(weight_decay_cfg, dict):
                return float(weight_decay_cfg.get(group_name, weight_decay_cfg.get("default", 0.01)))
            return float(weight_decay_cfg)

        fused_flag = opt_cfg.get("fused", None)
        supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters

        adam_kwargs = {"betas": (beta1, beta2), "eps": eps}
        if supports_fused and fused_flag is not None:
            adam_kwargs["fused"] = bool(fused_flag)
        elif supports_fused and fused_flag is None:
            adam_kwargs["fused"] = True

        groups = []
        if grouped["annotation"]:
            groups.append({
                "params": grouped["annotation"],
                "lr": float(lr_cfg.get("annotation", 1e-3)),
                "weight_decay": weight_decay_for("annotation"),
            })
        if grouped["annotation_gate"]:
            groups.append({
                "params": grouped["annotation_gate"],
                "lr": float(annotation_gate_lr),
                "weight_decay": weight_decay_for("annotation_gate"),
            })
        if grouped["generator"]:
            groups.append({
                "params": grouped["generator"],
                "lr": float(lr_cfg.get("generator", 1e-5)),
                "weight_decay": weight_decay_for("generator"),
            })
        if grouped["lm_head"]:
            groups.append({
                "params": grouped["lm_head"],
                "lr": float(lr_cfg.get("lm_head", 1e-5)),
                "weight_decay": weight_decay_for("lm_head"),
            })
        if not groups:
            raise ValueError("No trainable parameters found.")
        self.optimizer = torch.optim.AdamW(groups, **adam_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        optimizer = optimizer or self.optimizer
        train_cfg = self.config.get("training", {})
        sched_cfg = train_cfg.get("lr_scheduler", {})
        max_steps = int(train_cfg.get("max_steps", self.args.max_steps))
        schedule_steps = int(train_cfg.get("schedule_steps", max_steps))
        warmup_steps = int(train_cfg.get("warmup_steps", self.args.warmup_steps))
        scheduler_type = str(sched_cfg.get("type", "cosine")).lower()
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.1 if scheduler_type == "cosine" else 0.0))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, schedule_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            if scheduler_type == "linear":
                return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
            if scheduler_type == "cosine":
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            if scheduler_type == "constant":
                return 1.0
            raise ValueError(f"Unknown lr_scheduler.type: {scheduler_type!r}")

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler

    @staticmethod
    def _metric_bucket() -> dict[str, float]:
        return {"loss_sum": 0.0, "correct": 0.0, "tokens": 0.0}

    @staticmethod
    def _family_ids_from_labels(labels: torch.Tensor) -> torch.Tensor:
        """Map PSALM fine labels to coarse family IDs (0 for None/ignore)."""
        safe = labels.clamp_min(0)
        families = torch.zeros_like(safe)
        positive = safe > 0
        families[positive] = (safe[positive] - 1) // 3 + 1
        return families

    @staticmethod
    def _accumulate_token_metrics(
        bucket: dict[str, float],
        *,
        loss_sum: float,
        correct: int,
        n_tokens: int,
    ) -> None:
        bucket["loss_sum"] += float(loss_sum)
        bucket["correct"] += float(correct)
        bucket["tokens"] += float(n_tokens)

    @staticmethod
    def _update_token_metrics(
        bucket: dict[str, float],
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        accuracy_mode: str = "exact",
    ) -> tuple[int, float, int]:
        valid = labels != -100
        n_tokens = int(valid.sum().item())
        if n_tokens == 0:
            return 0, 0.0, 0
        loss_sum_tensor = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        if not torch.isfinite(loss_sum_tensor):
            return 0, 0.0, 0
        preds = logits.argmax(dim=-1)
        if accuracy_mode == "family":
            pred_families = InversePSALMTrainer._family_ids_from_labels(preds)
            label_families = InversePSALMTrainer._family_ids_from_labels(labels)
            correct = int((pred_families[valid] == label_families[valid]).sum().item())
        elif accuracy_mode == "exact":
            correct = int((preds[valid] == labels[valid]).sum().item())
        else:
            raise ValueError(f"Unknown accuracy_mode: {accuracy_mode!r}")
        loss_sum = max(0.0, float(loss_sum_tensor.detach().cpu().item()))
        InversePSALMTrainer._accumulate_token_metrics(
            bucket,
            loss_sum=loss_sum,
            correct=correct,
            n_tokens=n_tokens,
        )
        return n_tokens, loss_sum, correct

    @staticmethod
    def _update_prefilter_stats(
        bucket: dict[str, float],
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        valid = labels != -100
        if not bool(valid.any()):
            return
        pred_labels = logits.argmax(dim=-1)
        pred_families = InversePSALMTrainer._family_ids_from_labels(pred_labels)[valid]
        gt_families = InversePSALMTrainer._family_ids_from_labels(labels)[valid]
        pred_set = set(int(x) for x in pred_families.unique().tolist() if int(x) > 0)
        gt_set = set(int(x) for x in gt_families.unique().tolist() if int(x) > 0)
        if not gt_set:
            return
        matched = len(gt_set.intersection(pred_set))
        bucket["matched"] += float(matched)
        bucket["total"] += float(len(gt_set))

    @staticmethod
    def _finalize_prefilter_metrics(prefix: str, buckets: dict[str, dict[str, float]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, bucket in buckets.items():
            total = float(bucket.get("total", 0.0))
            if total <= 0:
                continue
            metrics[f"{prefix}_{name}_prefilter_accuracy"] = float(bucket.get("matched", 0.0)) / total
        return metrics

    @staticmethod
    def _finalize_token_metrics(
        prefix: str,
        buckets: dict[str, dict[str, float]],
        *,
        include_accuracy: bool,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, bucket in buckets.items():
            if bucket["tokens"] <= 0:
                continue
            tokens = bucket["tokens"]
            raw_loss = float(bucket["loss_sum"] / tokens)
            if not math.isfinite(raw_loss):
                continue
            # CE should be non-negative; clamp tiny numeric artifacts for stable perplexity.
            loss = max(0.0, raw_loss)
            metrics[f"{prefix}_{name}_loss"] = loss
            metrics[f"{prefix}_{name}_perplexity"] = math.exp(min(float(loss), 50.0))
            if include_accuracy:
                metrics[f"{prefix}_{name}_accuracy"] = bucket["correct"] / tokens
        return metrics

    def _evaluate_mode(
        self,
        *,
        model,
        loader,
        mode_name: str,
        mode_cfg: dict[str, Any],
        max_batches: int,
    ) -> dict[str, float]:
        allowed_classes = set(mode_cfg.get("classes", ["original", "shuffled", "negative", "domain_slice"]))
        masking_config = mode_cfg["masking"]
        seed_base = int(mode_cfg.get("seed", self.args.seed))
        metric_kind = str(mode_cfg.get("metric", "sequence")).lower()
        buckets = defaultdict(self._metric_bucket)
        prefilter_buckets = defaultdict(lambda: {"matched": 0.0, "total": 0.0})
        n_batches = 0

        iterator = loader
        progress = None
        if self.is_world_process_zero():
            progress = tqdm(total=max_batches, desc=f"eval:{mode_name}", leave=False)
        for batch in iterator:
            if n_batches >= max_batches:
                break
            batch = {
                key: value.to(self.args.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            example_types = list(batch.get("example_types", ["original"] * int(batch["input_ids"].size(0))))
            selected_rows = [idx for idx, cls in enumerate(example_types) if cls in allowed_classes]
            if not selected_rows:
                continue

            masked = self._mask_inputs_with_config(
                inputs=batch,
                masking_config=masking_config,
                seed=seed_base + n_batches,
                progress=1.0,
                original_conditioning_override="domain_only",
            )
            masked.pop("_mask_stats")
            if metric_kind == "sequence":
                masked["compute_verifier"] = False
            outputs = model(**self._strip_non_model_inputs(masked))
            generator_logits = outputs["logits"]
            mlm_labels = masked["mlm_labels"]
            if metric_kind == "verifier":
                verifier_logits = outputs["verifier_logits"]
                annotation_targets = masked["annotation_ids"].masked_fill(
                    ~masked["annotation_loss_mask"].bool(),
                    -100,
                )

            for idx in selected_rows:
                cls = str(example_types[idx])
                row = slice(idx, idx + 1)
                if metric_kind == "sequence":
                    labels = mlm_labels[row]
                    logits = generator_logits[row]
                elif metric_kind == "verifier":
                    labels = annotation_targets[row]
                    logits = verifier_logits[row]
                else:
                    raise ValueError(f"Unknown eval metric kind {metric_kind!r} for mode {mode_name!r}.")
                accuracy_mode = "family" if metric_kind == "verifier" else "exact"
                n_tokens, loss_sum, correct = self._update_token_metrics(
                    buckets[cls],
                    logits,
                    labels,
                    accuracy_mode=accuracy_mode,
                )
                if n_tokens > 0:
                    self._accumulate_token_metrics(
                        buckets["aggregate"],
                        loss_sum=loss_sum,
                        correct=correct,
                        n_tokens=n_tokens,
                    )
                    if metric_kind == "verifier":
                        self._update_prefilter_stats(prefilter_buckets[cls], logits=logits, labels=labels)
                        self._update_prefilter_stats(prefilter_buckets["aggregate"], logits=logits, labels=labels)
            n_batches += 1
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()

        metrics = self._finalize_token_metrics(
            f"eval_{mode_name}_{metric_kind}",
            buckets,
            include_accuracy=(metric_kind == "verifier"),
        )
        if metric_kind == "verifier":
            metrics.update(
                self._finalize_prefilter_metrics(
                    f"eval_{mode_name}_{metric_kind}",
                    prefilter_buckets,
                )
            )
        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        loader = self.get_eval_dataloader(eval_dataset)
        model = self.model.to(self.args.device)
        model.eval()
        eval_cfg = self.config.get("eval", {})
        max_batches = int(eval_cfg.get("max_batches", 100))
        metrics: dict[str, float] = {}
        with torch.no_grad():
            mode_defaults = {
                "mlm_15_random": "sequence",
                "mlm_50_random": "sequence",
                "mlm_85_random": "sequence",
                "anno_15_random": "verifier",
                "anno_50_random": "verifier",
                "anno_85_random": "verifier",
            }
            for mode_name, metric_kind in mode_defaults.items():
                mode_cfg = dict(eval_cfg.get(mode_name, {}))
                if not mode_cfg.get("enabled", True):
                    continue
                mode_cfg.setdefault("metric", metric_kind)
                if str(mode_cfg["metric"]).lower() == "verifier" and not annotation_loss_enabled(self.config):
                    continue
                metrics.update(
                    self._evaluate_mode(
                        model=model,
                        loader=loader,
                        mode_name=mode_name,
                        mode_cfg=mode_cfg,
                        max_batches=max_batches,
                    )
                )
                # Rebuild because loaders may be iterators.
                loader = self.get_eval_dataloader(eval_dataset)
        self.log(metrics)
        return metrics
