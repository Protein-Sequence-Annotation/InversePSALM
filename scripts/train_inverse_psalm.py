#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_psalm.config import deep_get, load_config
from inverse_psalm.models.inverse_psalm import build_inverse_psalm
from inverse_psalm.training.trainer import InversePSALMTrainer, make_data_collator


def _startup_log(message: str) -> None:
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    print(f"[startup][rank={rank} local_rank={local_rank}] {message}", flush=True)


def _resolve_val_dataset(path: str):
    if os.path.isfile(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)
    shard_paths = sorted(glob.glob(os.path.join(path, "shard-*")))
    if len(shard_paths) == 1:
        return load_from_disk(shard_paths[0])
    if not shard_paths:
        raise ValueError(f"No validation dataset found under {path}")
    return shard_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InversePSALM v1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    _startup_log("loading config and resolving datasets")
    cfg = load_config(args.config)
    train_shards = sorted(glob.glob(os.path.join(args.train_dir, "shard-*")))
    if not train_shards:
        raise ValueError(f"No train shards found under {args.train_dir}")
    eval_dataset = _resolve_val_dataset(args.val_dir)

    _startup_log("building model and tokenizer")
    model, tokenizer = build_inverse_psalm(cfg)
    _startup_log("model and tokenizer ready")
    train_cfg = cfg.get("training", {})
    report_to = train_cfg.get("report_to", ["wandb"])
    if "wandb" in report_to and train_cfg.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", str(train_cfg["wandb_project"]))
    mp = str(train_cfg.get("mixed_precision", "")).lower()

    _startup_log("creating training arguments")
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        overwrite_output_dir=True,
        eval_strategy=train_cfg.get("evaluation_strategy", "steps"),
        eval_steps=int(train_cfg.get("eval_steps", 1000)),
        save_steps=int(train_cfg.get("save_steps", 10000)),
        save_total_limit=train_cfg.get("save_total_limit", None),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        max_steps=int(train_cfg.get("max_steps", 0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(train_cfg.get("gradient_clipping", 0.0)),
        learning_rate=float(deep_get(cfg, "training.learning_rate.annotation", 1e-3)),
        fp16=(mp == "fp16"),
        bf16=(mp == "bf16"),
        report_to=report_to,
        run_name=train_cfg.get("run_name", None),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(train_cfg.get("dataloader_pin_memory", True)),
        dataloader_prefetch_factor=train_cfg.get("dataloader_prefetch_factor", None),
        seed=int(train_cfg.get("seed", 100)),
        ignore_data_skip=bool(train_cfg.get("ignore_data_skip", True)),
        ddp_find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", False)),
        save_safetensors=bool(train_cfg.get("save_safetensors", True)),
    )

    _startup_log("creating trainer")
    trainer = InversePSALMTrainer(
        config=cfg,
        tokenizer=tokenizer,
        train_dataset=train_shards,
        eval_dataset=eval_dataset,
        data_collator=make_data_collator(tokenizer),
        model=model,
        args=training_args,
    )
    _startup_log("starting trainer.train")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    _startup_log("trainer.train finished; saving final model")
    trainer.save_model()


if __name__ == "__main__":
    main()
