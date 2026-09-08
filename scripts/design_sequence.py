#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_psalm.config import load_config
from inverse_psalm.data.labels import load_mapping
from inverse_psalm.inference.annotations import parse_domain_spec


DEFAULT_CHECKPOINT = "runs/psage-base/checkpoint-107000/generator"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Design an annotation-conditioned protein sequence by iterative "
            "unmasking with an InversePSALM generator."
        )
    )
    parser.add_argument("--length", type=int, required=True, help="Full target sequence length in amino acids.")
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help=(
            "Annotation span as PF00042:25-175 using 1-indexed inclusive coordinates. "
            "May be repeated. Use None:176-250 for explicit non-domain background."
        ),
    )
    parser.add_argument("--label-mapping", required=True, help="Pickle or JSON label mapping used for training.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Generator checkpoint directory.")
    parser.add_argument("--config", default="configs/base.yaml", help="Config used for max length and ignore label.")
    parser.add_argument("--output-dir", required=True, help="Directory for final FASTA, trajectory, and metadata.")
    parser.add_argument("--output-id", default="design", help="FASTA record ID prefix.")

    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument("--step-aa", type=int, default=None, help="Residues to unmask at each step.")
    step_group.add_argument(
        "--step-frac",
        type=float,
        default=None,
        help="Fraction of the full sequence to unmask at each step, e.g. 0.05.",
    )

    parser.add_argument(
        "--position-policy",
        choices=["random", "confidence", "entropy", "margin"],
        default="confidence",
        help="Policy for selecting which masked coordinates are committed each step.",
    )
    parser.add_argument(
        "--aa-policy",
        choices=["argmax", "sample", "top-k", "top-p", "min-p"],
        default="top-p",
        help="Policy for choosing amino acids at selected coordinates.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for logits.")
    parser.add_argument("--top-k", type=int, default=None, help="K for --aa-policy top-k.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus cutoff for --aa-policy top-p.")
    parser.add_argument("--min-p", type=float, default=0.05, help="Relative min-p cutoff for --aa-policy min-p.")
    parser.add_argument("--refine-steps", type=int, default=0, help="Optional number of post-generation refinement steps.")
    refine_step_group = parser.add_mutually_exclusive_group()
    refine_step_group.add_argument("--refine-step-aa", type=int, default=None, help="Residues to revisit per refinement step.")
    refine_step_group.add_argument(
        "--refine-step-frac",
        type=float,
        default=None,
        help="Fraction of sequence to revisit per refinement step.",
    )
    parser.add_argument(
        "--refine-position-policy",
        choices=["random", "confidence", "entropy", "margin"],
        default="confidence",
        help="Refinement position policy. Non-random policies select uncertain positions first.",
    )
    parser.add_argument(
        "--refine-aa-policy",
        choices=["argmax", "sample", "top-k", "top-p", "min-p"],
        default="top-p",
        help="Amino-acid policy for refinement replacements.",
    )
    parser.add_argument("--refine-temperature", type=float, default=1.0, help="Refinement sampling temperature.")
    parser.add_argument("--refine-top-k", type=int, default=None, help="K for --refine-aa-policy top-k.")
    parser.add_argument("--refine-top-p", type=float, default=0.95, help="Nucleus cutoff for refinement top-p.")
    parser.add_argument("--refine-min-p", type=float, default=0.05, help="Relative cutoff for refinement min-p.")
    parser.add_argument(
        "--background-mode",
        choices=["none", "ignore"],
        default="ignore",
        help=(
            "How to treat unspecified background positions. Default ignore means no annotation injection; "
            "explicit --domain None:start-stop spans still inject the PSALM None label."
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-aa",
        action="store_true",
        help="Allow all single-letter uppercase tokenizer amino-acid tokens instead of canonical 20 AA only.",
    )
    parser.add_argument(
        "--masked-char",
        default="?",
        help="Character used in trajectory.fasta for residues that have not been unmasked yet.",
    )
    parser.add_argument(
        "--masked-fill-aa",
        default="A",
        help="Residue used in trajectory_foldable.fasta while positions remain masked.",
    )
    parser.add_argument("--seed", type=int, default=100, help="Random seed for position and amino-acid sampling.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for model inference. Use 'cpu' to force CPU; default uses CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        default="auto",
        help="Model dtype for inference. auto uses bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--use-fa",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override FAESM flash attention usage from the saved checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    domains = [parse_domain_spec(spec) for spec in args.domain]
    label_mapping = load_mapping(args.label_mapping)

    import torch
    from inverse_psalm.inference.design import DesignRequest, design_sequence
    from inverse_psalm.inference.outputs import write_design_result

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    request = DesignRequest(
        checkpoint=args.checkpoint,
        length=args.length,
        domains=domains,
        label_mapping=label_mapping,
        output_id=args.output_id,
        step_aa=args.step_aa,
        step_frac=args.step_frac,
        position_policy=args.position_policy,
        aa_policy=args.aa_policy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        refine_steps=args.refine_steps,
        refine_step_aa=args.refine_step_aa,
        refine_step_frac=args.refine_step_frac,
        refine_position_policy=args.refine_position_policy,
        refine_aa_policy=args.refine_aa_policy,
        refine_temperature=args.refine_temperature,
        refine_top_k=args.refine_top_k,
        refine_top_p=args.refine_top_p,
        refine_min_p=args.refine_min_p,
        background_mode=args.background_mode,
        canonical_only=not args.allow_noncanonical_aa,
        masked_char=args.masked_char,
        masked_fill_aa=args.masked_fill_aa,
        seed=args.seed,
        device=device,
        dtype=args.dtype,
        max_aa_length=int(data_cfg.get("max_aa_length", 4096)),
        ignore_label=int(data_cfg.get("ignore_label", -100)),
        use_fa=_parse_use_fa(args.use_fa),
    )
    result = design_sequence(request)
    write_design_result(result, args.output_dir)
    print(f"Wrote design trajectory to {args.output_dir}")
    print(f"Final sequence length: {len(result.final_sequence)}")
    print(f"Unmasking steps: {len(result.steps) - 1}")


def _parse_use_fa(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


if __name__ == "__main__":
    main()
