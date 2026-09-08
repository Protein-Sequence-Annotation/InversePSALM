# InversePSALM Shard Format

`scripts/build_shards.py` writes Hugging Face dataset directories named
`shard-XXXXX`. Rows are ordered batch-major: all examples for a `batch_id` are
contiguous.

Each row contains:

- `id`: FASTA record ID.
- `input_ids`: unmasked generator tokenizer IDs.
- `attention_mask`: standard attention mask.
- `labels`: copy of unmasked `input_ids`.
- `annotation_ids`: full PSALM fine labels, with `-100` on ignored special or
  padded positions.
- `residue_mask`: `1` where sequence masking and verifier loss are valid.
- `domain_mask`: `1` where the annotation is a trusted non-None domain state.
- `example_type` / `example_type_id`: one of `original`, `shuffled`,
  `negative`, or `domain_slice`.
- `sequence_length`: padded batch sequence length.
- `batch_id`: integer batch group.

Masks are not stored. The trainer creates random, span, or full masks on device
from the config schedule.

Per-class loss use is configured in `configs/base.yaml` under
`loss.example_classes`. By default, originals use annotation loss only on
`domain_mask`, while shuffled examples, negatives, and domain slices use
annotation loss on all real residues. MLM loss can be enabled or disabled per
class; disabled classes are excluded from MLM validation metrics.

To estimate epoch length from a shard build log:

```bash
python scripts/estimate_epoch_steps.py \
  --config configs/base.yaml \
  --build-log logs/build_shards.out \
  --world-size 8
```

This reports total examples, precomputed batches, estimated micro-steps per
epoch, and estimated optimizer steps per epoch after gradient accumulation.
