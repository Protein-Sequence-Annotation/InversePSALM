# InversePSALM v1

InversePSALM v1 is a training-first implementation of an optionally
annotation-conditioned FAESM/ESM-2 masked infiller with an optional frozen PSALM
verifier loss.

The first pass intentionally focuses on training only:

- FAESM-based ESM-2 650M generator.
- Full PSALM fine-label annotation conditioning by default.
- Frozen PSALM verifier branch by default.
- ESM2-FA-style batch-major shard streaming for same-node multi-GPU training.
- No inference, repair, iterative denoising, LoRA, or progressive unfreezing yet.

Use the environment where `PSALM-github` has FAESM available.

## Workflow

### 1. Augment raw data

```bash
python scripts/augment_data.py \
  --fasta raw.fa \
  --domain-dict raw_domains.pkl \
  --output-fasta augmented.fa \
  --output-dict augmented_domains.pkl \
  --max-length 4096
```

This script is a PSALM-style augmentation wrapper. The training-critical path
starts from the augmented FASTA and matching domain dict.

For larger inputs, use the parallel wrapper. It shards the raw FASTA/domain dict,
runs `augment_data.py` per shard, and merges the outputs:

```bash
python scripts/augment_data_parallel.py \
  --fasta raw.fa \
  --domain-dict raw_domains.pkl \
  --output-fasta augmented.fa \
  --output-dict augmented_domains.pkl \
  --max-length 4096 \
  --include-domain-slices \
  --negative-prob 0.05 \
  --workers 20
```

### 2. Build fast shards

```bash
python scripts/build_shards.py \
  --config configs/base.yaml \
  --fasta augmented.fa \
  --domain-dict augmented_domains.pkl \
  --label-mapping label_mapping.pkl \
  --output-dir data/train_shards
```

Shard rows are batch-major and include `input_ids`, `attention_mask`,
`annotation_ids`, `residue_mask`, `sequence_length`, `batch_id`, and `id`.

### 3. Train

```bash
accelerate launch --gpu_ids all scripts/train_inverse_psalm.py \
  --config configs/base.yaml \
  --train-dir data/train_shards \
  --val-dir data/val_shards
```

The dataloader partitions shards across `(rank, worker)` slots. Make sure:

```text
num_shards >= world_size * dataloader_num_workers
```

To estimate epoch length from the shard build log:

```bash
python scripts/estimate_epoch_steps.py \
  --config configs/base.yaml \
  --build-log logs/build_shards.out \
  --world-size 8
```

## Model Sketch

At each residue position, the masked sequence goes through the normal FAESM
embedding path. The full PSALM fine-label annotation id goes through a small
trainable annotation embedding/projection path. The two hidden states are summed
before the first transformer block:

```text
h0 = esm_embed(masked_sequence_ids) + alpha * annotation_projection(annotation_ids)
```

The annotation encoder is configurable. The default `full_token` encoder embeds
PSALM fine-label IDs directly and projects them to the ESM hidden size. The
`factorized` encoder splits labels into family and start/middle/stop state
embeddings before projection.

The generator predicts amino-acid logits with the pretrained LM head, frozen by
default. When `model.annotation_loss` is enabled, a frozen PSALM verifier
receives the soft expected input embedding from the generator distribution and
produces a fine-state annotation loss.

Validation logs 15% MLM, 100% MLM, and 100% masked verifier metrics per
augmented-data class and in aggregate where configured loss masks apply.

## Important Defaults

- `model.freeze_generator_trunk: true`
- `model.freeze_lm_head: true`
- `model.annotation_track: true`
- `model.annotation_loss: true`
- `model.canonical_aa_only_for_verifier: true`
- `training.wandb_project: "InversePSALM"`
- `loss.verifier_weight.value: 1.0`

Flip the freeze knobs in config after the annotation path and verifier loss are
confirmed to train correctly.
