#!/bin/bash
#SBATCH --job-name=build_invpsalm_shards
#SBATCH --output=logs/build_val_shards.out
#SBATCH --error=logs/build_val_shards.err
#SBATCH --time=72:00:00
#SBATCH --partition=eddy
#SBATCH -c 20
#SBATCH --mem=1000GB

module load python/3.12.5-fasrc01
conda deactivate
source activate /n/eddy_lab/Lab/protein_annotation_dl/PSALM-2/psalm2

set -euo pipefail

cd /n/eddy_lab/Lab/protein_annotation_dl/InversePSALM
mkdir -p logs data

OUTPUT_DIR="data/pfam_full_val_augmented_shards"

if [ -e "${OUTPUT_DIR}" ]; then
  echo "Refusing to overwrite existing output dir: ${OUTPUT_DIR}" >&2
  echo "Move or remove it before rerunning this job." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"

python scripts/build_shards.py \
  --config configs/base.yaml \
  --fasta data/pfam_full_val_augmented.fasta \
  --domain-dict data/pfam_full_val_augmented_domains.pkl \
  --label-mapping data/pfam_label_mapping.pkl \
  --output-dir "${OUTPUT_DIR}" \
  --max-aa-length 4096 \
  --max-tokens-per-batch 8192 \
  --chunk-size 100000 \
  --shard-size 25000 \
  --seed 100
