#!/bin/bash
#SBATCH --job-name=design_flavohemo
#SBATCH --output=logs/design_flavohemo.out
#SBATCH --error=logs/design_flavohemo.err
#SBATCH --time=1-00:00
#SBATCH --partition=seas_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=80GB

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

conda deactivate
source activate /n/eddy_lab/Lab/protein_annotation_dl/PSALM-2/psalm2

set -euo pipefail

cd /n/eddy_lab/Lab/protein_annotation_dl/InversePSALM
mkdir -p logs examples

export TOKENIZERS_PARALLELISM=false

LABEL_MAPPING="${LABEL_MAPPING:-data/pfam_label_mapping.pkl}"
CHECKPOINT="${CHECKPOINT:-runs/psage-base/checkpoint-107000/generator}"
OUTPUT_DIR="${OUTPUT_DIR:-examples/crazy}"

if [ ! -e "${LABEL_MAPPING}" ]; then
  echo "Label mapping not found: ${LABEL_MAPPING}" >&2
  echo "Set LABEL_MAPPING=/path/to/pfam_label_mapping.pkl when submitting." >&2
  exit 1
fi

rm -rf "${OUTPUT_DIR}"

python scripts/design_sequence.py \
  --config configs/base.yaml \
  --checkpoint "${CHECKPOINT}" \
  --label-mapping "${LABEL_MAPPING}" \
  --length 300 \
  --domain PF00013:26-85 \
  --domain PF00078:125-275 \
  --step-aa 1 \
  --position-policy random \
  --aa-policy min-p \
  --min-p 0.05 \
  --refine-steps 30 \
  --refine-step-aa 10 \
  --refine-position-policy confidence \
  --refine-aa-policy min-p \
  --refine-min-p 0.05 \
  --refine-temperature 1.0 \
  --temperature 1.0 \
  --seed 100 \
  --device cuda \
  --dtype bf16 \
  --masked-char "?" \
  --masked-fill-aa A \
  --output-id crazy_kh_rdrp \
  --output-dir "${OUTPUT_DIR}"

echo "Wrote trajectory FASTA to ${OUTPUT_DIR}/trajectory.fasta"
echo "Wrote ESMFold-ready FASTA to ${OUTPUT_DIR}/trajectory_foldable.fasta"
