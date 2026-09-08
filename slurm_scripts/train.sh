#!/bin/bash
#SBATCH --job-name=anno_test
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err
#SBATCH --time=10-00:00
#SBATCH --partition=eddy
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --mem=700GB #700 def works

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

conda deactivate
source activate /n/eddy_lab/Lab/protein_annotation_dl/PSALM-2/psalm2

set -euo pipefail

cd /n/eddy_lab/Lab/protein_annotation_dl/InversePSALM
mkdir -p logs runs

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-InversePSALM}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

NUM_PROCESSES="${SLURM_NTASKS:-4}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --gpu_ids all \
  scripts/train_inverse_psalm.py \
  --config configs/base.yaml \
  --train-dir data/pfam_ipr30_combined_augmented_shards \
  --val-dir data/pfam_full_val_augmented_shards