#!/bin/bash
#SBATCH --job-name=aug_fa_parallel
#SBATCH --output=logs/augment_val_data.out
#SBATCH --error=logs/augment_val_data.err
#SBATCH --time=50:00:00
#SBATCH --partition=eddy
#SBATCH -c 20
#SBATCH --mem=1000GB

# Load modules and activate environment (match augment_fasta.sh)
module load python/3.12.5-fasrc01
conda deactivate
source activate /n/eddy_lab/Lab/protein_annotation_dl/PSALM-2/psalm2

set -euo pipefail

cd /n/eddy_lab/Lab/protein_annotation_dl/InversePSALM
WORKERS="${SLURM_CPUS_PER_TASK:-20}"

python scripts/augment_data_parallel.py \
  --fasta data/processed_pfam_full_sampled_uniprot.fasta \
  --domain-dict data/pfam_full_sampled_domains_in_uniprot.pkl \
  --output-fasta data/pfam_full_val_augmented.fasta \
  --output-dict data/pfam_full_val_augmented_domains.pkl \
  --max-length 4096 \
  --include-domain-slices \
  --negative-prob 0.05 \
  --workers "${WORKERS}" \
  --temp-dir data/tmp/augment_parallel
