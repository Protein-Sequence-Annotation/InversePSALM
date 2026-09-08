#!/bin/bash
#SBATCH --job-name=esmfold_traj
#SBATCH --output=logs/fold_esmfold.out
#SBATCH --error=logs/fold_esmfold.err
#SBATCH --time=1-00:00
#SBATCH --partition=eddy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=80GB

set -euo pipefail

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

REPO_ROOT="/n/eddy_lab/Lab/protein_annotation_dl/InversePSALM"
cd "${REPO_ROOT}"
mkdir -p logs outputs

export INVERSE_PSALM_CACHE_ROOT="${REPO_ROOT}/.cache"
export XDG_CACHE_HOME="${INVERSE_PSALM_CACHE_ROOT}/xdg"
export HF_HOME="${INVERSE_PSALM_CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${INVERSE_PSALM_CACHE_ROOT}/torch"
export PIP_CACHE_DIR="${INVERSE_PSALM_CACHE_ROOT}/pip"
export CONDA_PKGS_DIRS="${INVERSE_PSALM_CACHE_ROOT}/conda/pkgs"

TRAJECTORY_FASTA="${TRAJECTORY_FASTA:?Set TRAJECTORY_FASTA to a design trajectory.fasta path.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the structure output directory.}"
ESMFOLD_ENV_PREFIX="${ESMFOLD_ENV_PREFIX:-${REPO_ROOT}/.conda/envs/esmfold}"

conda deactivate || true
if [[ -d "${ESMFOLD_ENV_PREFIX}" ]]; then
  source activate "${ESMFOLD_ENV_PREFIX}"
else
  source activate /n/eddy_lab/Lab/protein_annotation_dl/PSALM-2/psalm2
fi

cmd=(
  python scripts/structure/fold_trajectory_esmfold.py
  --trajectory-fasta "${TRAJECTORY_FASTA}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
)
if [[ -n "${ESMFOLD_CHUNK_SIZE:-}" ]]; then
  cmd+=(--chunk-size "${ESMFOLD_CHUNK_SIZE}")
fi
"${cmd[@]}"
