#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ROOT="${INVERSE_PSALM_CACHE_ROOT:-${REPO_ROOT}/.cache}"
ESMFOLD_ENV_PREFIX="${ESMFOLD_ENV_PREFIX:-${REPO_ROOT}/.conda/envs/esmfold}"

export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda/pkgs"

mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"

echo "[esmfold-setup] repo root: ${REPO_ROOT}"
echo "[esmfold-setup] cache root: ${CACHE_ROOT}"

if python - <<'PY'
import esm
import torch
assert hasattr(esm.pretrained, "esmfold_v1")
print(f"ESMFold import OK in current Python; torch={torch.__version__}")
PY
then
  echo "[esmfold-setup] current environment already provides ESMFold"
  exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[esmfold-setup] conda was not found. Activate the PSALM env first or load conda, then rerun." >&2
  exit 1
fi

if [[ ! -d "${ESMFOLD_ENV_PREFIX}" ]]; then
  echo "[esmfold-setup] creating repo-local conda env at ${ESMFOLD_ENV_PREFIX}"
  conda create -y -p "${ESMFOLD_ENV_PREFIX}" python=3.10 pip
fi

echo "[esmfold-setup] installing fair-esm ESMFold extra"
conda run -p "${ESMFOLD_ENV_PREFIX}" python -m pip install --upgrade pip
conda run -p "${ESMFOLD_ENV_PREFIX}" python -m pip install "fair-esm[esmfold]"

conda run -p "${ESMFOLD_ENV_PREFIX}" python - <<'PY'
import esm
import torch
assert hasattr(esm.pretrained, "esmfold_v1")
print(f"ESMFold import OK; torch={torch.__version__}")
PY

echo "[esmfold-setup] done"
echo "[esmfold-setup] use: conda activate ${ESMFOLD_ENV_PREFIX}"
