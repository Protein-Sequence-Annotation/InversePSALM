# InversePSALM Design Inference

This workflow designs a full protein sequence from a requested length and an
annotation track. It starts from a fully masked sequence, runs the generator,
commits a small set of positions, and repeats until no masked residues remain.

## Sequence Design

Run from the repo root in the PSALM environment:

```bash
conda activate /n/eddy_lab/Lab/protein_annotation_dl/PSALM-2/psalm2

python scripts/design_sequence.py \
  --length 500 \
  --domain PF00042:25-175 \
  --domain None:176-250 \
  --label-mapping /path/to/label_mapping.pkl \
  --checkpoint runs/psage-base/checkpoint-107000/generator \
  --step-aa 25 \
  --position-policy confidence \
  --aa-policy top-p \
  --top-p 0.95 \
  --temperature 1.0 \
  --seed 100 \
  --output-dir outputs/designs/pf00042_trial1
```

By default, the script uses CUDA if it is available. Add `--device cpu` to force
CPU generation, or `--device cuda` in a GPU SLURM job. CPU generation uses the
same sampling code but will be much slower for the 650M generator. `--dtype auto`
uses bf16 on CUDA and fp32 on CPU; FlashAttention on GPU requires bf16 or fp16.

Domain coordinates are 1-indexed and inclusive, matching the training data
format. Adjacent spans should not reuse the boundary residue: use `25-175` then
`176-250`, not `25-175` then `175-250`.

Unspecified residues receive no annotation injection by default. Use explicit
`--domain None:start-stop` spans when a region should receive the PSALM `None`
label. Use `--background-mode none` only if every unspecified residue should get
the PSALM `None` label.

## Sampling Controls

Step size is specified by exactly one of:

- `--step-aa N`: commit `N` residues per model forward pass.
- `--step-frac F`: commit about `F * length` residues per pass.

There are two independent sampling layers:

- `--position-policy`: chooses which currently masked coordinates to commit.
  Options are `random`, `confidence`, `entropy`, and `margin`.
- `--aa-policy`: chooses the amino acid written at each selected coordinate.
  Options are `argmax`, `sample`, `top-k`, `top-p`, and `min-p`.

Examples:

```bash
# Greedy, deterministic design.
--position-policy confidence --aa-policy argmax

# Random coordinate order with nucleus residue sampling.
--position-policy random --aa-policy top-p --top-p 0.95 --temperature 1.0

# Commit the least ambiguous positions first, but sample among top-k residues.
--position-policy margin --aa-policy top-k --top-k 5

# Relative min-p sampling keeps residues with p >= min_p * p_best.
--position-policy confidence --aa-policy min-p --min-p 0.05
```

Predictions are restricted to the canonical 20 amino acids by default. Add
`--allow-noncanonical-aa` to permit any single-letter uppercase tokenizer token.

## Outputs

The design script writes:

- `final.fasta`: the final generated sequence.
- `trajectory.fasta`: one FASTA record per design step. Positions that have not
  been unmasked yet are `?` by default; change this with `--masked-char`.
- `trajectory_foldable.fasta`: the same steps, but masked positions are alanine
  by default for structure prediction; change this with `--masked-fill-aa`.
- `trajectory.jsonl`: raw per-step state, including masked positions as `?`.
- `metadata.json`: checkpoint, policies, seed, domains, and step count.

`trajectory_foldable.fasta` is the intended ESMFold input. Step 0 is the fully
masked state with masked residues filled by alanine.

## Single-GPU SLURM Example

`slurm_scripts/design_pf00042_example.sh` requests one H200 GPU like the training
job and writes a length-150 PF00042 whole-domain example under `examples/`:

```bash
sbatch slurm_scripts/design_pf00042_example.sh
```

The script defaults to `data/pfam_label_mapping.pkl` and
`runs/psage-base/checkpoint-107000/generator`. Override either at submission
time if needed:

```bash
sbatch \
  --export=ALL,LABEL_MAPPING=/path/to/pfam_label_mapping.pkl,OUTPUT_DIR=examples/my_trial \
  slurm_scripts/design_pf00042_example.sh
```

It uses `--step-aa 1 --position-policy random --aa-policy sample`, so each
iteration chooses one still-masked position at random and samples an amino acid
from the model distribution at that position.

## Optional Refinement

Refinement is off by default. Enable it with `--refine-steps` plus an independent
refinement step size:

```bash
--refine-steps 20 \
--refine-step-aa 5 \
--refine-position-policy confidence \
--refine-aa-policy min-p \
--refine-min-p 0.05
```

After generation completes, each refinement step scores the current full
sequence, selects uncertain positions, masks them, predicts replacements, and
accepts sampled replacements only when `p(new) >= p(old)` under the masked
distribution. For refinement, `confidence` means low-confidence first, `entropy`
means high-entropy first, and `margin` means low-margin first. JSONL trajectory
rows include `phase: generation` or `phase: refinement`.

## ESMFold

ESMFold is intentionally outside the package. All cache and environment paths
are kept under the repo:

```bash
export INVERSE_PSALM_CACHE_ROOT="$PWD/.cache"
export HF_HOME="$INVERSE_PSALM_CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$INVERSE_PSALM_CACHE_ROOT/torch"
export XDG_CACHE_HOME="$INVERSE_PSALM_CACHE_ROOT/xdg"
export PIP_CACHE_DIR="$INVERSE_PSALM_CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$INVERSE_PSALM_CACHE_ROOT/conda/pkgs"
```

First try the current PSALM environment:

```bash
bash scripts/structure/setup_esmfold_env.sh
```

If ESMFold is not available there, the setup script creates a repo-local conda
environment at `.conda/envs/esmfold` and installs the `fair-esm[esmfold]` extra.

Fold a trajectory:

```bash
python scripts/structure/fold_trajectory_esmfold.py \
  --trajectory-fasta outputs/designs/pf00042_trial1/trajectory_foldable.fasta \
  --output-dir outputs/designs/pf00042_trial1/structures \
  --device cuda
```

The folder will contain ordered PDB files and `manifest.json`. Use the manifest
as the input for a later rendering script or external tool such as PyMOL or
ChimeraX to make a GIF.
