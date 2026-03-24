# Visual Evaluation Pipeline

This directory contains the visual quality evaluation pipeline for `spectral-init`.
It compares Rust spectral initialization output against Python UMAP reference output
across Tier 1 synthetic datasets, producing side-by-side scatter plots and quality metrics.

---

## Purpose

The fixture generation system (`tests/fixtures/`) validates numerical correctness
(residual checks, eigenvector comparison). This pipeline validates **visual quality**:
do the Rust spectral init coordinates look like the Python UMAP spectral init coordinates?

---

## Two-Phase Workflow

### Phase 1 — Baseline (`--phase baseline`)

Runs Python UMAP on each Tier 1 synthetic dataset and saves reference outputs:

- Spectral init coordinates (pre-SGD) from `umap.spectral.spectral_layout`
- Final UMAP embedding
- Fuzzy k-NN graph in Rust-compatible CSR format
- 2×2 comparison plot (spectral init, final embedding, eigenvalue spectrum, metrics)

### Phase 2 — Comparison (future)

Loads the Rust spectral init output and compares it visually and numerically against
the Phase 1 baseline. Generates overlay plots and alignment scores.

---

## How to Run Phase 1

Run from the repository root with the `spectral-test` conda environment active:

```bash
micromamba activate spectral-test

# All 5 Tier 1 datasets
python tests/visual_eval/generate_umap_comparisons.py --phase baseline

# Single dataset
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --dataset blobs_1000

# Custom output directory
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --output-dir /tmp/visual_eval
```

---

## Output Files

For each dataset `{name}`, Phase 1 writes the following files to `tests/visual_eval/output/`:

| File | Shape / Type | Description |
|------|-------------|-------------|
| `{name}_graph.npz` | CSR sparse | Fuzzy k-NN graph in Rust-compatible format (f32 data, i32 indices/indptr) |
| `{name}_py_spectral.npy` | `(n, 2)` float64 | Python spectral init coordinates (pre-SGD) |
| `{name}_py_final.npy` | `(n, 2)` float32 | Python UMAP final embedding |
| `{name}_labels.npy` | `(n,)` int32 | Dataset labels for coloring |
| `{name}_baseline.png` | 2×2 figure | Spectral init / final / eigenvalue spectrum / metrics |

The `output/` directory is gitignored (files are large and regenerated locally).
Only `output/.gitkeep` is tracked to ensure the directory exists after a fresh clone.

---

## Tier 1 Datasets

| Name | Generator | n | Features | Notes |
|------|-----------|---|----------|-------|
| `blobs_1000` | `make_blobs` | 1000 | 10 | 5 Gaussian clusters |
| `circles_1000` | `make_circles` | 1000 | 2 | Concentric circles |
| `swiss_roll_2000` | `make_swiss_roll` | 2000 | 3 | Labels = roll position, 5 bins |
| `two_moons_1000` | `make_moons` | 1000 | 2 | Two crescents |
| `blobs_hd_2000` | `make_blobs` | 2000 | 50 | 10 clusters, high-dimensional |

---

## Python Environment Setup

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
then create the environment from the shared spec file:

```bash
micromamba create -f tests/environment.yml
micromamba activate spectral-test
```

Key packages: Python 3.11, NumPy 2.2, SciPy 1.15, umap-learn 0.5.11, scikit-learn, matplotlib.
