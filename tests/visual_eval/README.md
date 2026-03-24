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

### Phase 2 — Comparison (`--phase compare`)

Loads Phase 1 artifacts and Rust spectral init coordinates, runs a three-way UMAP
SGD comparison (Python spectral, Rust spectral, random), and produces:

- A 2×3 comparison plot (pre-SGD inits and post-SGD embeddings)
- An overlay plot (Python vs Rust SGD results)
- Per-dataset metrics JSON with PASS/FAIL verdict

---

## How to Run

### End-to-end (recommended)

Run the full pipeline from Phase 1 through Phase 2 with a single script:

```bash
micromamba activate spectral-test
./tests/visual_eval/run_eval.sh
```

This runs Phase 1, the Rust export test, and Phase 2 in sequence, then prints a
per-dataset PASS/FAIL summary.

### Phase 1 only

```bash
micromamba activate spectral-test

# All 5 Tier 1 datasets
python tests/visual_eval/generate_umap_comparisons.py --phase baseline

# Single dataset
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --dataset blobs_1000

# Custom output directory
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --output-dir /tmp/visual_eval
```

### Phase 2 only (requires Phase 1 artifacts and Rust export)

```bash
micromamba activate spectral-test

# All 5 Tier 1 datasets
python tests/visual_eval/generate_umap_comparisons.py --phase compare

# Single dataset
python tests/visual_eval/generate_umap_comparisons.py --phase compare --dataset blobs_1000
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

Phase 2 reads `{name}_rust_init.npy` (written by the Rust export test) and adds:

| File | Shape / Type | Description |
|------|-------------|-------------|
| `{name}_comparison.png` | 2×3 figure | Pre-SGD inits (top row) and post-SGD embeddings (bottom row) |
| `{name}_overlay.png` | 1×1 figure | Python vs Rust SGD embeddings overlaid |
| `{name}_metrics.json` | JSON | Trustworthiness, silhouette, Procrustes, pairwise-dist correlation + PASS/FAIL |

The `output/` directory is gitignored (files are large and regenerated locally).
Only `output/.gitkeep` is tracked to ensure the directory exists after a fresh clone.

### Phase 2 success criteria

| Metric | Threshold | PASS condition |
|--------|-----------|---------------|
| Procrustes disparity (rust vs python) | < 0.05 | Rust embedding aligns with Python after Procrustes |
| Pairwise distance correlation (rust vs python) | > 0.99 | Pairwise distances are nearly identical |
| Trustworthiness difference | < 0.01 | Rust embedding preserves local structure as well as Python |
| Silhouette score difference | < 0.05 | Rust embedding separates clusters as well as Python |

Overall verdict is **PASS** only when all four metrics pass. A FAIL means the Rust spectral init
diverges from Python in a way that affects downstream SGD quality. A note is printed when the
random init also passes all thresholds (indicating the dataset may be too easy to distinguish
initialization strategies).

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
