# Visual Evaluation Pipeline

This directory contains the visual quality evaluation pipeline for `spectral-init`.
It compares Rust spectral initialization output against Python UMAP reference output,
producing side-by-side scatter plots, overlay comparisons, and quality metrics.

---

## Purpose

The fixture generation system (`tests/fixtures/`) validates numerical correctness
(residual checks, eigenvector comparison). This pipeline validates **visual quality**:
does the Rust spectral init produce embeddings of equal quality to Python UMAP's
spectral init, as measured by both human inspection and quantitative metrics?

---

## Understanding the Datasets

Each dataset tests a different aspect of spectral initialization quality. All Tier 1
datasets are deterministic (seeded with `random_state=42`).

### two_moons_1000

**Shape:** Two interleaving crescent (half-circle) shapes in 2D — like two cupped hands
facing each other. One moon curves upward; the other is flipped and offset so it
interlocks with the first. 1000 points, `noise=0.05`.

**Why it matters:** The two classes are not linearly separable — no straight line can
divide them. PCA or random projection smears both moons into a single blob. Spectral
init must recognize each moon as a locally connected 1D arc and place them apart.

**What to look for:**
- **Pre-SGD (top row):** Two smeared but recognizably separated elongated clouds, one
  per crescent. The Fiedler vector bisects the graph at the weakest connectivity gap
  between the moons.
- **Post-SGD (bottom row):** Two cleanly separated arcs or banana shapes with a visible
  gap. Random init may produce a tangled or less separated result.

### circles_1000

**Shape:** Two concentric circles in 2D — a small inner ring surrounded by a larger
outer ring (`factor=0.5`, so inner radius is half of outer). 1000 points, `noise=0.05`.

**Why it matters:** The inner ring is completely enclosed by the outer ring. No linear
method can separate them. The k-NN graph naturally forms two near-disconnected
components (the gap between rings exceeds the nearest-neighbor radius), making this
the simplest connectivity test for spectral init.

**What to look for:**
- **Pre-SGD:** Two clearly separated blobs or elongated loops — the Fiedler vector
  cleanly assigns positive values to one ring and negative to the other.
- **Post-SGD:** Two compact, well-separated clusters. If spectral init works at all,
  it will separate concentric circles.

### swiss_roll_2000

**Shape:** A 2D sheet rolled into a spiral in 3D — like a cinnamon roll. 2000 points on
the manifold surface (`noise=0.0`). Labels are the angular position along the spiral,
binned into 5 color bands.

**Why it matters:** This is the hardest Tier 1 dataset. In 3D Euclidean space, the
innermost wrap of the roll is physically close to the outermost wrap — but they are
topologically far apart along the manifold surface. A correct spectral init must
"unroll" the swiss roll, placing manifold-distant points far apart in the embedding.

**What to look for:**
- **Pre-SGD:** A partially unrolled or smeared layout where the 5 color bands progress
  in rough order. The Laplacian eigenvectors recover the two intrinsic manifold
  coordinates (angular sweep and axial height).
- **Post-SGD:** An unrolled sheet where the 5 bands appear as ordered stripes. Silhouette
  is typically lower (~0.4) than binary datasets because the bands are not as cleanly
  separated.

### blobs_1000

**Shape:** Five isotropic Gaussian clusters in 10-dimensional space. 1000 points, default
`cluster_std=1.0`. Clean Euclidean separation between centers.

**Why it matters:** This is a sanity check — the easiest possible case. The k-NN graph
has five near-disconnected components and spectral eigenvectors trivially separate them.
If this fails, something is fundamentally broken.

**Known limitation:** This dataset produces a disconnected graph with 5 components.
The Rust export currently passes `data=None`, but the multi-component meta-embedding
path requires raw data when `n_components > 2 * n_embedding_dims` (i.e., > 4 components).
Phase 2 comparison is not yet available for this dataset.

### blobs_hd_2000

**Shape:** Ten isotropic Gaussian clusters in 50-dimensional space. 2000 points. Due to
concentration of measure in high dimensions, clusters are extremely well-separated.

**Why it matters:** Tests computational scalability — the eigensolver must handle a
2000x2000 sparse Laplacian with 10 near-disconnected components, and the k-NN graph
is built from 50D data.

**Known limitation:** Same as `blobs_1000` — 10 components exceeds the threshold for
data-free meta-embedding. Phase 2 comparison is not yet available.

---

## How to Read the Plots

### Input Plot (`{name}_input.png`)

Shows the raw dataset before any UMAP processing. This is what the data actually
looks like — the ground truth you should compare everything else against.

- **2D datasets** (two_moons, circles): plotted directly with true coordinates.
- **3D datasets** (swiss_roll): shown as both a top-down 2D projection and a 3D view.
- **High-dimensional datasets** (blobs_1000, blobs_hd_2000): projected to 2D via PCA
  so you can see the cluster structure. The actual data lives in 10D or 50D.

Colors represent ground-truth class labels.

### Baseline Plot (`{name}_baseline.png`) — 2x2 Grid

| Panel | What it shows |
|-------|---------------|
| **Top-left** | Python spectral init (pre-SGD). Raw Laplacian eigenvector coordinates. Should already show rough cluster structure. |
| **Top-right** | Python UMAP final (post-SGD). The reference target that Rust must match in quality. |
| **Bottom-left** | Eigenvalue spectrum (first 10 eigenvalues). A large spectral gap (visible elbow) means strong community structure and reliable spectral init. The annotated value is `lambda_2 - lambda_1`. |
| **Bottom-right** | Metrics text: trustworthiness, silhouette, connected components, spectral gap, condition number. |

**Colors:** Ground-truth class labels using the `tab10` palette (up to 10 distinct colors).

### Comparison Plot (`{name}_comparison.png`) — 2x3 Grid

|  | Python Spectral Init | Rust Spectral Init | Random Init |
|--|---------------------|--------------------|-------------|
| **Top row (pre-SGD)** | Python eigenvector coords | Rust eigenvector coords | Uniform random in [-10, 10] |
| **Bottom row (post-SGD)** | Python init fed through SGD | Rust init fed through SGD | Random init fed through SGD |

**What to look for:**
- **Top row:** Python and Rust panels should show the same cluster shapes. They may be
  reflected or rotated (this is expected — see "Understanding Metric Results" below).
  The random panel will always look like a uniform cloud.
- **Bottom row:** Python and Rust panels should show similar cluster topology. The random
  panel may still produce recognizable clusters (UMAP SGD is robust) but with different
  global orientation.

**Colors:** Ground-truth class labels (`tab10` palette).

### Overlay Plot (`{name}_overlay.png`)

A single scatter plot overlaying the two post-SGD embeddings:
- **Blue circles:** Python-init-to-SGD final embedding
- **Red crosses:** Rust-init-to-SGD final embedding

**What good looks like:** Red and blue point clouds are nearly superimposed — you see a
purple-ish mixture with no systematic separation between colors.

**What bad looks like:** Red and blue form distinct, non-overlapping regions, meaning
the two initializations caused SGD to converge to qualitatively different local optima.

**Partial overlap** (correct shapes but mirrored/rotated) usually means eigenvector sign
or rotational ambiguity — a benign difference, not a bug.

---

## Understanding Metric Results

### Why Procrustes/pairwise can FAIL while trustworthiness/silhouette PASS

This is the most important thing to understand when interpreting results.

**Procrustes disparity** and **pairwise distance correlation** are *geometric alignment*
metrics — they ask "do these two embeddings look the same on a map?" They will fail
whenever the embeddings are reflected, rotated, or in different (but equally valid)
local minima of UMAP's loss landscape.

**Trustworthiness** and **silhouette** are *embedding quality* metrics — they compare
each embedding independently against ground truth (high-D neighborhoods and known
labels). They answer "is the embedding equally good?" regardless of orientation.

**Procrustes/pairwise FAIL + trustworthiness/silhouette PASS means:**
> "The Rust init produces an equally good embedding, but it looks different on the page —
> possibly mirrored, rotated, or in a different local minimum of equivalent quality."

This is a benign result. Eigenvectors are only unique up to sign (if `v` is an
eigenvector, so is `-v`), and near-degenerate eigenspaces allow full rotational
ambiguity. Both Rust and Python apply sign normalization (argmax convention), but they
may still disagree on degenerate subspaces.

**Trustworthiness/silhouette FAIL would mean** the Rust init causes SGD to converge to
a genuinely worse solution — fewer preserved neighborhoods, worse cluster separation.
That would indicate a real implementation bug.

### What to actually care about

1. **Trustworthiness and silhouette deltas** — the quality gate. If these fail, the Rust
   spectral init is producing measurably worse downstream embeddings.
2. **Visual inspection of the overlay plot** — if shapes match but are mirrored/rotated,
   the implementation is correct despite Procrustes failure.
3. **The "random also passes" warning** — if random init also passes all thresholds, the
   dataset is too easy to discriminate between strategies.

### Metric Details

| Metric | How it's computed | Threshold | What it measures |
|--------|-------------------|-----------|------------------|
| Procrustes disparity | `scipy.spatial.procrustes` on post-SGD embeddings | < 0.05 | Geometric shape similarity after optimal rotation/scale alignment |
| Pairwise distance correlation | Pearson r between `pdist(embed_py)` and `pdist(embed_rust)` | > 0.99 | Whether relative inter-point distances are preserved |
| Trustworthiness delta | `abs(tw_rust - tw_py)` where each is `sklearn.manifold.trustworthiness(X, embed, n_neighbors=15)` | < 0.01 | Whether local neighborhood preservation is equally good |
| Silhouette delta | `abs(sil_rust - sil_py)` where each is `sklearn.metrics.silhouette_score(embed, labels)` | < 0.05 | Whether cluster separation quality is equally good |

Overall verdict is **PASS** only when all four metrics pass.

---

## Two-Phase Workflow

### Phase 1 — Baseline (`--phase baseline`)

Runs Python UMAP on each dataset and saves reference outputs:

- Spectral init coordinates (pre-SGD) from `umap.spectral.spectral_layout`
- Final UMAP embedding
- Fuzzy k-NN graph in Rust-compatible CSR format
- 2x2 baseline plot (spectral init, final embedding, eigenvalue spectrum, metrics)

UMAP parameters: `n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean",
random_state=42, n_jobs=1`.

### Rust Export (between Phase 1 and Phase 2)

The `export_rust_init` test loads each `{name}_graph.npz` from Phase 1, runs
`spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())`, and writes
`{name}_rust_init.npy`. Parameters: `n_components=2, seed=42, PythonCompat` mode.

```bash
cargo test --test export_rust_init --features testing -- --ignored --nocapture
```

### Phase 2 — Comparison (`--phase compare`)

Loads Phase 1 artifacts and Rust init coordinates, runs a three-way UMAP SGD
comparison (Python spectral, Rust spectral, random), and produces:

- 2x3 comparison plot (pre-SGD inits and post-SGD embeddings)
- Overlay plot (Python vs Rust SGD results)
- Per-dataset metrics JSON with PASS/FAIL verdict

---

## How to Run

### End-to-end (recommended)

```bash
micromamba activate spectral-test
./tests/visual_eval/run_eval.sh
```

This runs Phase 1, the Rust export test, and Phase 2 in sequence, then prints a
per-dataset PASS/FAIL summary.

### Phase 1 only

```bash
micromamba activate spectral-test

# All Tier 1 datasets
python tests/visual_eval/generate_umap_comparisons.py --phase baseline

# Single dataset
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --dataset circles_1000

# Custom output directory
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --output-dir /tmp/visual_eval
```

### Phase 2 only (requires Phase 1 artifacts and Rust export)

```bash
micromamba activate spectral-test

# All datasets with available rust_init.npy
python tests/visual_eval/generate_umap_comparisons.py --phase compare

# Single dataset
python tests/visual_eval/generate_umap_comparisons.py --phase compare --dataset circles_1000
```

---

## Output Files

For each dataset `{name}`, Phase 1 writes the following files to `tests/visual_eval/output/`:

| File | Shape / Type | Description |
|------|-------------|-------------|
| `{name}_input.png` | figure | Raw input data visualization (the actual dataset before UMAP) |
| `{name}_graph.npz` | CSR sparse | Fuzzy k-NN graph in Rust-compatible format (f32 data, i32 indices/indptr) |
| `{name}_py_spectral.npy` | `(n, 2)` float64 | Python spectral init coordinates (pre-SGD) |
| `{name}_py_final.npy` | `(n, 2)` float32 | Python UMAP final embedding |
| `{name}_labels.npy` | `(n,)` int32 | Dataset labels for coloring |
| `{name}_baseline.png` | 2x2 figure | Spectral init / final / eigenvalue spectrum / metrics |

Phase 2 reads `{name}_rust_init.npy` (written by the Rust export test) and adds:

| File | Shape / Type | Description |
|------|-------------|-------------|
| `{name}_comparison.png` | 2x3 figure | Pre-SGD inits (top row) and post-SGD embeddings (bottom row) |
| `{name}_overlay.png` | 1x1 figure | Python vs Rust SGD embeddings overlaid (blue=Python, red=Rust) |
| `{name}_metrics.json` | JSON | Trustworthiness, silhouette, Procrustes, pairwise-dist correlation + PASS/FAIL |

The `output/` directory is gitignored (files are large and regenerated locally).
Only `output/.gitkeep` is tracked to ensure the directory exists after a fresh clone.

---

## Tier 1 Datasets

| Name | Generator | n | Features | Classes | What it tests |
|------|-----------|---|----------|---------|---------------|
| `blobs_1000` | `make_blobs(centers=5)` | 1000 | 10 | 5 | Sanity check: well-separated Gaussian clusters |
| `circles_1000` | `make_circles(factor=0.5, noise=0.05)` | 1000 | 2 | 2 | Nested topology: concentric rings |
| `swiss_roll_2000` | `make_swiss_roll(noise=0.0)` | 2000 | 3 | 5 | Geodesic vs Euclidean: manifold unrolling |
| `two_moons_1000` | `make_moons(noise=0.05)` | 1000 | 2 | 2 | Non-linear separability: interlocking arcs |
| `blobs_hd_2000` | `make_blobs(centers=10, n_features=50)` | 2000 | 50 | 10 | Scalability: high-dimensional eigensolver |

## Tier 2 Datasets (Real-World)

Tier 2 datasets provide a more rigorous stress test using real image data.
They require a one-time download; sklearn caches them in `~/scikit_learn_data/`.

| Name | Source | n | Features | Classes | Notes |
|------|--------|---|----------|---------|-------|
| `mnist_10k` | `fetch_openml('mnist_784')` | 10,000 | 784 | 10 digits | Subsampled from 70k; reproducible (seed=42) |
| `fashion_mnist_10k` | `fetch_openml('Fashion-MNIST')` | 10,000 | 784 | 10 categories | Harder; shirt/coat/pullover overlap |
| `pendigits` | `load_digits()` | 1,797 | 64 | 10 digits | Built-in; no download needed |

### Download and Caching

MNIST and Fashion-MNIST are downloaded from OpenML on first use and cached in
`~/scikit_learn_data/` (sklearn's default). Subsequent runs use the cache.

To use a custom cache directory:

```bash
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --tier 2 --data-dir /path/to/cache
```

---

## Running by Tier

```bash
# Tier 1 only (fast, synthetic, no download)
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --tier 1

# Tier 2 only (real-world; downloads MNIST and Fashion-MNIST on first run)
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --tier 2

# All datasets (default)
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --tier all
python tests/visual_eval/generate_umap_comparisons.py --phase baseline   # same as --tier all
```

---

## Tier 3 — Single-Cell (Placeholder)

A `singlecell` dataset stub exists for future use. It requires `scanpy` and
an `.h5ad` file, and currently raises `NotImplementedError`:

```bash
python tests/visual_eval/generate_umap_comparisons.py --phase baseline \
    --dataset singlecell --singlecell-path data/pbmc3k.h5ad
# NotImplementedError: Single-cell loading requires scanpy and a .h5ad file ...
```

---

## Python Environment Setup

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
then create the environment from the shared spec file:

```bash
micromamba create -f tests/environment.yml
micromamba activate spectral-test
```

Key packages: Python 3.11, NumPy 2.2, SciPy 1.15, umap-learn 0.5.11, scikit-learn, matplotlib.
