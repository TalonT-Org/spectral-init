# MERFISH Spatial Transcriptomics UMAP Benchmark Plan

## Comprehensive Research Plan: Python vs Rust UMAP on Allen Brain Cell Atlas MERFISH Data

**Date:** 2026-03-28 (updated 2026-03-29)
**Dataset:** Zhuang-ABCA-1 (Allen Brain Cell Atlas MERFISH, single adult mouse brain)
**Goal:** Evaluate whether Rust `spectral_init()` produces equivalent or superior UMAP embeddings compared to Python `umap-learn`, using physical spatial coordinates as ground truth for structure preservation assessment.

### Scope: 10K-First Approach

This plan is structured around an **incremental 10K-first strategy**. We start with a 10,000-cell subset only, build the full pipeline end-to-end at that scale, validate all tooling and metrics, and only then scale to larger subsets (100K, 500K, 1M, full). The 10K subset is small enough to commit to the repo (~15-25 MB compressed) and iterate rapidly. Larger subsets are retrieved at runtime via download scripts and are never tracked in git.

### Data Library Conventions

- **Polars** for all heavy data loading (CSV metadata, expression matrices, joins)
- **pandas** for metric result tables and benchmark summary DataFrames (small, tabular output)
- All metric computation results are stored as pandas DataFrames and serialized to CSV/JSON

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Glossary](#2-glossary)
3. [Dataset Overview](#3-dataset-overview)
4. [Codebase Context](#4-codebase-context)
5. [Phase 0: Data Infrastructure](#5-phase-0-data-infrastructure)
6. [Phase 1: Preprocessing Pipeline](#6-phase-1-preprocessing-pipeline)
7. [Phase 2: Baseline UMAP Runs](#7-phase-2-baseline-umap-runs)
8. [Phase 3: Quality Metrics](#8-phase-3-quality-metrics)
9. [Phase 4: Comparative Analysis](#9-phase-4-comparative-analysis)
10. [Metric Catalog](#10-metric-catalog)
11. [Tooling & Library Reference](#11-tooling--library-reference)
12. [Research Questions Decomposition](#12-research-questions-decomposition)
13. [Pre-work Checklist](#13-pre-work-checklist)
14. [Risk Register](#14-risk-register)
15. [Issue Decomposition](#15-issue-decomposition)

---

## 1. Executive Summary

This plan describes a multi-phase benchmark study comparing Rust and Python UMAP spectral initialization on the MERFISH Zhuang-ABCA-1 spatial transcriptomics dataset (4.2 million cells, 1,122 genes, 147 coronal brain sections). The dataset uniquely provides physical spatial coordinates as ground truth, enabling novel evaluation metrics beyond standard DR quality measures.

The study extends the existing `spectral-init` visual evaluation pipeline (Tier 1 synthetic + Tier 2 real-world datasets) to Tier 3 single-cell data, adding spatial correlation metrics, scaling analysis, and performance benchmarking.

### Implementation Strategy

1. **Start with 10K cells only.** Build the entire pipeline end-to-end, validate every metric, and commit the small subset to the repo. Larger subsets come later as separate work.
2. **Answer pre-work research questions first** (Section 12) before writing implementation code. These determine fundamental choices (normalization, PCA dims, distance metric).
3. **Break into GitHub issues.** Each phase maps to a set of concrete, independently-implementable issues (Section 15).
4. **Manual evaluation, not CI.** This is a person-initiated evaluation with a dedicated nextest profile (`merfish-eval`), separate from automated test suites.
5. **Set up `docs/metrics/`** as a permanent home for metric definitions and evaluation methodology.

---

## 2. Glossary

| Term | Definition |
|---|---|
| **ARI** | Adjusted Rand Index — measures cluster label agreement corrected for chance. Range [-0.5, 1.0]; 1.0 = perfect agreement. |
| **CCF** | Allen Common Coordinate Framework — a 3D reference atlas for the mouse brain at 10 µm resolution. |
| **CHAOS** | A spatial coherence metric: mean 1-nearest-neighbor distance within each cluster, using physical coordinates. Lower = more spatially compact clusters. |
| **DR** | Dimensionality Reduction — the general category of methods (PCA, UMAP, t-SNE, etc.) that project high-dimensional data into fewer dimensions. |
| **EVD** | Eigenvalue Decomposition — factoring a matrix into eigenvalues and eigenvectors. |
| **H5AD** | HDF5-backed AnnData file format — the standard on-disk format for single-cell genomics data, storing expression matrices + metadata. |
| **HVG** | Highly Variable Genes — a feature selection step in single-cell analysis. Genes with high cell-to-cell expression variance are more informative for distinguishing cell types. In whole-transcriptome data (~20,000 genes), you typically select the top 2,000–3,000 HVGs before PCA. For MERFISH's targeted 1,122-gene panel, HVG filtering is usually unnecessary because the panel was already designed to include informative genes. |
| **kNN** | k-Nearest Neighbors — a graph where each point is connected to its k closest points by some distance metric. UMAP builds a fuzzy kNN graph as its first step. |
| **LCMC** | Local Continuity Meta-Criterion — a metric measuring neighborhood overlap at the optimal scale k, corrected for random baseline. |
| **LOBPCG** | Locally Optimal Block Preconditioned Conjugate Gradient — an iterative eigensolver for sparse matrices, used by spectral-init at Level 1. |
| **MERFISH** | Multiplexed Error-Robust Fluorescence In Situ Hybridization — a spatial transcriptomics technology that images RNA molecules directly in tissue, providing both gene expression and physical cell coordinates. |
| **Moran's I** | A spatial autocorrelation statistic. Values near +1 indicate similar values cluster spatially; near 0 indicates random; near -1 indicates dissimilar values cluster. |
| **NMI** | Normalized Mutual Information — an information-theoretic measure of cluster agreement. Range [0, 1]; 1.0 = perfect agreement. |
| **PAS** | Percentage of Abnormal Spots — fraction of cells whose spatial neighbors mostly have a different cluster label. Lower = better spatial coherence. |
| **PCA** | Principal Component Analysis — linear DR that finds orthogonal axes of maximum variance. Used as a preprocessing step before UMAP. |
| **RSS** | Resident Set Size — the portion of a process's memory held in RAM. Used as a memory benchmark metric. |
| **rSVD** | Randomized Singular Value Decomposition — an approximate eigensolver using random projections (Halko-Tropp algorithm). |
| **SGD** | Stochastic Gradient Descent — the optimization algorithm UMAP uses to refine its low-dimensional layout. |
| **SNA** | Spatial Neighbor Agreement — a metric we define here: the Jaccard similarity between a cell's k-nearest spatial neighbors and its k-nearest embedding neighbors. |
| **SNS** | Scale-Normalized Stress — a corrected version of normalized stress that is truly scale-invariant (2024). |
| **UMAP** | Uniform Manifold Approximation and Projection — a nonlinear DR method that preserves local neighborhood structure. |

### Key Insight: Why MERFISH is Uniquely Valuable

Unlike standard scRNA-seq benchmarks, MERFISH provides **physical tissue coordinates** for every cell. This means we can ask questions no other UMAP benchmark can answer:

- Do cells that are physically adjacent in the brain remain adjacent in UMAP space?
- Does the Rust spectral initialization preserve spatial tissue organization as well as Python's?
- At what cell count does the Rust eigensolver's performance advantage become meaningful?

### What "Python UMAP" and "Rust UMAP" Mean Here

- **Python path:** `umap.spectral.spectral_layout()` → `scipy.sparse.linalg.eigsh` (ARPACK) → Python UMAP SGD
- **Rust path:** `spectral_init()` (this crate's 6-level solver escalation) → Python UMAP SGD
- The SGD optimization phase is **identical** in both paths — only the spectral initialization differs

---

## 3. Dataset Overview

### 2.1 The Zhuang-ABCA-1 Dataset

| Property | Value |
|---|---|
| Animal | Single adult C57BL/6J mouse |
| Technology | MERFISH (Multiplexed Error-Robust FISH) |
| Sections | 147 coronal sections (150 imaged, 3 fractured excluded) |
| Total cells (QC-passed) | 4,208,156 |
| Cells with subclass annotations | 3.4 million |
| Cells with cluster annotations | 2.8 million |
| Cells with CCF 3D coordinates | 2.6 million |
| Gene panel | 1,122 genes |
| License | CC BY 4.0 |

### 2.2 Data Files on AWS S3

All data is publicly accessible via HTTP GET — no AWS credentials required.

| Component | S3 Path | Size | Format |
|---|---|---|---|
| Expression (log2) | `expression_matrices/Zhuang-ABCA-1/20230830/Zhuang-ABCA-1-log2.h5ad` | 2.13 GB | H5AD |
| Expression (raw) | `expression_matrices/Zhuang-ABCA-1/20230830/Zhuang-ABCA-1-raw.h5ad` | 1.19 GB | H5AD |
| Cell metadata | `metadata/Zhuang-ABCA-1/20241115/cell_metadata.csv` | 630 MB | CSV |
| Gene list | `metadata/Zhuang-ABCA-1/20241115/gene.csv` | 83 KB | CSV |
| Annotated metadata | `metadata/Zhuang-ABCA-1/20241115/views/cell_metadata_with_cluster_annotation.csv` | 831 MB | CSV |
| CCF coordinates | `metadata/Zhuang-ABCA-1-CCF/20230830/ccf_coordinates.csv` | 211 MB | CSV |

**Total download:** ~5.07 GB

**Base URL:** `https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/`

### 2.3 Coordinate Systems

The dataset provides **three** coordinate systems:

| System | Columns | Coverage | Description |
|---|---|---|---|
| Section coordinates | `x`, `y`, `z` | All 4.2M cells | Experimentally measured, rotated/aligned to CCF |
| Reconstructed | `x_reconstructed`, `y_reconstructed`, `z_reconstructed` | Subset | Per-section 2D registration corrections applied |
| CCF coordinates | `x_ccf`, `y_ccf`, `z_ccf` | 2.6M cells | Full 3D Allen CCFv3 space (mm), via ANTs registration |

**For this benchmark:** Use section coordinates (`x`, `y`) as primary spatial ground truth (available for all cells). CCF coordinates provide a secondary, higher-fidelity ground truth for the 2.6M registered cells.

### 2.4 Cell Metadata Columns

`cell_label`, `brain_section_label`, `feature_matrix_label`, `donor_label`, `donor_genotype`, `donor_sex`, `cluster_alias`, `x`, `y`, `z`, `subclass_confidence_score`, `cluster_confidence_score`, `high_quality_transfer`, `abc_sample_id`

### 2.5 Why Spatial Coordinates Work as Ground Truth

The Zhuang lab's Nature 2023 paper directly establishes: *"Transcriptomically more similar cell types are located closer to each other spatially — neuronal subclasses within the same class are mostly colocalized within the same major brain region."* Each of ~5,200 transcriptomic clusters maps to a restricted spatial territory. This correspondence is strong at the class/subclass level but weaker at fine-grained cluster level within some brain regions.

**Caveats:**
- Non-neuronal cells (oligodendrocytes, astrocytes) show different spatial patterns than neurons
- The `z=0` issue for anterior sections means ~1.6M cells have degraded spatial z-information
- Use 2D (x, y) within individual sections for the strongest spatial signal

---

## 4. Codebase Context

### 3.1 What `spectral-init` Already Provides

**Rust spectral initialization crate:**
- Sole public function: `spectral_init(graph, n_components, seed, data, config) -> Array2<f32>`
- Input: sparse CSR graph `CsMatI<f32, u32, usize>` (the UMAP fuzzy kNN graph)
- 6-level eigensolver escalation: Dense EVD → LOBPCG → Shift-Invert LOBPCG → LOBPCG+reg → rSVD → Forced Dense
- Quality gating via eigenpair residual thresholds at each level
- Two compute modes: `PythonCompat` (matches Python bit-for-bit) and `RustNative`

**Existing visual eval pipeline (`tests/visual_eval/`):**
- Phase 1 (`run_baseline`): Python UMAP generates graphs, spectral init, final embeddings, eigenvalue spectra
- Rust export (`export_rust_init.rs`): loads `*_graph.npz`, runs `spectral_init()`, writes `*_rust_init.npy`
- Phase 2 (`run_compare`): Three-way comparison (Python spectral → SGD, Rust spectral → SGD, Random → SGD)
- Metrics: trustworthiness, silhouette, Procrustes disparity, pairwise distance correlation
- PASS/FAIL gates per metric per dataset

**Existing datasets:**
- Tier 1 (synthetic): blobs_1000, circles_1000, swiss_roll_2000, two_moons_1000, blobs_hd_2000
- Tier 2 (real-world): mnist_10k, fashion_mnist_10k, pendigits
- Tier 3 (single-cell): `NotImplementedError` stub for `load_singlecell()`

**Graph interchange format:** `export_graph()` writes CSR as `.npz` with keys: `data` (f32), `indices` (i32), `indptr` (i32), `shape` (i32), `format` (b"csr")

**No PyO3/FFI bridge exists.** Rust and Python communicate exclusively through `.npz`/`.npy` files on disk.

### 3.2 What Does NOT Exist (Gaps This Study Fills)

| Gap | Impact |
|---|---|
| No MERFISH data loader | Need `load_merfish()` in `generate_umap_comparisons.py` |
| No spatial correlation metrics | Existing metrics treat embeddings as abstract point clouds |
| No timing instrumentation at scale | `run_baseline()` doesn't separate kNN time from spectral init time |
| No memory measurement | No RSS/heap tracking during pipeline stages |
| No subset generation infrastructure | Need spatially-stratified subsets at 10K–4.2M |
| No per-run mean normalization | Imaging runs need per-run mean RNA count normalization (not Harmony — see Section 5.5) |
| Solver never tested at >5K nodes | LOBPCG/rSVD scalability to 100K–4.2M is unvalidated |

### 3.3 Key Architecture Constraint

The Rust crate takes a **pre-built kNN graph**, not raw expression data. The full pipeline is:

```
Raw counts → Normalize → PCA → kNN graph (Python) → spectral_init (Rust or Python) → SGD (Python)
```

Everything before spectral_init and everything after it runs in Python regardless. The benchmark isolates the spectral initialization step.

---

## 5. Phase 0: Data Infrastructure

### Objective
Download, validate, and stage MERFISH data in formats consumable by both the Python visual eval pipeline and the Rust `spectral_init()` function.

### 4.1 Local Data Location

**Data directory:** `data/merfish-abca1/` (gitignored via `/data/` in `.gitignore`)

| File | Local Path | Size | Status |
|---|---|---|---|
| Expression (log2) | `data/merfish-abca1/Zhuang-ABCA-1-log2.h5ad` | 2,128,478,610 bytes (2.0 GB) | Downloaded, validated |
| Cell metadata | `data/merfish-abca1/cell_metadata.csv` | 631 MB | Downloaded, validated |
| Gene list | `data/merfish-abca1/gene.csv` | 83 KB | Downloaded, validated |
| CCF coordinates | `data/merfish-abca1/ccf_coordinates.csv` | 211 MB | Downloaded, validated |

**Source URLs (public S3, no credentials):**
```
https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/expression_matrices/Zhuang-ABCA-1/20230830/Zhuang-ABCA-1-log2.h5ad
https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/metadata/Zhuang-ABCA-1/20241115/cell_metadata.csv
https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/metadata/Zhuang-ABCA-1/20241115/gene.csv
https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/metadata/Zhuang-ABCA-1-CCF/20230830/ccf_coordinates.csv
```

### 4.2 Data Validation (Completed)

| Check | Result |
|---|---|
| H5AD byte count matches S3 listing | 2,128,478,610 bytes — exact match |
| Cell metadata row count | 2,846,908 cells |
| Unique brain sections | 147 |
| CCF coordinate row count | 2,616,328 cells |
| Gene list row count | 1,122 genes |
| Metadata columns | `cell_label`, `brain_section_label`, `feature_matrix_label`, `donor_label`, `donor_genotype`, `donor_sex`, `cluster_alias`, `x`, `y`, `z`, `subclass_confidence_score`, `cluster_confidence_score`, `high_quality_transfer`, `abc_sample_id` |
| Spatial coordinates present | x, y, z columns in cell_metadata.csv (section coordinates in mm) |
| CCF coordinates columns | `cell_label`, `x`, `y`, `z`, `parcellation_index` |
| cluster_alias available | Yes — integer cell-type cluster IDs |

**Note:** H5AD loading requires `anndata` + `h5py`, which are not yet in the `spectral-test` conda environment. These need to be added (see issue 1.4 below). The metadata CSVs are readable with Python stdlib or Polars immediately.

### 4.3 Subset Generation (10K First)

**Start with 10K only.** Larger subsets are future work after the full pipeline is validated.

| Subset | Cells | Status | Storage | Solver Level Expected |
|---|---|---|---|---|
| `merfish_10k` | 10,000 | **First priority** | Git (compressed .npz, ~15-25 MB) | Dense EVD (n < 2000) or LOBPCG L1 |
| `merfish_100k` | 100,000 | Future | Runtime download | LOBPCG L1 |
| `merfish_500k` | 500,000 | Future | Runtime download | LOBPCG L1 or rSVD L4 |
| `merfish_1m` | 1,000,000 | Future | Runtime download | LOBPCG/rSVD |
| `merfish_full` | ~4,200,000 | Future | Runtime download | Unknown |

**File size estimate for 10K subset:**

| File | Raw Size | Compressed |
|---|---|---|
| Expression matrix (10K × 1122, f32 dense) | 44.9 MB | ~15-25 MB (.npz) |
| Spatial coordinates (10K × 2, f32) | 78 KB | ~40 KB |
| Labels (10K, i32) | 39 KB | ~5 KB |
| Section IDs (10K, i32) | 39 KB | ~5 KB |
| kNN graph (CSR, k=15) | 1.2 MB | ~300 KB |
| **Total** | **~46 MB** | **~16-26 MB** |

The compressed 10K subset fits comfortably in git. Use `np.savez_compressed()` for the expression matrix. For Git LFS tracking, add `tests/visual_eval/merfish_data/*.npz` to `.gitattributes` if the compressed size exceeds 25 MB.

**Stratification strategy:** Divide the (x, y) bounding box into a 50×50 spatial grid. Sample cells proportionally from each grid cell. This preserves spatial autocorrelation structure and cell-type composition.

```python
import numpy as np

def spatial_stratified_subsample(coords, n_target, grid_size=50, seed=42):
    """Spatially stratified subsampling preserving spatial structure."""
    rng = np.random.default_rng(seed)
    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), grid_size + 1)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), grid_size + 1)

    x_idx = np.digitize(coords[:, 0], x_bins) - 1
    y_idx = np.digitize(coords[:, 1], y_bins) - 1
    bin_ids = x_idx * grid_size + y_idx

    # Proportional allocation per bin
    unique_bins, bin_counts = np.unique(bin_ids, return_counts=True)
    fractions = bin_counts / bin_counts.sum()
    per_bin_n = np.maximum(1, np.round(fractions * n_target).astype(int))

    indices = []
    for bid, n_sample in zip(unique_bins, per_bin_n):
        mask = np.where(bin_ids == bid)[0]
        n_sample = min(n_sample, len(mask))
        indices.extend(rng.choice(mask, n_sample, replace=False))

    return np.array(indices[:n_target])
```

### 4.4 File Storage Strategy

**Committed to git (small, reproducible):**
- `tests/visual_eval/merfish_data/merfish_10k_expression.npz` — compressed expression matrix
- `tests/visual_eval/merfish_data/merfish_10k_spatial.npy` — spatial coordinates
- `tests/visual_eval/merfish_data/merfish_10k_labels.npy` — cell-type labels
- `tests/visual_eval/merfish_data/merfish_10k_section_ids.npy` — section assignments
- `tests/visual_eval/merfish_data/merfish_10k_meta.json` — subset metadata

**Runtime-downloaded (large, never tracked):**
- Full MERFISH H5AD files (2+ GB each)
- 100K+ subsets
- All generated from a download/extraction script

**Generated at evaluation time (gitignored, in `tests/visual_eval/output/`):**
- kNN graphs, spectral inits, UMAP embeddings, metrics JSON, plots

### 4.5 Runtime Data Download Script

The download script is implemented at `tests/visual_eval/download_merfish.py`. It uses only Python stdlib (no anndata/polars dependency) and downloads to a **fixed absolute path** so that worktrees share the same data:

```bash
# Download all files (skips any already present and valid)
python tests/visual_eval/download_merfish.py

# Validate existing files without downloading
python tests/visual_eval/download_merfish.py --check
```

**Canonical data path (hardcoded, shared across worktrees):**
```
/home/talon/projects/spectral-init/data/merfish-abca1/
```

All code that references MERFISH data should use this absolute path, NOT a relative path. This ensures worktrees, subagents, and scripts all read from the same location without redundant downloads.

### 4.6 Dependencies

- **Blocks:** Phase 1 (cannot preprocess without data), Phase 2 (cannot run UMAP without subsets)
- **Pre-requisites:** None (root of dependency graph)

---

## 6. Phase 1: Preprocessing Pipeline

### Research Question
> What normalization, dimensionality reduction, and data formatting choices produce a kNN graph from MERFISH expression data that captures biologically meaningful structure?

### 5.1 Normalization

**Recommended pipeline (matching Zhuang lab conventions, Zhang et al. 2021/2023):**

```python
import scanpy as sc

# 1. Library size normalization (Zhuang lab uses target_sum=1000 for integration)
sc.pp.normalize_total(adata, target_sum=1e3)

# 2. Log transform
sc.pp.log1p(adata)

# 3. Store normalized counts
adata.layers["log_normalized"] = adata.X.copy()
```

**Why `target_sum=1e3`:** The Zhuang lab integration scripts use `target_sum=1000` for MERFISH. This differs from scRNA-seq conventions (1e4 or 1e6) because MERFISH measures absolute RNA molecule counts per cell, not sequencing-depth-dependent read counts. The total counts per cell are much lower in MERFISH.

**Note on the distributed log2 H5AD:** The `Zhuang-ABCA-1-log2.h5ad` file uses `log2(count + 1)` on raw counts directly (no library-size normalization). If using the pre-computed log2 file, you skip `normalize_total` and `log1p` — but the scanpy pipeline above (normalize_total → log1p) is what the Zhuang lab actually used for their integration and UMAP. RQ-1.1 should determine which starting point produces a better spectral gap.

**Why NOT SCTransform:** It better handles mean-variance in smFISH but is computationally prohibitive at 4.2M cells in Python. Normalize_total + log1p is dramatically more tractable.

### 5.2 Feature Selection (HVG Filtering)

**HVG = Highly Variable Genes.** In single-cell genomics, not all genes are equally informative. Housekeeping genes (e.g., GAPDH) are expressed at similar levels across all cell types — they add noise, not signal. HVG filtering selects genes with high cell-to-cell variance, which are the genes that distinguish cell types from each other. In a typical whole-transcriptome scRNA-seq experiment (~20,000 genes), you select the top 2,000–3,000 HVGs before PCA to reduce noise and computation.

**For the 1,122-gene MERFISH panel: skip HVG filtering or apply very permissively.**

The MERFISH panel is fundamentally different from whole-transcriptome data — it was already designed by domain experts to include specifically informative genes. The 1,122 genes were chosen because they are biologically variable and distinguish cell types. Applying aggressive HVG filtering on top of this curated panel risks discarding intentionally included marker genes. The full 1,122-gene panel is already smaller than the typical HVG cutoff.

Options:
- **Option A (recommended):** Use all 1,122 genes for PCA — the panel is already curated
- **Option B:** Permissive HVG retaining top 1,000 genes by normalized dispersion (removes only truly invariant genes that happened to be included in the panel but show no variance in this particular sample)

### 5.3 Scaling

```python
sc.pp.scale(adata, max_value=10)  # Zero-mean, unit-variance, clip outliers
```

**Memory warning:** `scale()` forces dense conversion. At 4.2M × 1,122 float32: ~18.8 GB dense. For full-scale runs, use `sc.tl.pca(adata, zero_center=False)` to avoid dense materialization.

### 5.4 PCA

```python
sc.tl.pca(adata, n_comps=50, svd_solver='randomized')
```

**Parameter sweep on 10K subset:** Test n_pcs ∈ {10, 20, 30, 50}. For a 1,122-gene MERFISH panel, the intrinsic dimensionality is lower than whole-transcriptome data. The Allen Brain Atlas mouse MERFISH analysis uses ~20-30 PCs.

**Selection criterion:** Maximize spectral gap (`lambda_2 - lambda_1`) of the resulting kNN graph Laplacian. The spectral gap directly predicts how well spectral init captures structure.

At 4.2M cells, use `svd_solver='randomized'` (Halko algorithm). ARPACK is memory-prohibitive at this scale.

### 5.5 Batch Correction — NOT Recommended for This Dataset

**Do NOT apply Harmony (or Scanorama, scVI, etc.) between brain sections.**

This was investigated thoroughly (see research findings below) and the conclusion is unanimous across the literature, the Zhuang lab's own pipeline, and the Allen Brain Cell Atlas:

**Why not:**
1. **Batch-biology confounding.** The 147 coronal sections span the entire mouse brain. Each section samples a distinct neuroanatomical region — cortex, hippocampus, hypothalamus, cerebellum, brainstem. Telling Harmony that `section_id` is a batch variable asks it to make cortical neurons and cerebellar neurons look more similar in PCA space. That destroys real biology, not technical noise.
2. **The Zhuang lab doesn't do it.** The actual pipeline (Zhang et al. 2021, 2023) uses simple per-imaging-run mean normalization to handle the modest ~30% variation in mean RNA counts between imaging runs. No Harmony.
3. **Technical variation is small.** MERFISH replicates from the same tissue block correlate at R ≥ 0.95-0.99. There is little technical noise to remove relative to the large biological differences between brain regions.
4. **Overcorrection is silent.** A Harmony-corrected UMAP will still look clean and aesthetically pleasing — but it will show artificially homogenized cell populations with collapsed regional diversity. The damage is invisible without careful marker-gene validation.

**What the Zhuang lab does instead (Zhang 2021):**
- Normalize mean total RNA counts per cell to a common value per imaging run (each run = 4-6 sections). This addresses the main source of technical batch variation.
- This is a simple scalar correction, not a PCA-space transformation.

**When Harmony IS appropriate for MERFISH:**
- Sections from the **same narrow anatomical region** (e.g., 5 adjacent hypothalamus sections spanning 0.2mm, as in the Moffitt 2018 dataset). Cell-type composition is approximately constant across neighboring sections, so Harmony can safely target technical variation without confounding.
- Cross-modal integration (MERFISH ↔ scRNA-seq). The Zhuang lab uses CCA-based integration for this purpose, not Harmony.

**References:**
- Zhang et al. 2021 (Nature, motor cortex): no Harmony, uses per-run mean normalization
- Zhang/Yao et al. 2023 (Nature, whole brain): no Harmony, uses CCA for cross-modal integration
- Crescendo 2025 (Genome Biology): documents overcorrection risk, batch effects "smaller in magnitude compared to cell-type variance"
- GraphST 2023 (Nature Communications): "batch correction methods developed for scRNA-seq are not suitable for spatial transcriptomics"

### 5.6 kNN Graph Construction

```python
sc.pp.neighbors(
    adata,
    n_neighbors=15,
    n_pcs=30,
    use_rep='X_pca',
    metric='euclidean'
)
```

**Distance metric:** The Zhuang lab uses euclidean (scanpy default) on scaled PCA coordinates. Cosine is an alternative for unscaled log-normalized data. Post-scaling, cosine and euclidean are approximately equivalent since scaling removes magnitude differences. We use euclidean to match the published pipeline; RQ-1.4 can compare both on the 10K subset.

### 5.7 Graph Export for Rust

Use the existing `export_graph()` function from `generate_umap_comparisons.py`:

```python
def export_graph(graph: scipy.sparse.csr_matrix, path: Path) -> None:
    g = graph.tocsr()
    np.savez(path, data=g.data.astype(np.float32),
             indices=g.indices.astype(np.int32),
             indptr=g.indptr.astype(np.int32),
             shape=np.array(g.shape, dtype=np.int32),
             format=b"csr")
```

### 5.8 Preprocessing Parameter Sweep Design

Test on `merfish_10k` subset, validate on `merfish_100k`:

| Parameter | Values to Test |
|---|---|
| Normalization | `normalize_total+log1p`, raw counts, Pearson residuals |
| n_pcs | 10, 20, 30, 50 |
| Batch correction | Per-run mean normalization (no Harmony) |
| kNN metric | cosine, euclidean |
| n_neighbors | 10, 15, 30 |

**Winner selection:** Maximize spectral gap subject to trustworthiness > 0.85 and silhouette > 0.2.

### 5.9 Output Artifacts Per Configuration

- `merfish_{n}_{config}_graph.npz` — fuzzy kNN graph in Rust CSR format
- `merfish_{n}_{config}_py_spectral.npy` — Python spectral init coordinates
- `merfish_{n}_{config}_labels.npy` — cell-type labels
- `merfish_{n}_{config}_spatial.npy` — spatial coordinates (x, y)
- `merfish_{n}_{config}_eigenspectrum.json` — top 10 eigenvalues, spectral gap, condition number
- `merfish_{n}_{config}_baseline.png` — 2×2 baseline plot

### 5.10 Memory-Efficient Processing Strategy

For full 4.2M cells:

| Stage | Strategy |
|---|---|
| Load H5AD | `backed='r'` (memory-mapped) |
| Metadata joins | Polars (5-30x faster than pandas) |
| Normalization | In-place sparse operations |
| PCA | `svd_solver='randomized'`, `zero_center=False` to avoid dense |
| Per-run normalization | Scalar correction per imaging run (~30 runs × 1 scalar) |
| kNN | PyNNDescent (approximate, O(n log n)) |

**GPU option:** `rapids-singlecell` provides drop-in GPU replacements for all scanpy preprocessing functions, achieving 10-350x speedups on million-cell datasets.

---

## 7. Phase 2: Baseline UMAP Runs

### Research Question
> How does Rust `spectral_init` + Python SGD compare to Python spectral init + Python SGD on MERFISH data at each scale, and what are the timing and memory profiles?

### 6.1 Pipeline for Each Subset

```
For each n ∈ {10K, 100K, 500K, 1M, full}:

  [Python Phase 1]
  1. Load preprocessed subset
  2. Build kNN graph → export_graph() → merfish_{n}_graph.npz
  3. Run Python spectral init → merfish_{n}_py_spectral.npy
  4. Run Python UMAP SGD (from spectral init) → merfish_{n}_py_final.npy
  5. Run Python UMAP SGD (from random init) → merfish_{n}_random_final.npy
  6. Record timing and memory at each step

  [Rust Export]
  7. cargo test --test export_rust_init --features testing -- --ignored
     → loads merfish_{n}_graph.npz
     → runs spectral_init()
     → writes merfish_{n}_rust_init.npy
  8. Record Rust timing and peak RSS via /usr/bin/time -v

  [Python Phase 2]
  9. Load rust_init.npy
  10. Run Python UMAP SGD (from Rust init) → merfish_{n}_rust_final.npy
  11. Compute comparison metrics (3-way: Python, Rust, Random)
  12. Generate comparison plots
```

### 6.2 Timing Instrumentation

**Separate these phases in timing:**

| Phase | Python Tool | Rust Tool |
|---|---|---|
| kNN construction | `time.perf_counter()` | N/A (runs in Python) |
| Graph symmetrization | `time.perf_counter()` | N/A |
| Spectral init | `time.perf_counter()` | `std::time::Instant` + solver level timing |
| SGD optimization | `time.perf_counter()` | N/A (runs in Python) |
| Total end-to-end | `time.perf_counter()` | N/A |

The Rust spectral init timing is the core benchmark. The existing `[timing:level_N]` instrumentation in `src/solvers/mod.rs` (behind `#[cfg(feature = "testing")]`) provides per-solver-level wall-clock measurements.

### 6.3 Memory Instrumentation

```python
import resource
import psutil

# Before spectral init
rss_before = psutil.Process().memory_info().rss / (1024**2)  # MB

# ... run spectral init ...

rss_after = psutil.Process().memory_info().rss / (1024**2)
peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB on Linux
```

For Rust: `/usr/bin/time -v cargo test ... 2>&1 | grep "Maximum resident set size"`

### 6.4 Solver Level Recording

The Rust `solve_eigenproblem_pub()` test seam exposes which solver level was reached. Record this for each subset — it tells us whether the LOBPCG/rSVD chain scales correctly:

| Expected | n | Solver |
|---|---|---|
| Dense EVD | < 2,000 | Level 0 |
| LOBPCG | 2,000–100K | Level 1 |
| LOBPCG or rSVD | 100K–1M | Level 1 or 4 |
| Unknown | > 1M | First time at this scale |

### 6.5 Checkpointing

For 500K+ cell runs (hours per run), checkpoint intermediate results:
- If `merfish_{n}_graph.npz` exists → skip kNN construction
- If `merfish_{n}_py_spectral.npy` exists → skip Python spectral init
- If `merfish_{n}_rust_init.npy` exists → skip Rust export

### 6.6 Expected Timing Estimates

Based on published benchmarks and the solver complexity:

| Subset | Python kNN | Python spectral | Rust spectral | Python SGD |
|---|---|---|---|---|
| 10K | ~5s | ~1s | ~0.5s | ~10s |
| 100K | ~2min | ~30s | ~5-15s | ~5min |
| 500K | ~15min | ~5-10min | ~1-3min | ~30min |
| 1M | ~30-60min | ~20-40min | ~5-15min | ~1-2hr |
| 4.2M | ~2-4hr | ~2-8hr | Unknown | ~4-8hr |

These are rough estimates. The Rust advantage grows with n because LOBPCG scales better than ARPACK for the specific eigenvalue problem (few smallest eigenvalues of a sparse Laplacian).

### 6.7 Output Artifacts Per Subset

- `merfish_{n}_graph.npz` — kNN graph
- `merfish_{n}_py_spectral.npy` — Python spectral init (pre-SGD)
- `merfish_{n}_py_final.npy` — Python UMAP final (post-SGD)
- `merfish_{n}_rust_init.npy` — Rust spectral init
- `merfish_{n}_rust_final.npy` — Rust init → Python SGD final
- `merfish_{n}_random_final.npy` — Random init → Python SGD final
- `merfish_{n}_timing.json` — per-phase wall-clock breakdown
- `merfish_{n}_memory.json` — peak RSS at each stage
- `merfish_{n}_solver_info.json` — solver level, residuals, escalation history
- `merfish_{n}_baseline.png`, `merfish_{n}_comparison.png`, `merfish_{n}_overlay.png`

---

## 8. Phase 3: Quality Metrics

### Research Question
> To what degree do Python and Rust UMAP embeddings of MERFISH data preserve (a) local neighborhood structure, (b) spatial tissue organization, and (c) cell-type cluster structure?

### 7.1 Category A: Structure Preservation (Existing Metrics)

These are already implemented in `_compute_metrics()` and require no new code:

| Metric | Formula | Threshold | Measures |
|---|---|---|---|
| Trustworthiness | `sklearn.manifold.trustworthiness(X, emb, n_neighbors=15)` | delta < 0.01 | Local neighborhood preservation |
| Silhouette | `sklearn.metrics.silhouette_score(emb, labels)` | delta < 0.05 | Cluster separation quality |
| Procrustes disparity | `scipy.spatial.procrustes(emb_py, emb_rust)[2]` | < 0.05 | Geometric shape similarity |
| Pairwise distance correlation | `pearsonr(pdist(emb_py), pdist(emb_rust))` | > 0.99 | Relative distance preservation |

### 7.2 Category B: Spatial Correlation Metrics (NEW — Core Novelty)

These use physical spatial coordinates as ground truth. **This is the unique contribution of this study.**

#### B1. Spatial Neighbor Agreement (SNA)

For each cell, compute the Jaccard similarity between its k-nearest spatial neighbors (using physical coordinates) and its k-nearest embedding neighbors.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def spatial_neighbor_agreement(spatial_coords, embedding, k=15):
    """Fraction of spatial neighbors preserved in embedding space."""
    nn_spatial = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(spatial_coords)
    _, idx_spatial = nn_spatial.kneighbors(spatial_coords)
    idx_spatial = idx_spatial[:, 1:]  # exclude self

    nn_embed = NearestNeighbors(n_neighbors=k+1).fit(embedding)
    _, idx_embed = nn_embed.kneighbors(embedding)
    idx_embed = idx_embed[:, 1:]

    jaccard = np.mean([
        len(set(idx_spatial[i]) & set(idx_embed[i])) / len(set(idx_spatial[i]) | set(idx_embed[i]))
        for i in range(len(spatial_coords))
    ])
    return jaccard
```

**Range:** [0, 1]. Higher = better spatial preservation. Expected: 0.05–0.30 for UMAP (UMAP is designed for expression-space structure, not physical-space structure, so perfect spatial preservation is neither expected nor desired).

#### B2. Spatial Distance Rank Correlation

```python
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

def spatial_distance_correlation(spatial_coords, embedding, sample_size=5000, seed=42):
    """Spearman rank correlation between spatial and embedding distances."""
    rng = np.random.default_rng(seed)
    n = len(spatial_coords)
    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        spatial_coords = spatial_coords[idx]
        embedding = embedding[idx]

    d_spatial = pdist(spatial_coords)
    d_embed = pdist(embedding)
    return spearmanr(d_spatial, d_embed).correlation
```

**Range:** [-1, 1]. Higher = better global spatial preservation.

#### B3. Moran's I of Embedding Coordinates

Treat each UMAP dimension as a spatial signal. High Moran's I means the embedding coordinate varies smoothly across the tissue.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def morans_i(spatial_coords, values, k=6):
    """Global Moran's I for a vector of values against spatial neighbors."""
    n = len(values)
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(spatial_coords)
    _, indices = nn.kneighbors(spatial_coords)
    indices = indices[:, 1:]  # exclude self

    z = values - values.mean()
    numerator = 0.0
    for i in range(n):
        for j in indices[i]:
            numerator += z[i] * z[j]

    W = n * k  # total number of neighbor pairs
    denominator = np.sum(z ** 2)
    return (n / W) * (numerator / denominator)

# Compute for each UMAP dimension
mi_dim0 = morans_i(spatial_coords, embedding[:, 0])
mi_dim1 = morans_i(spatial_coords, embedding[:, 1])
```

**Range:** [-1, 1]. Values near 1 = strong positive spatial autocorrelation (expected for good embeddings). Compare: `max(MI_dim0, MI_dim1)` for Python vs Rust.

#### B4. Per-Section Spatial Coherence (CHAOS Score)

For each cell-type cluster, compute mean 1-nearest-neighbor edge length in physical space within that cluster. Lower = more spatially compact clusters.

```python
def chaos_score(spatial_coords, labels):
    """CHAOS: spatial continuity of clusters."""
    from sklearn.neighbors import NearestNeighbors
    unique_labels = np.unique(labels)
    total_score = 0.0
    total_weight = 0

    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() < 2:
            continue
        coords_cluster = spatial_coords[mask]
        nn = NearestNeighbors(n_neighbors=2).fit(coords_cluster)
        dists, _ = nn.kneighbors(coords_cluster)
        mean_1nn_dist = dists[:, 1].mean()
        total_score += mean_1nn_dist * mask.sum()
        total_weight += mask.sum()

    return total_score / total_weight if total_weight > 0 else float('inf')
```

#### B5. Percentage of Abnormal Spots (PAS)

For each cell, check if the majority of its spatial neighbors have a different cluster label.

```python
def pas_score(spatial_coords, labels, k=10, threshold=0.6):
    """Percentage of spots whose spatial neighbors disagree on label."""
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(spatial_coords)
    _, indices = nn.kneighbors(spatial_coords)
    indices = indices[:, 1:]

    abnormal = 0
    for i in range(len(labels)):
        neighbor_labels = labels[indices[i]]
        same_label_frac = np.mean(neighbor_labels == labels[i])
        if same_label_frac < (1 - threshold):
            abnormal += 1

    return abnormal / len(labels)
```

### 7.3 Category C: Cluster Preservation Metrics (NEW)

#### C1. Adjusted Rand Index (ARI)

Apply Leiden clustering to both the original kNN graph and the embedding's kNN graph. Measure agreement.

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Labels from original graph clustering vs embedding clustering
ari = adjusted_rand_score(labels_original, labels_embedding)
nmi = normalized_mutual_info_score(labels_original, labels_embedding)
```

#### C2. Cell-Type Purity

For each annotated cell type, compute the fraction of its cells that fall in the dominant Leiden cluster.

#### C3. Random Triplet Accuracy

Tests whether distance ordering between triples of points is preserved.

```python
def random_triplet_accuracy(X_high, X_low, n_triplets=50000, seed=42):
    """Fraction of triplets where distance ordering is preserved."""
    rng = np.random.default_rng(seed)
    n = X_high.shape[0]
    D_high = squareform(pdist(X_high[:min(n, 5000)]))
    D_low = squareform(pdist(X_low[:min(n, 5000)]))

    triplets = rng.integers(0, min(n, 5000), size=(n_triplets, 3))
    mask = (triplets[:, 0] != triplets[:, 1]) & \
           (triplets[:, 1] != triplets[:, 2]) & \
           (triplets[:, 0] != triplets[:, 2])
    triplets = triplets[mask]
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

    order_high = D_high[i, j] < D_high[i, k]
    order_low = D_low[i, j] < D_low[i, k]
    return np.mean(order_high == order_low)
```

**Range:** [0.5, 1.0] (0.5 = random baseline). UMAP typically 0.55–0.70.

### 7.4 Category D: Global Structure Metrics (NEW)

#### D1. Shepard Diagram Correlation

```python
def shepard_correlation(X_high, X_low, sample_n=2000, seed=42):
    """Pearson and Spearman correlation of pairwise distances."""
    rng = np.random.default_rng(seed)
    n = X_high.shape[0]
    if n > sample_n:
        idx = rng.choice(n, sample_n, replace=False)
        X_high, X_low = X_high[idx], X_low[idx]

    d_high = pdist(X_high)
    d_low = pdist(X_low)
    return {
        "pearson": np.corrcoef(d_high, d_low)[0, 1],
        "spearman": spearmanr(d_high, d_low).correlation,
    }
```

#### D2. Centroid Distance Correlation

Cheaper global metric: correlate inter-cluster centroid distances.

```python
def centroid_distance_correlation(X_high, X_low, labels):
    """Correlation between inter-cluster centroid distances."""
    unique_labels = np.unique(labels)
    centroids_high = np.array([X_high[labels == l].mean(axis=0) for l in unique_labels])
    centroids_low = np.array([X_low[labels == l].mean(axis=0) for l in unique_labels])
    return spearmanr(pdist(centroids_high), pdist(centroids_low)).correlation
```

### 7.5 Scalability Strategy for Metrics at 4.2M Cells

| Metric | Complexity | Strategy for 4.2M cells |
|---|---|---|
| Trustworthiness | O(n²) | Stratified subsample of 20K–50K cells |
| Silhouette | O(n²) | Subsample 50K cells |
| Procrustes | O(n) | Full dataset (trivially scalable) |
| Pairwise distance corr | O(n²) | Subsample 5K cells |
| SNA (spatial neighbor agreement) | O(n·k) | Full dataset (kd-tree is fast for 2D) |
| Spatial distance correlation | O(n²) | Subsample 5K cells |
| Moran's I | O(n·k) | Full dataset with sparse weights |
| CHAOS | O(n·k per cluster) | Full dataset |
| PAS | O(n·k) | Full dataset |
| ARI/NMI | O(n) | Full dataset |
| Random triplet accuracy | O(n_triplets) | Subsample 5K, 50K triplets |
| Shepard correlation | O(n²) | Subsample 2K–5K |
| Centroid distance | O(c²) | Full (only c clusters) |

### 7.6 Output Artifacts

For each subset:
- `merfish_{n}_metrics_full.json` — all A/B/C/D metrics for Python, Rust, and Random init
- `merfish_{n}_spatial_ground_truth.png` — cells colored by cell type in physical space
- `merfish_{n}_embedding_py.png` — Python UMAP colored by cell type
- `merfish_{n}_embedding_rust.png` — Rust UMAP colored by cell type
- `merfish_{n}_spatial_correlation.png` — spatial distance vs embedding distance scatter
- `merfish_{n}_morans_i.json` — per-dimension Moran's I for each embedding

---

## 9. Phase 4: Comparative Analysis

### Research Question
> Is Rust spectral initialization an acceptable substitute for Python spectral initialization on MERFISH data, and at what scale does the runtime advantage become meaningful?

### 8.1 Analysis Sections

**Section 1: Scaling Study (Performance)**

Plot wall-clock time vs. n_cells (log-log) for:
- Python kNN construction
- Python spectral init
- Rust spectral init
- Python SGD optimization

Fit power law: `T = a · n^b` via `scipy.stats.linregress(np.log(sizes), np.log(times))`.

Speedup ratio plot: `time_python_spectral / time_rust_spectral` vs. n_cells. Expected: increasing speedup with n.

**Section 2: Quality Gate (Pass/Fail Table)**

Extend the existing PASS/FAIL format to all metric categories:

| Metric | 10K | 100K | 500K | 1M | Full |
|---|---|---|---|---|---|
| Trustworthiness Δ < 0.01 | ? | ? | ? | ? | ? |
| Silhouette Δ < 0.05 | ? | ? | ? | ? | ? |
| Procrustes < 0.05 | ? | ? | ? | ? | ? |
| SNA (Rust ≥ Python - 0.02) | ? | ? | ? | ? | ? |
| Spatial dist corr (Rust ≥ Python - 0.02) | ? | ? | ? | ? | ? |
| Moran's I (Rust ≥ Python - 0.05) | ? | ? | ? | ? | ? |
| ARI > 0.90 | ? | ? | ? | ? | ? |

**Key gate:** Does Rust spectral init ever fail a quality metric that Python passes, at any scale?

**Section 3: Spatial Correlation Deep Dive**

- Plot SNA vs n_cells for Python, Rust, Random
- If Rust lags: apply Procrustes alignment first, then recompute SNA — if alignment restores the score, it's pure rotational/reflection ambiguity, not topological divergence
- Per-section SNA analysis: which brain regions show the largest Python-vs-Rust difference?
- Moran's I comparison: scatter plot of per-gene Moran's I (Python embedding) vs (Rust embedding)

**Section 4: Memory Efficiency**

- Peak RSS vs n_cells for both implementations
- Memory-per-point: `peak_rss / n_cells`
- At largest scale: does Rust's lower memory footprint (no Python interpreter, no GIL copies) matter?

**Section 5: Solver Escalation Analysis**

- Which solver level did Rust use at each scale?
- Did any subset trigger escalation beyond Level 1 (LOBPCG)?
- Correlation between solver level and embedding quality?
- Eigenpair residuals vs subset size

### 8.2 Statistical Significance

With a single dataset, there is only one set of embeddings per configuration — no replicates at the dataset level. Significance testing is meaningful for:
- **Leiden clustering:** Multiple random seeds (10 per configuration) → Wilcoxon signed-rank test on ARI values
- **Spatial sampling:** Multiple subsamples for correlation metrics → bootstrap confidence intervals
- **Triplet accuracy:** Different random triplet samples → standard error of mean

**Do NOT overstate** statistical conclusions from a single biological sample.

### 8.3 Visualization Recommendations

| Plot | X-axis | Y-axis | Purpose |
|---|---|---|---|
| Scaling (log-log) | n_cells | Wall time (s) | Show asymptotic scaling |
| Speedup ratio | n_cells | time_py / time_rust | Show Rust advantage vs scale |
| Quality heatmap | Metric | Scale | Color = delta (Python - Rust) |
| Spatial correlation scatter | Python SNA | Rust SNA | One point per section |
| Memory profile | n_cells | Peak RSS (MB) | Compare memory footprints |
| Solver level | n_cells | Level reached | Show escalation pattern |
| Moran's I scatter | MI (Python emb) | MI (Rust emb) | Per-gene, colored by gene class |

**For 4.2M-point embedding plots:** Use `datashader` for rasterization. Standard matplotlib scatterplots fail above ~100K points.

```python
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd

df = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'label': labels})
canvas = ds.Canvas(plot_width=800, plot_height=800)
agg = canvas.points(df, 'x', 'y', ds.count_cat('label'))
img = tf.shade(agg)  # Returns PIL Image
img.to_pil().save('merfish_embedding.png')
```

### 8.4 Final Report Structure

The report (`merfish_benchmark_report.md`) should contain:

1. **Abstract** — One paragraph summary of findings
2. **Dataset** — MERFISH description, subset sizes
3. **Methods** — Preprocessing, UMAP parameters, metric definitions
4. **Results: Performance** — Scaling plots, speedup ratios, memory profiles
5. **Results: Quality** — Pass/fail table, metric values at each scale
6. **Results: Spatial** — Spatial correlation analysis, Moran's I, per-section breakdown
7. **Discussion** — Where Rust wins, where it's equivalent, any failure modes
8. **Appendix** — Full metric tables in machine-readable format

---

## 10. Metric Catalog

### 9.1 Complete Metric Summary

| # | Metric | Category | Local/Global | Needs Labels | Needs Spatial | Complexity | Library |
|---|---|---|---|---|---|---|---|
| 1 | Trustworthiness | A (existing) | Local | No | No | O(n²) | sklearn |
| 2 | Continuity | A (new impl) | Local | No | No | O(n²) | pyDRMetrics/ZADU |
| 3 | Silhouette score | A (existing) | Cluster | Yes | No | O(n²) | sklearn |
| 4 | Procrustes disparity | A (existing) | Global | No | No | O(n) | scipy |
| 5 | Pairwise distance correlation | A (existing) | Global | No | No | O(n²) | scipy |
| 6 | Spatial Neighbor Agreement | B (new) | Local-spatial | No | Yes | O(n·k) | Custom |
| 7 | Spatial Distance Correlation | B (new) | Global-spatial | No | Yes | O(n²) | scipy |
| 8 | Moran's I (embedding dims) | B (new) | Global-spatial | No | Yes | O(n·k) | Custom/esda |
| 9 | CHAOS score | B (new) | Cluster-spatial | Yes | Yes | O(n·k) | Custom |
| 10 | PAS score | B (new) | Local-spatial | Yes | Yes | O(n·k) | Custom |
| 11 | ARI (graph vs embedding clusters) | C (new) | Cluster | Yes | No | O(n) | sklearn |
| 12 | NMI | C (new) | Cluster | Yes | No | O(n) | sklearn |
| 13 | Cell-type purity | C (new) | Cluster | Yes | No | O(n) | Custom |
| 14 | Random triplet accuracy | D (new) | Global | No | No | O(triplets) | Custom |
| 15 | Shepard diagram correlation | D (new) | Global | No | No | O(n²) | Custom |
| 16 | Centroid distance correlation | D (new) | Global | Yes | No | O(c²) | Custom |
| 17 | kNN preservation rate | D (new) | Local | No | No | O(n·k) | Custom |

### 9.2 Advanced Metrics (Optional, If Resources Allow)

| Metric | What It Measures | Library | Note |
|---|---|---|---|
| Q_local / Q_global (co-ranking) | Unified local/global decomposition | pyDRMetrics, coranking | Requires O(n²) co-ranking matrix |
| LCMC (Local Continuity Meta-Criterion) | Optimal-scale neighborhood preservation | coranking | Identifies "intrinsic scale" of embedding |
| Steadiness / Cohesiveness | Inter-cluster reliability | snc (pip install snc) | More expensive than ARI |
| scDEED reliability | Per-cell embedding quality | scDEED | O(n·k), scalable to 4.2M |
| Scale-Normalized Stress (SNS) | Scale-invariant distance preservation | Custom | Fixes known flaw in normalized stress |
| Mantel test | Permutation-tested distance matrix correlation | scikit-bio | Requires O(n²) distance matrix |
| CCA | Linear embedding agreement | sklearn CCA | More general than Procrustes |

### 9.3 Metric Interpretation Guide

**Trustworthiness / Continuity (T/C):**
- Measures: local neighborhood preservation
- Weakness: blind to global distortion. Embeddings can score >0.95 while exhibiting severe global distortion
- T > 0.90 = good; T > 0.95 = excellent

**Spatial Neighbor Agreement (SNA):**
- Measures: whether physically adjacent cells remain adjacent in UMAP
- Expected range for UMAP: 0.05–0.30 (UMAP optimizes for expression-space structure, not physical-space)
- The comparison between Python and Rust is what matters, not the absolute value

**Moran's I on Embedding Coordinates:**
- I near 1 = strong positive spatial autocorrelation (embedding varies smoothly across tissue)
- I near 0 = spatial randomness in embedding
- Apply to UMAP dim 0 and dim 1 separately; report max

**Random Triplet Accuracy:**
- 0.5 = random baseline; 1.0 = perfect ordering preservation
- UMAP: typically 0.55–0.70; PaCMAP/TriMap: 0.75–0.90
- Measures global structure preservation

**CHAOS / PAS:**
- From SpatialPCA (Nature Communications 2022)
- Lower CHAOS = more spatially compact clusters
- Lower PAS = fewer spatially abnormal cells
- Standard spatial transcriptomics benchmark metrics

---

## 11. Tooling & Library Reference

### 10.1 Required Python Packages

| Package | Version | Purpose |
|---|---|---|
| `anndata` | latest | H5AD loading, backed/memory-mapped access |
| `scanpy` | ≥1.10 | Preprocessing, PCA, neighbors, UMAP |
| `umap-learn` | 0.5.11 | Reference UMAP (pinned in environment.yml) |
| `numpy` | ≥2.2 | Core array operations |
| `scipy` | ≥1.15 | Sparse matrices, Procrustes, statistics |
| `scikit-learn` | ≥1.8 | Metrics (trustworthiness, silhouette, ARI, NMI) |
| `polars` | latest | Fast metadata loading (5-30x faster than pandas) |
| `pynndescent` | 0.5.x | Approximate nearest neighbors (used by UMAP) |
| ~~`harmonypy`~~ | ~~latest~~ | ~~Not needed — see Section 5.5~~ |
| `datashader` | ≥0.18 | Visualization of 4.2M point embeddings |
| `matplotlib` | ≥3.10 | Static plots |
| `psutil` | latest | Process memory measurement |

**Optional (enhanced analysis):**
| Package | Purpose |
|---|---|
| `abc_atlas_access` | Allen Brain Cell Atlas data download |
| `squidpy` | Spatial statistics (wraps esda for Moran's I) |
| `esda` / `libpysal` | PySAL spatial autocorrelation |
| `pyDRMetrics` | Co-ranking framework, continuity |
| `zadu` | Unified DR metrics suite |
| `snc` | Steadiness / Cohesiveness |
| `rapids-singlecell` | GPU-accelerated preprocessing |
| `pacmap` | Alternative DR for comparison |
| `holoviews` | Interactive visualization with datashader |

### 10.2 Rust Crate Dependencies (Already in Cargo.toml)

| Crate | Role |
|---|---|
| `sprs` 0.11 | Sparse CSR matrices (graph format) |
| `ndarray` 0.17 | Dense arrays (eigenvectors, embeddings) |
| `faer` 0.24 | Dense/sparse linear algebra (EVD, Cholesky) |
| `linfa-linalg` 0.2 | LOBPCG eigensolver |
| `ndarray-npy` 0.10 | .npy/.npz file I/O for Python interop |

### 10.3 Approximate Nearest Neighbor Libraries (Scale Reference)

| Library | Backend | Scale | Notes |
|---|---|---|---|
| pynndescent | Python/Numba | ~10M | Default for UMAP; flexible metrics |
| hnswlib | C++/Python | Billions | Best query latency; tune M, ef_construction |
| FAISS | C++/Python/GPU | Billions | IVF-PQ for billion-scale; GPU-accelerated |
| USearch | C++/Rust/Python | Billions | Single-file; up to 20x faster than FAISS flat |

For 4.2M cells: **pynndescent** is the no-friction default (already used by UMAP/scanpy).

### 10.4 Performance Benchmarking Tools

| Tool | Purpose |
|---|---|
| `pytest-benchmark` | Python timing with warmup, IQR, JSON export |
| `hyperfine` | CLI timing for Rust binary |
| `resource.getrusage` | Peak RSS (Linux, cross-language) |
| `psutil` | Process RSS snapshot |
| `tracemalloc` | Python heap only (does NOT capture Rust allocations) |
| `/usr/bin/time -v` | Peak RSS for Rust subprocess |
| Criterion | Rust microbenchmarks (already in `benches/`) |

---

## 12. Research Questions Decomposition

### 11.1 Pre-Work Research Questions (Before Writing Code)

These are actual research questions — things we need to investigate to make informed design choices. Data download and staging are tasks, not research questions, and are covered in Section 15 (Issue Decomposition).

| # | Question | Phase | How to Answer |
|---|---|---|---|
| RQ-1.1 | What normalization maximizes spectral gap of the kNN graph Laplacian? | 1 | Parameter sweep on 10K subset |
| RQ-1.2 | How many PCA dimensions are needed for 1,122 MERFISH genes? | 1 | Elbow plot + spectral gap analysis |
| RQ-1.3 | Does per-imaging-run mean normalization visibly improve the 10K UMAP? | 1 | Compare with/without per-run normalization on 10K subset |
| RQ-1.4 | Does cosine vs euclidean metric affect spatial preservation? | 1 | Side-by-side on 10K subset |
| RQ-1.5 | Does the log2-normalized H5AD need additional normalization? | 1 | Check Allen Institute docs; inspect value distributions |

### 11.2 Core Research Questions (Require Full Pipeline)

| # | Question | Phase | Success Criterion |
|---|---|---|---|
| RQ-2.1 | Does Rust spectral init produce equivalent downstream UMAP embeddings? | 2+3 | All Category A metrics PASS at every scale |
| RQ-2.2 | At what n does Rust spectral init become faster than Python? | 2 | Timing crossover point on scaling plot |
| RQ-2.3 | How does Rust memory scale vs Python memory? | 2 | Peak RSS comparison |
| RQ-2.4 | Which solver level does Rust use at each scale? | 2 | Solver level recording |
| RQ-3.1 | Does Rust init preserve spatial tissue organization as well as Python? | 3 | SNA delta < 0.02, Moran's I delta < 0.05 |
| RQ-3.2 | Are spatial correlation metrics sensitive to eigenvector sign/rotation ambiguity? | 3 | Compare SNA before and after Procrustes alignment |
| RQ-3.3 | Which brain regions show the largest Python-vs-Rust difference? | 3 | Per-section SNA analysis |
| RQ-3.4 | Does random initialization provide a useful lower bound for spatial metrics? | 3 | Random SNA << Python SNA ≈ Rust SNA |
| RQ-4.1 | Is there a scale at which Rust quality degrades relative to Python? | 4 | Quality gate table (all scales) |
| RQ-4.2 | Does solver escalation correlate with quality degradation? | 4 | Solver level vs metric scatter |

### 11.3 Stretch Research Questions (If Time Permits)

| # | Question | Phase | Notes |
|---|---|---|---|
| RQ-S.1 | Does PaCMAP outperform UMAP for spatial structure preservation? | 3+ | Would require adding PaCMAP to the pipeline |
| RQ-S.2 | Can we use spectral init quality metrics to predict downstream UMAP quality? | 3+ | Eigenpair residual vs SNA correlation |
| RQ-S.3 | Does the `RustNative` compute mode outperform `PythonCompat` mode? | 2+ | Compare both modes |
| RQ-S.4 | How do cuML GPU UMAP embeddings compare? | 2+ | Requires GPU hardware |

---

## 13. Pre-work Checklist

### 12.0 Research Questions to Answer FIRST

These are genuine research questions — things we need to investigate to make informed design choices. Data download and staging are just tasks (covered in Section 15), not research questions.

Answer these with small experiments or literature review **before** building the full pipeline.

| # | Question | How to Answer | Blocking? |
|---|---|---|---|
| RQ-1.1 | Does the log2-normalized H5AD need additional normalization? | Read Allen Institute docs; inspect value distributions in the 10K subset | Yes — determines preprocessing pipeline |
| RQ-1.2 | How many PCA dimensions for 1,122 MERFISH genes? | Compute PCA on 10K subset, plot explained variance (elbow plot), measure spectral gap at each n_pcs | Yes — determines kNN graph quality |
| RQ-1.3 | Does per-run mean normalization matter for the 10K subset? | Compare UMAP with/without per-run normalization | Low — likely negligible at 10K from few runs |
| RQ-1.4 | Does cosine vs euclidean metric affect spatial preservation? | Run both on 10K subset, compare trustworthiness | Low — can pick a default and revisit |

**Recommended approach:** Create a Jupyter notebook or Python script that answers these once the 10K subset is staged. This is pure exploration — no permanent code changes.

### 12.1 Hardware Requirements (10K Focus)

| Requirement | For 10K subset | For future scaling |
|---|---|---|
| RAM | 4 GB | 64+ GB (4.2M cells) |
| Disk | 5 GB free | 50 GB free |
| CPU | Any | 8+ cores |
| GPU | Not needed | RTX 5080 for rapids-singlecell |

### 12.2 Software Setup

- [ ] Rust toolchain with `target-cpu=native` (already in `.cargo/config.toml`)
- [ ] Python environment with `scanpy`, `anndata`, `umap-learn 0.5.11`
- [ ] `polars` for metadata loading (replaces pandas for heavy CSV/Parquet work)
- [ ] `pandas` for metric result tables (small DataFrames only)
- [ ] ~~`harmonypy`~~ — not needed (see Section 5.5: batch correction not recommended for whole-brain sections)
- [ ] `psutil` for memory measurement
- [ ] Verify `anndata.read_h5ad(backed='r')` works with available HDF5 version
- [ ] `datashader` — install but not needed until scaling beyond 10K

### 12.3 Data Download (10K First)

- [ ] Download `Zhuang-ABCA-1-log2.h5ad` (2.13 GB) — needed to generate 10K subset
- [ ] Download `cell_metadata.csv` (630 MB) — spatial coordinates and labels
- [ ] Generate 10K subset and commit compressed `.npz` to repo
- [ ] After 10K subset is committed, the raw H5AD is no longer needed locally for development

### 12.4 Code Changes Required (10K Scope)

- [ ] Implement `load_merfish()` in `generate_umap_comparisons.py` (replacing `NotImplementedError` stub)
- [ ] Add spatial coordinate output to the pipeline (third return value from loaders)
- [ ] Implement spatial metrics functions (SNA, Moran's I, CHAOS, PAS)
- [ ] Add timing instrumentation to `run_baseline()` and `run_compare()`
- [ ] Extend `_compute_metrics()` with Categories B, C, D
- [ ] Create `tests/visual_eval/download_merfish.py` for runtime data retrieval
- [ ] Create `tests/visual_eval/run_merfish_eval.sh` for manual evaluation
- [ ] Add `merfish-eval` nextest profile to `.config/nextest.toml`
- [ ] Create `docs/metrics/` with metric documentation

### 12.5 Decisions Required Before Starting

| Decision | Options | Recommended |
|---|---|---|
| Which H5AD? | log2 or raw | log2 (already normalized — answer RQ-1.1 to confirm) |
| Full UMAP or spectral only at 10K? | Full SGD vs spectral init only | Full SGD (10K is fast enough for full runs) |
| Spatial coordinates | Section (x,y) or CCF (x_ccf, y_ccf, z_ccf) | Section (x,y) — available for all cells |
| n_pcs for 10K | Fixed 30 vs sweep | Sweep {10, 20, 30, 50}, pick winner (RQ-1.1) |
| Metric result format | pandas DataFrame → CSV/JSON | pandas for metric tables, JSON for per-run metadata |

---

## 14. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Full 4.2M graph doesn't fit in RAM for Rust eigensolver | Blocks full-scale benchmark | Medium | Use 1M as max; monitor peak RSS carefully |
| LOBPCG doesn't converge at >1M nodes | Escalates to rSVD/forced-dense (very slow) | Medium | rSVD is designed as fallback; record solver level |
| Python kNN construction takes >4 hours at 4.2M | Blocks Phase 2 full-scale runs | High | Use checkpointing; consider rapids-singlecell GPU |
| Per-run normalization constants unknown for 10K subset | May not have imaging-run metadata | Low | Check if `feature_matrix_label` encodes imaging run; if not, skip per-run normalization for 10K |
| Eigenvector sign/rotation ambiguity masks real quality differences | Phase 3 spatial metrics show false equivalence | Medium | Procrustes alignment; use rotation-invariant metrics (ARI, SNA) |
| Single biological sample limits statistical power | Cannot claim generalizability | High | Explicitly scope conclusions; suggest replication on Zhuang-ABCA-2/3/4 |
| Out-of-memory when computing pairwise distances for metrics | Crashes during metric computation | High | Strict subsampling protocol; use O(n·k) metrics on full data |
| H5AD format version incompatibility | Data loading fails | Low | Pin anndata version; test download immediately |

---

## Appendix A: Cross-Phase Dependency Graph

```
Phase 0: Data Infrastructure
  ├── Download and validate MERFISH data
  ├── Generate spatially-stratified subsets (10K, 100K, 500K, 1M, full)
  └── Export spatial coordinates and labels as .npy files
        │
        ▼
Phase 1: Preprocessing Pipeline
  ├── Parameter sweep on 10K (normalization, n_pcs, distance metric)
  ├── Validate winning config on 100K
  └── Export kNN graphs as .npz for all subsets
        │
        ▼
Phase 2: Baseline UMAP Runs
  ├── Python spectral init → SGD (all subsets)
  ├── Rust spectral init (via export_rust_init.rs) → SGD (all subsets)
  ├── Random init → SGD (all subsets)
  ├── Timing instrumentation (per-phase wall-clock)
  └── Memory instrumentation (peak RSS)
        │
        ├───────────────────────────┐
        ▼                           ▼
Phase 3: Quality Metrics      Phase 4: Performance Analysis
  ├── Category A (existing)      ├── Scaling plots (log-log)
  ├── Category B (spatial)       ├── Speedup ratios
  ├── Category C (cluster)       ├── Memory profiles
  └── Category D (global)        └── Solver escalation analysis
        │                           │
        └───────────┬───────────────┘
                    ▼
           Phase 4: Final Report
             ├── Quality gate table
             ├── Spatial correlation deep dive
             ├── Performance comparison
             └── merfish_benchmark_report.md
```

## Appendix B: Existing Pipeline Entry Points

| Script | Purpose | How to Run |
|---|---|---|
| `tests/visual_eval/generate_umap_comparisons.py --phase baseline` | Python Phase 1 | `python tests/visual_eval/generate_umap_comparisons.py --phase baseline --dataset merfish` |
| `tests/visual_eval/export_rust_init.rs` | Rust export | `cargo test --test export_rust_init --features testing -- --ignored --nocapture` |
| `tests/visual_eval/generate_umap_comparisons.py --phase compare` | Python Phase 2 | `python tests/visual_eval/generate_umap_comparisons.py --phase compare --dataset merfish` |
| `tests/visual_eval/run_eval.sh` | End-to-end | `./tests/visual_eval/run_eval.sh` |

## Appendix C: File Naming Convention

All output files follow the pattern:
```
merfish_{subset_size}_{variant}.{ext}

Examples:
  merfish_10k_graph.npz
  merfish_100k_py_spectral.npy
  merfish_500k_rust_init.npy
  merfish_1m_rust_final.npy
  merfish_full_metrics_full.json
  merfish_100k_timing.json
  merfish_1m_solver_info.json
  merfish_full_spatial_correlation.png
```

## Appendix D: Key Literature References

| Reference | Relevance |
|---|---|
| Zhang et al., Nature 2023 (10.1101/2023.03.06.531348) | MERFISH whole-brain atlas, spatial-transcriptomic correspondence |
| Kobak 2021, Nature Biotechnology | Initialization is critical for UMAP global structure |
| Shang et al., Nature Communications 2022 | SpatialPCA: CHAOS and PAS metrics |
| Lee & Verleysen, Neurocomputing 2009 | Co-ranking matrix framework (Q_local, Q_global, LCMC) |
| Venna & Kaski, 2006 | Trustworthiness and Continuity metrics |
| Böhm et al., bioRxiv 2019 | UMAP global structure is due to initialization, not the algorithm |
| SRTBenchmark, iMeta 2025 | Comprehensive spatial transcriptomics clustering benchmark |
| RAPIDS cuML, NVIDIA 2024 | GPU UMAP benchmarks at 20M+ cells |
| arXiv 2408.07724, 2024 | Scale-Normalized Stress (fixes normalized stress flaw) |
| arXiv 2509.04222, 2025 | Precision-Recall for DR cluster visibility |
| DREAMS, arXiv 2508.13747, 2025 | Scale-decomposed local-global DR score |

## Appendix E: Allen Brain Cell Atlas Data Access Quick Reference

```python
# Installation
pip install git+https://github.com/alleninstitute/abc_atlas_access.git

# Basic usage
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
cache = AbcProjectCache.from_cache_dir(Path("data/abc_atlas"))

# Check available files
cache.list_metadata_files("Zhuang-ABCA-1")
cache.list_expression_matrix_files("Zhuang-ABCA-1")
cache.get_directory_metadata_size("Zhuang-ABCA-1")  # e.g., "1.33 GB"

# Download expression matrix (returns local path; cached after first download)
expr_path = cache.get_file_path("Zhuang-ABCA-1", "Zhuang-ABCA-1/log2")

# Load with memory-mapped backing (critical for 4.2M cells)
import anndata
adata = anndata.read_h5ad(expr_path, backed='r')

# Load metadata as DataFrame
cell_meta = cache.get_metadata_dataframe("Zhuang-ABCA-1", "cell_metadata",
                                          dtype={"cell_label": str})

# Direct S3 access (no credentials needed)
# aws s3 cp s3://allen-brain-cell-atlas/expression_matrices/Zhuang-ABCA-1/20230830/Zhuang-ABCA-1-log2.h5ad . --no-sign-request
```

## Appendix F: Manual Evaluation Infrastructure

### Nextest Profile for MERFISH Evaluation

Add a `merfish-eval` profile to `.config/nextest.toml`:

```toml
[profile.merfish-eval]
# MERFISH evaluation tests — manual only, never in CI
default-filter = 'test(merfish)'
slow-timeout = { period = "600s", terminate-after = 2 }
fail-fast = false
```

Rust tests for MERFISH use `#[ignore = "requires MERFISH 10K subset data"]`:
```rust
#[test]
#[ignore = "requires MERFISH 10K subset data"]
fn export_merfish_10k_rust_init() {
    // Load merfish_10k_graph.npz, run spectral_init, write merfish_10k_rust_init.npy
}
```

Run manually:
```bash
# Full MERFISH evaluation pipeline
./tests/visual_eval/run_merfish_eval.sh

# Rust spectral init export only
cargo nextest run --profile merfish-eval --run-ignored all --features testing

# Python phases only
python tests/visual_eval/generate_umap_comparisons.py --phase baseline --dataset merfish_10k
python tests/visual_eval/generate_umap_comparisons.py --phase compare --dataset merfish_10k
```

### docs/ Setup for Metrics Documentation

Create `docs/metrics/` as a permanent home for metric definitions:

```
docs/
├── umap-spectral-initialization-rust-implementation-report.md  # (existing)
└── metrics/
    ├── README.md                    # Overview of all metrics used in the project
    ├── structure-preservation.md    # Trustworthiness, Continuity, kNN preservation
    ├── spatial-correlation.md       # SNA, Moran's I, CHAOS, PAS (MERFISH-specific)
    ├── cluster-preservation.md      # ARI, NMI, cell-type purity
    └── global-structure.md          # Shepard correlation, triplet accuracy, centroid correlation
```

Each metric document follows a standard template:
- **Definition**: mathematical formula
- **What it measures**: plain-English interpretation
- **Range and interpretation**: what values mean good/bad
- **Computational complexity**: O(?) and scalability notes
- **Implementation**: which Python library/function computes it
- **Thresholds**: what PASS/FAIL gates we use

### GPU Acceleration Note (RTX 5080)

With an RTX 5080 available, `rapids-singlecell` + `cuML` provides substantial speedups:
- UMAP: ~60x faster than CPU umap-learn
- PCA: GPU-accelerated incremental PCA for large matrices
- kNN: GPU nn-descent for fast approximate neighbor search

Install: `pip install 'rapids-singlecell[rapids12]' --extra-index-url=https://pypi.nvidia.com`

This is optional infrastructure — the pipeline should work on CPU-only first, with GPU as an acceleration path for larger subsets.

---

## 15. Issue Decomposition

This section maps the plan into concrete GitHub issues for sequential implementation. Issues are grouped into batches that can be processed independently.

### Batch 1: Foundation (No Data Required)

| Issue | Title | Description |
|---|---|---|
| 1.1 | Create `docs/metrics/` documentation structure | Create `docs/metrics/README.md` with metric catalog from Section 10. Add per-category metric docs. |
| 1.2 | Add `merfish-eval` nextest profile | Add profile to `.config/nextest.toml`. Add `#[ignore]` pattern for MERFISH tests. |
| 1.3 | Add MERFISH data directory structure | Create `tests/visual_eval/merfish_data/` with `.gitkeep`. Update `.gitignore` for large MERFISH files. |

### Batch 2: Data Acquisition

| Issue | Title | Description |
|---|---|---|
| 2.1 | Implement MERFISH download script | `tests/visual_eval/download_merfish.py` — downloads from S3, validates checksums, extracts subsets. Uses Polars for metadata CSV loading. |
| 2.2 | Generate and commit 10K subset | Run download script, generate spatially-stratified 10K subset, commit compressed `.npz` to repo. Validate cell-type composition matches full dataset proportions. |

### Batch 3: Pipeline Integration (Answers RQ-1.1 through RQ-1.4)

| Issue | Title | Description |
|---|---|---|
| 3.1 | Implement `load_merfish()` in `generate_umap_comparisons.py` | Replace `NotImplementedError` stub. Load from `merfish_data/` directory. Return (expression, labels, spatial_coords, name). |
| 3.2 | Add spatial coordinates to pipeline data flow | Extend the pipeline to carry spatial coordinates through all phases. Add `_spatial.npy` output alongside existing artifacts. |
| 3.3 | Run preprocessing parameter sweep on 10K subset | Test normalization × n_pcs × metric combinations. Record spectral gap for each. Document winning configuration. |

### Batch 4: Metric Implementation

| Issue | Title | Description |
|---|---|---|
| 4.1 | Implement spatial correlation metrics (Category B) | SNA, spatial distance correlation, Moran's I, CHAOS, PAS. All metrics stored as pandas DataFrames. |
| 4.2 | Implement cluster preservation metrics (Category C) | ARI, NMI, cell-type purity against embedding-derived clusters. |
| 4.3 | Implement global structure metrics (Category D) | Random triplet accuracy, Shepard correlation, centroid distance correlation. |
| 4.4 | Extend `_compute_metrics()` with Categories B/C/D | Wire new metrics into the existing comparison pipeline. Update `_metrics.json` output format. |

### Batch 5: End-to-End 10K Evaluation

| Issue | Title | Description |
|---|---|---|
| 5.1 | Run full 10K evaluation pipeline | Execute `run_merfish_eval.sh` end-to-end. Generate all plots and metrics. Validate PASS/FAIL gates. |
| 5.2 | Add timing and memory instrumentation | Per-phase wall-clock timing, peak RSS measurement. Output to `_timing.json` and `_memory.json`. |
| 5.3 | Generate 10K benchmark report | Compile metrics, plots, and timing into `merfish_10k_benchmark_report.md`. |

### Future Batches (After 10K Validation)

| Batch | Focus |
|---|---|
| 6 | Scale to 100K subset (runtime download, not committed) |
| 7 | Scale to 500K and 1M subsets |
| 8 | Full 4.2M evaluation (may require GPU acceleration) |
| 9 | Comparative analysis report across all scales |
