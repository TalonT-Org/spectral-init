# spectral-init Test Infrastructure

This directory contains the Python-based fixture generation system for `spectral-init`.
Fixtures are `.npz` files that capture each step of the spectral initialization pipeline,
generated from Python UMAP's reference implementation and used as ground-truth by Rust tests.

---

## Setup

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
then create the environment from the spec file:

    micromamba create -f tests/environment.yml
    micromamba activate spectral-test

Key packages: Python 3.11, NumPy 2.2, SciPy 1.15, umap-learn 0.5.11, pytest.

---

## Generating Fixtures

Run from the repository root:

    python tests/generate_fixtures.py

Options:
- `--output-dir DIR`   Root output directory (default: `tests/fixtures`)
- `--datasets NAME …`  Generate only the named datasets (space-separated)
- `--verify`           Run `verify_fixtures.py` after generation; exit 1 if any check fails
- `--n-neighbors N`    KNN neighbor count (default: 15)
- `--n-components N`   Embedding dimensions (default: 2)
- `--knn-method M`     KNN method: `approx` (default), `exact`, or `both` (see section below)

Examples:

    # Generate all datasets (approx KNN, default)
    python tests/generate_fixtures.py

    # Generate only blobs_50 and disconnected_200
    python tests/generate_fixtures.py --datasets blobs_50 disconnected_200

    # Generate and immediately verify
    python tests/generate_fixtures.py --verify

---

## KNN Methods and the 4096 Threshold

Python UMAP uses two KNN algorithms depending on dataset size:

- **n < 4096**: `sklearn.neighbors.NearestNeighbors` (exact pairwise distances via
  `sklearn.metrics.pairwise_distances`).
- **n ≥ 4096**: PyNNDescent (approximate, faster).

The fixture pipeline captures both paths via the `--knn-method` flag:

| Flag value | What runs | Files generated |
|---|---|---|
| `approx` (default) | `umap.nearest_neighbors` (PyNNDescent) | `step1_knn.npz` … `step5a_pruned.npz` |
| `exact` | `sklearn.metrics.pairwise_distances` | `step1_knn_exact.npz` … `step5a_pruned_exact.npz` |
| `both` | Both paths | All approx + all exact files |

Use `--knn-method exact` (or `both`) only for datasets with n < 4096 — the O(n²) distance
matrix is impractical for large datasets.

Examples:

    # Generate exact-path fixtures for all small datasets
    python tests/generate_fixtures.py --datasets blobs_50 blobs_500 moons_200 \
        circles_300 near_dupes_100 disconnected_200 --knn-method exact --verify

    # Generate both paths for blobs_50
    python tests/generate_fixtures.py --datasets blobs_50 --knn-method both

---

## Verifying Fixtures

Run the standalone verification script against an existing fixture directory:

    python tests/verify_fixtures.py
    python tests/verify_fixtures.py --output-dir tests/fixtures --datasets blobs_50

Exit code 0 = all checks pass, 1 = at least one check failed.

---

## Regenerating After a Python UMAP Version Bump

1. Update `umap-learn` version in `tests/environment.yml`.
2. Recreate the environment: `micromamba env update -f tests/environment.yml`.
3. Delete old fixture files: `find tests/fixtures -name "*.npz" -delete`
4. Regenerate: `python tests/generate_fixtures.py --verify`
5. Commit the updated `manifest.json` (the `.npz` files are gitignored and generated locally).
6. Update `tests/README.md` if the pipeline steps or format change.

**Note:** `full_umap_e2e.npz` embeddings will differ across UMAP versions because SGD
optimization is version-sensitive. `full_spectral.npz` should be stable across minor
versions but may shift with algorithmic changes to `umap.spectral.spectral_layout`.

---

## Fixture Format Reference

All 14 `.npz` files are generated per dataset. The 12 intermediate step files capture
each stage of the spectral init pipeline; the 2 full-pipeline files are end-to-end
references from Python UMAP.

### KNN Pipeline (Steps 0–5a)

| File | Keys | Dtypes | Shape | Notes |
|------|------|--------|-------|-------|
| `step0_raw_data.npz` | `X`, `n_samples`, `n_features` | f64, i32, i32 | (n,d), (), () | Raw dataset |
| `step1_knn.npz` | `knn_indices`, `knn_dists`, `n_neighbors` | i32, f32, i32 | (n,k), (n,k), () | Nearest-neighbor indices and distances |
| `step2_smooth_knn.npz` | `sigmas`, `rhos`, `n_neighbors` | f32, f32, i32 | (n,), (n,), () | Bandwidth parameters from `smooth_knn_dist` |
| `step3_membership.npz` | *(scipy sparse COO)* | f32 | (n,n) | Directed membership-strength graph; sparse, asymmetric |
| `step4_symmetrized.npz` | *(scipy sparse CSR)* | f32 | (n,n) | Fuzzy union: A + A^T − A∘A^T; symmetric |
| `step5a_pruned.npz` | *(scipy sparse CSR)* | f32 | (n,n) | Edges pruned below max/n_epochs; symmetric |

### Laplacian Components (comp_a–f)

| File | Keys | Dtypes | Shape | Notes |
|------|------|--------|-------|-------|
| `comp_a_degrees.npz` | `degrees`, `sqrt_deg` | f64, f64 | (n,), (n,) | Row-sum degrees of pruned graph |
| `comp_b_laplacian.npz` | *(scipy sparse CSR)* | f64 | (n,n) | Normalized Laplacian L = I − D^{-½}AD^{-½}; diagonal = 1 |
| `comp_c_components.npz` | `n_components`, `labels` | i32, i32 | (), (n,) | Connected components of pruned graph |
| `comp_d_eigensolver.npz` | `eigenvalues`, `eigenvectors`, `residuals`, `k` | f64, f64, f64, i32 | (k,), (n,k), (k,), () | k = dim+1 smallest eigenpairs; residuals = ‖Lv−λv‖/‖v‖ |
| `comp_e_selection.npz` | `order`, `embedding` | i32, f64 | (dim,), (n,dim) | Indices of selected (non-trivial) eigenvectors |
| `comp_f_scaling.npz` | `pre_noise`, `noise`, `final`, `expansion` | f32, f32, f32, f64 | (n,dim) each, () | Scaled to max-abs=10, then seeded Gaussian noise added |

### Full-Pipeline References

| File | Keys | Dtype | Shape | Notes |
|------|------|-------|-------|-------|
| `full_spectral.npz` | `embedding` | f64 | (n,2) | `umap.spectral.spectral_layout` on pruned graph; gold standard for Rust correctness |
| `full_umap_e2e.npz` | `embedding` | f32 | (n,2) | `UMAP(init="spectral", n_jobs=1, random_state=42).fit_transform(X)`; end-to-end quality reference |

### Sparse File Format

`step3_membership.npz`, `step4_symmetrized.npz`, `step5a_pruned.npz`, and
`comp_b_laplacian.npz` are written by `scipy.sparse.save_npz` and must be read
with `scipy.sparse.load_npz`. The internal keys (`data`, `indices`, `indptr`,
`format`, `shape`) are scipy-managed and not consumed directly by Rust tests.

---

## Python UMAP Function Mapping

| Pipeline step | Python UMAP function | Module |
|---|---|---|
| step1 — KNN | `nearest_neighbors` | `umap.umap_` |
| step2 — smooth KNN | `smooth_knn_dist` | `umap.umap_` |
| step3 — membership | `compute_membership_strengths` | `umap.umap_` |
| step4 — fuzzy union | `A + A^T − A∘A^T` | direct scipy |
| step5a — pruning | threshold at `max / n_epochs` | direct scipy |
| comp_b — Laplacian | `I − D^{-½}AD^{-½}` | direct scipy |
| comp_c — components | `connected_components` | `scipy.sparse.csgraph` |
| comp_d — eigensolver | `eigsh` (ARPACK) | `scipy.sparse.linalg` |
| full_spectral | `spectral_layout` | `umap.spectral` |
| full_umap_e2e | `UMAP.fit_transform` | `umap` |

---

## How Rust Tests Consume Fixtures

Rust integration tests locate fixtures via `CARGO_MANIFEST_DIR`:

```rust
use std::path::Path;

fn fixture_path(dataset: &str, file: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dataset)
        .join(file)
}

#[test]
fn test_laplacian_matches_reference() {
    let path = fixture_path("blobs_50", "comp_b_laplacian.npz");
    // load with ndarray-npz or similar, then compare to Rust output
}
```

Dense `.npz` files (single-array): open the archive, extract the array key (e.g. `"X"`),
and compare with `assert_relative_eq!` from the `approx` crate.

Sparse `.npz` files (scipy format): these are not directly loadable by standard Rust npz
readers; for Rust tests, prefer loading the dense intermediate steps (comp_a through
comp_f) which are all standard numpy `.npz` archives.

---

## Datasets

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `blobs_50` | (50, 2) | Gaussian blobs | Small connected graph; primary reference dataset |
| `blobs_500` | (500, 2) | Gaussian blobs | Medium connected graph |
| `blobs_5000` | (5000, 2) | Gaussian blobs | Large connected graph |
| `moons_200` | (200, 2) | Two moons | Nonlinear structure |
| `circles_300` | (300, 2) | Concentric circles | Nonlinear structure |
| `near_dupes_100` | (100, 2) | Near-duplicates | Tests degenerate eigenvalue handling |
| `disconnected_200` | (200, 2) | 4 isolated clusters | Tests multi-component graph handling |
| `blobs_connected_200` | (200, 2) | Gaussian blobs | Guaranteed-connected graph; tests eigensolver on medium connected input |
| `blobs_connected_2000` | (2000+, 2) | Gaussian blobs | Guaranteed-connected graph; tests eigensolver on large connected input |
