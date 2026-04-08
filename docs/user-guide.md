# User Guide: spectral-init

> For crate-level API documentation see [docs.rs/spectral-init](https://docs.rs/spectral-init).

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Background](#background)
3. [Installation](#installation)
4. [Path A: Integration with umap-rs](#path-a-integration-with-umap-rs)
5. [Path B: Standalone Usage](#path-b-standalone-usage)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tips](#performance-tips)

---

## Prerequisites

**Rust 1.85 or later (MSRV).** Install Rust via [rustup](https://www.rust-lang.org/tools/install).

**No LAPACK/BLAS required** — this crate is pure Rust. All linear algebra is
handled by `linfa-linalg` (LOBPCG) and `faer` (dense EVD).

---

## Background

**What is spectral initialization?**
Spectral initialization uses the eigenvectors of the graph Laplacian to produce a
globally-aware starting point for UMAP's SGD optimization phase. Instead of placing
points at random, it encodes the manifold topology directly into the initial
coordinates — the primary reason Python UMAP produces superior embeddings. For
theoretical background see [McInnes et al., "UMAP: Uniform Manifold Approximation
and Projection for Dimension Reduction" (2018)](https://arxiv.org/abs/1802.03426).

**What is a fuzzy k-NN graph?**
A sparse symmetric matrix `W` where entry `W[i,j]` represents the strength of
connectivity between data points `i` and `j`. Higher values mean more similar.
For UMAP, values are fuzzy membership weights in `[0, 1]`.

---

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
spectral-init = "0.1"
```

For standalone graph construction (Path B), also add:

```toml
sprs = "0.11"
```

### Feature Flags

| Flag | Purpose |
|------|---------|
| `testing` | Exposes internal solver functions. **Not stable API.** Only for integration testing. |
| `cli` | Builds the `trustworthiness` and `tw_profiler` quality-assessment binaries. |
| `profiling` | Enables timing instrumentation in the solver escalation chain. |

Normal users only need the default features (none enabled).

---

## Path A: Integration with umap-rs

### Overview

`umap_rs::Umap::graph()` returns `&CsMatI<f32, u32, usize>`, which is the exact
type `spectral_init` accepts. No adapter or type conversion is needed.

### Dependencies

```toml
[dependencies]
spectral-init = "0.1"
umap-rs = "..."       # check crates.io for current version
ndarray = "0.17"      # for data loading
```

### Full Worked Example

```rust
use spectral_init::{spectral_init, SpectralInitConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Load or prepare your feature data.
    // Shape: [n_samples, n_features], dtype f32.
    // (Replace this with your actual data loading.)
    let data: ndarray::Array2<f32> = ndarray::Array2::zeros((100, 10));

    // Step 2: Build the fuzzy k-NN graph with umap-rs.
    // umap-rs handles nearest-neighbour search and fuzzy membership internally.
    let umap = umap_rs::Umap::fit(&data, umap_rs::UmapConfig::default())?;

    // Step 3: Extract the graph. The return type is &CsMatI<f32, u32, usize> —
    // exactly what spectral_init expects; no conversion needed.
    let graph = umap.graph();

    // Step 4: Compute spectral initialization.
    // - n_components: embedding dimensionality (2 for 2-D visualization)
    // - seed: RNG seed for reproducibility
    // - data: None (only needed if graph has disconnected components)
    // - config: SpectralInitConfig::default() uses PythonCompat mode
    let init_coords = spectral_init(graph, 2, 42, None, SpectralInitConfig::default())?;

    // init_coords: Array2<f32> with shape [n_samples, 2]
    // Values are in approximately [-10, 10] with small Gaussian noise.
    println!("Initial embedding shape: {:?}", init_coords.shape());
    assert_eq!(init_coords.shape(), &[100, 2]);
    assert!(init_coords.iter().all(|&v| v.is_finite()));

    // Step 5: Pass init_coords as the starting embedding to your SGD optimizer.
    // (Integrate with your UMAP SGD loop here.)

    Ok(())
}
```

### Verifying Output Quality

- **Shape:** `[n_samples, n_components]`
- **Finite values:** all entries pass `v.is_finite()`
- **Scale:** values are scaled to approximately `[-10, 10]` with small noise (σ = 10⁻⁴)
- **Structure:** for connected graphs, each column is a Laplacian eigenvector encoding global manifold topology
- Residual quality is validated internally; if `ConvergenceFailure` is returned, something unexpected occurred — see [Troubleshooting](#troubleshooting)

---

## Path B: Standalone Usage

### When to Use This Path

Use Path B when:
- You are not using `umap-rs` (custom UMAP, non-UMAP dimensionality reduction, visualization pipelines)
- You already have a precomputed sparse similarity graph from another source
- You want spectral initialization for a custom SGD optimizer that does not come with built-in initialization

### Building the Input Graph

The input type is `CsMatI<f32, u32, usize>` from the `sprs` crate:

- `f32` weights — non-negative, no NaN or Inf
- `u32` column indices
- `usize` row pointer offsets (standard CSR format)
- The matrix **must be symmetric**: if edge `(i,j,w)` exists, edge `(j,i,w)` must also exist

#### From an adjacency list (most common)

```rust
use sprs::{CsMatI, TriMatI};

// Your kNN edges: (source_node, target_node, similarity_weight)
// Weights must be non-negative. For UMAP fuzzy membership, values are in [0,1].
let edges: Vec<(usize, usize, f32)> = vec![
    (0, 1, 0.9),
    (0, 2, 0.7),
    (1, 3, 0.8),
    (2, 3, 0.6),
    // ... more edges
];
let n = 4; // number of nodes

// Build a COO-format sparse matrix, then symmetrize and convert to CSR.
let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
for &(i, j, w) in &edges {
    tri.add_triplet(i, j, w); // forward edge
    tri.add_triplet(j, i, w); // reverse edge — REQUIRED for symmetry
}
let graph: CsMatI<f32, u32, usize> = tri.to_csr();
```

#### From a dense similarity matrix

```rust
use sprs::{CsMatI, TriMatI};

// If you have an n×n dense similarity matrix (e.g., from a Gaussian kernel),
// convert only the non-zero entries.
let n = 4;
let dense_sim: Vec<Vec<f32>> = vec![
    vec![0.0, 0.9, 0.7, 0.0],
    vec![0.9, 0.0, 0.0, 0.8],
    vec![0.7, 0.0, 0.0, 0.6],
    vec![0.0, 0.8, 0.6, 0.0],
];

let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
for i in 0..n {
    for j in 0..n {
        let w = dense_sim[i][j];
        if w > 0.0 {
            tri.add_triplet(i, j, w);
        }
    }
}
// Already symmetric if dense_sim[i][j] == dense_sim[j][i].
let graph: CsMatI<f32, u32, usize> = tri.to_csr();
```

> **Weight requirements:**
> - Non-negative (`w >= 0.0`). Negative weights cause `InvalidGraph`.
> - Finite (no `f32::NAN`, `f32::INFINITY`). NaN/Inf cause `InvalidGraph`.
> - Symmetric: `W[i,j] == W[j,i]`. The API does not enforce this; asymmetric graphs produce incorrect eigenvectors silently.
> - A weight of 0.0 is equivalent to no edge. Diagonal self-loops are ignored.

### Full Worked Example (Standalone Brute-Force kNN)

A complete, self-contained example that computes brute-force nearest neighbors on
a set of 2-D points, builds the sparse graph, and calls `spectral_init`. Requires
only `spectral-init` and `sprs` as dependencies.

```rust
use spectral_init::{spectral_init, SpectralInitConfig};
use sprs::{CsMatI, TriMatI};

fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 20 synthetic 2-D points arranged in two clusters.
    let points: Vec<[f32; 2]> = (0..10)
        .map(|i| [i as f32 * 0.1, 0.0_f32])
        .chain((0..10).map(|i| [5.0 + i as f32 * 0.1, 0.0_f32]))
        .collect();
    let n = points.len();
    let k = 5; // number of nearest neighbours per point

    // Brute-force kNN: find the k closest neighbours for each point.
    // In production, use a proper kNN library (e.g., hora, hnsw_rs).
    let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
    for i in 0..n {
        // Compute distances to all other points.
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_sq(&points[i], &points[j])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Add edges to the k nearest neighbours.
        // Weight = 1.0 / (1.0 + distance) converts distance to similarity.
        for &(j, dist) in dists.iter().take(k) {
            let w = 1.0 / (1.0 + dist.sqrt());
            tri.add_triplet(i, j, w);
            tri.add_triplet(j, i, w);
        }
    }
    let graph: CsMatI<f32, u32, usize> = tri.to_csr();

    // Compute spectral initialization.
    let init_coords = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())?;

    println!("Embedding shape: {:?}", init_coords.shape());
    for (i, row) in init_coords.outer_iter().enumerate() {
        println!("  point {i:>2}: [{:.4}, {:.4}]", row[0], row[1]);
    }

    assert_eq!(init_coords.shape(), &[n, 2]);
    assert!(init_coords.iter().all(|&v| v.is_finite()));
    Ok(())
}
```

### Use Cases for Standalone Path

- Feeding initial coordinates into a custom UMAP SGD optimizer
- Spectral embedding for data visualization (layout pre-computation)
- Initializing t-SNE or ForceAtlas2 with a topology-aware starting point
- Any pipeline where you already have a precomputed sparse similarity graph

---

## Configuration

### ComputeMode

`SpectralInitConfig` is `#[non_exhaustive]`, which means struct literal construction
is not allowed outside the crate. The only valid construction pattern is:

```rust
use spectral_init::{ComputeMode, SpectralInitConfig};

let mut config = SpectralInitConfig::default(); // always start from default
config.compute_mode = ComputeMode::RustNative;  // then override fields
```

**Decision guide:**

| Requirement | Recommended mode |
|-------------|-----------------|
| Match Python UMAP output exactly (reproducibility, CI validation) | `PythonCompat` (default) |
| Maximum throughput on large graphs (n > 2 000) on x86\_64 with AVX2 | `RustNative` |
| Running on non-x86\_64 platforms (ARM, RISC-V, etc.) | Either — `RustNative` silently falls back to the same scalar path as `PythonCompat` |
| Embeddings are subspace-equivalent between modes | True — they span the same eigenspace, may differ by sign per column |

### Disconnected Graphs and the `data` Parameter

- For fully connected graphs (the typical UMAP case), always pass `None`.
- If your graph has disconnected components (isolated subgraphs, isolated nodes),
  `spectral_init` handles each component independently. For components too small
  to run spectral initialization, it uses PCA on the raw feature data.
- Pass `Some(data.view())` only when your graph may have disconnected components
  **and** you want PCA-based fallback for small components.

```rust
use ndarray::ArrayView2;

// If graph may be disconnected and you have the raw feature matrix:
let data: ndarray::Array2<f32> = /* your feature matrix, shape [n_samples, n_features] */;
let init_coords = spectral_init(
    &graph,
    2,
    42,
    Some(data.view()), // enables PCA fallback for small components
    SpectralInitConfig::default(),
)?;
```

When `data` is `None` and the graph is disconnected, small components that cannot
support spectral init are initialized with random coordinates.

---

## Troubleshooting

| Error | Typical cause | Fix |
|-------|--------------|-----|
| `SpectralError::TooFewNodes { n, dims }` | Graph has `n` nodes but `n_components >= n` | Reduce `n_components` below `n`, or use a larger dataset |
| `SpectralError::InvalidGraph(msg)` | Malformed adjacency matrix | Inspect `msg`; check for negative weights, NaN/Inf, non-square shape, zero-size graph |
| `SpectralError::ConvergenceFailure` | Degenerate graph or solver bug | This should not occur on any valid graph; file a bug at the repository |

### `TooFewNodes`

```rust
// This will fail: 3 nodes, 5 embedding dimensions
let result = spectral_init(&small_graph, 5, 42, None, SpectralInitConfig::default());
// Err(TooFewNodes { n: 3, dims: 5 })
//
// Fix: use n_components < n
let result = spectral_init(&small_graph, 2, 42, None, SpectralInitConfig::default()); // ok
```

### `InvalidGraph`

Common causes:

- Negative edge weights → `InvalidGraph("negative weight at (i, j)")`
- NaN or Inf weights → `InvalidGraph("non-finite weight at (i, j)")`
- Non-square matrix → `InvalidGraph("matrix is not square")`
- Zero nodes (`n == 0`) or zero components (`n_components == 0`) → `InvalidGraph`

### `ConvergenceFailure`

This error fires only if the full 6-level solver escalation chain (dense EVD →
LOBPCG → shift-invert LOBPCG → LOBPCG+regularization → randomized SVD → forced
dense EVD) fails on every level. In practice this should not happen on any valid,
non-degenerate graph. If encountered, it is a bug — file an issue at the repository
with the graph's shape, density, and a reproducible construction.

---

## Performance Tips

### Use `RustNative` for Large Graphs

On x86\_64 machines with AVX2 support, `ComputeMode::RustNative` routes SpMV operations
through an AVX2+FMA SIMD kernel, improving throughput for large graphs:

```rust
let mut config = SpectralInitConfig::default();
config.compute_mode = ComputeMode::RustNative;
```

Measured on a 2 000-node ring graph:

| Stage | Time |
|-------|------|
| Full pipeline | ~223 ms |
| Laplacian construction | ~228 µs |
| LOBPCG solve (k = 3) | ~306 ms |
| Randomized SVD (k = 3) | ~286 ms |

### Compile with `target-cpu=native`

To fully enable AVX2+FMA on your specific CPU:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This is especially effective when `ComputeMode::RustNative` is selected.

### Solver Selection is Automatic

The eigensolver escalates through up to six levels automatically; you do not need
to tune it. For graphs with `n < 2 000`, dense EVD is used directly (fast and exact).
For larger graphs, LOBPCG iterative solvers are used. The dense EVD threshold can be
overridden via the `SPECTRAL_DENSE_N_THRESHOLD` environment variable, but this is
only available in test builds (`--features testing`) and is intended for integration
tests — not for production or general benchmarking use.
