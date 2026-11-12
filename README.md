# spectral-init

> Spectral initialization for UMAP embeddings in Rust.

[![crates.io](https://img.shields.io/crates/v/spectral-init)](https://crates.io/crates/spectral-init)
[![docs.rs](https://img.shields.io/docsrs/spectral-init)](https://docs.rs/spectral-init)
[![CI](https://github.com/TalonT-Org/spectral-init/actions/workflows/ci.yml/badge.svg)](https://github.com/TalonT-Org/spectral-init/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV: 1.88](https://img.shields.io/badge/MSRV-1.88-blue)](https://blog.rust-lang.org/2025/06/26/Rust-1.88.0/)

[API docs](https://docs.rs/spectral-init) · [User guide](docs/user-guide.md) · [Using it with umap-rs](docs/umap-rs-integration.md) · [Changelog](CHANGELOG.md)

## What it computes

`spectral-init` computes the coordinates umap-learn starts from by default before it
optimizes an embedding: the spectral layout of the fuzzy k-nearest-neighbour graph.

Given a symmetric sparse graph `W` with non-negative weights, it builds the symmetric
normalized Laplacian

```text
L = I − D^(−1/2) · W · D^(−1/2)        D = diag(row sums of W)
```

and returns, for every node, its entries in the eigenvectors of the `n_components`
smallest non-zero eigenvalues of `L`. The coordinates are scaled so that the largest
magnitude is 10, and Gaussian noise with standard deviation 10⁻⁴ is added.

- **Input.** A `sprs` matrix view with `f32` weights and `u32` column indices, in CSR or
  CSC storage. The matrix must be square and symmetric, with finite, non-negative
  weights. Stored zeros are absent edges. Diagonal entries (self-loops) count towards
  both degree and adjacency.
- **Output.** An `(n, n_components)` `ndarray::Array2<f32>` in standard (row-major)
  layout.
- **Disconnected graphs.** Each connected component is embedded on its own, and the
  components are then placed relative to one another, in the manner of umap-learn's
  multi-component layout.
- **Semantics.** Those of `umap.spectral.spectral_layout` in umap-learn 0.5.x, which
  umap-learn 0.6 keeps as its default layout.

The crate computes initial coordinates and nothing else: it contains no nearest-neighbour
search, no graph construction and no optimizer, and it depends on no UMAP implementation.

## Quick start

```sh
cargo add spectral-init
```

```rust
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

fn main() -> Result<(), SpectralError> {
    // A ring of 10 nodes: every node is joined to its two neighbours.
    // Replace this with the graph of your UMAP implementation.
    let n = 10;
    let mut triplets: TriMatI<f32, u32> = TriMatI::new((n, n));
    for i in 0..n {
        let next = (i + 1) % n;
        triplets.add_triplet(i, next, 1.0);
        triplets.add_triplet(next, i, 1.0);
    }
    let graph: CsMatI<f32, u32, usize> = triplets.to_csr();

    // Two embedding dimensions, seed 42.
    let config = SpectralInitConfig::new(2).with_seed(42);
    let coords = spectral_init(graph.view(), None, &config)?;

    assert_eq!(coords.dim(), (n, 2));
    Ok(())
}
```

`sprs` and `ndarray` are re-exported, so the snippet needs no other dependency. The
[`examples/`](examples/) directory has runnable programs: `basic`, `from_adjacency_list`,
`compute_modes` and `umap_rs`.

## Using it with a UMAP implementation

There are two routes.

**Standalone.** Compute the coordinates with this crate and pass them to any UMAP
implementation that accepts an initial embedding. With
[`umap-rs`](https://crates.io/crates/umap-rs), the graph comes from `learn_manifold`:

```rust
use spectral_init::ndarray::{Array2, ArrayView2};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};
use umap_rs::{Umap, UmapConfig};

/// Initial coordinates for a `umap-rs` fit.
fn spectral_coordinates(
    data: ArrayView2<'_, f32>,
    knn_indices: ArrayView2<'_, u32>,
    knn_dists: ArrayView2<'_, f32>,
    config: &UmapConfig,
    seed: u64,
) -> Result<Array2<f32>, SpectralError> {
    let manifold = Umap::new(config.clone()).learn_manifold(data, knn_indices, knn_dists);
    let init_config = SpectralInitConfig::new(config.n_components).with_seed(seed);
    spectral_init(manifold.graph().view(), Some(data), &init_config)
}
```

The [integration guide](docs/umap-rs-integration.md) continues from here to the
optimizer and lists what to know about `umap-rs`. Implementations that keep their graph
as raw CSR arrays can lend them to `spectral_init` without copying; the guide shows how.

**Built in.** A fork of `umap-rs` with spectral initialization built in is at
<https://github.com/TalonT-Org/umap-rs/tree/spectral-init>.

## Compute modes

| Mode | Behaviour |
|------|-----------|
| `ComputeMode::PythonCompat` (default) | Follows umap-learn's arithmetic where it affects the result: degrees are accumulated in single precision in SciPy's order, and the iterative solvers use a scalar sparse matrix-vector product. The reference tests run in this mode. |
| `ComputeMode::RustNative` | Sums degrees in double precision on every target. On x86_64 CPUs with AVX2 and FMA, detected at run time, the iterative solvers use a SIMD sparse matrix-vector product. The result spans the same subspace as `PythonCompat` and differs from it by rounding. |

```rust
use spectral_init::{ComputeMode, SpectralInitConfig};

let config = SpectralInitConfig::new(2)
    .with_seed(42)
    .with_compute_mode(ComputeMode::RustNative);
assert_eq!(config.compute_mode(), ComputeMode::RustNative);
```

## Solvers and memory

The eigenvectors come from a chain of six solvers. Each result is checked by its residual
`‖L·v − λ·v‖ / ‖v‖`, and a result that misses the threshold of its solver is discarded in
favour of the next solver.

| Level | Solver | Used when |
|-------|--------|-----------|
| 0 | Dense eigendecomposition | fewer than 2000 nodes |
| 1 | LOBPCG | 2000 nodes or more, or level 0 was not accepted |
| 2 | Shift-and-invert LOBPCG (sparse Cholesky) | level 1 was not accepted |
| 3 | LOBPCG with a small diagonal shift | level 2 was not accepted |
| 4 | Randomized SVD | level 3 was not accepted |
| 5 | Dense eigendecomposition | level 4 was not accepted |

There is no random fallback. If the last level fails its check, or needs a dense matrix
larger than the dense memory budget, `spectral_init` returns an error.

The **dense memory budget** (`SpectralInitConfig::with_dense_memory_budget`, 4 GiB by
default) caps every dense matrix a solver allocates. A solver that would exceed it is
skipped. The budget bounds each dense matrix individually, not the peak memory of the
process: the dense levels hold a few matrices of that size at once. An allocation within
the budget that the system cannot satisfy aborts the process, as any Rust allocation
does.

## Errors

`spectral_init` does not panic on any input. Every failure is a `SpectralError`:

| Variant | Meaning |
|---------|---------|
| `InvalidParameter { name, reason }` | `n_components` is 0. |
| `NotSquare { rows, cols }` | The graph matrix is not square. |
| `TooFewNodes { n, n_components }` | The graph does not have more nodes than `n_components`; this covers the empty graph. |
| `NonFiniteWeight { row, col }` | A weight is NaN or infinite. |
| `NegativeWeight { row, col }` | A weight is negative. |
| `NotSymmetric { row, col }` | An entry has no matching mirror entry. Relative differences up to 10⁻⁶ are tolerated. |
| `DegreeOverflow { node }` | The summed weights of a node overflow `f32`. |
| `DataShapeMismatch { expected_rows, actual_rows }` | `data` does not have one row per node. |
| `NonFiniteData { row, col }` | `data` contains NaN or an infinity. |
| `DataRequired { connected_components, n_components }` | The graph has more than `2 · n_components` connected components and `data` is `None`. |
| `DenseBudgetExceeded { required_bytes, budget_bytes }` | The iterative solvers were not accepted and the dense one needs more than the dense memory budget, or the graph has so many components that their affinity matrix exceeds it. |
| `ConvergenceFailure { n }` | No solver produced eigenvectors of acceptable quality. |
| `DegenerateEmbedding` | All computed coordinates are zero. |

The first nine are found by validating the input before any computation, in the order
listed. The enum and its struct variants are `#[non_exhaustive]`: match them with `..`
and a wildcard arm.

umap-learn falls back to a random layout when its spectral layout fails. This crate
reports the failure, and the caller decides what to substitute:

```rust
use spectral_init::ndarray::Array2;
use spectral_init::sprs::CsMatViewI;
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

/// Spectral coordinates, or the caller's own layout when the solvers give up.
fn initial_coordinates(
    graph: CsMatViewI<'_, f32, u32, usize>,
    config: &SpectralInitConfig,
    fallback: impl FnOnce() -> Array2<f32>,
) -> Result<Array2<f32>, SpectralError> {
    match spectral_init(graph, None, config) {
        Ok(coords) => Ok(coords),
        Err(
            SpectralError::ConvergenceFailure { .. }
            | SpectralError::DenseBudgetExceeded { .. }
            | SpectralError::DegenerateEmbedding,
        ) => Ok(fallback()),
        // Anything else is a problem with the input: report it.
        Err(other) => Err(other),
    }
}
```

## Platforms

- Pure Rust: no BLAS, LAPACK or C toolchain is needed.
- Every target has a scalar code path. On x86_64 the AVX2+FMA kernel of
  `ComputeMode::RustNative` is selected by run-time detection; no compiler flags are
  involved.
- CI runs the test suite on Linux (x86_64 and aarch64), macOS and Windows.
- Minimum supported Rust version: 1.88.

## Reproducibility

For a fixed `Cargo.lock`, target, CPU feature set and thread configuration, the output is
a deterministic function of the graph, the data and the configuration. Random numbers
come from the ChaCha12 generator (`rand_chacha::ChaCha12Rng`), seeded with
`SpectralInitConfig::with_seed`.

Results are not promised to agree bit for bit across targets, CPU feature sets or thread
counts; they agree to rounding. Measured on x86_64 with faer 0.24.4, one thread against
four changed no coordinate for a 500-node graph and changed coordinates by at most
1.2 × 10⁻¹⁰ for a 1500-node graph, on coordinates of magnitude up to 10. To pin the
thread configuration, call `faer::set_global_parallelism` before `spectral_init` or set
`RAYON_NUM_THREADS`.

## Evidence

Every stage of the pipeline is tested against values recorded from umap-learn 0.5.11 on
nine reference datasets. On the connected reference graphs the computed eigenvectors span
the same subspace as umap-learn's (Gram determinant at least 0.99999998) with eigenpair
residuals of at most 9.1 × 10⁻⁶. On the disconnected ones both implementations find the
same components. The trustworthiness (k = 15) of the raw initial coordinates is between
0.83 and 0.99, against 0.50 to 0.55 for a uniform random initialization. The tables, the
method and the versions are in the
[integration guide](docs/umap-rs-integration.md#evidence).

These figures describe the initialization. They are not a comparison of final UMAP
embeddings.

## Stability

This is a 0.x release, and the API may change in 0.2.

`sprs` 0.11 and `ndarray` 0.17 appear in the signature of `spectral_init` and are
re-exported; moving to a semver-incompatible release of either is a breaking release of
this crate. The crate has no public Cargo features.

## Repository

The repository is a Cargo workspace. The root package is the published crate.
[`trustworthiness/`](trustworthiness/) is a standalone embedding-quality tool,
[`eval/`](eval/) is the research harness, and `research/` holds experiment reports; none
of these is part of the published crate. [CONTRIBUTING](.github/CONTRIBUTING.md) explains
how to build and test.

The design of the algorithm is described in the
[implementation report](https://github.com/TalonT-Org/spectral-init/blob/research-baseline-pre-0.1/docs/umap-spectral-initialization-rust-implementation-report.md).

## Acknowledgements

The algorithm and its reference behaviour are those of
[umap-learn](https://github.com/lmcinnes/umap) by Leland McInnes and contributors
(McInnes, Healy and Melville, *UMAP: Uniform Manifold Approximation and Projection for
Dimension Reduction*, [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)). The sign
convention for eigenvectors and the Laplacian of the component layout follow
[scikit-learn](https://scikit-learn.org/). The degree accumulation follows
[SciPy](https://scipy.org/)'s sparse matrix-vector product, and the reference
eigenvectors in the test fixtures were computed with SciPy.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
