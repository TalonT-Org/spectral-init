# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First release.

### Added

- `spectral_init(graph, data, &config)`: spectral initialization coordinates for a
  symmetric, non-negative sparse graph, with the semantics of
  `umap.spectral.spectral_layout` in umap-learn 0.5.x. The graph is a `sprs` view with
  `f32` weights and `u32` indices in CSR or CSC storage; the result is an
  `(n, n_components)` `Array2<f32>` in standard layout, scaled to a largest magnitude of
  10 with Gaussian noise of standard deviation 10⁻⁴.
- `SpectralInitConfig`: the number of embedding dimensions, the seed (default 0), the
  compute mode (default `PythonCompat`) and the dense memory budget (default 4 GiB).
- `ComputeMode::PythonCompat`, which follows umap-learn's arithmetic, and
  `ComputeMode::RustNative`, which sums degrees in double precision and uses an AVX2+FMA
  sparse matrix-vector kernel on x86_64 CPUs that support it.
- `SpectralError`, with one variant per failure. The input is validated before any
  computation: the graph must be square, symmetric (within a relative tolerance of 10⁻⁶)
  and have finite, non-negative weights, with more nodes than embedding dimensions.
  Stored zeros are absent edges; self-loops count towards degree and adjacency.
- A chain of six eigensolvers — dense eigendecomposition, LOBPCG, shift-and-invert
  LOBPCG, regularized LOBPCG, randomized SVD and a final dense eigendecomposition — in
  which every result has to pass a residual check.
- A dense memory budget. No solver allocates a dense matrix larger than the budget: the
  first dense level and the randomized SVD are skipped when they would exceed it, and
  the final dense level and the component layout return
  `SpectralError::DenseBudgetExceeded`. The budget bounds each dense matrix individually,
  not the peak memory of the process, and an allocation within the budget that the
  system cannot satisfy aborts the process.
- Layout of disconnected graphs: every connected component is embedded on its own, and
  the components are placed relative to one another. `data` is required only when the
  graph has more than `2 · n_components` components.
- Re-exports of `sprs` and `ndarray`, whose types appear in the API.
- A reproducibility contract: for a fixed `Cargo.lock`, target, CPU feature set and
  thread configuration the output is deterministic. Random numbers come from
  `rand_chacha::ChaCha12Rng`.
- A user guide, a guide to using the crate with `umap-rs`, and four examples.

### Notes

- `spectral_init` does not panic on any input, and it never substitutes a random layout:
  a failure of every solver is reported as `SpectralError::ConvergenceFailure`.
- The minimum supported Rust version is 1.88.
- The crate has no public Cargo features.

[0.1.0]: https://github.com/TalonT-Org/spectral-init/releases/tag/v0.1.0
