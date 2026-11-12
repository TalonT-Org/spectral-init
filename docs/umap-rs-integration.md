# Using spectral-init with umap-rs

[`umap-rs`](https://crates.io/crates/umap-rs) optimizes a UMAP embedding and leaves the
starting coordinates to the caller. `spectral-init` computes the starting coordinates
umap-learn uses by default — the spectral layout of the fuzzy graph — from the graph
`umap-rs` itself builds.

`spectral-init` does not depend on `umap-rs`. This guide is written against `umap-rs`
0.4.5, and every Rust snippet in it is compiled by this crate's test suite. A runnable
program is in [`examples/umap_rs.rs`](https://github.com/TalonT-Org/spectral-init/blob/main/examples/umap_rs.rs).

## Dependencies

```toml
[dependencies]
spectral-init = "0.1"
umap-rs = "0.4.5"
```

`umap-rs` is imported as `umap_rs`. Both crates use `ndarray` 0.17 and `sprs` 0.11, so
arrays and the graph pass between them without conversion. `spectral-init` re-exports the
two as `spectral_init::ndarray` and `spectral_init::sprs`.

## The sequence

1. Find the k nearest neighbours of every point, with any library. `umap-rs` takes them
   as two `(n, n_neighbors)` arrays and needs `n > n_neighbors`. A row may list the point
   itself; `umap-rs` skips that entry.
2. `Umap::learn_manifold` builds the fuzzy graph.
3. `spectral_init` computes the initial coordinates from `manifold.graph()`.
4. `Optimizer` runs the stochastic gradient descent from those coordinates.

```rust
use spectral_init::ndarray::{Array2, ArrayView2};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};
use umap_rs::{EuclideanMetric, Metric, Optimizer, Umap, UmapConfig};

/// Embeds `data` with `umap-rs`, starting from the spectral initialization.
fn embed(
    data: ArrayView2<'_, f32>,
    knn_indices: ArrayView2<'_, u32>,
    knn_dists: ArrayView2<'_, f32>,
    config: UmapConfig,
    seed: u64,
) -> Result<Array2<f32>, SpectralError> {
    let n = data.nrows();

    // The fuzzy graph.
    let manifold = Umap::new(config.clone()).learn_manifold(data, knn_indices, knn_dists);

    // The initial coordinates, for the same number of components as `config`.
    let init_config = SpectralInitConfig::new(config.n_components).with_seed(seed);
    let init = spectral_init(manifold.graph().view(), Some(data), &init_config)?;

    // The number of epochs `Umap::fit` would choose.
    let epochs = config
        .optimization
        .n_epochs
        .unwrap_or(if n <= 10_000 { 500 } else { 200 });

    let metric = EuclideanMetric;
    let mut optimizer = Optimizer::new(manifold, init, epochs, &config, metric.metric_type());
    optimizer.step_epochs(epochs, &metric);
    let fitted = optimizer.into_fitted(config);

    // `Umap::fit` reports vertices without any edge as NaN; do the same here.
    let isolated: Vec<usize> = fitted
        .manifold()
        .graph()
        .outer_iterator()
        .enumerate()
        .filter(|(_, row)| row.data().iter().sum::<f32>() == 0.0)
        .map(|(vertex, _)| vertex)
        .collect();
    let mut embedding = fitted.into_embedding();
    for vertex in isolated {
        embedding.row_mut(vertex).fill(f32::NAN);
    }
    Ok(embedding)
}
```

`Optimizer::new` does not check the shape of the initialization and reads it as one
contiguous slice. `spectral_init` always returns an `(n, n_components)` array in standard
(row-major) layout, which is what it needs.

### The one-call variant

`Umap::fit` takes the initialization and runs the optimizer itself. It builds the graph
internally, so the graph is constructed twice: once for the initialization and once
inside `fit`. Prefer the sequence above when graph construction is expensive.

```rust
use spectral_init::ndarray::{Array2, ArrayView2};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};
use umap_rs::{Umap, UmapConfig};

fn embed_with_fit(
    data: ArrayView2<'_, f32>,
    knn_indices: ArrayView2<'_, u32>,
    knn_dists: ArrayView2<'_, f32>,
    config: UmapConfig,
    seed: u64,
) -> Result<Array2<f32>, SpectralError> {
    let umap = Umap::new(config.clone());
    let manifold = umap.learn_manifold(data, knn_indices, knn_dists);
    let init_config = SpectralInitConfig::new(config.n_components).with_seed(seed);
    let init = spectral_init(manifold.graph().view(), Some(data), &init_config)?;
    Ok(umap
        .fit(data, knn_indices, knn_dists, init.view())
        .into_embedding())
}
```

## What to know

- **`symmetrize` must stay `true`.** With `GraphParams { symmetrize: false, .. }` the graph
  is directed, and `spectral_init` returns `SpectralError::NotSymmetric`.
- **`set_op_mix_ratio`.** For ratios other than 0, 0.5 and 1, the single-precision
  symmetrization in `umap-rs` can leave the two halves of an edge one rounding error
  apart (1.2 × 10⁻⁷ relative). `spectral_init` accepts relative differences up to 10⁻⁶.
- **`data`.** Pass `Some(data)`. It is read only when the graph has more than
  `2 · n_components` connected components, where it places the components relative to one
  another; without it that case is `SpectralError::DataRequired`.
- **Rescaling.** `Optimizer::new` rescales every axis of the initialization to `[0, 10]`,
  as umap-learn does before it optimizes. The optimizer therefore sees the relative
  positions along each axis, not the absolute scale `spectral_init` returns.
- **Isolated vertices.** A vertex without edges is a connected component of its own and
  gets coordinates like any other component. `umap-rs` never moves it, and `Umap::fit`
  reports it as NaN.
- **Errors.** `spectral_init` returns a `SpectralError` for invalid input and for solver
  failure; it does not panic. `umap-rs` panics on invalid input.
- **Reproducibility.** The seed in `SpectralInitConfig` fixes the initialization. The
  optimizer of `umap-rs` draws from an unseeded generator, so two fits from the same
  initialization differ.

## Matching umap-learn's edge pruning

Before it computes the spectral layout, umap-learn removes every edge lighter than the
heaviest edge divided by the number of epochs. `manifold.graph()` still holds those edges.
They carry little weight, so the initialization changes little, but the graph umap-learn
uses can be reproduced:

```rust
use spectral_init::sprs::CsMatI;

/// Zeroes the edges umap-learn prunes before its spectral layout.
fn prune_weak_edges(
    graph: &CsMatI<f32, u32, usize>,
    n_epochs: usize,
) -> CsMatI<f32, u32, usize> {
    let default_epochs = if graph.rows() <= 10_000 { 500 } else { 200 };
    let epochs = if n_epochs > 10 { n_epochs } else { default_epochs };
    let max_weight = graph.data().iter().copied().fold(0.0_f32, f32::max);
    let threshold = max_weight / epochs as f32;
    graph.map(|&weight| if weight < threshold { 0.0 } else { weight })
}
```

`spectral_init` treats stored zeros as absent edges, so the zeroed entries need not be
removed. With a `set_op_mix_ratio` other than 0, 0.5 or 1, an edge whose two halves
straddle the threshold is pruned on one side only and the result is rejected as
`NotSymmetric`.

## State of umap-rs

As of 2026-11-11:

- 0.4.5, released on 2025-12-01, is the latest version; the repository has had no commit
  since 2025-12-02.
- The optimizer differs from umap-learn's in two ways. Both are corrected by pull request
  [#6](https://github.com/wilsonzlin/umap-rs/pull/6), which is open.
  1. The gradient denominators carry an extra factor of the squared distance:
     `a·d²ᵇ·d² + 1` where umap-learn has `a·d²ᵇ + 1`.
  2. `Optimizer::step_epochs` optimizes against a copy of the embedding made at the start
     of each call, where umap-learn updates one array in place. The result depends on how
     the epochs are divided among calls.
- Issues [#7](https://github.com/wilsonzlin/umap-rs/issues/7) ("Weird ring 2D embedding")
  and [#8](https://github.com/wilsonzlin/umap-rs/issues/8) ("Optimizer produces
  structureless embeddings…") are open.

Final embeddings from `umap-rs` 0.4.5 therefore say little about any initialization, and
the evidence below does not involve an optimizer.

A fork of `umap-rs` that carries both optimizer fixes and has spectral initialization
built in is at <https://github.com/TalonT-Org/umap-rs/tree/spectral-init>.

## Other UMAP implementations

Any implementation that can expose its fuzzy graph as CSR arrays can use `spectral-init`
without copying the graph:

```rust
use spectral_init::ndarray::Array2;
use spectral_init::sprs::CsMatViewI;
use spectral_init::{SpectralInitConfig, spectral_init};

/// Spectral initialization from borrowed CSR arrays of an `n × n` graph.
fn init_from_csr(
    n: usize,
    indptr: &[usize],
    indices: &[u32],
    weights: &[f32],
    n_components: usize,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    // A view over the three slices. `try_new` checks the CSR structure: `indptr` has
    // `n + 1` non-decreasing entries, and the column indices of each row are sorted
    // and below `n`.
    let graph = CsMatViewI::<f32, u32, usize>::try_new((n, n), indptr, indices, weights)
        .map_err(|(_, _, _, error)| error)?;
    Ok(spectral_init(
        graph,
        None,
        &SpectralInitConfig::new(n_components),
    )?)
}
```

The graph must be symmetric with finite, non-negative weights; CSC arrays work the same
way through `try_new_csc`.

## Evidence

Measured with `cargo run --release -p spectral-init-eval --bin init_evidence` on
2026-11-12 (UTC), with `spectral-init` 0.1.0 on x86_64 Linux, against reference data
generated by umap-learn 0.5.11 (the fixtures under `tests/fixtures/`). No UMAP optimizer
is involved.

**Agreement with umap-learn on connected graphs.** The solver runs on umap-learn's
Laplacian. The residual is `‖L·v − λ·v‖ / ‖v‖`, the eigenvalue error is against
umap-learn's eigenvalues, and the Gram determinant compares the spans of the two sets of
non-trivial eigenvectors (1 means the same subspace).

| Dataset | n | Solver | Max eigenpair residual | Max eigenvalue error | Subspace Gram determinant |
|---|---|---|---|---|---|
| blobs_connected_200 | 200 | Dense EVD | 1.3e-15 | 2.6e-17 | 1.000000000 |
| blobs_connected_2000 | 2000 | LOBPCG | 9.1e-6 | 6.6e-10 | 0.999999988 |
| circles_300 | 300 | Dense EVD | 1.2e-15 | 3.0e-12 | 1.000000000 |
| moons_200 | 200 | Dense EVD | 1.7e-10 | 2.4e-16 | 1.000000000 |
| near_dupes_100 | 100 | Dense EVD | 1.1e-15 | 4.2e-16 | 1.000000000 |

**Disconnected graphs.** Both implementations find the same connected components. The
separation ratio is the smallest distance between component centroids over the largest
spread within a component.

| Dataset | n | Components (umap-learn) | Components (spectral-init) | Separation ratio |
|---|---|---|---|---|
| blobs_50 | 50 | 3 | 3 | 1.90 |
| blobs_500 | 500 | 4 | 4 | 1.56 |
| blobs_5000 | 5000 | 2 | 2 | 1.88 |
| disconnected_200 | 200 | 4 | 4 | 1.58 |

**Trustworthiness of the initial coordinates** (k = 15), before any optimization, for
the spectral initialization and for umap-learn's random one (uniform in [−10, 10]). Eight
of the nine datasets have two-dimensional input, where a two-dimensional layout can
preserve neighbourhoods exactly; `blobs_connected_2000` has ten input dimensions.

| Dataset | n | Input dimensions | Spectral | Random (mean of 5 seeds) |
|---|---|---|---|---|
| blobs_50 | 50 | 2 | 0.8918 | 0.5535 |
| blobs_500 | 500 | 2 | 0.9714 | 0.5075 |
| blobs_5000 | 5000 | 2 | 0.9628 | 0.4998 |
| blobs_connected_200 | 200 | 2 | 0.9183 | 0.5140 |
| blobs_connected_2000 | 2000 | 10 | 0.8314 | 0.5000 |
| circles_300 | 300 | 2 | 0.8497 | 0.5134 |
| disconnected_200 | 200 | 2 | 0.9442 | 0.5218 |
| moons_200 | 200 | 2 | 0.9866 | 0.5194 |
| near_dupes_100 | 100 | 2 | 0.9130 | 0.5332 |
