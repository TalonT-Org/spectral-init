# User guide: spectral-init

The API reference is on [docs.rs/spectral-init](https://docs.rs/spectral-init). This guide
covers how to build the input, how to configure the call, what happens with disconnected
graphs, and what each error means. Every Rust snippet here is compiled and run by the
crate's test suite.

## Contents

1. [Requirements](#requirements)
2. [Background](#background)
3. [Installation](#installation)
4. [The call](#the-call)
5. [Building the input graph](#building-the-input-graph)
6. [Using it with umap-rs](#using-it-with-umap-rs)
7. [Configuration](#configuration)
8. [Disconnected graphs and the `data` argument](#disconnected-graphs-and-the-data-argument)
9. [Errors](#errors)
10. [Reproducibility](#reproducibility)
11. [Performance notes](#performance-notes)

## Requirements

- Rust 1.88 or later.
- No BLAS, LAPACK or C toolchain. The linear algebra comes from `faer` (dense and sparse
  factorizations) and `linfa-linalg` (LOBPCG), both pure Rust.
- `linfa-linalg` is built on ndarray 0.16, while this crate (and `umap-rs`) use
  ndarray 0.17, so a build also compiles ndarray 0.16, rand 0.8 and thiserror 1. None of
  them appears in the public API.

## Background

**Spectral initialization** places every node of a graph at its entries in a few
eigenvectors of the graph Laplacian. The eigenvectors of the smallest non-zero eigenvalues
change slowly from node to node, so nodes that are well connected end up close together
and the layout reflects the large-scale shape of the graph. umap-learn uses this layout as
the default starting point of its optimization (McInnes, Healy and Melville, *UMAP:
Uniform Manifold Approximation and Projection for Dimension Reduction*,
[arXiv:1802.03426](https://arxiv.org/abs/1802.03426)).

**The graph** is a symmetric sparse matrix `W`. `W[i, j]` is the strength of the
connection between nodes `i` and `j`; a larger value means the two are more similar. In
UMAP it is the fuzzy k-nearest-neighbour graph, whose weights lie in `[0, 1]`.

## Installation

```toml
[dependencies]
spectral-init = "0.1"
```

The crate re-exports the two crates whose types appear in its API, as
`spectral_init::sprs` and `spectral_init::ndarray`. If you depend on them directly, use
`sprs` 0.11 and `ndarray` 0.17 so that the types are the same. The crate has no Cargo
features to choose from.

## The call

```rust
use spectral_init::sprs::CsMatI;
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

fn main() -> Result<(), SpectralError> {
    // A path of four nodes: 0 – 1 – 2 – 3.
    let graph = CsMatI::<f32, u32, usize>::new(
        (4, 4),
        vec![0, 1, 3, 5, 6],
        vec![1, 0, 2, 1, 3, 2],
        vec![1.0; 6],
    );

    let config = SpectralInitConfig::new(2).with_seed(42);
    let coords = spectral_init(graph.view(), None, &config)?;

    // One row per node, one column per embedding dimension.
    assert_eq!(coords.dim(), (4, 2));
    assert!(coords.is_standard_layout());
    Ok(())
}
```

`spectral_init` takes three arguments:

- `graph` — a view of the graph matrix (`CsMatViewI<f32, u32, usize>`); `.view()` on an
  owned `CsMatI` gives one.
- `data` — `Option<ArrayView2<f32>>`, the points the graph was built from, one row per
  node. It is needed only for graphs with many connected components; see
  [below](#disconnected-graphs-and-the-data-argument).
- `config` — a [`SpectralInitConfig`](#configuration), which carries the number of
  embedding dimensions and the seed.

The result is an `(n, n_components)` array of `f32` in standard (row-major) layout. The
coordinates are scaled so that the largest magnitude is 10, and carry Gaussian noise with
standard deviation 10⁻⁴.

## Building the input graph

The graph must be

- **square**, with one row and column per node, and more nodes than embedding
  dimensions;
- **symmetric**: `W[i, j]` and `W[j, i]` are both stored and equal (a relative difference
  up to 10⁻⁶ is tolerated);
- **non-negative and finite**: no negative weight, NaN or infinity.

A stored zero is treated as an absent edge. A diagonal entry (a self-loop) counts towards
the degree of its node and towards the adjacency, as it does in umap-learn. CSR and CSC
storage are both accepted. Anything else is reported as an [error](#errors) before any
computation starts.

### From a list of edges

```rust
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

fn main() -> Result<(), SpectralError> {
    // Undirected edges as (node, node, weight).
    let edges = [
        (0, 1, 0.9_f32),
        (0, 2, 0.7),
        (1, 3, 0.8),
        (2, 3, 0.6),
        (3, 4, 0.5),
    ];
    let n = 5;

    // Store every edge in both directions: the matrix has to be symmetric.
    let mut triplets: TriMatI<f32, u32> = TriMatI::new((n, n));
    for &(i, j, weight) in &edges {
        triplets.add_triplet(i, j, weight);
        triplets.add_triplet(j, i, weight);
    }
    let graph: CsMatI<f32, u32, usize> = triplets.to_csr();

    let coords = spectral_init(graph.view(), None, &SpectralInitConfig::new(2))?;
    assert_eq!(coords.dim(), (n, 2));
    Ok(())
}
```

`TriMatI` sums the weights of an edge that is added twice. Add each undirected edge once
in each direction.

### From a dense similarity matrix

```rust
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

fn main() -> Result<(), SpectralError> {
    // A symmetric similarity matrix; zeros mean "no edge".
    let similarity = [
        [0.0_f32, 0.9, 0.7, 0.0],
        [0.9, 0.0, 0.0, 0.8],
        [0.7, 0.0, 0.0, 0.6],
        [0.0, 0.8, 0.6, 0.0],
    ];
    let n = similarity.len();

    let mut triplets: TriMatI<f32, u32> = TriMatI::new((n, n));
    for (i, row) in similarity.iter().enumerate() {
        for (j, &weight) in row.iter().enumerate() {
            if weight > 0.0 {
                triplets.add_triplet(i, j, weight);
            }
        }
    }
    let graph: CsMatI<f32, u32, usize> = triplets.to_csr();

    let coords = spectral_init(graph.view(), None, &SpectralInitConfig::new(2))?;
    assert_eq!(coords.dim(), (n, 2));
    Ok(())
}
```

### From points, with a brute-force neighbour search

This builds a k-nearest-neighbour graph of two well-separated clusters and embeds it. For
real data, use a nearest-neighbour library instead of the quadratic search.

```rust
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

fn squared_distance(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn main() -> Result<(), SpectralError> {
    // Two clusters of ten points each, five units apart.
    let points: Vec<[f32; 2]> = (0..10)
        .map(|i| [i as f32 * 0.1, 0.0])
        .chain((0..10).map(|i| [5.0 + i as f32 * 0.1, 0.0]))
        .collect();
    let n = points.len();
    let k = 5;

    // A weight for every pair (i, j) where j is one of the k nearest neighbours of i
    // or the other way round. The larger of the two directed weights is kept, which
    // makes the matrix symmetric.
    let mut weights = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        let mut neighbours: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (squared_distance(&points[i], &points[j]), j))
            .collect();
        neighbours.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(distance_squared, j) in neighbours.iter().take(k) {
            let weight = 1.0 / (1.0 + distance_squared.sqrt());
            weights[i][j] = weights[i][j].max(weight);
            weights[j][i] = weights[j][i].max(weight);
        }
    }
    let mut triplets: TriMatI<f32, u32> = TriMatI::new((n, n));
    for (i, row) in weights.iter().enumerate() {
        for (j, &weight) in row.iter().enumerate() {
            if weight > 0.0 {
                triplets.add_triplet(i, j, weight);
            }
        }
    }
    let graph: CsMatI<f32, u32, usize> = triplets.to_csr();

    let config = SpectralInitConfig::new(2).with_seed(42);
    let coords = spectral_init(graph.view(), None, &config)?;
    assert_eq!(coords.dim(), (n, 2));
    assert!(coords.iter().all(|v| v.is_finite()));

    // The clusters do not share an edge, so the graph has two connected components
    // and each is placed around its own position.
    let centre = |range: std::ops::Range<usize>| -> [f32; 2] {
        let count = range.len() as f32;
        let sum = range.fold([0.0, 0.0], |acc, i| {
            [acc[0] + coords[[i, 0]], acc[1] + coords[[i, 1]]]
        });
        [sum[0] / count, sum[1] / count]
    };
    let (a, b) = (centre(0..10), centre(10..20));
    assert!(squared_distance(&a, &b).sqrt() > 1.0);
    Ok(())
}
```

### From raw CSR arrays

If the graph already exists as CSR arrays (`indptr`, `indices`, `data`), borrow them as a
view instead of copying them:

```rust
use spectral_init::sprs::CsMatViewI;
use spectral_init::{SpectralInitConfig, spectral_init};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The path 0 – 1 – 2 – 3 in CSR form.
    let indptr: Vec<usize> = vec![0, 1, 3, 5, 6];
    let indices: Vec<u32> = vec![1, 0, 2, 1, 3, 2];
    let weights: Vec<f32> = vec![1.0; 6];

    // `try_new` checks the structure: `indptr` has `n + 1` non-decreasing entries, and
    // the column indices of every row are sorted and below `n`.
    let graph = CsMatViewI::<f32, u32, usize>::try_new((4, 4), &indptr, &indices, &weights)
        .map_err(|(_, _, _, error)| error)?;

    let coords = spectral_init(graph, None, &SpectralInitConfig::new(2))?;
    assert_eq!(coords.dim(), (4, 2));
    Ok(())
}
```

`try_new_csc` does the same for CSC arrays.

## Using it with umap-rs

[`umap-rs`](https://crates.io/crates/umap-rs) takes the initial coordinates from the
caller and exposes the graph they are computed from:

1. `Umap::learn_manifold(data, knn_indices, knn_dists)` builds the fuzzy graph.
2. `spectral_init(manifold.graph().view(), Some(data), &config)` computes the initial
   coordinates.
3. `Optimizer::new(manifold, init, ..)` or `Umap::fit(.., init.view())` optimizes from
   them.

The [integration guide](umap-rs-integration.md) has the complete code for both
variants, the settings that matter (`symmetrize` must stay `true`), and the current state
of `umap-rs`.

## Configuration

`SpectralInitConfig::new(n_components)` creates a configuration with the defaults; the
`with_*` methods change one setting each and can be chained.

```rust
use spectral_init::{ComputeMode, SpectralInitConfig};

let config = SpectralInitConfig::new(2)
    .with_seed(42)
    .with_compute_mode(ComputeMode::RustNative)
    .with_dense_memory_budget(8 << 30); // 8 GiB

assert_eq!(config.n_components(), 2);
assert_eq!(config.seed(), 42);
assert_eq!(config.compute_mode(), ComputeMode::RustNative);
assert_eq!(config.dense_memory_budget(), 8 << 30);
```

| Setting | Default | Meaning |
|---------|---------|---------|
| `n_components` | (required) | Number of embedding dimensions. Must be at least 1 and less than the number of nodes. |
| `with_seed` | `0` | Seed of the random numbers: the starting vectors of the iterative solvers and the final noise. |
| `with_compute_mode` | `PythonCompat` | See below. |
| `with_dense_memory_budget` | 4 GiB | Largest dense matrix, in bytes, a solver may allocate. See [Performance notes](#performance-notes). |

### Compute mode

| | `PythonCompat` | `RustNative` |
|---|---|---|
| Degree sums | single precision, in SciPy's order | double precision |
| Sparse matrix-vector product in the iterative solvers | scalar | AVX2+FMA on x86_64 CPUs that have both (detected at run time), scalar elsewhere |
| Relation to umap-learn | the mode the reference tests run in | same subspace, coordinates differ by rounding |

Choose `PythonCompat` to stay as close as possible to umap-learn, and `RustNative` when
that is not a concern. On targets other than x86_64 the two modes still differ in how
degrees are summed.

## Disconnected graphs and the `data` argument

A graph that falls into several connected components has no single spectral layout: the
eigenvectors of eigenvalue zero only tell the components apart. `spectral_init` therefore

1. embeds every component on its own, with the same solvers;
2. chooses a position for every component;
3. scales each component to half the distance to its nearest neighbour component and
   moves it to its position.

How the positions are chosen depends on the number of components:

- **Up to `2 · n_components` components.** The components are placed on the coordinate
  axes, at `+e₁, +e₂, …, −e₁, −e₂, …`. `data` is not consulted and may be `None`.
- **More than `2 · n_components` components.** The components are laid out by a spectral
  embedding of their centroids, and the centroids are computed from `data`. Without
  `data` the call returns `SpectralError::DataRequired`.

A component with fewer than `2 · n_components` nodes is not embedded: all its nodes are
put at the position of the component, and only the final noise separates them.

`data`, when given, must have one row per node and only finite values. It is validated
even when it ends up unused.

```rust
use spectral_init::ndarray::Array2;
use spectral_init::sprs::{CsMatI, TriMatI};
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

fn main() -> Result<(), SpectralError> {
    // Five separate pairs of nodes: five connected components.
    let n = 10;
    let mut triplets: TriMatI<f32, u32> = TriMatI::new((n, n));
    for pair in 0..5 {
        triplets.add_triplet(2 * pair, 2 * pair + 1, 1.0);
        triplets.add_triplet(2 * pair + 1, 2 * pair, 1.0);
    }
    let graph: CsMatI<f32, u32, usize> = triplets.to_csr();
    let config = SpectralInitConfig::new(2);

    // Five components are more than 2 · 2, so the data is required.
    let without_data = spectral_init(graph.view(), None, &config);
    assert!(matches!(
        without_data,
        Err(SpectralError::DataRequired {
            connected_components: 5,
            n_components: 2,
            ..
        })
    ));

    // The points the graph was built from: one row per node.
    let data = Array2::from_shape_fn((n, 3), |(node, feature)| {
        (node / 2) as f32 + 0.1 * feature as f32
    });
    let coords = spectral_init(graph.view(), Some(data.view()), &config)?;
    assert_eq!(coords.dim(), (n, 2));
    Ok(())
}
```

## Errors

`spectral_init` does not panic. It returns a `SpectralError`, whose variants carry the
details as fields. The enum and its variants are `#[non_exhaustive]`, so patterns need
`..` and a wildcard arm:

```rust
use spectral_init::sprs::CsMatI;
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

// One edge from node 0 to node 1 and none back.
let graph = CsMatI::<f32, u32, usize>::new((3, 3), vec![0, 1, 1, 1], vec![1], vec![1.0]);

match spectral_init(graph.view(), None, &SpectralInitConfig::new(2)) {
    Ok(_) => unreachable!("the graph is not symmetric"),
    Err(SpectralError::NotSymmetric { row, col, .. }) => assert_eq!((row, col), (0, 1)),
    Err(other) => panic!("unexpected error: {other}"),
}
```

### Errors in the input

These are found before any computation, in this order.

| Variant | Cause | What to do |
|---------|-------|------------|
| `InvalidParameter { name, reason }` | `n_components` is 0. | Ask for at least one dimension. |
| `NotSquare { rows, cols }` | The matrix is not square. | Pass the `n × n` graph, not a rectangular neighbour table. |
| `TooFewNodes { n, n_components }` | The graph has `n ≤ n_components` nodes (an empty graph included). | Use fewer dimensions; `n_components` must be below `n`. |
| `NonFiniteWeight { row, col }` | The weight at `(row, col)` is NaN or infinite. | Fix the weight computation; distances of zero or overflow are the usual source. |
| `NegativeWeight { row, col }` | The weight at `(row, col)` is negative. | Use similarities, not signed values. |
| `NotSymmetric { row, col }` | The entry at `(row, col)` has no equal entry at `(col, row)`. | Store every edge in both directions with the same weight. A directed k-nearest-neighbour graph has to be symmetrized first. |
| `DegreeOverflow { node }` | The weights of `node` sum to more than an `f32` can hold. | Rescale the weights. |
| `DataShapeMismatch { expected_rows, actual_rows }` | `data` does not have one row per node. | Pass the rows the graph was built from, in the same order. |
| `NonFiniteData { row, col }` | `data` contains NaN or an infinity. | Clean the data, or pass `None` if the graph has few components. |

Row and column numbers refer to the matrix as it was passed, whatever its storage order.

### Errors of the computation

| Variant | Cause | What to do |
|---------|-------|------------|
| `DataRequired { connected_components, n_components }` | The graph has more than `2 · n_components` connected components and `data` is `None`. | Pass `Some(data)`, or check why the graph is so fragmented (too few neighbours, isolated points). |
| `DenseBudgetExceeded { required_bytes, budget_bytes }` | The iterative solvers were not accepted and the dense one needs a larger matrix than the budget allows; or the graph has so many components that the matrix of their affinities exceeds it. | Raise the budget with `with_dense_memory_budget` if the memory is available, or substitute another initialization. |
| `ConvergenceFailure { n }` | No solver produced eigenvectors of acceptable quality for a problem of size `n`. | Substitute another initialization. If the graph is an ordinary one, please report it. |
| `DegenerateEmbedding` | Every computed coordinate is zero. | Substitute another initialization. |

umap-learn falls back to a random layout in the cases of the second table. This crate
never does that silently; the README shows a `match` that substitutes a fallback.

## Reproducibility

For a fixed `Cargo.lock`, target, CPU feature set and thread configuration, the result is
a deterministic function of the graph, the data and the configuration. Random numbers
come from the ChaCha12 generator (`rand_chacha::ChaCha12Rng`).

Across thread configurations, CPU feature sets and targets the results agree to rounding
but not bit for bit, because the dense solvers split their work differently. Measured on
x86_64 with faer 0.24.4, one thread against four changed no coordinate for a 500-node
graph and changed coordinates by at most 1.2 × 10⁻¹⁰ for a 1500-node graph, on
coordinates of magnitude up to 10.

The solvers use faer's global parallelism setting, which defaults to the Rayon thread
pool. To pin it, either set the `RAYON_NUM_THREADS` environment variable or, with
`faer = "0.24"` among your dependencies, set it in code:

```rust
use spectral_init::sprs::CsMatI;
use spectral_init::{SpectralError, SpectralInitConfig, spectral_init};

fn main() -> Result<(), SpectralError> {
    // Run the dense solvers on the calling thread only.
    faer::set_global_parallelism(faer::Par::Seq);

    let graph = CsMatI::<f32, u32, usize>::new(
        (4, 4),
        vec![0, 1, 3, 5, 6],
        vec![1, 0, 2, 1, 3, 2],
        vec![1.0; 6],
    );
    let config = SpectralInitConfig::new(2).with_seed(42);
    let first = spectral_init(graph.view(), None, &config)?;
    let second = spectral_init(graph.view(), None, &config)?;
    assert_eq!(first, second);
    Ok(())
}
```

## Performance notes

**The solver is chosen automatically.** Graphs with fewer than 2000 nodes are solved by a
dense eigendecomposition, which is exact and costs time cubic in the number of nodes.
Larger graphs go to the iterative solvers, which only multiply by the sparse Laplacian;
if one of them does not reach its accuracy threshold, the next is tried. The crate logs
the solver that produced the result through the [`log`](https://crates.io/crates/log)
crate at debug level.

**The dense memory budget** matters only when the iterative solvers are not accepted on a
large graph. With the default of 4 GiB:

- the randomized SVD needs an `n × (n/10 + n_components + 1)` matrix and is skipped
  above roughly 73,000 nodes;
- the final dense eigendecomposition needs an `n × n` matrix and is refused above 23,170
  nodes, in which case the call returns `DenseBudgetExceeded`.

The budget caps each matrix, not the total: the dense solvers hold a few matrices of
that size at once.

**`RustNative` on x86_64** uses an AVX2+FMA kernel for the sparse matrix-vector product
of the iterative solvers. It is selected at run time; building with
`-C target-cpu=native` is not needed for it.

**Each connected component is solved separately**, so a graph of many small components
is cheaper than a connected graph of the same size.
