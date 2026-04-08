# spectral-init

> Spectral initialization for UMAP embeddings in Rust.

[![crates.io](https://img.shields.io/crates/v/spectral-init)](https://crates.io/crates/spectral-init)
[![docs.rs](https://img.shields.io/docsrs/spectral-init)](https://docs.rs/spectral-init)
[![CI](https://github.com/TalonT-Org/spectral-init/actions/workflows/test.yml/badge.svg)](https://github.com/TalonT-Org/spectral-init/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MSRV: 1.85](https://img.shields.io/badge/MSRV-1.85-blue)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

[Docs](https://docs.rs/spectral-init) · [Crate](https://crates.io/crates/spectral-init)

## What is this?

Spectral initialization seeds UMAP's SGD optimization from Laplacian eigenvectors
instead of random coordinates, giving the optimizer a globally-aware starting point
that reflects the true topology of the data. This is the single factor responsible
for Python UMAP's embedding quality advantage over random-init implementations.

This crate computes the symmetric normalized Laplacian

```
L = I - D^(-1/2) W D^(-1/2)
```

of the fuzzy k-NN graph and returns its smallest non-trivial eigenvectors as
initial UMAP coordinates. It is designed to integrate with
[`umap-rs`](https://github.com/wilsonzlin/umap-rs) via its `graph()` accessor, and
is usable with any Rust UMAP implementation that provides a sparse adjacency matrix.

## Quick Start

```sh
cargo add spectral-init
```

```rust
use spectral_init::{spectral_init, SpectralInitConfig};
use sprs::{CsMatI, TriMatI};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a small synthetic graph (replace with your real UMAP graph)
    let n = 10usize;
    let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
    for i in 0..n {
        let next = (i + 1) % n;
        tri.add_triplet(i, next, 1.0_f32);
        tri.add_triplet(next, i, 1.0_f32);
    }
    let graph: CsMatI<f32, u32, usize> = tri.to_csr();

    let embedding = spectral_init(&graph, 2, 42, None, SpectralInitConfig::default())?;
    assert_eq!(embedding.shape(), &[n, 2]);
    Ok(())
}
```

See [`examples/basic.rs`](examples/basic.rs) for the full runnable version.

## Integration with umap-rs

```rust
use spectral_init::{spectral_init, SpectralInitConfig};

let umap = umap_rs::Umap::fit(&data, umap_rs::UmapConfig::default())?;
let graph = umap.graph(); // &CsMatI<f32, u32, usize>
let init = spectral_init(graph, 2, 42, None, SpectralInitConfig::default())?;
// Pass `init` to your SGD optimizer as the initial embedding
```

## Standalone Usage — Building the Input Graph

If you are not using `umap-rs`, construct a `CsMatI<f32, u32, usize>` via `sprs::TriMatI`.
The matrix must be symmetric and non-negative. See
[`examples/from_adjacency_list.rs`](examples/from_adjacency_list.rs) for a complete example.

```rust
let mut tri: TriMatI<f32, u32> = TriMatI::new((n, n));
for &(i, j, w) in &edges {
    tri.add_triplet(i, j, w);
    tri.add_triplet(j, i, w); // symmetrize
}
let graph: CsMatI<f32, u32, usize> = tri.to_csr();
```

## ComputeMode

| Mode | Description |
|------|-------------|
| `PythonCompat` (default) | Matches Python UMAP's degree accumulation and solver path exactly; use when reproducibility against Python output is required. |
| `RustNative` | Enables AVX2+FMA SIMD degree accumulation on x86\_64; silently falls back on other platforms. Embeddings are subspace-equivalent to `PythonCompat`. |

```rust
use spectral_init::{spectral_init, ComputeMode, SpectralInitConfig};

let mut config = SpectralInitConfig::default();
config.compute_mode = ComputeMode::RustNative;
let embedding = spectral_init(&graph, 2, 42, None, config)?;
```

See [`examples/compute_modes.rs`](examples/compute_modes.rs) for a side-by-side comparison.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `testing` | Exposes internal solver functions for integration tests. **Not stable API.** |
| `cli` | Builds `trustworthiness` and `tw_profiler` binaries for embedding quality assessment. |
| `profiling` | Enables timing instrumentation in the solver escalation chain. |

## Performance

Benchmarks run on a 2000-node ring graph (4 neighbours per node):

| Stage | Time |
|-------|------|
| Full pipeline (`spectral_init`) | ~223 ms |
| Laplacian construction | ~228 µs |
| LOBPCG solve (k=3) | ~306 ms |
| Randomized SVD (k=3) | ~286 ms |

> These are pre-optimization baselines on a regular ring graph. Real k-NN graphs
> have non-uniform degree distributions; expect 2–4× slower SpMV in practice.
> See [`benches/README.md`](benches/README.md) for methodology and inputs.

## Correctness

Outputs are validated against Python UMAP reference fixtures stored as `.npz` files
in `tests/fixtures/`. Verification uses:

- **Residual checks**: `||L·v - λ·v|| / ||v|| < 0.005` for each eigenpair
- **Subspace comparison**: Gram determinant of `V_rust^T V_python` for near-degenerate eigenvalues
- **Sign-invariant matching**: eigenvectors are compared up to global sign flip

See the [implementation report](docs/umap-spectral-initialization-rust-implementation-report.md)
for the full algorithmic design.

## Requirements

- **MSRV:** Rust 1.85
- **No LAPACK/BLAS dependency** — pure Rust via `linfa-linalg` (LOBPCG) and `faer` (dense EVD)
- **AVX2+FMA** is optional and only used when `ComputeMode::RustNative` is selected on x86\_64
- Tested on Linux x86\_64; should work on any platform supported by the above crates

## Testing

Install [`cargo-nextest`](https://nexte.st/) if not already present:

```sh
cargo install cargo-nextest
```

Run the test suite (synthetic tests only — no fixture generation required):

```sh
cargo nextest run --features testing
```

Run with CI profile:

```sh
cargo nextest run --profile ci --features testing
```

Run fixture-dependent tests (requires Python fixture generation first):

```sh
python tests/generate_fixtures.py
cargo nextest run --profile with-fixtures --run-ignored all --features testing
```

## License

MIT — see [LICENSE](LICENSE).
