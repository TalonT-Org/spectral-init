# spectral-init

> **Warning:** This project is under active construction and is not ready for use.

Spectral initialization for UMAP embeddings in Rust.

Computes Laplacian eigenvectors of the fuzzy k-NN graph to provide globally-aware starting coordinates for SGD optimization — the single factor that makes Python UMAP produce superior embeddings.

## Status

Early development. See [the implementation report](docs/umap-spectral-initialization-rust-implementation-report.md) for the full design.

## Testing

This project uses [cargo-nextest](https://nexte.st/) as the test runner.

**Install nextest:**
```sh
cargo install cargo-nextest
```

**Run tests locally:**
```sh
cargo nextest run --features testing
```

**Run tests with CI profile (produces JUnit XML at `target/nextest/ci/junit.xml`):**
```sh
cargo nextest run --profile ci --features testing
```

**Run fixture-dependent tests (requires `.npz` files — see `tests/generate_fixtures.py`):**
```sh
cargo nextest run --profile with-fixtures --run-ignored all --features testing
```

## License

MIT
