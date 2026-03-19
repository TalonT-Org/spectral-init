# spectral-init

Spectral initialization for UMAP embeddings in Rust.

Computes Laplacian eigenvectors of the fuzzy k-NN graph to provide globally-aware starting coordinates for SGD optimization — the single factor that makes Python UMAP produce superior embeddings.

## Status

Early development. See [the implementation report](docs/umap-spectral-initialization-rust-implementation-report.md) for the full design.

## License

MIT
