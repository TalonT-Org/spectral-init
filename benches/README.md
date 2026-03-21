# Benchmark Baselines — spectral-init

These baselines were captured before Phase 3 optimization (SIMD, SELL-C-sigma, Rayon).
Use them to measure improvement after optimizations land.

## Running

```sh
cargo bench --features testing
```

HTML reports are written to `target/criterion/`.

## Benchmarks

| Benchmark | Input | What It Measures | Baseline (mean) |
|-----------|-------|------------------|----------------|
| `spmv_csr/200` | 200-node ring Laplacian | SpMV Phase 3 SIMD target | — |
| `spmv_csr/2000` | 2000-node ring Laplacian | SpMV at scale | — |
| `dense_evd_200` | 200-node ring Laplacian | Dense EVD via faer | — |
| `lobpcg_2000` | 2000-node ring Laplacian, k=3 | LOBPCG full solve | — |
| `rsvd_2000` | 2000-node ring Laplacian, k=3 | Randomized SVD full solve | — |
| `laplacian_build_2000` | 2000-node ring graph | Laplacian construction | — |
| `components_bfs_2000` | 2000-node ring graph | BFS connected components | — |
| `full_pipeline/200` | 200-node ring graph | End-to-end spectral_init | — |
| `full_pipeline/2000` | 2000-node ring graph | End-to-end at medium scale | — |

## Input Graphs

All benchmarks use deterministic synthetic ring graphs (`make_ring_graph(n, 2)`)
— no fixture generation is required. Each node has 4 undirected neighbours.

> **Note:** Ring graphs have uniform degree and regular memory-access patterns, making
> SpMV timings optimistic compared to real UMAP kNN graphs. Real graphs have non-uniform
> degree distributions and irregular access; expect 2–4× slower SpMV in practice.
> The baseline numbers here are pre-optimization reference points, not real-world targets.

## Phase 3 Target

`spmv_csr` is the primary Phase 3 optimization target. Compare
`spmv_csr/200` and `spmv_csr/2000` before and after SIMD/SELL-C-sigma changes.
