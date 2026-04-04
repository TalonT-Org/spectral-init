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
| `spmv_csr/200` | 200-node ring Laplacian | SpMV Phase 3 SIMD target | 396 ns |
| `spmv_csr/2000` | 2000-node ring Laplacian | SpMV at scale | 4.83 µs |
| `dense_evd_200` | 200-node ring Laplacian | Dense EVD via faer | 12.6 ms |
| `lobpcg_2000` | 2000-node ring Laplacian, k=3 | LOBPCG full solve | 306 ms |
| `rsvd_2000` | 2000-node ring Laplacian, k=3 | Randomized SVD full solve | 286 ms |
| `laplacian_build_2000` | 2000-node ring graph | Laplacian construction | 228 µs |
| `components_bfs_2000` | 2000-node ring graph | BFS connected components | 7.24 µs |
| `full_pipeline/200` | 200-node ring graph | End-to-end spectral_init | 14.2 ms |
| `full_pipeline/2000` | 2000-node ring graph | End-to-end at medium scale | 223 ms |
| `lobpcg_blobs5000/solve_eigenproblem` | 5,000-node blobs Laplacian | `solve_eigenproblem_pub()` full escalation chain | (TBD) |
| `lobpcg_merfish_10k/solve_eigenproblem` | MERFISH 10K Laplacian | `solve_eigenproblem_pub()` full escalation chain | (optional fixture) |
| `lobpcg_merfish_100k/solve_eigenproblem` | MERFISH 100K Laplacian | `solve_eigenproblem_pub()` full escalation chain | (optional fixture) |

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

## LOBPCG Microbenchmark

`lobpcg_bench` provides algorithm-level timing for `solve_eigenproblem_pub()`,
bypassing the `cargo nextest` process-spawn overhead that dominated the 100K
MERFISH wall-time measurement in the H4 benchmark (PR #216).

### Run (mandatory fixture only)

```sh
cargo bench --bench lobpcg_bench --features testing
```

### Run (with optional large fixtures)

Generate the Laplacian NPZ files first (see `benches/lobpcg_bench.rs` doc
comment for the Python export snippet), then:

```sh
# Generate optional fixtures (Python, run once)
python - <<'EOF'
import scipy.sparse as sp
# ... build lap as scipy CSR float64 ...
sp.save_npz("temp/lobpcg_bench/merfish_10k_laplacian.npz", lap)
EOF

cargo bench --bench lobpcg_bench --features testing
```

### Comparing with Python eigsh

The `solve_eigenproblem_pub()` timing from this benchmark is directly comparable
to the Python `eigsh()` call timing reported in the MERFISH benchmark reports
under `research/`. Both measure only the iterative eigensolver, not graph
construction or process startup.

### Sample Sizes and Warm-up

| Fixture | sample_size | warm_up_time | measurement_time | Expected per-call |
|---------|-------------|--------------|------------------|-------------------|
| blobs_5000 | 10 | 5 s | 60 s | ~300 ms |
| merfish_10k | 10 | 10 s | 300 s | ~1–5 s |
| merfish_100k | 10 | 30 s | 600 s | ~8–60 s |
