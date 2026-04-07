# Analysis Report: kdtree-y-knn-trustworthiness

## Run Scope

- **Dry run:** True
- **N values analyzed:** [1000]
- **RAYON_NUM_THREADS:** 16
- **Rust channel:** nightly-x86_64-unknown-linux-gnu
- **Timestamp:** 2026-04-07T01:28:50-07:00
- **Scope qualifier:** All conclusions scoped to `RAYON_NUM_THREADS=16` threads on the benchmark machine.

> **DRY RUN — insufficient data for H1/H4 verdict; pipeline integrity verified**

## Total Speedup (flat_simd ns / kdtree ns; >1 = kdtree faster)

| dist | n | speedup | flat_simd CV | kdtree CV |
|------|---|---------|-------------|-----------|
| uniform | 1000 | 0.710 | 0.023 | 0.049 |
| gauss | 1000 | 0.710 | 0.046 | 0.032 |

## KD-tree Build Fraction (build / (build + query))

| dist | n | build_fraction | note |
|------|---|---------------|------|
| uniform | 1000 | 0.018 |  |
| gauss | 1000 | 0.019 |  |

## Query Speedup (flat_simd y_dist / kdtree y_kdtree_query)

| dist | n | query_speedup |
|------|---|--------------|
| uniform | 1000 | 3.161 |
| gauss | 1000 | 3.228 |

## Crossover Analysis (H2)

- **T_cross estimate (uniform):** None — speedup does not cross 1.0 in available N range
- **T_cross by rep:** rep1=N/A  rep2=N/A  rep3=N/A
- **T_cross range ratio (max/min):** N/A
- **T_cross stable (ratio ≤ 2×):** False

## n=75K Held-Out Check (RT8)

- n=75K not in analysis N_VALUES (dry-run mode or not collected).

## Hypothesis Evaluation

### H1 — KD-tree speedup at large n
- speedup_50k_uniform = N/A
- speedup_100k_uniform = N/A
- **H1 met:** None

### H2 — Crossover exists in [1K, 50K]
- T_cross = None
- **H2 met:** None

### H3 — Correctness (external)
- **H3 note:** ASSUMED PASS — run `cargo test t_tw_11 --features testing` separately

### H4 — Build fraction ≤ 10% at large n
- **H4 met:** None

## Five Success Criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | speedup_50k_uniform ≥ 5.0 | N/A (dry run) |
| 2 | speedup_100k_uniform ≥ 10.0 | N/A (dry run) |
| 3 | Correctness (t_tw_11/t_tw_08/t_tw_10) | External prerequisite: cargo test t_tw_11/t_tw_08/t_tw_10 --features testing |
| 4 | T_cross variance ≤ 2× across 3 reps | N/A (dry run) |
| 5 | build_fraction ≤ 10% at n=50K and n=100K | N/A (dry run) |

## Verdict

**INCONCLUSIVE**

DRY RUN — insufficient data for H1/H4 verdict; pipeline integrity verified

_All conclusions scoped to `RAYON_NUM_THREADS=16` threads on the benchmark machine._
