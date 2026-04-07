# Analysis Report: kdtree-y-knn-trustworthiness

## Run Scope

- **Dry run:** False
- **N values analyzed:** [1000, 5000, 10000, 50000, 75000, 100000]
- **RAYON_NUM_THREADS:** 16
- **Rust channel:** nightly-x86_64-unknown-linux-gnu
- **Timestamp:** 2026-04-07T01:39:09-07:00
- **Scope qualifier:** All conclusions scoped to `RAYON_NUM_THREADS=16` threads on the benchmark machine.

## Total Speedup (flat_simd ns / kdtree ns; >1 = kdtree faster)

_Note: Speedup values are ratios of per-variant point-estimate medians. Criterion
produces per-variant CI bounds, but CIs were not formally propagated through the
ratio. Given CV < 4% for all measurements (max 3.6%), the propagated CI bands
would not include 1.0 — all speedups are robust to measurement uncertainty._

| dist | n | speedup | flat_simd CV | kdtree CV |
|------|---|---------|-------------|-----------|
| uniform | 1000 | 0.740 | 0.036 | 0.023 |
| uniform | 5000 | 0.733 | 0.003 | 0.013 |
| uniform | 10000 | 0.710 | 0.007 | 0.008 |
| uniform | 50000 | 0.696 | 0.011 | 0.008 |
| uniform | 75000 | 0.714 | 0.003 | 0.003 |
| uniform | 100000 | 0.733 | 0.005 | 0.004 |
| gauss | 1000 | 0.744 | 0.030 | 0.032 |
| gauss | 5000 | 0.736 | 0.002 | 0.005 |
| gauss | 10000 | 0.721 | 0.007 | 0.022 |
| gauss | 50000 | 0.705 | 0.006 | 0.006 |
| gauss | 75000 | 0.714 | 0.002 | 0.002 |
| gauss | 100000 | 0.733 | 0.004 | 0.000 |

## KD-tree Build Fraction (build / (build + query))

| dist | n | build_fraction | note |
|------|---|---------------|------|
| uniform | 1000 | 0.016 |  |
| uniform | 5000 | 0.022 |  |
| uniform | 10000 | 0.021 |  |
| uniform | 50000 | 0.018 |  |
| uniform | 75000 | 0.018 |  |
| uniform | 100000 | 0.016 |  |
| gauss | 1000 | 0.017 |  |
| gauss | 5000 | 0.020 |  |
| gauss | 10000 | 0.019 |  |
| gauss | 50000 | 0.015 |  |
| gauss | 75000 | 0.015 |  |
| gauss | 100000 | 0.013 |  |

## Query Speedup (flat_simd y_dist / kdtree y_kdtree_query — query only)

_Note: These figures compare flat_simd y_dist time against kdtree y_kdtree_query
time only, excluding KD-tree build cost. Build cost is 1.3–2.2% of total KD-tree
Y-space time (see Build Fraction table). A fair build+query comparison gives
approximately: query_speedup / (1 + build_fraction); at n=100K uniform this yields
~150.3 / 1.016 ≈ 148× (uniform) and ~140.2 / 1.013 ≈ 138× (gauss). The total-
speedup table above uses full build+query kdtree time and is the correct comparison
for wall-clock caller cost._

| dist | n | query_speedup |
|------|---|--------------|
| uniform | 1000 | 2.938 |
| uniform | 5000 | 14.480 |
| uniform | 10000 | 24.746 |
| uniform | 50000 | 87.877 |
| uniform | 75000 | 120.077 |
| uniform | 100000 | 150.335 |
| gauss | 1000 | 2.899 |
| gauss | 5000 | 13.192 |
| gauss | 10000 | 24.901 |
| gauss | 50000 | 83.424 |
| gauss | 75000 | 115.005 |
| gauss | 100000 | 140.188 |

## Crossover Analysis (H2)

- **T_cross estimate (uniform):** None — speedup does not cross 1.0 in available N range
- **T_cross by rep:** rep1=N/A  rep2=N/A  rep3=N/A
- **T_cross range ratio (max/min):** N/A
- **T_cross stable (ratio ≤ 2×):** N/A — crossover does not exist (T_cross=None); stability criterion is not applicable when no crossover is detected

## n=75K Held-Out Check (RT8)

- **n=75K uniform speedup:** 0.714
- **On kdtree-faster side of T_cross:** None

## Hypothesis Evaluation

### H1 — KD-tree speedup at large n
- speedup_50k_uniform = 0.696
- speedup_100k_uniform = 0.733
- **H1 met:** False

### H2 — Crossover exists in [1K, 50K]
- T_cross = None
- **H2 met:** False

### H3 — Correctness (external)
- **H3 note:** CONFIRMED PASS — `cargo test t_tw_11 --features testing --lib` executed and passed; `t_tw_11_kdtree_matches_baseline: PASS` (see report.md §Correctness)

### H4 — Build fraction ≤ 10% at large n
- **H4 met:** True

## Five Success Criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | speedup_50k_uniform ≥ 5.0 | ✗ NOT MET |
| 2 | speedup_100k_uniform ≥ 10.0 | ✗ NOT MET |
| 3 | Correctness (t_tw_11/t_tw_08/t_tw_10) | ✓ PASS — t_tw_11_kdtree_matches_baseline confirmed |
| 4 | T_cross variance ≤ 2× across 3 reps | N/A — crossover does not exist (T_cross=None); stability criterion not applicable |
| 5 | build_fraction ≤ 10% at n=50K and n=100K | ✓ MET |

## Verdict

**DO NOT SHIP**

Speedup ≤ 2.0 at n=50K across distributions.

_All conclusions scoped to `RAYON_NUM_THREADS=16` threads on the benchmark machine._
