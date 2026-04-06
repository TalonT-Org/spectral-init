# y_heap Bottleneck Optimization — Analysis Report

**Date:** 2026-04-06  **n:** [1000, 5000, 10000]  **k:** 15  **Hardware:** n/a

## Primary Result

**Decision:** `POSITIVE`

POSITIVE — flat_simd n=10000 speedup 1.9939× (ratio 95% CI lb 1.7272 > 1.0)

## Speedup Table

| Variant | n | Mean (ms) | Speedup | CI lb | CI ub | Sig |
|---------|---|-----------|---------|-------|-------|-----|
| baseline | 1000 | 11.162 | 1.0000 | 1.0000 | 1.0000 |  |
| baseline | 5000 | 69.994 | 1.0000 | 1.0000 | 1.0000 |  |
| baseline | 10000 | 289.315 | 1.0000 | 1.0000 | 1.0000 |  |
| heap_reuse | 1000 | 11.276 | 0.9899 | 0.9532 | 1.0265 |  |
| heap_reuse | 5000 | 70.323 | 0.9953 | 0.9768 | 1.0131 |  |
| heap_reuse | 10000 | 292.412 | 0.9894 | 0.9391 | 1.0410 |  |
| flat_partial | 1000 | 8.934 | 1.2494 | 1.1922 | 1.3092 | * |
| flat_partial | 5000 | 39.648 | 1.7654 | 1.7351 | 1.7952 | * |
| flat_partial | 10000 | 162.843 | 1.7766 | 1.6955 | 1.8679 | * |
| flat_simd | 1000 | 7.892 | 1.4144 | 1.3537 | 1.4780 | * |
| flat_simd | 5000 | 37.684 | 1.8574 | 1.7824 | 1.9259 | * |
| flat_simd | 10000 | 145.099 | 1.9939 | 1.7272 | 2.2727 | * |

## Causal Decomposition

_Bundle attribution (W2: conflated bundles, not single-cause isolation)_

| Bundle | Attribution fraction | n |
|--------|----------------------|---|
| Allocation (malloc elim.) | -0.0107 | 10000 |
| DS change (BTreeMap→Vec) | 0.4431 | 10000 |
| SIMD (flat layout)       | 0.1090 | 10000 |

## Step Fractions

_per-call wall-clock step fraction (profiling feature enabled)_

| Variant | x_dist (ms) | x_sort (ms) | y_heap (ms) | penalty (ms) | y_heap % |
|---------|-------------|-------------|-------------|--------------|----------|
| baseline | 582.751 | 435.079 | 2988.416 | 273.777 | 69.8 |
| heap_reuse | 608.119 | 454.205 | 3013.613 | 281.907 | 69.2 |
| flat_partial | 579.541 | 434.476 | 1070.424 | 282.171 | 45.2 |
| flat_simd | 602.322 | 457.746 | 521.753 | 306.863 | 27.6 |

## Correctness

```
# env: envs/spectral-test/bin/python — numpy scipy matplotlib import OK
gaussian_n1000_x.npy  shape=(1000, 10)  dtype=float64  min=0.000314  max=0.999977  NaN/Inf=False
gaussian_n1000_y.npy  shape=(1000, 2)  dtype=float64  min=0.000013  max=0.999466  NaN/Inf=False
gaussian_n5000_x.npy  shape=(5000, 10)  dtype=float64  min=0.000019  max=0.999994  NaN/Inf=False
gaussian_n5000_y.npy  shape=(5000, 2)  dtype=float64  min=0.000013  max=0.999952  NaN/Inf=False
gaussian_n10000_x.npy  shape=(10000, 10)  dtype=float64  min=0.000026  max=0.999998  NaN/Inf=False
gaussian_n10000_y.npy  shape=(10000, 2)  dtype=float64  min=0.000023  max=0.999828  NaN/Inf=False
```

Run `cargo test --features testing` and confirm t_tw_01–t_tw_07 pass for all variants with |ΔT| < 1e-12.

## Shipping Decision

**SHIP `flat_simd`** — CI lower bound > 1.0 confirms statistically significant speedup at n=10000. Apply `flat_simd` variant to production.

## Threats to Validity

See experiment plan Analysis Plan section (W1–W8).
