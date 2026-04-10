# Subsampled Trustworthiness Tradeoff — Results Summary

## H1: Accuracy at Pre-specified Operating Points

| Hypothesis | Config | H1 Verdict |
|-----------|--------|-----------|
| H1_A | Approach A, MERFISH n=10K, m=2000 | PASS |
| H1_B | Approach B, MERFISH n=10K, m=5000 | PASS |

**Operational consequence:** Use Approach A (m=2000) as default; Approach B acceptable at m=5000.

## H2: Variance Scaling (std_T_sub vs. m)

Expected slope ≈ −0.5 (std ∝ m^{-0.5}).

| Approach | Slope | R² |
|---------|-------|----|
| A | -0.6573 | 0.8524 |
| B | -0.5816 | 0.8024 |

## H3: Dataset Variability Ratio (std_MERFISH / std_Gaussian)

| Approach | n | m | std_MERFISH | std_Gaussian | Ratio |
|---------|---|---|-------------|--------------|-------|
| A | 10000 | 2000 | 0.00215 | 0.00113 | 1.904 |
| A | 10000 | 5000 | 0.00065 | 0.00053 | 1.244 |
| A | 20000 | 2000 | 0.00260 | 0.00141 | 1.853 |
| A | 20000 | 5000 | 0.00105 | 0.00098 | 1.070 |
| B | 10000 | 2000 | 0.00303 | 0.00252 | 1.205 |
| B | 10000 | 5000 | 0.00150 | 0.00058 | 2.580 |
| B | 20000 | 2000 | 0.00460 | 0.00176 | 2.618 |
| B | 20000 | 5000 | 0.00200 | 0.00107 | 1.877 |

## H4: Speed Scaling (speedup vs. n/m)

Expected slope ≈ 1 for Approach A (O(mn) complexity), ≈ 2 for Approach B (O(m²)).

| Approach | Slope | R² |
|---------|-------|----|
| A | 0.9555 | 0.9915 |
| B | 1.984 | 0.9948 |

## H5: Crossover m/n Ratio

Smallest m such that mean_abs_delta_T < 0.01 on MERFISH (Approach A/B reference).

| Approach | n | Crossover m | m/n ratio |
|---------|---|-------------|-----------|
| A | 10000 | 250 | 0.0250 |
| A | 20000 | 250 | 0.0125 |
| B | 10000 | 250 | 0.0250 |
| B | 20000 | 250 | 0.0125 |
  Approach A: crossover ratios within 2×? YES
  Approach B: crossover ratios within 2×? YES

## H6: Extrapolation to n=100K ⚠️ OUT-OF-DISTRIBUTION PROJECTION (fit from n=10K and n=20K)

Fit: |ΔT| = 0.000180 × n^0.2404  (Approach A, m=2000, MERFISH, 2 data points)

**Predicted |ΔT| at n=100K: 0.002870**

> This is an extrapolation (out-of-distribution). Treat as an order-of-magnitude estimate only.
