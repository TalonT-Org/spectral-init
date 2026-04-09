# Subsampled Trustworthiness Tradeoff — Results Summary

## H1: Accuracy at Pre-specified Operating Points

| Hypothesis | Config | H1 Verdict |
|-----------|--------|-----------|
| H1_A | Approach A, MERFISH n=10K, m=2000 | PASS |
| H1_B | Approach B, MERFISH n=10K, m=5000 | N/A (no data) |

**Operational consequence:** Inconclusive (missing data).

## H2: Variance Scaling (std_T_sub vs. m)

Expected slope ≈ −0.5 (std ∝ m^{-0.5}).

| Approach | Slope | R² |
|---------|-------|----|
| A | N/A | N/A |
| B | N/A | N/A |

## H3: Dataset Variability Ratio (std_MERFISH / std_Gaussian)

| Approach | n | m | std_MERFISH | std_Gaussian | Ratio |
|---------|---|---|-------------|--------------|-------|
| A | 10000 | 2000 | N/A | N/A | N/A |
| A | 10000 | 5000 | N/A | N/A | N/A |
| A | 50000 | 2000 | N/A | N/A | N/A |
| A | 50000 | 5000 | N/A | N/A | N/A |
| B | 10000 | 2000 | N/A | N/A | N/A |
| B | 10000 | 5000 | N/A | N/A | N/A |
| B | 50000 | 2000 | N/A | N/A | N/A |
| B | 50000 | 5000 | N/A | N/A | N/A |

## H4: Speed Scaling (speedup vs. n/m)

Expected slope ≈ 1 for Approach A (O(mn) complexity), ≈ 2 for Approach B (O(m²)).

| Approach | Slope | R² |
|---------|-------|----|
| A | N/A | N/A |
| B | N/A | N/A |

## H5: Crossover m/n Ratio

Smallest m such that mean_abs_delta_T < 0.01 on MERFISH (Approach A/B reference).

| Approach | n | Crossover m | m/n ratio |
|---------|---|-------------|-----------|
| A | 10000 | 2000 | 0.2000 |
| A | 50000 | not reached | N/A |
| B | 10000 | 2000 | 0.2000 |
| B | 50000 | not reached | N/A |

## H6: Extrapolation to n=100K

Insufficient data for power-law fit (need both n=10K and n=50K MERFISH results).
