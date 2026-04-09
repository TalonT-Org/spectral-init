## Speedup Results

_Amdahl x_dist fraction: 0.6891_

| Variant | x_dist speedup | Total speedup | Amdahl predicted | H1 pass (>=1.5x) | Correctness delta |
|---------|---------------|--------------|-----------------|-----------------|------------------|
| avx2_looped | 2.09x | 1.57x | 1.56x | Y | 0.00e+00 |
| avx512_looped | 1.98x | 1.54x | 1.52x | Y | 0.00e+00 |

**AVX-512 marginal gain over looped AVX2:** 0.98x (<1.2x -- ship AVX2 only)
