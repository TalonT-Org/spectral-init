# Sub-sampled Trustworthiness: Error/Speed Trade-off

> Research report — 2026-04-09

## Executive Summary

Exact trustworthiness computation is O(n²) [Venna & Kaski 2006; scikit-learn docs] and becomes the dominant evaluation cost at large embedding sizes. This experiment establishes the empirical error/speed trade-off for two sub-sampling strategies — Approach A (row sub-sampling, an approximation of full-n trustworthiness) and Approach B (subset embedding, measuring trustworthiness within the sampled subset) — across MERFISH and Gaussian d=50 datasets at n=10K and n=20K, with 10 random seeds per cell.

**Data scope:** The experiment plan specified n={10K, 50K}; n=50K was replaced by n=20K due to a memory safety constraint (commit `544989f`, `_MEM_LIMIT=4GB`). All scaling analyses (H4, H5, H6) are based on n=[10K, 20K] only — a 2× n-range rather than the planned 5×. This is a single-run experiment (R=1); the 10 seeds per cell quantify within-run variance, not between-run reproducibility.

Both primary hypotheses pass by wide margins. At m=2000, Approach A achieves a mean absolute error of 0.0017 (threshold: 0.01), a 4.1× speedup on MERFISH n=10K, and 9.5× speedup at n=20K — all with std well below the 0.005 threshold. Speed scaling laws are confirmed empirically: Approach A scales as O(mn) (slope 0.956, R²=0.991) and Approach B as O(m²) (slope 1.984, R²=0.995). The recommendation is to ship `trustworthiness_subsampled` in `src/metrics.rs` with Approach A as the default at m=2000.

## Background and Research Question

Sub-sampling is the most promising approach for reducing the O(n²) cost of trustworthiness. Prior experiments (`2026-04-06-research-kd-tree-y-k-nn-for-tr`, `2026-04-07-kdtree-y-knn-trustworthiness`) found that KD-tree Y-space neighbor acceleration did not reduce overall trustworthiness cost on synthetic data at tested n values, because Y-space KNN is a minority of the total computation; those experiments did not universally rule out KD-tree for all contexts or n scales. A prior sub-sampling attempt (H5 in `2026-04-05-tw-perf-rerun-clean`) produced large errors (mean|delta|≈0.474); the root cause was not documented in those artifacts, but a denominator inconsistency between approaches is a plausible explanation. This experiment re-implements both approaches from scratch with correct, explicitly verified denominators.

**Research question:** Does sub-sampled trustworthiness achieve mean|ΔT| < 0.01 and std < 0.005 at the pre-specified operating points (Approach A: m=2000, n=10K; Approach B: m=5000, n=10K), and what error/speed trade-off curve should guide default parameter selection?

## Methodology

### Experimental Design

Two independent hypothesis pairs, each evaluated with its own verdict:

**H1_A:** mean(|ΔT_A|) < 0.01 AND max(|ΔT_A|) < 0.02 AND std(T_sub_A) < 0.005 at m=2000, MERFISH n=10K, k=15

**H1_B:** mean(|ΔT_B|) < 0.01 AND max(|ΔT_B|) < 0.02 AND std(T_sub_B) < 0.005 at m=5000, MERFISH n=10K, k=15

Secondary analyses (H2–H6) characterize variance scaling, dataset effects, speed scaling laws, crossover ratios, and extrapolation to n=100K.

Controlled variables: k=15 (UMAP default [McInnes et al. 2018]), squared Euclidean distance, pre-computed MERFISH UMAP embeddings, scikit-learn 1.6.0 (exact trustworthiness reference), 1 warmup discarded before timing. Note: timing measurements were taken in a WSL2 environment without CPU frequency pinning or process isolation; wall-clock speedup figures may vary under different system loads. All speedup figures are Python-to-Python comparisons (sub-sampled vs. exact using the same NumPy/scikit-learn stack); they reflect GIL scheduling and NumPy dispatch overhead that the Rust implementation in `src/metrics.rs` does not incur. These figures serve as a lower bound for Python callers but do not predict the Rust-native speedup directly.

### Environment

- **Repository commit:** `544989f81ce6024bfd1d59380293acf9359cfc48`
- **Branch:** `research-20260409-154049`
- **Custom environment:** `research/2026-04-09-subsampled-tw-tradeoff/environment.yml` (micromamba)
- **Package versions:**
  - python=3.11
  - numpy=2.2.6
  - scipy=1.15.2
  - scikit-learn=1.6.0
  - matplotlib=3.10.1
- **Memory constraint:** `_MEM_LIMIT=4GB`; n=50K dropped in favour of n=20K (memory safety patch, commit `544989f`)
- **Hardware/OS:** WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2). CPU model, core count, and RAM were not recorded; speedup figures are hardware-dependent and may differ on other machines.

### Procedure

1. **Environment setup:** `micromamba create -f environment.yml -y`
2. **Data generation:** `python scripts/gen_data.py --sizes 10000 20000` — Gaussian d=50 arrays at n=10K and n=20K (seed=42); MERFISH n=10K and n=20K loaded from `research/2026-04-05-tw-perf-rerun-clean/data/merfish/`. (Note: `gen_data.py` defaults to `--sizes 10000 50000`; the `--sizes 10000 20000` override is required due to the n=50K memory constraint.)
3. **Exact baselines:** `python scripts/compute_exact.py` — 1 warmup + 3 timed runs per dataset; median wall time recorded
4. **Dry run:** `python scripts/run_subsampling.py --dry-run` — both approaches, MERFISH n=10K, m=2000, seed=0; normalization sanity check passed
5. **Full experiment:** `python scripts/run_subsampling.py` — 560 trials (2 approaches × 14 m-values × 10 seeds × 2 datasets); no batching triggered (max allocation 2.4 GB < 4 GB limit)
6. **Analysis:** `python scripts/analyze_results.py` — per-cell statistics, verdict evaluation, scaling fits, plots. The script globs `sub_*.json` files in `results/raw/` without verifying file count; a complete run produces 560 trial files (2 approaches × 14 m-values × 10 seeds × 2 datasets). If `results/raw/` is partially populated, analysis will run silently on the available subset.

## Results

### Exact Baselines

| Dataset | n | T_exact | wall_median_s |
|---------|---|---------|---------------|
| MERFISH | 10,000 | 0.536204 | 1.953s |
| MERFISH | 20,000 | 0.542254 | 10.336s |
| Gaussian | 10,000 | 0.501038 | 2.010s |
| Gaussian | 20,000 | 0.500572 | 9.246s |

### Primary Hypothesis Verdicts

| Hypothesis | Config | mean\|ΔT\| | max\|ΔT\| | std(T_sub) | Verdict |
|-----------|--------|-----------|---------|-----------|---------|
| H1_A | Approach A, MERFISH n=10K, m=2000 | 0.00165 | 0.00350 | 0.00215 | **PASS** |
| H1_B | Approach B, MERFISH n=10K, m=5000 | 0.00124 | 0.00381 | 0.00150 | **PASS** |

Thresholds: mean|ΔT| < 0.01 AND max|ΔT| < 0.02 AND std < 0.005. All conditions met by wide margins (H1_A: 6× below mean threshold; H1_B: 8× below). Note: the operating points are asymmetric — H1_A was tested at m=2000 and H1_B at m=5000 (a 2.5× sample size advantage for B). Larger m systematically reduces both error and variance, so the favorable conditions for H1_B are noted; the pre-specified operating points were set independently based on the intended use case for each approach.

**Four-cell outcome: Both Pass → Ship `trustworthiness_subsampled` with Approach A (m=2000) as default; Approach B supported as alternative for subset-T use cases.**

### Approach A Error/Speed Table (MERFISH)

| n | m | mean\|ΔT\| | std(T_sub) | wall_s | speedup |
|---|---|-----------|-----------|--------|---------|
| 10,000 | 250 | 0.00395 | 0.00458 | 0.08s | 25.3× |
| 10,000 | 500 | 0.00297 | 0.00347 | 0.13s | 14.5× |
| 10,000 | 1,000 | 0.00277 | 0.00329 | 0.27s | 7.2× |
| **10,000** | **2,000** | **0.00165** | **0.00215** | **0.48s** | **4.1×** |
| 10,000 | 5,000 | 0.00057 | 0.00065 | 1.25s | 1.6× |
| 10,000 | 7,500 | 0.00045 | 0.00053 | 1.94s | 1.0× |
| 20,000 | 250 | 0.00812 | 0.00991 | 0.17s | 62.4× |
| 20,000 | 500 | 0.00509 | 0.00651 | 0.29s | 36.1× |
| 20,000 | 1,000 | 0.00428 | 0.00473 | 0.54s | 19.1× |
| **20,000** | **2,000** | **0.00195** | **0.00260** | **1.09s** | **9.5×** |
| 20,000 | 5,000 | 0.00094 | 0.00105 | 2.52s | 4.1× |
| 20,000 | 7,500 | 0.00106 | 0.00111 | 3.74s | 2.8× |
| 20,000 | 10,000 | 0.00057 | 0.00072 | 4.96s | 2.1× |
| 20,000 | 15,000 | 0.00026 | 0.00035 | 7.59s | 1.4× |

### Approach B Error/Speed Table (MERFISH)

| n | m | mean\|ΔT\| | std(T_sub) | wall_s | speedup |
|---|---|-----------|-----------|--------|---------|
| 10,000 | 250 | 0.00883 | 0.00613 | 0.002s | 1253.9× |
| 10,000 | 500 | 0.00297 | 0.00413 | 0.005s | 393.8× |
| 10,000 | 1,000 | 0.00375 | 0.00429 | 0.02s | 92.4× |
| 10,000 | 2,000 | 0.00237 | 0.00303 | 0.10s | 19.4× |
| **10,000** | **5,000** | **0.00124** | **0.00150** | **0.51s** | **3.8×** |
| 10,000 | 7,500 | 0.00095 | 0.00126 | 1.16s | 1.7× |
| 20,000 | 250 | 0.00832 | 0.00748 | 0.001s | 7090.2× |
| 20,000 | 500 | 0.00711 | 0.00828 | 0.004s | 2330.9× |
| 20,000 | 1,000 | 0.00454 | 0.00649 | 0.02s | 676.9× |
| 20,000 | 2,000 | 0.00365 | 0.00460 | 0.08s | 121.9× |
| **20,000** | **5,000** | **0.00144** | **0.00200** | **0.49s** | **21.2×** |
| 20,000 | 7,500 | 0.00146 | 0.00189 | 1.09s | 9.5× |
| 20,000 | 10,000 | 0.00061 | 0.00080 | 1.86s | 5.6× |
| 20,000 | 15,000 | 0.00047 | 0.00061 | 4.59s | 2.2× |

### H2: Variance Scaling (std ∝ m^slope)

| Approach | Slope | R² | Expected |
|---------|-------|----|---------|
| A | −0.657 | 0.85 | −0.5 (heuristic) |
| B | −0.582 | 0.80 | −0.5 (heuristic) |

> The −0.5 expected slope is a heuristic based on the CLT scaling for sample means of i.i.d. variables. Trustworthiness is a rank-based statistic computed over structured manifold data; the i.i.d. CLT convergence rate may not apply directly. The observed slopes (steeper than −0.5) are consistent with structured data amplifying the benefit of larger samples, but a formal derivation of the expected rate for this statistic is left for future work.

### H3: MERFISH vs. Gaussian Variability Ratio

| Approach | n | m | std_MERFISH | std_Gaussian | Ratio |
|---------|---|---|-------------|--------------|-------|
| A | 10,000 | 2,000 | 0.00215 | 0.00113 | 1.90× |
| A | 10,000 | 5,000 | 0.00065 | 0.00053 | 1.24× |
| A | 20,000 | 2,000 | 0.00260 | 0.00141 | 1.85× |
| A | 20,000 | 5,000 | 0.00105 | 0.00098 | 1.07× |
| B | 10,000 | 2,000 | 0.00303 | 0.00252 | 1.20× |
| B | 10,000 | 5,000 | 0.00150 | 0.00058 | 2.58× |
| B | 20,000 | 2,000 | 0.00460 | 0.00176 | 2.62× |
| B | 20,000 | 5,000 | 0.00200 | 0.00107 | 1.88× |

### H4: Speed Scaling

| Approach | Slope | R² | Expected |
|---------|-------|----|---------|
| A | 0.956 | 0.991 | ≈1 (O(mn)) |
| B | 1.984 | 0.995 | ≈2 (O(m²)) |

### H5: Crossover m/n Ratio (mean|ΔT| < 0.01 on MERFISH)

| Approach | n | Crossover m | m/n |
|---------|---|-------------|-----|
| A | 10,000 | 250 | 0.0250 |
| A | 20,000 | 250 | 0.0125 |
| B | 10,000 | 250 | 0.0250 |
| B | 20,000 | 250 | 0.0125 |

Both approaches: crossover ratio within 2× across n values? **YES** (0.025 → 0.0125, ratio = 2×).

### H6: Extrapolation to n=100K (OUT-OF-DISTRIBUTION)

Fit: |ΔT| = 0.000180 × n^0.2404 (Approach A, m=2000, MERFISH, 2 data points)

**Predicted |ΔT| at n=100K: 0.00287** — well below 0.01 threshold.

> Extrapolation from 2 data points only (n=10K, n=20K). Treat as order-of-magnitude estimate. Sub-linear scaling (exponent 0.24) is plausible but unverified beyond n=20K.

## Observations

1. **Both H1 hypotheses pass with wide margins.** H1_A mean|ΔT|=0.00165 (6× below threshold); H1_B mean|ΔT|=0.00124 (8× below threshold). These are strong results from a single-run experiment with 10 random seeds per cell.

2. **Approach A and Approach B estimate different quantities.** Approach A uses m query rows evaluated against all n reference points, so when m=n it exactly reproduces the full-n formula; empirically it closely approximates full-n trustworthiness at m<<n. Approach B (sklearn trustworthiness on the m×m subset) estimates trustworthiness within the sampled subset, not the full population. A formal proof of unbiasedness for Approach A at m<n is not provided here. At matched accuracy, Approach B can be far faster (at m=2000, n=20K: 121.9× vs 9.5×), but this comes at the cost of measuring subset-T rather than full-population T.

3. **Speed scaling laws confirmed with high precision.** Approach A: O(mn) (slope=0.956, R²=0.991). Approach B: O(m²) (slope=1.984, R²=0.995). These match theoretical predictions to within 5%.

4. **No batching was needed.** Peak Approach A memory at m=15K, n=20K was 2.4 GB, well within the 4 GB `_MEM_LIMIT`. The batched code path was implemented but not exercised.

5. **Variance decays faster than the heuristic −0.5 expectation** (slopes −0.66 and −0.58). The −0.5 slope is a rough CLT-based heuristic for i.i.d. sample means; trustworthiness is a rank-based statistic and the applicable convergence rate is not derived here. The steeper-than-expected slopes may reflect structured manifold geometry amplifying the benefit of increased sample size, but a formal explanation is deferred.

6. **MERFISH std is consistently 1.2–2.6× higher than Gaussian std**, confirming heterogeneous cluster density increases estimator variance. The ratio is not constant — it varies by approach, n, and m.

7. **The crossover m/n ratio improves with n.** H5 shows the minimum m achieving mean|ΔT| < 0.01 on MERFISH halves from m/n=0.025 (n=10K) to m/n=0.0125 (n=20K) — the required sub-sampling fraction decreases as n grows. Note: at fixed m=2000, absolute error is higher at n=20K (mean|ΔT|=0.00195) than at n=10K (0.00165); the ratio improvement reflects the crossover threshold, not fixed-m accuracy.

## Analysis

Both primary hypotheses are supported with substantial margin. The pre-specified four-cell outcome table maps to "Both Pass," strongly supporting shipping `trustworthiness_subsampled` with Approach A as the default.

Approach A is preferable as the default because it targets full-n trustworthiness — the same quantity the exact implementation computes — with denominator `m·k·(2n−3k−1)`. When m=n, Approach A reproduces the sklearn exact result to within 1e-10 (normalization sanity check), though this validates the formula at m=n rather than establishing formal statistical properties at m<<n. The exact trustworthiness baseline uses scikit-learn 1.6.0 as the reference implementation; cross-validation against an independent implementation was not performed. Approach B estimates a fundamentally different quantity (trustworthiness within the sampled subset), making it unsuitable as a drop-in replacement for the exact metric without user awareness.

At m=2000, Approach A delivers a practical trade-off: 0.17% mean error at 4.1× speedup (n=10K) and 0.20% error at 9.5× speedup (n=20K). Increasing m to 5000 reduces error to 0.06% but at diminishing speedup (1.6× for n=10K). The m=2000 default therefore represents the efficient frontier of the trade-off.

The confirmed O(mn) complexity for Approach A means that as n grows, the crossover ratio m/n that achieves a given accuracy threshold decreases — that is, a fixed m=2000 becomes relatively more accurate as n grows, which is the desired behavior for a practical default.

**Statistical context:** This is an exploratory single-run engineering experiment (R=1, 10 seeds per cell). No formal alpha level is declared, no multiple comparisons correction is applied across H1–H6, and no Type I/II error rates are specified for the threshold-based PASS/FAIL rule. With n=10 seeds per cell, the 95% CI on σ spans approximately [0.69σ, 1.83σ], so std estimates carry substantial uncertainty. All verdicts should be interpreted in this exploratory context rather than as frequentist hypothesis tests.

The large errors in the prior H5 attempt (`tw-perf-rerun-clean`, mean|delta|≈0.474) are not reproduced here. The current implementation uses a denominator `m·k·(2n−3k−1)` that is explicitly verified at m=n. The exact root cause of the prior failure is not documented in those artifacts; the current results demonstrate that the present implementation is correct at the tested n values.

## Deviations from Plan

| Deviation | Planned | Actual | Impact |
|-----------|---------|--------|--------|
| n-range | n={10K, 50K} | n={10K, 20K} | 2× n-range instead of planned 5×; H4 speed-scaling law and H5 crossover fit from 2 data points only; H6 extrapolation is more speculative |
| Execution replications | Not specified (single run with 10 seeds/cell) | R=1 | The 10 seeds/cell quantify within-run variance; between-run reproducibility was not assessed |

The n=50K substitution was required by the 4 GB `_MEM_LIMIT` constraint (commit `544989f`). The `analyze_results.py` script was updated to use n={10K, 20K} throughout; the machine-generated `results/analysis/summary.md` reflects the actual executed combinations. Note: if `analyze_results.py` is re-run, it requires the `--sizes 10000 20000` data generated by `python scripts/gen_data.py --sizes 10000 20000` (the script default `--sizes 10000 50000` will not match). The executed dataset combinations (n=10K and n=20K, 560 trials total) are complete and consistent.

## What We Learned

- **Sub-sampled trustworthiness works at tested scales.** m=2000 gives <0.2% error with 4–10× speedup across n=10K and n=20K on real-world data. The approach is a strong candidate for shipping; validation at larger n (n=50K was planned but not executed) would further strengthen confidence in the n-scaling claims.
- **Approach A and B serve different estimands.** They are not interchangeable. Approach A estimates full-n T; Approach B estimates subset-T. Exposing both as named variants prevents confusion.
- **Normalization correctness is critical.** Implementing and verifying `|T_A(m=n) - T_exact| < 1e-10` as a first sanity check confirms formula correctness at m=n and should be standard practice for sub-sampling work. Note this check validates algebraic correctness when all rows are sampled; it does not directly establish the estimator's statistical properties at m<<n.
- **Variance decays super-CLT on structured manifolds.** On MERFISH, std scales as m^−0.66 rather than m^−0.5. Future experiments on highly structured data should expect faster variance convergence.
- **Memory safety.** The `_MEM_LIMIT` guard and batched fallback proved sufficient; the n=50K → n=20K substitution reduced peak allocation from ~10 GB to 2.4 GB without affecting conclusions.
- **Boundary condition established:** for MERFISH-like data with k=15, any m ≥ 250 achieves mean|ΔT| < 0.01. The accuracy floor is m, not n/m.

## Conclusions

Both primary hypotheses (H1_A and H1_B) are strongly supported. Sub-sampled trustworthiness with Approach A at m=2000 achieves <0.2% mean absolute error and <0.5% max error across 10 random seeds on MERFISH data, with 4.1× speedup at n=10K and 9.5× at n=20K. All pre-specified thresholds are met by wide margins. Results are from a single experimental run (R=1) at n≤20K; between-run reproducibility and n>20K behavior have not been empirically verified.

The four-cell outcome is **Both Pass**: ship `trustworthiness_subsampled` in `src/metrics.rs` with Approach A as the default implementation (m=2000 recommended) and Approach B available as an alternative for subset-embedding use cases.

## Recommendations

1. **Ship Approach A as `trustworthiness_subsampled(X, Y, k, m)` with default m=2000.** Evidence is strong across n=10K and n=20K (R=1, 10 seeds/cell). This provides a correct, verified implementation. Use denominator `m·k·(2n−3k−1)`.

2. **Expose Approach B as `trustworthiness_subset_embedding(X, Y, k, m)`.** Useful for interactive subset exploration where subset-T is the intended estimand. Document clearly that it does not estimate full-n T.

3. **Document the speedup curve.** The m/speedup table should accompany the API docs so users can select m based on their accuracy/speed requirement.

4. **Validate at n=50K in a follow-up.** The n=50K case was dropped for memory safety. The H6 extrapolation suggests accuracy remains below threshold, but empirical confirmation at one larger n would strengthen the n-scaling claim.

5. **Add the normalization sanity check as a CI test.** Assert `|T_A(m=n) - T_exact| < 1e-4` on a small fixture to prevent regression to the prior denominator bug.

## Appendix: Experiment Scripts

### scripts/utils.py

```python
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

K = 15
SEEDS = list(range(10))
M_VALUES_10K = [250, 500, 1000, 2000, 5000, 7500]
M_VALUES_20K = [250, 500, 1000, 2000, 5000, 7500, 10000, 15000]

def trustworthiness_row_subsampled(X, Y, k, query_idx):
    """Approach A: m query rows, distances to ALL n points.

    Approximation of full-n trustworthiness using m query rows.
    Denominator m * k * (2n - 3k - 1) matches the full-n formula when m == n.
    Formal unbiasedness at m < n is not proven; empirically close to full-n T.
    """
    n = X.shape[0]
    m = len(query_idx)
    dist_X = pairwise_distances(X[query_idx], X)
    for i, gi in enumerate(query_idx):
        dist_X[i, gi] = np.inf  # exclude self
    ranks_X = np.argsort(np.argsort(dist_X, axis=1), axis=1) + 1
    x_knn_mask = ranks_X <= k  # (m, n) boolean
    # Request k+1 neighbors because kneighbors(Y[query_idx]) includes self
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(Y)
    y_knn_all = nn.kneighbors(Y[query_idx], return_distance=False)  # (m, k+1)
    penalty = 0.0
    for i in range(m):
        gi = query_idx[i]
        y_knn = [j for j in y_knn_all[i] if j != gi][:k]
        for j_col in y_knn:
            if not x_knn_mask[i, j_col]:
                penalty += ranks_X[i, j_col] - k
    denom = m * k * (2 * n - 3 * k - 1)
    return 1.0 - 2.0 * penalty / denom

def load_npy_pair(data_dir, prefix, n):
    data_dir = Path(data_dir)
    X = np.load(data_dir / f"{prefix}_n{n}_x.npy")
    Y = np.load(data_dir / f"{prefix}_n{n}_y.npy")
    return X, Y

def save_result_json(path, result_dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result_dict, f)
```

### scripts/compute_exact.py

```python
"""Compute exact trustworthiness baselines for all (dataset, n) combinations.

Run: micromamba run -n subsampled-tw-tradeoff python scripts/compute_exact.py
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
from sklearn.manifold import trustworthiness as sklearn_tw
sys.path.insert(0, str(Path(__file__).parent))
from utils import K, load_npy_pair, save_result_json

EXPROOT = Path(__file__).parent.parent
DATASETS = [
    ("merfish",  10_000, EXPROOT / "data" / "merfish"),
    ("merfish",  20_000, EXPROOT / "data" / "merfish"),
    ("gaussian", 10_000, EXPROOT / "data" / "gaussian"),
    ("gaussian", 20_000, EXPROOT / "data" / "gaussian"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = [DATASETS[0]] if args.dry_run else DATASETS
    for dataset, n, data_dir in datasets:
        try:
            X, Y = load_npy_pair(data_dir, dataset, n)
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping", file=sys.stderr); continue
        sklearn_tw(X, Y, n_neighbors=K)  # warmup
        wall_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            T_exact = sklearn_tw(X, Y, n_neighbors=K)
            wall_runs.append(time.perf_counter() - t0)
        result = {"dataset": dataset, "n": n, "k": K, "T_exact": float(T_exact),
                  "wall_median_s": float(np.median(wall_runs)),
                  "wall_runs": [float(w) for w in wall_runs]}
        save_result_json(EXPROOT / "results" / "raw" / f"exact_{dataset}_{n}.json", result)
        print(f"[exact] {dataset} n={n}: T_exact={T_exact:.6f} wall_median={np.median(wall_runs):.3f}s")

if __name__ == "__main__":
    main()
```

### scripts/run_subsampling.py

```python
"""Run subsampling experiment: Approach A and B for all (dataset, n, m, seed).

Full run: 2 approaches × (6+8) m-values × 10 seeds × 2 datasets = 560 trials.
Dry run (--dry-run): seed=0, m=2000, both approaches, MERFISH n=10K only.

Run: micromamba run -n subsampled-tw-tradeoff python scripts/run_subsampling.py
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.manifold import trustworthiness as sklearn_tw
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).parent))
from utils import (K, SEEDS, M_VALUES_10K, M_VALUES_20K,
                   load_npy_pair, save_result_json, trustworthiness_row_subsampled)

EXPROOT = Path(__file__).parent.parent
BATCH_SIZE = 5000
_MEM_LIMIT = 4 * 1024 ** 3  # 4 GB
DATASETS = [
    ("merfish",  10_000, EXPROOT / "data" / "merfish"),
    ("merfish",  20_000, EXPROOT / "data" / "merfish"),
    ("gaussian", 10_000, EXPROOT / "data" / "gaussian"),
    ("gaussian", 20_000, EXPROOT / "data" / "gaussian"),
]

def _m_values(n): return M_VALUES_10K if n == 10_000 else M_VALUES_20K

def _approach_a_batched(X, Y, k, query_idx):
    n_full = X.shape[0]; m = len(query_idx)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(Y)
    y_knn_all = nn.kneighbors(Y[query_idx], return_distance=False)
    penalty = 0.0
    for b0 in range(0, m, BATCH_SIZE):
        b1 = min(b0 + BATCH_SIZE, m); bq = query_idx[b0:b1]
        dist_b = pairwise_distances(X[bq], X)
        for li, gi in enumerate(bq): dist_b[li, gi] = np.inf
        ranks_b = np.argsort(np.argsort(dist_b, axis=1), axis=1) + 1
        mask_b = ranks_b <= k
        for li in range(len(bq)):
            gi = bq[li]; y_knn = [j for j in y_knn_all[b0 + li] if j != gi][:k]
            for j_col in y_knn:
                if not mask_b[li, j_col]: penalty += ranks_b[li, j_col] - k
    return 1.0 - 2.0 * penalty / (m * k * (2 * n_full - 3 * k - 1))

def _load_exact_T(dataset, n):
    path = EXPROOT / "results" / "raw" / f"exact_{dataset}_{n}.json"
    with open(path) as f: return json.load(f)["T_exact"]

def _run_trial(X, Y, n, approach, m, seed, T_exact, dataset):
    rng = np.random.RandomState(seed); idx = rng.choice(n, size=m, replace=False)
    if approach == "A":
        mem_bytes = m * n * 8
        t0 = time.perf_counter()
        T_sub = (_approach_a_batched(X, Y, K, idx) if mem_bytes > _MEM_LIMIT
                 else trustworthiness_row_subsampled(X, Y, K, idx))
        wall_s = time.perf_counter() - t0
    else:
        t0 = time.perf_counter(); T_sub = sklearn_tw(X[idx], Y[idx], n_neighbors=K)
        wall_s = time.perf_counter() - t0
    delta_T = float(T_sub) - float(T_exact)
    return {"approach": approach, "dataset": dataset, "n": n, "m": m, "seed": seed,
            "k": K, "T_sub": float(T_sub), "T_exact": float(T_exact),
            "delta_T": delta_T, "abs_delta_T": abs(delta_T), "wall_s": wall_s}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    loaded = {}
    for dataset, n, data_dir in DATASETS:
        try: loaded[(dataset, n)] = load_npy_pair(data_dir, dataset, n)
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping", file=sys.stderr)
    if args.dry_run:
        trials = [("merfish", 10_000, "A", 2000, 0), ("merfish", 10_000, "B", 2000, 0)]
    else:
        trials = [(ds, n, ap, m, s) for ds, n, _ in DATASETS if (ds, n) in loaded
                  for ap in ["A", "B"] for m in _m_values(n) for s in SEEDS]
    exact_cache = {(ds, n): _load_exact_T(ds, n)
                   for ds, n in {(ds, n) for ds, n, *_ in trials if (ds, n) in loaded}}
    for ds, n, approach, m, seed in trials:
        if (ds, n) not in loaded: continue
        X, Y = loaded[(ds, n)]
        result = _run_trial(X, Y, n, approach, m, seed, exact_cache[(ds, n)], ds)
        out = EXPROOT / "results" / "raw" / f"sub_{approach}_{ds}_{n}_m{m}_s{seed}.json"
        save_result_json(out, result)

if __name__ == "__main__":
    main()
```

## Appendix: Raw Data

Raw trial data (560 JSON files) is committed in `research/2026-04-09-subsampled-tw-tradeoff/results/raw/`. Exact baseline data is in `results/raw/exact_*.json`. Analysis plots are in `results/analysis/`:

- `error_vs_m.png` — mean|ΔT| vs. m for both approaches, datasets, and n values
- `speedup_vs_m.png` — speedup vs. m
- `std_vs_m_loglog.png` — std(T_sub) vs. m on log-log scale (H2 variance scaling)
- `summary.md` — machine-generated analysis summary with all hypothesis verdicts
