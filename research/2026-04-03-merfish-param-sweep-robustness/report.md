# MERFISH Param-Sweep Robustness — Results Report

## Hypothesis Verdicts

| Hypothesis | Verdict | Rationale |
|------------|---------|-----------|
| H1: Spectral init CV stability | INCONCLUSIVE | CV_ratio trust=0.997, triplet=0.469 |
| H2: Rust–Python Procrustes | (see table) | PASS=0, WARNING=0, FAIL=14 |
| H3: Random degrades faster at low N | REFUTED | slope_rust=-0.00021180125973754296, slope_random=-0.00020299830676742128 |
| H4: SNA metric stability | VARIABLE | CV range: {'rust_spectral': 0.022573886382838994, 'python_spectral': 0.03147758666283018, 'pca': 0.04364320891657896, 'random': 0.048750858128172654} |
| H5: Procrustes stable across min_dist | REFUTED | std=0.011292206479344523 |

## Quantitative CV Table

CV computed across n_neighbors sweep dimension.

| init_method | trustworthiness CV | triplet_accuracy CV |
|-------------|-------------------|---------------------|
| rust_spectral | 0.0017 | 0.0218 |
| python_spectral | 0.0017 | 0.0259 |
| pca | 0.0018 | 0.0259 |
| random | 0.0017 | 0.0464 |

## Procrustes Alignment (H2)

| config_key | param_swept | param_value | procrustes_rust_vs_python | solver_level | status |
|---|---|---|---|---|---|
| n_neighbors_5_euclidean | n_neighbors | 5 | 0.2685851577787182 | 1 | FAIL |
| n_neighbors_10_euclidean | n_neighbors | 10 | 0.2830243915081189 | 1 | FAIL |
| n_neighbors_15_euclidean | n_neighbors | 15 | 0.2995690702256447 | 1 | FAIL |
| n_neighbors_30_euclidean | n_neighbors | 30 | 0.3006524647968094 | 1 | FAIL |
| n_neighbors_50_euclidean | n_neighbors | 50 | 0.2789736456865636 | 1 | FAIL |
| n_neighbors_100_euclidean | n_neighbors | 100 | 0.4176256262760164 | 1 | FAIL |
| min_dist_0.0_euclidean | min_dist | 0.0 | 0.2945638319539537 | 1 | FAIL |
| min_dist_0.01_euclidean | min_dist | 0.01 | 0.3045602619808689 | 1 | FAIL |
| min_dist_0.1_euclidean | min_dist | 0.1 | 0.2995690702256447 | 1 | FAIL |
| min_dist_0.25_euclidean | min_dist | 0.25 | 0.3012184919793498 | 1 | FAIL |
| min_dist_0.5_euclidean | min_dist | 0.5 | 0.3083281124399231 | 1 | FAIL |
| min_dist_0.8_euclidean | min_dist | 0.8 | 0.3268615833816294 | 1 | FAIL |
| metric_euclidean_euclidean | metric | euclidean | 0.2995690702256447 | 1 | FAIL |
| metric_cosine_cosine | metric | cosine | 0.2457360122122163 | 1 | FAIL |

## t-SNE Reference Comparison

Best t-SNE trustworthiness: 0.9960 at perplexity=30.
Best rust_spectral trustworthiness: 0.9916.

## Solver Level Diagnostics

| config_key | solver_level |
|------------|-------------|
| n_neighbors_5_euclidean | 1 |
| n_neighbors_10_euclidean | 1 |
| n_neighbors_15_euclidean | 1 |
| n_neighbors_30_euclidean | 1 |
| n_neighbors_50_euclidean | 1 |
| n_neighbors_100_euclidean | 1 |
| min_dist_0.0_euclidean | 1 |
| min_dist_0.01_euclidean | 1 |
| min_dist_0.1_euclidean | 1 |
| min_dist_0.25_euclidean | 1 |
| min_dist_0.5_euclidean | 1 |
| min_dist_0.8_euclidean | 1 |
| metric_euclidean_euclidean | 1 |
| metric_cosine_cosine | 1 |

## Threats to Validity

- **Incomplete sweep:** results_sweep.csv may contain partial data if the sweep was interrupted. Verdicts derived from fewer than all 56 configs should be treated as provisional.
- **Single dataset:** All results are from the MERFISH 10k cell dataset. Generalization to other datasets is unverified.
- **Single random seed:** `RANDOM_STATE=42` throughout. Variance estimates may not reflect true stochasticity.

## Success Criteria Checklist

- [x] `trustworthiness_vs_n_neighbors.png`
- [x] `triplet_accuracy_vs_n_neighbors.png`
- [x] `trustworthiness_vs_min_dist.png`
- [x] `cv_comparison_bar.png`
- [x] `procrustes_rust_vs_python_heatmap.png`
- [x] `tsne_reference.png`
- [x] `solver_levels.json`
