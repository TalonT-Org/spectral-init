# tw-perf-rerun-clean Analysis Report

## H5: Approximate Trustworthiness Accuracy
  Seeds loaded: 10
  Speedup ratio (wall_approx/wall_exact): median=0.9062, range=[0.8725, 0.9512]
  |delta| mean=0.474926, 95% t-CI (df=9, t=2.2622): [0.474889, 0.474962]
  Verdict: **NEGATIVE (approx inaccurate: CI above threshold)**

## H-100K: Criterion Variant Speedups (n=10K, FALLBACK method)
  Note: "H-100K" refers to the production-scale target; benchmarks ran at n=10K.
  Provenance note: partial_rank ratio is 0.9775, derived from a supplemental
  Criterion run (tw_partial_rank/partial_rank/10000, second run) appended to
  criterion_output.json after the initial pipeline execution. The original
  pipeline run produced ratio 0.9134; the supplemental run supersedes this.
  W6 Note: Criterion CIs are non-deterministic across runs due to bootstrapped estimation; point estimates are the stable quantity.
  W5 FALLBACK CI: raw samples unavailable for thread_local or baseline.
  W5 FALLBACK CI: raw samples unavailable for partial_rank or baseline.
  W5 FALLBACK CI: raw samples unavailable for avx2 or baseline.
  W5 FALLBACK CI: raw samples unavailable for combined or baseline.
  Baseline n=10K id: tw_baseline/baseline/10000
  Bootstrap samples: 10000

  | Variant | Mean Ratio | 95% Boot CI | p-value | Holm adj p | Reject H0 | Method |
  |---------|------------|-------------|---------|------------|-----------|--------|
  | thread_local | 1.5444 | [1.4672, 1.6217] | 1.0000 | 1.0000 | False | FALLBACK |
  | partial_rank | 0.9775 | [0.9286, 1.0263] | 0.0000 | 0.0000 | True | FALLBACK |
  | avx2 | 1.4898 | [1.4153, 1.5643] | 1.0000 | 1.0000 | False | FALLBACK |
  | combined | 1.0300 | [0.9785, 1.0815] | 1.0000 | 1.0000 | False | FALLBACK |
  W4 ANOMALY: Cache warm-state bias >5%: tw_combined: 21.5% difference; tw_baseline: 19.2% difference

## H-partial-MERFISH: Partial Rank MERFISH vs Gaussian CI Width (n=50K)
  Provenance note: Gaussian n=50K data originates from a supplemental Criterion
  run (tw_partial_rank/partial_rank/50000) appended to criterion_output.json
  after initial pipeline execution. The original pipeline run produced ERROR
  (no n=50K Gaussian record). The MERFISH n=50K entry
  (tw_partial_rank_merfish/partial_rank_merfish/50000) comes from the MERFISH
  bench run included in the original run_criterion_clean.sh execution.
  Gaussian n=50K CI half-width: 160701624.62 ns  (id: tw_partial_rank/partial_rank/50000)
  MERFISH  n=50K CI half-width: 499169901.51 ns  (id: tw_partial_rank_merfish/partial_rank_merfish/50000)
  MERFISH/Gaussian ratio: 3.11
  Verdict: **ELEVATED VARIANCE (MERFISH CI width 3.1× Gaussian; investigate data heterogeneity)**

## H0/H1-clean: Step CPU-Time Fractions (Baseline, n=10K)
  Iterations: 30, df=29, t_crit(0.975,df=29)=2.0452

  | Step | Mean Fraction | 95% t-CI |
  |------|--------------|----------|
  | x_dist         | 0.1296 (13.0%) | [0.1291, 0.1301] |
  | x_sort         | 0.0997 (10.0%) | [0.0995, 0.0999] |
  | rank_scatter   | 0.0000 (0.0%) | [0.0000, 0.0000] |
  | x_knn_set      | 0.0036 (0.4%) | [0.0035, 0.0036] |
  | y_heap         | 0.7034 (70.3%) | [0.7028, 0.7040] |
  | penalty        | 0.0637 (6.4%) | [0.0635, 0.0638] |
  W3: Step ordering anomaly (descriptive only — no p-value):
       Expected: x_dist > x_sort > rank_scatter > y_heap > x_knn_set > penalty
       Observed: y_heap > x_dist > x_sort > penalty > x_knn_set > rank_scatter
