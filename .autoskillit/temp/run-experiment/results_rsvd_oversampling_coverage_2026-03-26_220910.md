# Experiment Results: rSVD Oversampling Sweep and Cross-Level Solver Coverage

## Run Metadata
- Date: 2026-03-26 22:09:10
- Worktree: /home/talon/projects/worktrees/research-20260326-220910
- Commit: f3c24a6606d9cbab69e0e1f5e4c25811959116a8
- Environment: rustc 1.93.0-nightly (27b076af7 2025-11-21), cargo-nextest 0.9.132
- Machine: WSL2 Linux 6.6.87.2-microsoft-standard-WSL2

## Configuration

### RQ1 — rSVD Oversampling Sweep
- Sweep p values: 5, 10, 15, 20, 25, 30, 50, 100, 200
- Fixtures: near_dupes_100, blobs_connected_200, moons_200, circles_300, blobs_500, blobs_connected_2000
- Production solver: `rsvd_solve` (nbiter=2), env-var seam `SPECTRAL_RSVD_OVERSAMPLING`
- Accurate solver: `rsvd_solve_accurate` (nbiter=6) for quick-scan upper bound
- Quality threshold: `max_eigenpair_residual < 1e-2`
- n_components: 2, seed: 42

### RQ2 — Mutation Detection via Per-Solver Invariant Tests
- Test suite: `test_exp_solver_invariants` (8 tests)
- Mutations: B1–B8 via `inject_bug.sh`
- Feature flag: `--features testing`

---

## Results

### RQ1 — Quick Scan (rsvd_solve_accurate, nbiter=6)

| fixture | n | min_p (passes) | residual at min_p | notes |
|---------|---|----------------|-------------------|-------|
| near_dupes_100 | 100 | 10 | 8.66e-3 | ✓ passes |
| blobs_connected_200 | 200 | 20 | 6.19e-3 | ✓ passes |
| moons_200 | 200 | 20 | 8.18e-3 | ✓ passes |
| circles_300 | 300 | 50 | 8.64e-4 | ✓ passes |
| blobs_500 | 500 | 100 | 9.30e-4 | ✓ passes |
| blobs_connected_2000 | 2000 | NOT REACHED | 1.26e-2 at p=100 | ✗ still fails |

Even with nbiter=6 (upper bound accuracy), `blobs_connected_2000` requires p>100 to meet the 1e-2 threshold.

### RQ1 — Production Sweep (rsvd_solve, nbiter=2)

Summary table (min passing p across all tested values up to 200):

| fixture | n | min_p (passes) | residual | t@p=200 (s) | t@min_p (s) | speedup |
|---------|---|----------------|----------|------------|------------|---------|
| near_dupes_100 | 100 | 100 | 1.35e-14 (k=n) | 0.357 | 0.477 | 0.75x |
| blobs_connected_200 | 200 | 200 | 1.14e-14 (k=n) | 0.866 | 0.866 | 1.00x |
| moons_200 | 200 | 200 | 7.07e-15 (k=n) | 1.141 | 1.141 | 1.00x |
| circles_300 | 300 | NOT REACHED | 1.11e-2 at p=200 | — | — | — |
| blobs_500 | 500 | NOT REACHED | 3.20e-2 at p=200 | — | — | — |
| blobs_connected_2000 | 2000 | NOT REACHED | 1.38e-1 at p=200 | — | — | — |

**Critical observation:** The only passing cases for production (nbiter=2) are when k≥n (trivially exact SVD): near_dupes_100 at p=100 → k=min(103,100)=100=n, and both n=200 fixtures at p=200 → k=min(203,200)=200=n. No genuine low-rank approximation passes the 1e-2 threshold with nbiter=2.

**At the current production p values (n/10 formula, nbiter=2):**
| fixture | n | production p | residual | passes |
|---------|---|-------------|----------|--------|
| near_dupes_100 | 100 | 10 | 1.23e-1 | false |
| blobs_connected_200 | 200 | 20 | 1.19e-1 | false |
| moons_200 | 200 | 20 | 1.18e-1 | false |
| circles_300 | 300 | 30 | 8.72e-2 | false |
| blobs_500 | 500 | 50 | 9.67e-2 | false |
| blobs_connected_2000 | 2000 | 200 | 1.38e-1 | false |

**All six production-default runs fail the 1e-2 threshold with nbiter=2.**

### RQ1 — Quality/Time Tradeoff Curve (blobs_connected_2000)

| p | k | residual (nbiter=2) | residual (nbiter=6) | wall_time_s (prod) |
|---|---|---------------------|---------------------|-------------------|
| 5 | 8 | 2.60e-1 | 1.00e-1 | 0.163 |
| 10 | 13 | 2.91e-1 | 7.18e-2 | 0.037 |
| 15 | 18 | 2.88e-1 | 6.23e-2 | 0.071 |
| 20 | 23 | 2.89e-1 | 5.26e-2 | 0.093 |
| 25 | 28 | 2.55e-1 | 4.39e-2 | 0.137 |
| 30 | 33 | 2.75e-1 | 4.53e-2 | 0.212 |
| 50 | 53 | 2.43e-1 | 2.64e-2 | 0.240 |
| 100 | 103 | 2.05e-1 | 1.26e-2 | 0.548 |
| 200 | 203 | 1.38e-1 | (not tested at nbiter=6) | 0.823 |

Wall times are fast (0.04–0.82s) even at p=200, confirming the previous 111.8s slowness was NOT rSVD itself — it was downstream forced dense EVD (Level 5) triggered by quality check failure.

### RQ1 — Power Iteration Benefit (nbiter=2 vs nbiter=6 ratio)

Selected ratios showing how much nbiter=6 improves over nbiter=2:

| fixture | p | prod_res (nbiter=2) | acc_res (nbiter=6) | ratio |
|---------|---|---------------------|-------------------|-------|
| moons_200 | 100 | 1.94e-2 | 1.93e-5 | 1001x |
| circles_300 | 100 | 3.88e-2 | 6.00e-5 | 647x |
| blobs_connected_200 | 100 | 2.44e-2 | 4.39e-5 | 555x |
| near_dupes_100 | 50 | 2.15e-2 | 2.77e-5 | 776x |
| blobs_connected_2000 | 100 | 2.05e-1 | 1.26e-2 | 16x |

Power iterations dominate quality: switching nbiter from 2 to 6 reduces residuals by 16–1001×, dwarfing any oversampling effect.

---

### RQ2 — Mutation Detection Matrix

| bug_id | target_file | detected | suite_wall_time_s |
|--------|-------------|----------|-------------------|
| B1 | dense.rs | YES | 7.050 |
| B2 | mod.rs | YES | 7.790 |
| B3 | scaling.rs | YES | 6.750 |
| B4 | multi_component.rs | NO | 7.450 |
| B5 | laplacian.rs | YES | 7.494 |
| B6 | lobpcg.rs | NO | 6.604 |
| B7 | rsvd.rs | YES | 7.965 |
| B8 | sinv.rs | YES | 7.550 |

**Detection rate: 6/8 (75%)**
**Single suite wall time: 5.1s on clean source, 6.6–8.0s under mutations**

Improvement over previous property test suite: 1/8 → 6/8.

**Previously undetected bug now caught:**
- B3 (scaling.rs sign flip): NOW DETECTED by `test_inv_sign_convention` ✓ — closes the blind spot identified in the previous CI experiment.

**Still undetected:**
- B4 (multi_component.rs): `test_inv_multi_component_completeness` uses the `disconnected_200` fixture. The mutation skips writing some component embeddings — requires that the fixture graph actually has disconnected components and the test verifies all rows are non-zero. Needs investigation whether the fixture actually triggers multi-component code paths or the invariant check is insufficient.
- B6 (lobpcg.rs warm-restart truncation): Warm restart truncated to 1 iteration still converges on well-conditioned graphs. Only detectable with adversarial ill-conditioned input that specifically requires multiple warm-restart cycles.

---

## Observations

1. **nbiter=2 is fundamentally insufficient for low-rank rSVD quality.** All six production-default configurations (n/10 oversampling, nbiter=2) produce residuals well above 1e-2 (range: 8.7e-2 to 2.9e-1). The quality threshold is NEVER met for genuine low-rank approximations (k<n) with nbiter=2.

2. **The previously-reported 111.8s slowness is NOT in rSVD itself.** Production rSVD runs take 0.04–0.82s even at p=200. The 111.8s bottleneck was downstream: after rSVD fails the quality check, the escalation chain proceeds to Level 5 (forced dense EVD), which is O(n³) and expensive at n=2000. Reducing oversampling p would NOT address the true performance bottleneck.

3. **Power iterations, not oversampling, are the dominant quality lever.** The ratio of residuals (nbiter=6 vs nbiter=2) ranges from 16× to 1001×. Oversampling changes residual by a much smaller factor. To achieve <1e-2 quality with nbiter=2, p would need to be implausibly large (k≈n for n≤200, still insufficient for n≥300).

4. **blobs_connected_2000 is irreducibly hard.** Even nbiter=6 with p=100 (k=103) only achieves 1.26e-2 residual, just above threshold. This fixture likely has a very slowly-decaying spectral gap, making it inherently difficult for randomized methods with small k.

5. **RQ2 invariant tests are effective.** 6/8 mutations detected vs. 1/8 with property tests. Single suite run in 5.1s on clean source, 6.6–8.0s under mutations — well under the 30s target. B3 (sign flip), the blind spot from the previous experiment, is now caught.

6. **B4 non-detection warrants investigation.** The multi-component invariant test passed under the B4 mutation, suggesting either the fixture doesn't trigger multi-component logic or the completeness check is insufficient. This is the same bug that WAS detected by the full 221-test suite in the previous experiment (5 tests failed). The isolation test is not replicating that detection.

---

## Recommendation

### RQ1 Decision: H0 CONFIRMED — formula reduction not justified

- **Do NOT reduce p from n/10.** With nbiter=2, rSVD quality is insufficient regardless of oversampling. Reducing p would only save a few milliseconds on rSVD before escalating to the expensive Level 5.
- **The real fix is escalation, not oversampling.** The production code already gracefully handles poor-quality rSVD by escalating to forced dense EVD (Level 5). This is working as designed. If the 111.8s test is the problem, tier it to `--profile slow` (the CI optimization from the previous experiment already addressed this via opt-level=2).
- **Consider increasing nbiter to 4–6 under ComputeMode::RustNative.** The power-iteration benefit is massive (16–1001×). At nbiter=4–6, p=30 might suffice for all fixtures, enabling genuine speedup. This would be a separate experiment.
- **Near-term action:** Tier `comp_d_rsvd_blobs_connected_2000` and `test_level_3_rsvd_valid_on_large_path` to a `--profile slow` nextest profile rather than trying to optimize rSVD parameters.

### RQ2 Decision: H1 CONFIRMED — promote invariant tests (with B4 fix)

- **Promote `test_exp_solver_invariants` to the permanent test suite** (rename to `test_solver_invariants`). 6/8 detection rate, ~5s suite time.
- **Fix B4 non-detection** before promoting: either (a) verify `disconnected_200` fixture triggers multi-component code, (b) add an assertion that explicitly counts non-zero rows in the result, or (c) add a synthetic 2-component graph with known structure.
- **Accept B6 as a known gap.** Warm-restart truncation requires a dedicated adversarial test with a graph that provably requires multiple restart cycles. Not worth the complexity for marginal coverage.

---

## Status
CONCLUSIVE_POSITIVE (RQ2) / CONCLUSIVE_NEGATIVE (RQ1)

**RQ1:** H0 CONFIRMED. The oversampling formula cannot be safely reduced with nbiter=2. The true bottleneck is downstream Level 5 (forced dense EVD), not rSVD computation itself. The hypothesis that rSVD was the 111.8s bottleneck was incorrect — rSVD completes in <1s even at p=200; the slowness is in Level 5 escalation.

**RQ2:** H1 CONFIRMED. Per-solver invariant tests detect 6/8 mutations (vs. 1/8 with property tests) in 5.1s wall time (vs. 143s for property tests at opt2). Promote with a fix for B4 non-detection. B3 blind spot from the previous experiment is now closed.
