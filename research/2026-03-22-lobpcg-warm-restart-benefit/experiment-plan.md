# Experiment Plan: LOBPCG Warm-Restart Benefit on Adversarial Graphs

## Motivation

The unconvergence detection / warm-restart feature added in PR #115 (`lobpcg_solve`,
`src/solvers/lobpcg.rs` lines 287–420) has never been observed to fire on any of the
9 existing test fixtures. If the warm-restart loop is unreachable on any realistic UMAP
Laplacian, it is dead code that adds maintenance cost for no benefit. If it does fire on
adversarial graphs (path, ring, barbell), the question is whether it measurably improves
eigenvector residuals compared to a single-pass solve.

The decision this experiment informs: **keep, document, or revert the warm-restart
mechanism** (referenced in issue #120). The experiment also fixes the tolerance
discrepancy in issue #120 (`LOBPCG_ACCEPT_TOL = 1e-5`, not `1e-4` as the issue claims).

---

## Hypothesis

**Null hypothesis (H0):** For all normalized UMAP Laplacians with `n ≥ 2000` (the LOBPCG
level threshold), ChFSI prefiltering produces a sufficiently good initial subspace that
LOBPCG converges in a single pass. The warm-restart loop fires zero times (`restart_count
= 0`) on every adversarial graph (path, ring, barbell, epsilon-bridge) at sizes reachable
by the LOBPCG escalation level. The feature is correct but unreachable code for UMAP
inputs.

**Alternative hypothesis (H1):** At least one adversarial graph with `n ≥ 2000` causes
`lobpcg_solve` to fire the warm-restart loop (`restart_count ≥ 1`). When warm-restart
fires, the final per-vector residuals are `< 1e-5` (below `LOBPCG_ACCEPT_TOL`), whereas
a single-pass solve (0 restarts) on the same graph returns residuals `≥ 1e-5`. The
improvement is measurable as a ratio of max residuals: `residual_no_restart /
residual_with_restart ≥ 10`.

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Graph type | path P\_n, ring C\_n, epsilon-bridge | Different stress mechanisms (small eigengap, degenerate eigenspace, controllable Fiedler gap) |
| n (graph size) | 2000, 3000, 5000 | All above `DENSE_N_THRESHOLD = 2000`; tests LOBPCG behavior at increasing stress |
| Bridge weight (epsilon-bridge only) | 1.0, 1e-2, 1e-4, 1e-6, 1e-8 | Continuously sweeps the eigengap from well-conditioned to near-zero |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method |
|--------|------|-------------------|
| Warm restart count | integer (0–3) | Added to `lobpcg_solve` return type (see Phase 1) |
| Max per-vector residual (post-full-solve) | dimensionless f64 | `common::eigenpair_residual` called on each eigenpair after `lobpcg_solve` |
| Max per-vector residual (single-pass, no warm restart) | dimensionless f64 | Direct `linfa_linalg::lobpcg` call in test, bypassing the warm-restart loop |
| Residual improvement ratio | ratio (≥ 1.0) | `max_residual_no_restart / max_residual_with_restart`; > 1 means WR helped |

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| `n_components` | 2 | Standard UMAP 2D embedding; matches all existing tests |
| `seed` | 42 | Reproducibility |
| `regularize` | false | Tests Level 1 (unregularized LOBPCG); regularization is a separate escalation |
| `LOBPCG_ACCEPT_TOL` | 1e-5 | Actual code value; issue #120's claim of `1e-4` is incorrect |
| `MAX_WARM_RESTARTS` | 3 | Actual code value |
| ChFSI prefiltering | always active for n ≥ 2000 | `CHEB_MIN_N = 1000`; always present in LOBPCG level |

---

## Inputs and Data

All data is generated synthetically inline in Rust test code. No Python fixtures required.

- **Path graph P\_n**: The normalized Laplacian is analytically known. `large_path_laplacian(n)` already exists in `tests/integration/test_adversarial_graphs.rs` (lines 255–276) and builds `CsMatI<f64, usize>` directly. The spectral gap is `λ_2 ≈ π²/(n+1)²`, which at n=2000 is `≈ 2.5×10⁻⁶` — potentially too small for convergence within `maxiter = min(n*5, 300) = 300` iterations.
- **Ring graph C\_n**: Normalized Laplacian has all diagonal entries = 1.0 and off-diagonals = −0.5 (every node has degree 2). λ_2 = λ_3 (exactly degenerate eigenspace) — the Gram matrix conditioning stress case. A `large_ring_laplacian(n)` function must be written analogously to `large_path_laplacian`.
- **Epsilon-bridge (two clusters)**: `make_epsilon_bridge(cluster_size, bridge_weight)` exists and produces an f32/u32 adjacency matrix. To test at the solver level directly, a `large_epsilon_bridge_laplacian(cluster_size, bridge_weight)` builder must be written that computes the normalized Laplacian in f64.
- **`sqrt_deg` arrays**: For path P\_n (unit weights): `[1.0, √2, √2, …, √2, 1.0]` (endpoints have degree 1, interior nodes degree 2). For ring C\_n: `[√2, √2, …, √2]` (all degree 2). For epsilon-bridge: computed from actual edge weights.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| Path P\_2000, P\_3000, P\_5000 | Inline Rust (`large_path_laplacian`) | Tiny eigengap; analytically known spectrum | Primary stress test for rnorm-gate warm restart |
| Ring C\_2000, C\_3000 | Inline Rust (new `large_ring_laplacian`) | Exactly degenerate λ\_2 = λ\_3; Gram ill-conditioning stress | Tests post-RR unconvergence detection path |
| Epsilon-bridge, bridge\_weight = 1.0 to 1e-8 (5 values) | Inline Rust (new `large_epsilon_bridge_laplacian`) | Controllable Fiedler eigengap | Sweeps warm-restart trigger threshold |

---

## Experiment Directory Layout

```
research/2026-03-22-lobpcg-warm-restart-benefit/
├── scripts/
│   ├── run_experiment.sh       # Full experiment command with output capture
│   └── collect_results.sh      # Runs tests, extracts metrics, writes results/raw.txt
├── data/                       # Empty — all inputs are synthetic inline
├── results/
│   ├── raw.txt                 # Captured test output (stdout + stderr)
│   └── metrics.md              # Manually or script-extracted metric table
└── report.md                   # Written by /write-report
```

The experiment also touches files **outside** `research/` (source changes):
- `src/solvers/lobpcg.rs` — add restart count to return value
- `src/solvers/mod.rs` — update call sites
- `tests/integration/test_adversarial_graphs.rs` — add `large_ring_laplacian`, `large_epsilon_bridge_laplacian` helper functions; update any `lobpcg_solve` call sites for new signature
- `tests/integration/test_comp_f_lobpcg.rs` — update `lobpcg_solve` call sites
- `tests/integration/test_warm_restart_benefit.rs` — NEW experiment test file (gated behind `--features testing`)
- `benches/spectral_bench.rs` — fix existing signature drift (missing `sqrt_deg` arg) as a prerequisite

---

## Environment

**No custom environment needed.**

The experiment is purely Rust. All required dependencies (`sprs`, `ndarray`, `linfa-linalg`,
`faer`, `log`, `approx`) are already in `Cargo.toml`. The test harness is `cargo test`.
No Python, no conda, no `environment.yml`.

The `testing` feature flag already exists (`testing = []` in `Cargo.toml`) and gates
access to internal solver functions from integration tests. All new experiment tests must
be compiled with `--features testing`.

---

## Implementation Phases

### Phase 1: Fix Bench Compilation and Instrument `lobpcg_solve`

**Goal:** Ensure the codebase compiles cleanly before adding new tests, and expose restart
count from `lobpcg_solve` for whitebox testing.

Files to modify:
1. **`benches/spectral_bench.rs`**: Add the missing `sqrt_deg: &Array1<f64>` argument to
   the existing `lobpcg_solve` call. Construct `sqrt_deg` from the ring graph degrees
   (all = 2 for a ring, so `sqrt_deg = vec![2f64.sqrt(); n]`). This unblocks `cargo bench`.

2. **`src/solvers/lobpcg.rs`**: Change the return type of `lobpcg_solve` from
   `Option<EigenResult>` to `Option<(EigenResult, usize)>`, where the `usize` is the
   number of warm restarts that fired (0 = first-pass success, 1 = one restart needed,
   etc.). Specifically:
   - Track `let mut restart_count: usize = 0;` initialized before the loop.
   - On each successful exit via `return Some(result)` inside the loop body (not the
     `last_result` fallback), return `Some((result, restart_count))`.
   - On the `last_result` fallback path after loop exhaustion, return
     `Some((last_result, restart_count))` (or `None` if `last_result` is `None`).
   - Increment `restart_count` at the same point the `"warm restart {restart}/..."` log
     message is emitted.

3. **`src/solvers/mod.rs`**: Update the two call sites of `lobpcg_solve` to destructure
   the new return type: `if let Some((result, _restarts)) = lobpcg_solve(...)`.
   The restart count is discarded in production code but exposed for tests.

4. **`tests/integration/test_comp_f_lobpcg.rs`**: Update the `make_lobpcg_tests!` macro
   or `run_lobpcg_test` helper to destructure `(result, _restarts)`. Existing test
   assertions are unchanged — just unwrap the tuple.

5. **`tests/integration/test_adversarial_graphs.rs`**: Update any `lobpcg_solve` call
   sites for the new return type.

Verify: `cargo test --features testing` passes all existing tests.

### Phase 2: Add Adversarial Laplacian Builders

**Goal:** Provide solver-level (`CsMatI<f64, usize>`) normalized Laplacians for ring and
epsilon-bridge graphs, so experiment tests can call `lobpcg_solve` directly without going
through the full `spectral_init` pipeline.

Add to `tests/integration/test_adversarial_graphs.rs` (or a new private test helper module
`tests/common/adversarial_laplacians.rs`):

1. **`large_ring_laplacian(n: usize) -> CsMatI<f64, usize>`**: Builds the normalized
   Laplacian of ring C\_n. All diagonal entries = 1.0; each node has exactly two
   off-diagonal entries = −0.5 (since every node has degree 2, `D^{-1/2}AD^{-1/2} = A/2`).
   The matrix is `n×n` CSR.

2. **`ring_sqrt_deg(n: usize) -> Array1<f64>`**: Returns `Array1::from_elem(n, 2f64.sqrt())`.

3. **`large_epsilon_bridge_laplacian(cluster_size: usize, bridge_weight: f64) -> CsMatI<f64, usize>`**:
   Two complete subgraphs of size `cluster_size` connected by a single edge of weight
   `bridge_weight`. Compute degrees exactly, then build the normalized Laplacian. Total
   `n = 2 * cluster_size`.

4. **`epsilon_bridge_sqrt_deg(cluster_size: usize, bridge_weight: f64) -> Array1<f64>`**:
   Returns the square-root degree vector for the epsilon-bridge graph. The two bridge
   nodes have degree `cluster_size - 1 + bridge_weight`; all other nodes have degree
   `cluster_size - 1`.

Verify: `cargo test --features testing -- large_ring_laplacian large_epsilon_bridge_laplacian`
(add a trivial sanity test asserting the matrix is square and has correct nonzero count).

### Phase 3: Write Experiment Tests

**Goal:** Create the whitebox tests that measure warm-restart count and residuals on all
adversarial graph types.

Create **`tests/integration/test_warm_restart_benefit.rs`** (requires `--features testing`):

```
#[cfg(test)]
// NOTE: compile only with --features testing
// Run with: cargo test --features testing warm_restart -- --nocapture
```

Test structure for each adversarial graph:

```
fn run_warm_restart_test(
    op: &CsrOperator,
    sqrt_deg: &Array1<f64>,
    n_components: usize,
    label: &str,
) {
    // 1. Call lobpcg_solve (with warm restart, the production path)
    let (result, restart_count) = lobpcg_solve(op, n_components, 42, false, sqrt_deg)
        .expect("lobpcg_solve returned None");
    let (eigenvalues, eigenvectors) = result;

    // 2. Compute per-vector residuals using common::eigenpair_residual
    let max_residual = (0..n_components + 1)
        .map(|i| common::eigenpair_residual(op, eigenvectors.column(i), eigenvalues[i]))
        .fold(0f64, f64::max);

    // 3. Call linfa_linalg::lobpcg directly (single-pass, no warm restart)
    // ... compute single-pass residuals ...

    // 4. Print structured metrics
    println!("METRIC graph={label} restart_count={restart_count} \
              max_residual_with_restart={max_residual:.6e} \
              max_residual_no_restart={single_pass_max:.6e}");

    // 5. Assert correctness
    assert!(max_residual < 1e-5, "final residuals exceeded LOBPCG_ACCEPT_TOL");
}
```

Tests to implement:

| Test function | Graph | n | Assertion |
|---|---|---|---|
| `test_path_2000_warm_restart` | P\_2000 | 2000 | `assert!(max_residual < 1e-5)`; print restart_count |
| `test_path_3000_warm_restart` | P\_3000 | 3000 | same |
| `test_path_5000_warm_restart` | P\_5000 | 5000 | same |
| `test_ring_2000_warm_restart` | C\_2000 | 2000 | same |
| `test_ring_3000_warm_restart` | C\_3000 | 3000 | same |
| `test_epsilon_bridge_sweep` | ε-bridge, cluster=1000, bridge∈{1.0,1e-2,1e-4,1e-6,1e-8} | 2000 | `assert!(max_residual < 1e-5)`; print restart_count per weight |

**Single-pass baseline within each test**: Before calling `lobpcg_solve`, directly call
`linfa_linalg::lobpcg` (imported via `linfa_linalg::Lobpcg::params(n_components+1).max_iterations(...)`)
on the same operator with the same seed/initial block — but stop after one call with no restart.
Record the residuals from that single call. This establishes the "no warm restart" baseline.

Add to `Cargo.toml` under `[[test]]`:
```toml
[[test]]
name = "test_warm_restart_benefit"
path = "tests/integration/test_warm_restart_benefit.rs"
required-features = ["testing"]
```

### Phase 4: Dry Run and Metric Capture Scripts

**Goal:** Verify end-to-end pipeline, create the `results/` capture scripts.

Create **`research/2026-03-22-lobpcg-warm-restart-benefit/scripts/run_experiment.sh`**:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS_DIR="research/2026-03-22-lobpcg-warm-restart-benefit/results"
mkdir -p "$RESULTS_DIR"

echo "=== Running warm restart benefit experiment ===" | tee "$RESULTS_DIR/raw.txt"
cargo test --features testing --release warm_restart -- --nocapture 2>&1 \
    | tee -a "$RESULTS_DIR/raw.txt"
echo "=== Done. Results in $RESULTS_DIR/raw.txt ==="
```

Create **`research/2026-03-22-lobpcg-warm-restart-benefit/scripts/collect_results.sh`**:

```bash
#!/usr/bin/env bash
# Extract METRIC lines from raw.txt and format as a markdown table
grep '^METRIC' results/raw.txt | \
    awk 'BEGIN {print "| graph | restart_count | max_residual_with_restart | max_residual_no_restart |"}
         {
           split($0, a, " ");
           # parse key=value pairs
           ...
           print "| " graph " | " rc " | " res_wr " | " res_nr " |"
         }'
```

Dry run command:
```bash
cargo test --features testing test_path_2000_warm_restart -- --nocapture
```

Expected dry run output for path P_2000 (if H0 holds):
```
METRIC graph=path_2000 restart_count=0 max_residual_with_restart=3.14e-07 max_residual_no_restart=3.14e-07
```

Expected dry run output if H1 holds:
```
METRIC graph=path_2000 restart_count=1 max_residual_with_restart=8.23e-06 max_residual_no_restart=2.17e-03
```

---

## Execution Protocol

After all phases are implemented and the dry run succeeds:

1. Run the full experiment from the project root:
   ```bash
   bash research/2026-03-22-lobpcg-warm-restart-benefit/scripts/run_experiment.sh
   ```

2. Review `results/raw.txt` for METRIC lines and any test failures.

3. Extract the metric table:
   ```bash
   bash research/2026-03-22-lobpcg-warm-restart-benefit/scripts/collect_results.sh \
       > research/2026-03-22-lobpcg-warm-restart-benefit/results/metrics.md
   ```

4. Inspect for the key signals:
   - Is `restart_count > 0` on any graph? → H1 candidate.
   - Is `max_residual_no_restart ≥ 1e-5` on any graph? → Warm restart was needed.
   - Is `max_residual_with_restart < 1e-5` on the same graph? → Warm restart helped.

5. Verify all existing tests still pass:
   ```bash
   cargo test --features testing
   ```

---

## Analysis Plan

The central question is answered by the `restart_count` column:

**If all `restart_count == 0`**: ChFSI prefiltering always provides a subspace good enough
for single-pass convergence. Compare `max_residual_no_restart` vs. `max_residual_with_restart`
— they should be equal (the warm-restart path was never entered). This conclusively supports
H0. The recommendation is to document the warm-restart mechanism as defensive code for
non-ChFSI contexts, or to revert it with an explanation.

**If any `restart_count ≥ 1`**: Warm restart fired. Check whether:
- `max_residual_with_restart < 1e-5` (final quality is good): WR recovered convergence.
- `max_residual_no_restart ≥ 1e-5` (single pass was insufficient): WR provided measurable benefit.
- Compute `improvement_ratio = max_residual_no_restart / max_residual_with_restart`. A ratio
  of ≥ 10 is considered "measurably beneficial".

**Epsilon-bridge sweep analysis**: Plot or tabulate `(bridge_weight, restart_count,
max_residual_with_restart)`. Find the threshold `bridge_weight*` below which warm restart
fires. Report whether this weight is achievable by UMAP k-NN graph construction in practice
(compare to the 9 existing fixture datasets' Fiedler eigenvalues).

---

## Success Criteria

- **Conclusive positive (H1 supported):** At least one test shows `restart_count ≥ 1` AND
  `max_residual_with_restart < 1e-5` AND `max_residual_no_restart ≥ 1e-5`. Specifically:
  `improvement_ratio ≥ 10` (one order of magnitude improvement).

- **Conclusive negative (H0 supported):** All graph/n/weight combinations show
  `restart_count = 0` and `max_residual_with_restart < 1e-5`. This includes the most
  extreme adversarial cases: P\_5000, C\_3000, and epsilon-bridge with `bridge_weight =
  1e-8`.

- **Inconclusive:** Warm restart fires (`restart_count ≥ 1`) but `max_residual_with_restart ≥ 1e-5`
  even after 3 restarts — the loop exhausted without recovery. This would require
  investigation with larger n or with regularization enabled.

---

## Threats to Validity

### Internal

- **linfa-linalg random initialization**: linfa-linalg may use its own internal RNG not
  fully controlled by the `seed` parameter passed to `lobpcg_solve`. Results could differ
  across runs if ChFSI only approximately controls the initial subspace. Mitigate by
  running each test 3 times with seeds 42, 43, 44 and reporting the worst case.

- **ChFSI dominance masking the effect**: If ChFSI always produces a near-optimal
  subspace, warm restart will never fire regardless of graph topology. The experiment
  would then show H0 is supported for ChFSI-preprocessed inputs but cannot speak to
  non-ChFSI use. This is not a confound — it is the correct answer to the research
  question.

- **maxiter capping at 300**: For n=5000, `min(n*5, 300) = 300` — the cap means LOBPCG
  has the same iteration budget as for n=60. Path P\_5000 might not converge in 300
  iterations even with warm restart. If all 3 restarts exhaust without convergence, the
  result is inconclusive rather than evidence against warm restart's utility.

- **`restart_count` instrumentation accuracy**: The `usize` field is incremented at the
  same log emission point — verify in code review that the increment point correctly
  captures "a restart was needed" rather than "a restart loop iteration was entered".

### External

- **UMAP k-NN graph structure**: Real UMAP graphs are far from path/ring graphs. The
  experiment proves or disproves the feature's value on synthetic adversarial inputs, not
  on production UMAP Laplacians. A conclusive positive here may not translate to benefit
  in production use.

- **linfa-linalg version**: The experiment uses linfa-linalg 0.2 as specified in
  `Cargo.toml`. A future linfa-linalg version may have different convergence behavior
  that changes the result.

- **`f32` vs. `f64` precision**: linfa-linalg uses `f32` internally. The experiment
  is sensitive to the `f32`→`f64` residual gap. A future linfa-linalg using `f64`
  internally would likely show different warm-restart trigger rates.

---

## Estimated Resource Requirements

- **Compute time**: Each `lobpcg_solve` on a 5000-node graph runs in approximately 2–5
  seconds in release mode. Full experiment (≈ 15 graph/n/weight combinations × 3 seeds)
  ≈ 5–15 minutes total in `--release` mode. Debug mode ≈ 5× slower.
- **Disk space**: Source modifications + experiment folder: < 100 KB. Results output: < 1 MB.
- **Dependencies**: None beyond existing `Cargo.toml` entries. No new crates needed.
- **Feature flag**: All experiment tests require `--features testing`.

