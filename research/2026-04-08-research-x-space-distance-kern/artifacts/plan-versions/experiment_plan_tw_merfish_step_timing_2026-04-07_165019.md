# Experiment Plan: Trustworthiness Step-Timing on MERFISH Real Data

## Motivation

PR #238 concluded that X-space computation (x_dist + x_sort) is the productive next optimization target for `trustworthiness()`, based entirely on synthetic Gaussian/uniform data at n=10K. This experiment validates or refutes that conclusion on real MERFISH spatial transcriptomics data. The result informs whether to proceed with X-space ANN approximation as the next engineering investment, or whether real biological data geometry shifts the bottleneck elsewhere.

A secondary question is whether the MERFISH fixture's actual `d_x` differs materially from the d_x=10 assumed when writing PR #238 — file-size estimates suggest d_x may be ~48, which would push x_dist's fraction substantially higher and strengthen the optimization case.

---

## Hypothesis

**Null hypothesis (H0):** On MERFISH n=10K data, the X-space fraction (x_dist% + x_sort%) is not meaningfully different from the synthetic Gaussian baseline (56.1%), i.e., real data geometry does not change the step-timing distribution by more than ±5 percentage points.

**Alternative hypothesis (H1):** The X-space fraction on MERFISH n=10K differs from the synthetic baseline by more than ±5 pp — either higher (if d_x > 10 increases x_dist dominance) or structurally redistributed (if clustered Y-space geometry reduces y_dist's fraction). Regardless of direction, X-space (x_dist + x_sort) remains ≥50% of total runtime, validating X-space ANN as the productive optimization target on real data.

---

## Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Dataset | MERFISH n=10K, MERFISH n=50K | Real biological data vs. synthetic; scale stability |
| `k` | 15 (primary), optionally 5 and 30 (stretch) | Standard UMAP default; k-sweep to characterize penalty scaling |

---

## Dependent Variables (Metrics)

| Metric | Unit | Collection Method | Canonical Name |
|--------|------|-------------------|----------------|
| `x_dist_pct` | % of total thread-ns | Sum `step_timing.x_dist` / sum all steps × 100 | NEW (computed post-hoc from profiler JSON) |
| `x_sort_pct` | % of total thread-ns | Sum `step_timing.x_sort` / sum all steps × 100 | NEW (computed post-hoc) |
| `y_dist_pct` | % of total thread-ns | Sum `step_timing.y_dist` / sum all steps × 100 | NEW (computed post-hoc; key is `y_dist` in current source, `y_heap` in historical baselines) |
| `penalty_pct` | % of total thread-ns | Sum `step_timing.penalty` / sum all steps × 100 | NEW (computed post-hoc) |
| `x_space_pct` | % of total thread-ns | `x_dist_pct + x_sort_pct` | NEW (composite, computed post-hoc) |
| `d_x` | integer dimensions | `np.load(...).shape[1]` from npy header | NEW (not emitted by tw_profiler JSON) |
| Wall-clock `mean_s` | seconds | `profiler_json["mean_s"]` | Existing field in tw_profiler JSON |

**Notes on "NEW" metrics:** These are all computed post-hoc from the `step_timing` dict in the profiler JSON output. No changes to `src/metrics.rs` are required — they are research analysis artifacts, not CI-gated quality dimensions. The step-timing fractions inform optimization priority decisions only.

**Step-name warning:** The current codebase (post-flat_simd) emits `[timing:y_dist]` to stderr; the profiler parses this as the `y_dist` key in `step_timing`. Historical baseline files (`2026-04-06-y-heap-bottleneck-optimization`, `2026-04-05-tw-perf-rerun-clean`) use `y_heap`. The analysis script must normalize this: when loading a historical baseline, treat `y_heap` as `y_dist`.

---

## Controlled Variables

| Variable | Fixed Value | Rationale |
|----------|-------------|-----------|
| Profiler binary | `tw_profiler` at HEAD, `--features cli,profiling` | Same binary as synthetic baselines |
| Warmup iterations | 2 | Matches 2026-04-06-y-heap baseline config |
| Timed iterations | 5 | Sufficient for stable mean; matches scope report recommendation |
| `k` (primary run) | 15 | Standard UMAP default; same as 2026-04-06 baseline |
| Machine | Local development machine (same hardware as prior baselines) | Ensures valid cross-run fraction comparison |
| Build profile | `--release` | Required for meaningful performance data |

---

## Inputs and Data

The experiment uses pre-existing MERFISH fixtures — no data generation is needed. All four files are confirmed present in the codebase at their source paths.

| Dataset | Source | Properties | Purpose |
|---------|--------|------------|---------|
| `merfish_n10k_x.npy` | `research/2026-04-05-tw-perf-rerun-clean/data/merfish/` | ~10K cells, PCA-reduced, d_x TBD (~48 at f64 from file size, ~8-20 per MERFISH norms) | Primary test: step fractions on real data |
| `merfish_n10k_y.npy` | Same | ~10K cells, 2D UMAP embedding | Y-space input |
| `merfish_n50k_x.npy` | Same | ~50K cells, PCA-reduced | Stretch: scale-stability check |
| `merfish_n50k_y.npy` | Same | ~50K cells, 2D UMAP embedding | Stretch: Y-space input for n=50K |
| Synthetic baseline | `research/2026-04-06-y-heap-bottleneck-optimization/results/profiler/profiler_baseline_n10000.json` | Gaussian n=10K, k=15, step fractions: x_dist≈31.9%, x_sort≈24.2%, y_heap≈27.6%, penalty≈16.2% | Reference comparison |

**d_x determination:** The npy array shape must be read before drawing any conclusions. The pre-flight step reads the header: `python3 -c "import numpy as np; a=np.load('...merfish_n10k_x.npy'); print(a.shape, a.dtype)"`. The shape and dtype together determine actual d_x and storage format (f32 vs f64). This value must be recorded in results.

**Data validity:** All files were verified by file-size ratios (n=50K files are ~5× the n=10K equivalents), and Y files are consistent with 2D embeddings (~15× smaller than X files). No data generation scripts are needed.

---

## Experiment Directory Layout

```
research/2026-04-07-tw-merfish-step-timing/
├── environment.yml                  # Conda env (python 3.11, numpy, scipy, matplotlib)
├── scripts/
│   ├── check_shapes.py              # Read npy headers; print d_x, n, dtype for all fixtures
│   ├── run_profiler.sh              # Build tw_profiler; run on MERFISH fixtures; save JSON to results/
│   └── analyze_merfish_timing.py    # Load profiler JSON(s); compute step fractions; build comparison table
├── results/
│   ├── shapes.txt                   # Output of check_shapes.py (d_x, n, dtype per fixture)
│   ├── merfish_n10k_k15.json        # tw_profiler output for n=10K, k=15
│   ├── merfish_n10k_k15_stderr.txt  # Raw stderr (step timing lines) for n=10K, k=15
│   ├── merfish_n50k_k15.json        # (stretch) tw_profiler output for n=50K
│   ├── merfish_n50k_k15_stderr.txt  # (stretch) stderr for n=50K
│   └── step_fractions_comparison.md # Comparison table: MERFISH vs synthetic baseline
└── report.md                        # Final report (written by /write-report skill)
```

**No `data/` directory** — the experiment references existing fixtures by absolute path; no copying is needed.

---

## Environment

**Custom environment required (following established project pattern).**

Prior experiments all include a per-experiment `environment.yml`. The system shell has Python 3.13.2 + numpy 2.2.6, but established pattern pins Python 3.11 for consistency across experiments. The analysis script requires `scipy` (for confidence interval computation) and `matplotlib` (for optional bar chart). Neither is guaranteed present in the system environment.

```yaml
name: tw-merfish-step-timing
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=2.2
  - scipy=1.15
  - matplotlib=3.10
```

**Rationale:**
- `python=3.11` — matches all prior experiment envs (`y-heap-bench`, `tw-perf-rerun-clean`, `tw-perf-scaling`)
- `numpy=2.2` — required for `.npy` file I/O and array operations
- `scipy=1.15` — required for `scipy.stats.t.interval` CI computation on step fractions (follows `analyze_results.py` from `y-heap-bottleneck`)
- `matplotlib=3.10` — for optional bar chart output (non-interactive `Agg` backend)

The Rust toolchain (`cargo`, `rustc`) is the project standard and requires no conda packaging.

---

## Implementation Phases

### Phase 1: Directory Structure and Environment

1. Create `research/2026-04-07-tw-merfish-step-timing/` with subdirectories `scripts/` and `results/`
2. Create `environment.yml` with the spec above
3. Create `scripts/check_shapes.py`:
   ```python
   import numpy as np, pathlib
   BASE = pathlib.Path("research/2026-04-05-tw-perf-rerun-clean/data/merfish")
   for fname in ["merfish_n10k_x.npy", "merfish_n10k_y.npy",
                  "merfish_n50k_x.npy", "merfish_n50k_y.npy"]:
       a = np.load(BASE / fname)
       print(f"{fname}: shape={a.shape}, dtype={a.dtype}, size_MB={a.nbytes/1e6:.1f}")
   ```
   Run from project root: `python3 scripts/check_shapes.py > results/shapes.txt`
4. Verify environment builds: `micromamba create -f environment.yml`

**Deliverable:** `shapes.txt` with confirmed d_x and dtype for each fixture.

### Phase 2: Profiler Build and Run

Create `scripts/run_profiler.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESEARCH_DIR="$ROOT/research/2026-04-07-tw-merfish-step-timing"
MERFISH_DIR="$ROOT/research/2026-04-05-tw-perf-rerun-clean/data/merfish"

# Build
cd "$ROOT"
cargo build --release --features cli,profiling --bin tw_profiler

BIN="$ROOT/target/release/tw_profiler"

# n=10K, k=15 (primary)
"$BIN" \
  --x "$MERFISH_DIR/merfish_n10k_x.npy" \
  --y "$MERFISH_DIR/merfish_n10k_y.npy" \
  --output "$RESEARCH_DIR/results/merfish_n10k_k15.json" \
  --k 15 --iters 5 --warmup 2 \
  --stderr-capture "$RESEARCH_DIR/results/merfish_n10k_k15_stderr.txt"

echo "n=10K done"

# n=50K, k=15 (stretch — comment out if time-constrained)
"$BIN" \
  --x "$MERFISH_DIR/merfish_n50k_x.npy" \
  --y "$MERFISH_DIR/merfish_n50k_y.npy" \
  --output "$RESEARCH_DIR/results/merfish_n50k_k15.json" \
  --k 15 --iters 5 --warmup 2 \
  --stderr-capture "$RESEARCH_DIR/results/merfish_n50k_k15_stderr.txt"

echo "n=50K done"
```

Run from project root: `bash research/2026-04-07-tw-merfish-step-timing/scripts/run_profiler.sh`

**Key constraint:** `--stderr-capture` is mandatory for step timing. Without it, `step_timing` will be absent from the JSON and the experiment will fail.

**Deliverable:** `results/merfish_n10k_k15.json` (and optionally `merfish_n50k_k15.json`).

### Phase 3: Analysis Script

Create `scripts/analyze_merfish_timing.py`. This script must:

1. Load the MERFISH profiler JSON and the synthetic baseline JSON
2. Compute step fractions for each:
   ```python
   def step_fractions(step_timing: dict) -> dict:
       # step_timing maps step_name -> list of per-iteration ns values
       # Current source: keys are "x_dist", "x_sort", "y_dist", "penalty"
       # Baseline file: key is "y_heap" — normalize to "y_dist"
       totals = {k.replace("y_heap", "y_dist"): sum(v)
                 for k, v in step_timing.items()}
       grand = sum(totals.values())
       return {k: 100.0 * v / grand for k, v in totals.items()}
   ```
3. Compute composite x_space_pct = x_dist_pct + x_sort_pct
4. Produce `results/step_fractions_comparison.md` with a markdown table:

   | Step | Synthetic n=10K (Gaussian) | MERFISH n=10K | Delta (pp) |
   |------|---------------------------|---------------|------------|
   | x_dist | 31.9% | ? | ? |
   | x_sort | 24.2% | ? | ? |
   | y_dist | 27.6% | ? | ? |
   | penalty | 16.2% | ? | ? |
   | **x_space total** | **56.1%** | **?** | **?** |

5. Print d_x (loaded from `results/shapes.txt`) and annotate the table with it
6. Optionally (with matplotlib): produce a bar chart comparing the two profiles

**Deliverable:** `results/step_fractions_comparison.md` with a complete comparison table.

### Phase 4: Dry Run and Verification

Before committing to the full 5-iteration run, do a minimal dry run:

```bash
# Quick sanity check: 1 iter, no warmup
./target/release/tw_profiler \
  --x research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_x.npy \
  --y research/2026-04-05-tw-perf-rerun-clean/data/merfish/merfish_n10k_y.npy \
  --output temp/merfish_dryrun.json \
  --k 15 --iters 1 --warmup 0 \
  --stderr-capture temp/merfish_dryrun_stderr.txt
```

Verify:
- `temp/merfish_dryrun.json` exists and has `step_timing` key (not absent)
- `step_timing` has keys `x_dist`, `x_sort`, `y_dist`, `penalty` (not `y_heap`)
- Values are plausible (> 0)
- `score` is a valid trustworthiness value in (0, 1]

If `step_timing` is absent, the build lacks the `profiling` feature — rebuild with `--features cli,profiling`.
If `y_heap` appears instead of `y_dist`, the binary was built from an older checkout — verify HEAD.

---

## Execution Protocol

After implementation is complete and the dry run passes:

1. **Confirm d_x:** `python3 research/2026-04-07-tw-merfish-step-timing/scripts/check_shapes.py > research/2026-04-07-tw-merfish-step-timing/results/shapes.txt && cat research/2026-04-07-tw-merfish-step-timing/results/shapes.txt`

2. **Build and run primary:** `bash research/2026-04-07-tw-merfish-step-timing/scripts/run_profiler.sh`
   - Expected wall time for n=10K: ~1–3 minutes (5 iters × ~0.28s/iter × 1 warmup overhead)
   - Expected wall time for n=50K: ~20–80 minutes (scales O(n²); 25× longer than 10K)
   - If n=50K is too slow, abort after confirming n=10K results are sufficient

3. **Analyze:** `python3 research/2026-04-07-tw-merfish-step-timing/scripts/analyze_merfish_timing.py`

4. **Review output:** Read `research/2026-04-07-tw-merfish-step-timing/results/step_fractions_comparison.md`

---

## Analysis Plan

### Step Fraction Comparison

For each step, compute mean fraction across the 5 timed iterations (after discard of 2 warmup). The step_timing values in the JSON are thread-aggregate nanosecond totals (AtomicU64 sum over all Rayon threads × all n rows per call). Ratios between steps are valid for bottleneck analysis even though absolute values are not wall-clock time.

Baseline for comparison: `research/2026-04-06-y-heap-bottleneck-optimization/results/profiler/profiler_baseline_n10000.json` (key `y_heap` → normalize to `y_dist`).

### d_x Effect

Record actual d_x from the npy header. The theoretical x_dist FLOP count grows as O(n² × d_x). At fixed d_y=2:
- If d_x = 10: expect x_dist/y_dist ratio ≈ 5:1 (matching synthetic baseline)
- If d_x = 48: expect x_dist/y_dist ratio ≈ 24:1, x_dist fraction potentially 50–60%

The observed x_dist_pct should be compared to the theoretical prediction `(d_x / (d_x + d_y + constant)) × 100` as a sanity check on the measurement.

### Clustering Geometry Effect on y_dist

Measure y_dist_pct on MERFISH vs synthetic. If MERFISH shows y_dist_pct lower by >5pp, attribute to cache effects from clustered 2D layout. Below 5pp delta, treat as noise.

### x_space_pct Threshold Check

The binary verdict: is `x_dist_pct + x_sort_pct ≥ 50%`?
- If YES: X-space ANN is validated as productive optimization target on real data
- If NO: Report which step dominates and why (likely only if d_x is very small, e.g. <5)

---

## Success Criteria

- **Conclusive positive (H1 validated):** MERFISH n=10K x_space_pct ≥ 50%, with all four step fractions present and measurable. X-space ANN target is confirmed valid for real biological data.
- **Conclusive negative (H0):** MERFISH n=10K x_space_pct < 50%, with another step (e.g. y_dist or penalty) dominating. Would require revising the optimization roadmap.
- **d_x resolved:** Actual d_x from npy header recorded; the file-size ambiguity (d_x ≈ 10 vs ≈ 48) is resolved. If d_x > 20, H2 (x_dist increases even further) is also testable.
- **Comparison table complete:** Side-by-side MERFISH vs Gaussian n=10K fractions produced.
- **Stretch — scale stability:** MERFISH n=50K fractions within ±5pp of n=10K fractions, confirming the conclusion holds across scales.
- **Inconclusive:** If `step_timing` is absent from the JSON (missing `profiling` feature in build), or if the profiler crashes on MERFISH data, no conclusion can be drawn. The dry run in Phase 4 guards against this.

---

## Threats to Validity

### Internal

1. **Thread-aggregate ns ≠ wall-clock**: `step_timing` values are sums over all Rayon threads. If one step parallelizes better than another (different work distribution across rows), its ns total may be inflated or deflated relative to actual wall time. Fractions are valid relative metrics, not absolute time measurements.
2. **Single machine, single run order**: Results are not independent across iterations (CPU thermal state, TLB warm-up). The 2-warmup / 5-timed protocol mitigates but does not eliminate this.
3. **AtomicU64 overflow**: Very long runs (n=50K at 5 iters) could theoretically overflow u64 nanosecond counters (~18 seconds of single-thread time per step per call). Extremely unlikely but worth checking if any step value is near u64::MAX.
4. **`y_dist` vs `y_heap` key mismatch**: If the build uses an older binary that emits `y_heap`, the analysis script will need to handle both. The dry-run verification step catches this.

### External

1. **Single dataset**: One MERFISH dataset (mouse hypothalamus, ~10K cells). Results may not generalize to other biological datasets with different intrinsic dimensionality or cluster structure.
2. **Fixed UMAP embedding**: The Y-space fixture is a pre-computed UMAP embedding. A different embedding of the same data (different random seed or hyperparameters) would produce different 2D geometry and potentially different y_dist cache effects.
3. **Hardware-specific SIMD behavior**: x_dist uses AVX2+FMA (8-wide f64); y_dist uses a batched 2-point kernel. The step fraction ratios observed on this machine may differ on hardware without AVX2 or with different cache sizes.
4. **n=10K scope**: PR #238's 61% figure may be from n=100K. The O(n²·k) penalty cost grows relative to O(n²·d) distance costs as n increases. The fractions at n=10K may understate X-dominance at production-scale n.

---

## Estimated Resource Requirements

- **Compute:** n=10K primary run: ~5–15 minutes total (build + 5 iters × ~0.28s + overhead). n=50K stretch: ~30–90 minutes.
- **Disk:** Profiler JSON outputs are small (<10 KB each). No large data generation needed.
- **Dependencies:** Rust toolchain at HEAD (already present), Python 3.11 + numpy + scipy + matplotlib (via environment.yml or system Python if scipy is available).
- **No external services** required.
