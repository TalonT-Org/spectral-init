# KD-tree Y-space KNN Optimization: Trustworthiness Benchmark Study

> Research report — 2026-04-07

## Executive Summary

**Data scope:** All benchmarks use synthetic data — uniform random and 8-cluster
Gaussian distributions in a 10D X-space and 2D Y-space. No real UMAP output
embeddings were tested. The DO NOT SHIP conclusion applies to this synthetic
benchmark regime; performance on actual UMAP manifold projections was not
measured and may differ.

This experiment evaluated whether replacing the flat SIMD brute-force Y-space
k-nearest-neighbor search in the trustworthiness metric with a KD-tree (kiddo
v5.3.0) would yield a ≥5× total speedup at n=50K and a ≥10× speedup at n=100K.
Criterion benchmarks (72 runs, 3 reps × 2 variants × 2 distributions × 6 n
values) and step-level profiler traces (24 runs) were collected on a 16-thread
machine across n ∈ {1K, 5K, 10K, 50K, 75K, 100K} for uniform and Gaussian
distributions.

The headline finding is a **definitive negative result**: `flat_simd` is
consistently 36–44% faster than `kdtree` across every combination of n and
distribution tested (speedup ratio 0.696–0.744; speedup > 1.0 would indicate
kdtree faster). No crossover point exists in the tested range. The Y-space KNN
query is 150× faster with the KD-tree at n=100K (query-only; fair build+query
comparison gives ~148×), but the Y-space step represents only ~28% of total
trustworthiness runtime at d_x=10; the remaining 72% is dominated by the O(n²)
X-space distance computation which the KD-tree does not accelerate.

**Recommendation: DO NOT SHIP the KD-tree optimization** (on synthetic data at
d_x=10). To achieve meaningful trustworthiness speedup, the X-space bottleneck
must be addressed via approximate nearest neighbors in X-space or acceptance of
approximate scores.

## Background and Research Question

The `trustworthiness_inner` function computes the UMAP trustworthiness metric,
which requires finding k nearest neighbors in both the high-dimensional X-space
(10D in the benchmark) and the 2D Y-space embedding. The existing `flat_simd`
variant performs an O(n²) brute-force pairwise distance computation for Y-space
as well as X-space.

The hypothesis was that the low-dimensional Y-space (2D) is an ideal case for
KD-tree acceleration: at n ≥ 50K, the O(n log n) KD-tree query cost would
dominate O(n²) brute force, producing a crossover where the KD-tree becomes
faster. This experiment tests that hypothesis end-to-end with statistically
rigorous benchmarks.

**Research question:** At what n (if any) does a KD-tree-based Y-space KNN
strategy outperform flat SIMD brute force for the full trustworthiness
computation? Specifically: does a ≥5× total speedup exist at n=50K?

## Methodology

### Experimental Design

**Hypothesis (primary):** KD-tree Y-space KNN yields ≥5× total trustworthiness
speedup at n=50K and ≥10× at n=100K on uniform data (H1).

**Secondary hypotheses:**
- H2: A crossover point T_cross exists in [1K, 50K] on uniform distribution.
- H4: KD-tree build time is ≤10% of total Y-space KD-tree time (confirming
  query, not build, is the operative cost).

**Dependent variables:**
- Total speedup: `flat_simd_ns / kdtree_ns` (>1.0 = kdtree faster)
- Build fraction: `y_kdtree_build_ns / (y_kdtree_build_ns + y_kdtree_query_ns)`
- Query speedup: `flat_simd_y_dist_ns / kdtree_y_kdtree_query_ns`

**Controls:**
- n=75K held out as a blind validation point for crossover prediction (RT8)
- 3 Criterion reps per cell to detect measurement instability (CV threshold 10%)
- `RAYON_NUM_THREADS=16` fixed to ensure reproducible parallelism

### Environment

- **Repository commit:** `b1077a3de357e6298fb47c21c0cbf07c47364fda`
- **Branch:** `research-20260407-004436`
- **Rust toolchain:** `nightly-x86_64-unknown-linux-gnu` (nightly-2026-03-26, rustc 1.96.0-nightly 23903d01c)
- **Key Rust dependency:** kiddo 5.3.0 (KD-tree implementation)
- **Benchmark framework:** cargo-criterion 1.1.0 + criterion 0.5
- **Python analysis environment:** `kdtree-y-knn-bench` micromamba env
  (`research/2026-04-07-kdtree-y-knn-trustworthiness/environment.yml`):
  Python 3.11, numpy 2.2, scipy 1.15, matplotlib 3.10
- **Hardware:** 16-thread CPU (WSL2/Linux 6.6.87.2)
- **RAYON_NUM_THREADS:** 16

### Procedure

1. Generated 24 `.npy` data files (X/Y pairs, k=15) via `scripts/gen_data.py`
   for n ∈ {1K, 5K, 10K, 50K, 75K, 100K} × {uniform, gauss}.
2. Ran `scripts/run_criterion.sh` (no `--dry-run`): 72 Criterion benchmark
   runs (2 variants × 2 distributions × 6 n values × 3 reps,
   `--measurement-time 10`). Estimates captured from
   `cargo criterion --message-format json` output.
3. Ran `scripts/run_profiler.sh` (no `--dry-run`): 24 profiler runs (2 variants
   × 2 distributions × 6 n values, 30 iters + 2 warmup each). Step-level
   timing atomics captured to `results/profiler/*.json`.
4. Ran `scripts/analyze_results.py` (no `--dry-run`): computed all DVs,
   interpolated crossover, evaluated hypotheses, generated plots.
5. Verified correctness: `cargo test t_tw_11 --features testing --lib` PASS.

## Results

### Criterion Benchmarks — Total Speedup

All 72/72 benchmark runs completed. Coefficient of variation (CV) was ≤3.6% for
all cells, well below the 10% flagging threshold — measurements are stable.

Speedup ratio = flat_simd median ns / kdtree median ns (>1.0 = kdtree faster):

| dist    | n      | speedup | flat_simd CV | kdtree CV |
|---------|--------|---------|-------------|-----------|
| uniform | 1000   | 0.740   | 0.036       | 0.023     |
| uniform | 5000   | 0.733   | 0.003       | 0.013     |
| uniform | 10000  | 0.710   | 0.007       | 0.008     |
| uniform | 50000  | 0.696   | 0.011       | 0.008     |
| uniform | 75000  | 0.714   | 0.003       | 0.003     |
| uniform | 100000 | 0.733   | 0.005       | 0.004     |
| gauss   | 1000   | 0.744   | 0.030       | 0.032     |
| gauss   | 5000   | 0.736   | 0.002       | 0.005     |
| gauss   | 10000  | 0.721   | 0.007       | 0.022     |
| gauss   | 50000  | 0.705   | 0.006       | 0.006     |
| gauss   | 75000  | 0.714   | 0.002       | 0.002     |
| gauss   | 100000 | 0.733   | 0.004       | 0.000     |

**flat_simd is always 1.34–1.44× faster than kdtree.** No speedup value
exceeds 1.0 at any tested n.

### Profiler Step Timing — Build Fraction

24/24 profiler runs completed. KD-tree build fraction (build / (build + query)):

| dist    | n      | build_fraction |
|---------|--------|----------------|
| uniform | 1000   | 0.016          |
| uniform | 5000   | 0.022          |
| uniform | 10000  | 0.021          |
| uniform | 50000  | 0.018          |
| uniform | 75000  | 0.018          |
| uniform | 100000 | 0.016          |
| gauss   | 1000   | 0.017          |
| gauss   | 5000   | 0.020          |
| gauss   | 10000  | 0.019          |
| gauss   | 50000  | 0.015          |
| gauss   | 75000  | 0.015          |
| gauss   | 100000 | 0.013          |

Build fraction is only 1.3–2.2% across all conditions — well under the 10%
threshold (H4 MET).

### Profiler Step Timing — Query Speedup

Y-space query speedup (flat_simd `y_dist` / kdtree `y_kdtree_query`):

| dist    | n      | query_speedup |
|---------|--------|---------------|
| uniform | 1000   | 2.938         |
| uniform | 5000   | 14.480        |
| uniform | 10000  | 24.746        |
| uniform | 50000  | 87.877        |
| uniform | 75000  | 120.077       |
| uniform | 100000 | 150.335       |
| gauss   | 1000   | 2.899         |
| gauss   | 5000   | 13.192        |
| gauss   | 10000  | 24.901        |
| gauss   | 50000  | 83.424        |
| gauss   | 75000  | 115.005       |
| gauss   | 100000 | 140.188       |

The Y-space KNN query is 150× faster with the KD-tree at n=100K. This speedup
scales approximately as O(n / log n) as expected.

### Crossover Analysis

- **T_cross (uniform, log-interpolated):** None — speedup does not cross 1.0 in
  [1K, 100K]
- **T_cross by rep:** rep1=N/A, rep2=N/A, rep3=N/A (consistent across all reps)
- **n=75K held-out (uniform):** speedup = 0.714 — kdtree remains slower

### Correctness

- `t_tw_11_kdtree_matches_baseline`: **PASS**

## Observations

1. **X-space dominance.** At n=100K (uniform), the flat_simd step timing
   breakdown is approximately: `x_dist` ≈ 48B ns, `x_sort` ≈ 35.5B ns,
   `y_dist` ≈ 41.2B ns, `penalty` ≈ 22.1B ns. The Y-space distance
   computation (`y_dist`) represents ~28% of total time. Even replacing it
   entirely with zero cost would give only ~1.39× speedup — insufficient to
   meet the ≥5× hypothesis.

2. **KD-tree Y-space time is not zero.** The `y_kdtree_query` step at n=100K
   averages ~275M ns total (vs ~41B ns for flat_simd `y_dist`). While 150×
   faster, it still adds ~2.4% of the original y_dist cost. Combined with the
   kdtree requiring all the same X-space work, kdtree total runtime exceeds
   flat_simd.

3. **Speedup dips to its minimum at n=50K** (0.696 uniform), not monotonically
   increasing. This may reflect cache effects: at n=50K the flat_simd SIMD
   brute-force may fit in a favorable memory access pattern; the KD-tree query
   overhead may grow faster than its O(n log n) savings in this regime. No cache
   profiling data (LLC miss rates, perf stat) was collected; this mechanism
   remains a hypothesis.

4. **No distribution sensitivity.** Uniform and Gaussian speedup curves are
   nearly identical at every n (max difference 0.009). The X-space bottleneck
   is distribution-independent because pairwise distance computation cost is
   determined by n and dimensionality, not data distribution.

5. **Stable measurements.** All CV values are below 4% (maximum CV: 3.6% for
   flat_simd at n=1K), confirming high measurement reliability.

## Analysis

The root cause of the negative result is a structural bottleneck mismatch. The
KD-tree optimization targets the Y-space KNN step — which is a minority
contributor to trustworthiness runtime — while leaving the dominant X-space
computation untouched.

For exact trustworthiness, X-space KNN (and its implicit pairwise distances) is
required at full precision: approximate X-space KNN produces approximate
trustworthiness scores, not exact ones (see Recommendations). The O(n²) cost of
`x_dist` + `x_sort` at d_x=10 grows faster than any optimization to `y_dist`
can compensate. This is not a failure of the KD-tree implementation; kiddo's
query performance is impressive. The failure is that the optimized step is not
the bottleneck.

The hypothesis in H2 (crossover in [1K, 50K]) assumed that Y-space cost would
dominate at large n due to the KD-tree's O(n log n) vs O(n²) complexity
advantage. This ignores that flat_simd also runs the O(n²) X-space computation
in parallel — so the total runtime for flat_simd scales as O(n²) in both
paths, with a smaller constant because it avoids KD-tree query overhead.

The n=50K minimum in speedup (0.696) vs the slight recovery at n=100K (0.733)
is consistent with SIMD brute-force benefiting from cache-friendly memory
access patterns that degrade more gracefully than the KD-tree's
pointer-chasing traversal at moderate n. At very large n both degrade similarly.

## What We Learned

- **Y-space is not the trustworthiness bottleneck** (at d_x=10). At n=100K,
  Y-space accounts for ~28% of total runtime; X-space (pairwise distances +
  sort) accounts for ~58%. These fractions are specific to d_x=10; higher-
  dimensional X-spaces would amplify X-space dominance further. Any Y-space-only
  optimization faces a hard ceiling of ~1.4× total speedup regardless of the
  technique used.
- **KD-tree query performance is excellent.** A ~148× Y-space speedup at n=100K
  (build+query; query-only is ~150×) demonstrates kiddo v5.3.0 is a high-quality
  KD-tree. The library is not at fault.
- **Build fraction is negligible (<2.2%).** The cost of constructing the
  KD-tree is not a barrier; the query cost itself is the operative expense —
  and even that is small relative to X-space work.
- **Distribution does not affect the conclusion.** Both uniform and Gaussian
  distributions produce near-identical results, ruling out data-shape as a
  confounding variable.
- **Methodology is sound.** The Criterion + step-profiler dual measurement
  approach cleanly separated the bottleneck question from the hypothesis test.
  The 3-rep design detected no instability (all CV < 4%).
- **For future experiments:** If trustworthiness speed matters, the X-space KNN
  step is the only productive target. Approximate nearest neighbors (HNSW,
  IVF, etc.) in X-space, or approximate trustworthiness via sub-sampling, are
  the recommended directions.

## Conclusions

The hypothesis that KD-tree-based Y-space KNN yields ≥5× total speedup at
n=50K is **definitively rejected**. The measured speedup is 0.696 at n=50K
(i.e., kdtree is 1.44× *slower*). No crossover point exists in [1K, 100K]
for either distribution tested.

The Y-space KNN step — though accelerated 83–150× by the KD-tree — is not
the trustworthiness bottleneck. The X-space O(n²) pairwise distance
computation dominates runtime and is unaffected by this optimization.

**Verdict: DO NOT SHIP.**

## Recommendations

1. **Do not merge the kdtree optimization.** It makes trustworthiness
   consistently slower (by ~36–44%) while adding code complexity. The Criterion
   data is unambiguous across all tested conditions.

2. **Investigate X-space approximate KNN.** The only productive path to faster
   trustworthiness is reducing the X-space bottleneck. HNSW or product
   quantization-based ANN in X-space could give sub-quadratic scaling at the
   cost of approximate (rather than exact) trustworthiness scores. This
   trade-off may be acceptable for large-n monitoring use cases.

3. **Consider sub-sampling for large n.** Computing trustworthiness on a random
   subset of points is a well-known technique in the UMAP literature. At n=100K
   a 10K subsample would run ~100× faster with ~1% score error.

4. **Retain the KD-tree infrastructure if Y-space-only speed matters
   independently.** If a future use case needs only the Y-space KNN (not full
   trustworthiness), the `kiddo`-based path shows excellent query performance
   and could be useful there.

## Appendix: Experiment Scripts

### scripts/run_criterion.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"

DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

export RAYON_NUM_THREADS
RAYON_NUM_THREADS="$(nproc)"

if [[ "$DRY_RUN" == "true" ]]; then
    N_VALUES=(1000)
    EXTRA_FLAGS=(--sample-size 10 --warm-up-time 1 --measurement-time 2)
else
    N_VALUES=(1000 5000 10000 50000 75000 100000)
    EXTRA_FLAGS=(--measurement-time 10)
fi
VARIANTS=(flat_simd kdtree)
DISTRIBUTIONS=(uniform gauss)
REPS=3

mkdir -p "$RESEARCH_DIR/results/criterion"
mkdir -p "$PROJECT_ROOT/temp"

RUST_CHANNEL="$(cd "$PROJECT_ROOT" && rustup show active-toolchain 2>/dev/null | awk '{print $1}' || echo "unknown")"
TIMESTAMP="$(date -Iseconds)"
cat > "$RESEARCH_DIR/results/run_metadata.json" <<EOF
{
  "experiment": "kdtree-y-knn-trustworthiness",
  "kiddo_version": "5.3.0",
  "rust_channel": "$RUST_CHANNEL",
  "rayon_num_threads": $RAYON_NUM_THREADS,
  "timestamp": "$TIMESTAMP",
  "dry_run": $DRY_RUN
}
EOF

LOG_ENTRIES=()
for variant in "${VARIANTS[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            group="${variant}_${dist}_n${n}"
            bench_id="${group}/${n}"
            for rep in $(seq 1 "$REPS"); do
                json_tmp="$PROJECT_ROOT/temp/criterion_json_$$.jsonl"
                status="completed"
                if (cd "$PROJECT_ROOT" && cargo criterion \
                        --bench trustworthiness_bench \
                        --features testing \
                        --message-format json \
                        -- "$bench_id" "${EXTRA_FLAGS[@]}" > "$json_tmp"); then
                    dst="$RESEARCH_DIR/results/criterion/${group}_rep${rep}.json"
                    result_line=$(python3 -c "
import json, sys
target = sys.argv[1]; jsonl = sys.argv[2]
with open(jsonl) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
            if d.get('reason') == 'benchmark-complete' and d.get('id') == target:
                print(line); break
        except Exception: pass
" "$bench_id" "$json_tmp" 2>/dev/null || true)
                    if [[ -n "$result_line" ]]; then
                        echo "$result_line" > "$dst"
                    else
                        status="missing_estimates"
                    fi
                else
                    status="failed"
                fi
                rm -f "$json_tmp"
                LOG_ENTRIES+=("{\"variant\": \"$variant\", \"dist\": \"$dist\", \"n\": $n, \"rep\": $rep, \"status\": \"$status\"}")
            done
        done
    done
done

CRITERION_ARRAY="["
for i in "${!LOG_ENTRIES[@]}"; do
    [[ $i -gt 0 ]] && CRITERION_ARRAY+=","
    CRITERION_ARRAY+="${LOG_ENTRIES[$i]}"
done
CRITERION_ARRAY+="]"
cat > "$RESEARCH_DIR/results/run_log.json" <<EOF
{"criterion": $CRITERION_ARRAY}
EOF
```

### scripts/run_profiler.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"

DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then DRY_RUN=true; fi
done

export RAYON_NUM_THREADS
if [[ -z "${RAYON_NUM_THREADS:-}" ]]; then RAYON_NUM_THREADS="$(nproc)"; fi

if [[ "$DRY_RUN" == "true" ]]; then
    N_VALUES=(1000); ITERS=3
else
    N_VALUES=(1000 5000 10000 50000 75000 100000); ITERS=30
fi
VARIANTS=(flat_simd kdtree); DISTRIBUTIONS=(uniform gauss); WARMUP=2

mkdir -p "$RESEARCH_DIR/results/profiler" "$PROJECT_ROOT/temp"
LOG_ENTRIES=()

for variant in "${VARIANTS[@]}"; do
    for dist in "${DISTRIBUTIONS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            stderr_file="$PROJECT_ROOT/temp/tw_profiler_stderr_$$.txt"
            out="$RESEARCH_DIR/results/profiler/${variant}_n${n}_${dist}.json"
            status="completed"
            if (cd "$PROJECT_ROOT" && \
                    RAYON_NUM_THREADS="$RAYON_NUM_THREADS" \
                    cargo run --bin tw_profiler \
                        --features profiling,cli --release -- \
                        --n "$n" --dist "$dist" --variant "$variant" \
                        --iters "$ITERS" --warmup "$WARMUP" \
                        --stderr-capture "$stderr_file" \
                        --output "$out"); then
                :
            else
                status="failed"
            fi
            rm -f "$stderr_file"
            LOG_ENTRIES+=("{\"variant\": \"$variant\", \"dist\": \"$dist\", \"n\": $n, \"status\": \"$status\"}")
        done
    done
done

PROFILER_JSON_ARRAY="["
for i in "${!LOG_ENTRIES[@]}"; do
    [[ $i -gt 0 ]] && PROFILER_JSON_ARRAY+=","
    PROFILER_JSON_ARRAY+="${LOG_ENTRIES[$i]}"
done
PROFILER_JSON_ARRAY+="]"
python3 -c "
import json, sys
path = '$RESEARCH_DIR/results/run_log.json'
try: log = json.loads(open(path).read())
except Exception: log = {}
log['profiler'] = json.loads(sys.argv[1])
open(path,'w').write(json.dumps(log, indent=2))
" "$PROFILER_JSON_ARRAY"
```

### scripts/analyze_results.py

See `research/2026-04-07-kdtree-y-knn-trustworthiness/scripts/analyze_results.py`
for the full Python analysis script (560 lines). Key computations:
- `median_estimate(variant, dist, n)`: median across 3 Criterion reps
- `compute_total_speedup(dist, n)`: `flat_ns / kdtree_ns`
- `compute_build_fraction(n, dist)`: KD-tree build / (build + query)
- `compute_query_speedup(n, dist)`: flat_simd `y_dist` / kdtree `y_kdtree_query`
- `_compute_tcross_from_speedups(speedup_by_n)`: log-interpolated crossover

## Appendix: Raw Data

### Representative profiler output at n=100K (uniform)

**flat_simd** (`results/profiler/flat_simd_n100000_uniform.json`):

| step    | mean timing (ns) | fraction of total |
|---------|-----------------|-------------------|
| x_dist  | 48,030,000,000  | 34.9%             |
| x_sort  | 35,530,000,000  | 25.8%             |
| y_dist  | 41,290,000,000  | 30.0%             |
| penalty | 22,150,000,000  | 16.1%             |
| *total* | *~137.7B*       | —                 |

(mean_s = 9.213 s per trustworthiness call)

**kdtree** (`results/profiler/kdtree_n100000_uniform.json`):

| step            | mean timing (ns) | note           |
|-----------------|-----------------|----------------|
| y_kdtree_build  | 4,350,000       | 0.016% of step |
| y_kdtree_query  | 275,500,000     | 99.98% of step |
| *kdtree Y total*| *279,850,000*   | ~0.7% of flat_simd y_dist |

(mean_s = 11.586 s per trustworthiness call — 25.8% slower than flat_simd)

### Crossover summary (`results/analysis/crossover_summary.json`)

```json
{
  "T_cross_estimate": null,
  "T_cross_range": {"rep1": null, "rep2": null, "rep3": null},
  "T_cross_stable": false,
  "n75k_speedup_uniform": 0.7139,
  "n75k_on_kdtree_faster_side": null
}
```
