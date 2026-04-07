# Revision Guidance — tw-perf-scaling

**Source dashboard:** evaluation_dashboard_tw-perf-scaling_2026-04-04_172000.md
**Source plan:** experiment_plan_tw_perf_scaling_2026-04-04_170000.md
**Generated:** 2026-04-04 17:25:00
**Resolution:** revised — all 4 stop-trigger findings are ADDRESSABLE

---

## Required Fixes

### Fix 1 — Extend per-variant n=100K validation (STOP-1)

**Finding:** Individual variants H2 (thread_local), H3 (avx2_kernel), H4 (avx-512) can clear the ≥1.5× GO criterion using Criterion benchmarks capped at n=50K. The production target is n=100K–250K. Only H6 has an explicit n=100K check.

**Required change — `run_profiling.sh`:**

Add `tw_profiler` invocations for each individual variant at n=100K, mirroring the existing H6 baseline structure:

```bash
# Run each variant at n=100K via tw_profiler (mirrors H6 treatment)
for VARIANT in thread_local partial_rank avx2_kernel; do
  $BIN \
    --x "$RESEARCH_DIR/data/gaussian/gaussian_n100000_x.npy" \
    --y "$RESEARCH_DIR/data/gaussian/gaussian_n100000_y.npy" \
    --k 15 --iters 5 --warmup 2 \
    --variant "$VARIANT" \
    --output "$RESEARCH_DIR/results/step_timing/gaussian_n100000_${VARIANT}.json"
done
```

(Exact CLI flag for variant selection depends on `tw_profiler` implementation — the binary spec may need a `--variant` flag or separate binaries per variant.)

**Required change — per-variant GO criteria:**

Add to each of H2, H3, H4 success criteria:

> Requires ≥1.5× wall-clock speedup at n=100K (tw_profiler, ≥3 warm iterations) in addition to Criterion CI gate at n≤50K.

**Required change — controlled variables table:**

Add row:
| Per-variant n=100K validation | Required for H2, H3, H4 GO | Prevents Goodhart exploitation at sub-production scale |

---

### Fix 2 — Align H3 hypothesis with measurement method (STOP-2)

**Finding:** H3's quantitative claim ("< 2× over auto-vectorized baseline at d=10 f64") is resolved via binary asm inspection ("If AVX2 present → NO-GO"), not by measuring the ratio. The measurement and the conclusion are about different things.

**Choose one of two options:**

**Option A — Redefine H3 as binary detection (simpler):**

Change H3 from:
> "Manual AVX2 for the X-distance inner loop yields < 2× over the current auto-vectorized baseline at d=10 f64 (confirmed by `cargo asm` + benchmark)."

To:
> "The current `trustworthiness()` inner loop is auto-vectorized to AVX2 by the Rust compiler at `target-cpu=native` (confirmed by `cargo asm` instruction inspection). If auto-vectorization is confirmed, manual AVX2 intrinsics provide negligible additional benefit and should not be implemented."

Also remove the `< 2×` threshold from the Analysis Plan H3 decision rule; replace with: "If ≥1 AVX2/FMA instruction (`vmovupd`, `vfmadd`, `ymm`) in the distance inner loop → H3 confirmed (auto-vectorized) → NO-GO for manual SIMD."

**Option B — Add a benchmark that actually measures the ratio (rigorous):**

Implement `bench_avx2_kernel` (already listed as a Phase 2b placeholder) as a Criterion benchmark group comparing the manual AVX2 variant against the auto-vectorized baseline at d=10 f64, n ∈ {1K, 5K, 10K, 50K}. Use the measured speedup ratio as the H3 decision threshold: if ratio < 2× → H3 supported (manual SIMD not worth the maintenance cost).

Update Analysis Plan H3 rule: "If `bench_avx2_kernel` speedup ratio at n=50K < 2× → H3 supported → NO-GO for manual SIMD." Remove `cargo asm` as the primary decision evidence for H3 (retain as supplementary).

**Recommendation:** Option A is simpler and more honest — the original design intent was always "check if auto-vectorization is working," not "measure the manual SIMD ratio." Option B is more rigorous if the plan author wants an actual quantitative bound.

---

### Fix 3 — Establish reproducible MERFISH provenance (STOP-3)

**Finding:** `temp/merfish_100k/merfish_100k_expression.npz` is gitignored with no URL, DOI, or download script. H5 PASS/FAIL cannot be reproduced independently.

**Choose one of two options:**

**Option A — Commit derived artifact (recommended):**

After generating `merfish_n10k_x.npy` and `merfish_n10k_y.npy` via `prepare_merfish.py` (10K×50 f64, ~4MB total), commit them to `tests/fixtures/merfish/`:

```
tests/fixtures/merfish/
├── merfish_n10k_x.npy     # (10000, 50) f64 — PCA-reduced MERFISH expression
└── merfish_n10k_y.npy     # (10000, 2)  f64 — spatial coordinates
```

Update `prepare_merfish.py` to check for the committed fixture before attempting to derive from `temp/`:

```python
FIXTURE_X = "tests/fixtures/merfish/merfish_n10k_x.npy"
FIXTURE_Y = "tests/fixtures/merfish/merfish_n10k_y.npy"
if os.path.exists(FIXTURE_X) and os.path.exists(FIXTURE_Y):
    print("Using committed fixture — skipping derivation from temp/")
    # Copy to research/data/merfish/ for experiment use
    ...
```

Update the data table: change "Source" for merfish rows from `temp/...` to `tests/fixtures/merfish/` with a note that the original was derived from Allen Brain Cell Atlas (Mouse Brain MERFISH, 2023).

**Option B — Add documented fetch step:**

Add to `prepare_merfish.py` a fetch block with a stable URL/DOI:

```python
MERFISH_URL = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-raw.h5ad"
MERFISH_DOI = "10.xxxxx/xxxxx"  # fill in actual DOI from Allen Brain Cell Atlas
```

Include `print(f"Download URL: {MERFISH_URL}")` and a hash check of the downloaded file.

Update the data table Source column with the URL/DOI.

---

### Fix 4 — Fix non-deterministic H5 confirmatory gate (STOP-4)

**Finding:** `run_h5_confirmatory.sh` invokes `subsampling_sweep.py --n-trials 5` without `--seed`. The sealed `h5_confirmatory_result.json` is non-reproducible; re-running produces a different `delta_max`.

**Required change — `run_h5_confirmatory.sh`:**

Change:
```bash
python scripts/subsampling_sweep.py \
  --x data/merfish/merfish_n10k_x.npy \
  --y data/merfish/merfish_n10k_y.npy \
  --m 5000 --n-trials 5 \
  --output results/subsampling/h5_confirmatory_result.json
```

To:
```bash
python scripts/subsampling_sweep.py \
  --x data/merfish/merfish_n10k_x.npy \
  --y data/merfish/merfish_n10k_y.npy \
  --m 5000 --n-trials 5 --seed 42 \
  --output results/subsampling/h5_confirmatory_result.json
```

**Required change — controlled variables table:**

Add row:
| H5 subsampling seed | 42 | Reproducibility of sealed GO gate |

**Required change — `subsampling_sweep.py`:**

If `--seed` is not already accepted, add:
```python
parser.add_argument("--seed", type=int, default=None,
                    help="RNG seed for row subsampling reproducibility")
```
And pass it to `np.random.RandomState(args.seed)` used for row index selection.

---

## Design Questions for Human Review

*(None — all findings were ADDRESSABLE with mechanical fixes.)*

---

## Structural Findings (for context)

*(None — no findings classified as STRUCTURAL.)*
