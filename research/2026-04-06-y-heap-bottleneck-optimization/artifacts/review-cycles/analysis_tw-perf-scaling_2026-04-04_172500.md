# Resolve Design Review — Triage Analysis Report

**Experiment slug:** tw-perf-scaling
**Dashboard:** evaluation_dashboard_tw-perf-scaling_2026-04-04_172000.md
**Plan:** experiment_plan_tw_perf_scaling_2026-04-04_170000.md
**Triage timestamp:** 2026-04-04 17:25:00

---

```
Triage complete (BEFORE any guidance written)
ADDRESSABLE: 4 | STRUCTURAL: 0 | DISCUSS: 0
```

---

## Finding-by-Finding Classification

### STOP-1 — Red-team: Goodhart exploitation (benchmark scale mismatch)

**Verdict: ADDRESSABLE**

**Evidence:** `tw_profiler` already runs n=100K for baseline in Phase 2a (`run_profiling.sh`). H6 already has an explicit n=100K `tw_profiler` requirement. The controlled variables table shows Criterion benchmarks iterate n in {1_000, 5_000, 10_000, 50_000} — capped at 50K. Success criteria reference "≥1.5× Criterion speedup" without pre-registering a specific n, leaving H2/H3/H4 validatable entirely from n≤50K data.

**Fix sketch:** Add `tw_profiler` n=100K runs for H2, H3, and H4 to `run_profiling.sh` (mirroring the existing H6 and baseline invocations). Amend the per-variant GO criteria to require ≥1.5× wall-clock speedup at n=100K in `tw_profiler` output in addition to (or instead of) the Criterion CI gate. No new tooling needed — `tw_profiler --iters` and `--warmup` flags already support this.

---

### STOP-2 — L4 Measurement Alignment: H3 method mismatch

**Verdict: ADDRESSABLE**

**Evidence:** H3 states "< 2× over the current auto-vectorized baseline at d=10 f64 (confirmed by `cargo asm` + benchmark)" but the Analysis Plan resolves H3 via binary asm inspection ("If AVX2 instructions present: mark H3 NO-GO"). The plan also lists `bench_avx2_kernel` as an existing Criterion benchmark group placeholder in Phase 2b, meaning the benchmark infrastructure is already scoped.

**Fix sketch:** Either (1) redefine H3 as "auto-vectorization is present in the current build" and drop the ≤2× ratio claim, updating the Analysis Plan criterion to match; or (2) implement the `bench_avx2_kernel` Criterion benchmark comparing the manual AVX2 variant against the auto-vectorized baseline at d=10 f64 and use the measured ratio as the H3 decision threshold. Both paths require only a one-line hypothesis rewrite or a single benchmark addition — no structural redesign.

---

### STOP-3 — L4 Reproducibility: MERFISH provenance

**Verdict: ADDRESSABLE**

**Evidence:** Plan states `merfish_n10k_x.npy` / `merfish_n10k_y.npy` derives from `temp/merfish_100k/merfish_100k_expression.npz` first 10K rows with PCA reduction to 50 dims (~4MB total); `prepare_merfish.py` outputs to `research/2026-04-04-tw-perf-scaling/data/merfish/`; Allen Brain Cell Atlas MERFISH data is publicly available via a documented portal.

**Fix sketch:** Either (a) commit the derived `merfish_n10k_x.npy` and `merfish_n10k_y.npy` artifacts directly to `tests/fixtures/` since they are ~4MB and contain no raw expression data, or (b) add a documented fetch step to `prepare_merfish.py` with the stable Allen Brain Cell Atlas URL/DOI so any party can reproduce the input. Either path fully resolves the provenance gap with a single script or file addition.

---

### STOP-4 — L3 Variance Protocol: missing seed on confirmatory gate

**Verdict: ADDRESSABLE**

**Evidence:** Controlled variables table already registers "RNG seed | 42 | Reproducibility for synthetic datasets"; `run_h5_confirmatory.sh` specifies `--n-trials 5` but omits `--seed`; `subsampling_sweep.py` accepts or can be extended to accept `--seed`; fix is purely additive with no design implications.

**Fix sketch:** Add `--seed 42` to the `subsampling_sweep.py` invocation in `run_h5_confirmatory.sh`, and add a row "H5 subsampling seed | 42 | Reproducibility of sealed GO gate" to the controlled variables table.

---

## Resolution

All 4 stop-trigger findings are ADDRESSABLE. Resolution = **revised**.
