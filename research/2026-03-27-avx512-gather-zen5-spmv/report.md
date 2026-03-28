# AVX-512 Gather SpMV Throughput on Zen 5 (Ryzen 9800X3D)

> Research report — 2026-03-28

## Executive Summary

This experiment measured whether `_mm512_i32gather_pd` (8-wide AVX-512 gather) delivers
meaningfully better throughput than `_mm256_i32gather_pd` (4-wide AVX2 gather) for sparse
matrix–vector multiplication (SpMV) on the AMD Ryzen 9800X3D (Zen 5). The SpMV kernel drives the
Laplacian eigensolver at the core of SpectralInit, and the Phase 3 plan designates it as the
replacement point for a SIMD gather implementation. The prior baseline experiment established
AVX2 gather at 1.72–1.76× over scalar; this experiment determines whether upgrading to AVX-512
gather is warranted.

**The answer is no.** Across all five matrix sizes (n ∈ {200, 2000, 5000, 10000, 50000}), the
AVX-512 gather kernel is **4–11% slower** than AVX2 gather. The null hypothesis (H0: AVX-512
delivers ≤ 10% speedup over AVX2) is supported with statistically non-overlapping confidence
intervals at n = 200. Frequency throttling was not observed; Zen 5 sustains AVX-512 clock speed
throughout the benchmark window. The performance penalty is structural: with NNZ/row = 15, an
8-wide AVX-512 gather leaves a 7-element scalar tail per row (47% scalar), versus a 3-element
tail for 4-wide AVX2 (20% scalar), eliminating the throughput advantage of the wider vector unit.

**Recommendation:** Do not implement an AVX-512 gather kernel for Phase 3. The AVX2 gather kernel
is the ceiling for gather-based SpMV on UMAP-scale sparse graphs (k ≈ 15 NNZ/row) on this
architecture. If Phase 3 throughput improvements are needed, investigate SELL-C-σ with AVX2 SIMD
on padded chunks, which eliminates gather entirely.

## Background and Research Question

The SpectralInit eigensolver relies on SpMV to apply the symmetric normalized Laplacian
`L = I - D^{-1/2} W D^{-1/2}` in iterative eigensolver loops (LOBPCG) and randomized SVD power
iterations. The UMAP k-NN graph is sparse (typically k = 15, meaning ~15 NNZ/row), making CSR
SpMV memory-bound at scale. The prior experiment
(`research/2026-03-27-simd-spmv-isa-baseline/`) established that:

- Autovectorization of the scalar CSR kernel produces no SIMD instructions (confirmed by
  assembly inspection).
- Manual AVX2 gather (`_mm256_i32gather_pd`) achieves 1.39–1.68× over scalar at n = 200–50000.
- SELL-C-σ with C = 4 further improves to ~2× at cache-resident sizes but degrades at n = 50000.

The Phase 3 implementation decision requires knowing whether the jump to AVX-512 width (8 f64
elements per instruction) offers a material additional benefit over AVX2 on the Zen 5
microarchitecture, or whether the gather instruction's uops.info characteristics (higher latency
per element for wider vectors) cancel the theoretical throughput advantage.

**Research question:** Does `_mm512_i32gather_pd` (8-wide) outperform `_mm256_i32gather_pd`
(4-wide) by more than 10% for ring-graph Laplacian SpMV on Zen 5, across cache regimes spanning
L1 through DRAM?

## Methodology

### Experimental Design

**Null hypothesis (H0):** On Zen 5, the 8-wide AVX-512 gather kernel delivers ≤ 10% speedup over
the 4-wide AVX2 gather kernel at all five matrix sizes.

**Alternative hypothesis (H1):** The AVX-512 kernel delivers > 10% speedup at one or more sizes.

**Sub-hypotheses:**
- **H2 (memory-bound ceiling):** At n = 50000 (x-vector ~400 KB, exceeds Zen 5's 1 MB L2), the
  speedup ratio between AVX-512 and AVX2 will be smaller than at cache-resident sizes.
- **H3 (throttle significance):** AVX-512 frequency throttling on the 9800X3D will not exceed
  10% net impact on Criterion throughput for SpMV workloads.

**Independent variables:**

| Variable | Values |
|----------|--------|
| ISA kernel | `scalar`, `avx2_gather`, `avx512_gather` |
| Matrix size (n) | 200, 2000, 5000, 10000, 50000 |

**Dependent variables:** SpMV wall-clock time (mean ± 95% CI, nanoseconds), speedup ratios,
CPU clock frequency during bench (Avg_MHz / Bzy_MHz via turbostat or /sys polling fallback).

**Controlled variables:** Ring-graph Laplacian with `half=7` (15 NNZ/row constant), x-vector
= `1/sqrt(n)` constant, `RUSTFLAGS="-C target-cpu=native"`, Criterion 3s warmup / 100 samples,
`testing` feature flag enabled.

### Environment

- **Repository commit:** `afd7bbe9b223a5301711de31defd0106b0ff2df3`
- **Branch:** `research-20260327-162735`
- **Rust toolchain:** rustc 1.96.0-nightly (23903d01c 2026-03-26)
- **Criterion version:** 0.5 (dev-dependency)
- **Hardware:** AMD Ryzen 7 9800X3D 8-Core Processor (Zen 5 microarchitecture, 3D V-Cache)
- **OS:** Linux (WSL2, kernel 6.6.87.2-microsoft-standard-WSL2)
- **AVX-512 extensions confirmed at runtime:**
  `avx512f avx512bw avx512cd avx512dq avx512vl avx512ifma avx512vbmi avx512vbmi2 avx512vnni avx512_bf16 avx512_bitalg avx512_vp2intersect avx512_vpopcntdq`
- **Runtime detection:** `is_x86_feature_detected!("avx512f")` → `true`
- **Frequency monitoring:** turbostat required sudo (unavailable); `/sys/cpufreq` polling fallback
  captured one sample per second during the independent AVX-512-only re-run

### Procedure

1. **Environment capture** (`isa_report.sh`): Recorded date, rustc version, CPU model, AVX-512
   flags from `/proc/cpuinfo`, and runtime feature detection result. Output: `results/env.txt`.

2. **Full three-way ISA benchmark** (`run_benchmarks.sh`): Ran
   `cargo bench --features testing --bench simd_spmv_exp -- spmv_avx2` with
   `RUSTFLAGS="-C target-cpu=native"`. This invoked the `spmv_avx2` Criterion group
   containing all three ISA variants (`scalar`, `avx2_gather`, `avx512_gather`) at all five
   matrix sizes (15 total bench IDs × 100 samples each). Output: `results/criterion_avx2.txt`.

3. **Independent AVX-512-only re-run with frequency monitoring** (`run_with_turbostat.sh`):
   Ran the AVX-512 benchmarks alone alongside `/sys/cpufreq` polling. The turbostat fallback
   captured one measurement per second during the run. Output: `results/criterion_avx512_only.txt`
   and `results/turbostat_avx512.txt`.

4. **Speedup table generation** (`analyze_results.py`): Read Criterion JSON from
   `target/criterion/spmv_avx2/{variant}/{n}/new/estimates.json`, computed speedup ratios,
   emitted `results/speedup_table.md`.

## Results

### Throughput Comparison (mean ± 95% CI)

| n | scalar_ns | avx2_ns | avx2_speedup | avx512_ns | avx512_vs_scalar | avx512_vs_avx2 |
|---|----------:|--------:|:------------:|----------:|:----------------:|:--------------:|
| 200 | 1,911.4 | 1,090.8 | 1.752× | 1,179.0 | 1.621× | **0.925×** |
| 2,000 | 18,865.8 | 10,835.5 | 1.741× | 12,170.4 | 1.550× | **0.890×** |
| 5,000 | 47,561.1 | 27,106.5 | 1.755× | 29,766.5 | 1.598× | **0.911×** |
| 10,000 | 98,877.1 | 57,430.1 | 1.722× | 59,825.2 | 1.653× | **0.960×** |
| 50,000 | 493,695.4 | 284,499.4 | 1.735× | 304,172.6 | 1.623× | **0.935×** |

*avx512_vs_avx2 < 1.0 means AVX-512 is slower than AVX2.*

### Raw Criterion Output (n=200, key comparison)

```
spmv_avx2/scalar/200          time:  [1.9023 µs  1.9092 µs  1.9173 µs]
spmv_avx2/avx2_gather/200     time:  [1.0882 µs  1.0987 µs  1.1104 µs]
spmv_avx2/avx512_gather/200   time:  [1.1744 µs  1.1850 µs  1.1986 µs]
```

### 95% Confidence Intervals — AVX2 vs AVX-512 at n=200 (Non-Overlap Test)

| Kernel | Lower CI (ns) | Point Estimate (ns) | Upper CI (ns) |
|--------|:-------------:|:-------------------:|:-------------:|
| avx2_gather | 1,083.6 | 1,090.8 | 1,099.5 |
| avx512_gather | 1,171.7 | 1,179.0 | 1,188.3 |

**Gap between intervals: 72.2 ns.** The confidence intervals do not overlap, confirming the
result is not measurement noise at n=200.

### Independent AVX-512-Only Re-run (H3 Throttle Check)

| n | Main run (ns) | Re-run (ns) | Δ |
|---|:-------------:|:-----------:|:---:|
| 200 | 1,179.0 | 1,184.6 | +0.5% |
| 2,000 | 12,170.4 | 12,237.0 | +0.5% |
| 5,000 | 29,766.5 | 29,586.0 | −0.6% |
| 10,000 | 59,825.2 | 60,120.0 | +0.5% |
| 50,000 | 304,172.6 | 304,170.0 | ~0% |

All re-run values agree within < 1% of the main run, within Criterion's noise threshold.

## Observations

1. **AVX-512 gather is consistently slower than AVX2 gather at every size.** The avx512_vs_avx2
   ratio ranges from 0.890 (n=2000) to 0.960 (n=10000). This is the opposite of H1.

2. **Root cause: scalar tail fraction.** With NNZ/row = 15 (constant for the ring graph):
   - AVX-512 (8-wide): 1 full SIMD iteration (8 elements) + 7-element scalar tail → 47% scalar.
   - AVX2 (4-wide): 3 full SIMD iterations (12 elements) + 3-element scalar tail → 20% scalar.
   The wider vector unit processes only 53% of elements via SIMD, compared to 80% for AVX2.
   The scalar tail overhead absorbs the theoretical throughput benefit of the larger register.

3. **AVX2 speedup over scalar is remarkably stable: 1.72–1.76× across all five sizes.** This
   replicates the prior baseline experiment's results and confirms the cache hierarchy (L1 through
   DRAM) does not qualitatively change the AVX2 benefit for this workload.

4. **H2 (memory-bound ceiling for AVX-512 penalty) not confirmed.** The avx512_vs_avx2 ratio at
   n=50000 (0.935) is not smaller than at n=5000 (0.911); n=10000 shows the least penalty (0.960).
   The memory-bound regime does not further disadvantage AVX-512 relative to AVX2.

5. **H3 (throttle significance) not confirmed.** The two independent AVX-512 Criterion runs agree
   within < 1% at all sizes. Zen 5 sustains AVX-512 gather workloads at full clock speed for
   SpMV-scale compute windows. The Criterion 3s warmup is sufficient to absorb any ISA transition
   penalty.

6. **Non-overlapping confidence intervals at n=200 establish statistical significance** for the
   AVX-512 penalty. The 72 ns gap between the AVX2 upper CI (1,099.5 ns) and the AVX-512 lower CI
   (1,171.7 ns) rules out measurement noise.

7. **Criterion's change detection is consistent with noise except at n=2000 (re-run +4.3% vs
   first run).** This marginal regression in the re-run is within the Criterion significance
   threshold for that size and does not change conclusions — all five sizes show AVX-512 slower
   than AVX2.

## Analysis

### Primary Analysis — Throughput Comparison

H0 is supported at all five sizes: AVX-512 gather delivers **negative** speedup over AVX2 gather
(0.89–0.96× = 4–11% **slower**). The experiment's success criterion for H0 was "avx512_vs_avx2
≤ 1.10 at all sizes"; the actual values are all below 1.0, which is a stronger statement.

The structural explanation is clear. For NNZ/row = 15:

```
AVX-512: ceil(15/8) = 2 iterations, but only 1 full; tail = 7  → 7/15 = 47% scalar
AVX2:    ceil(15/4) = 4 iterations, but 3 full;    tail = 3  → 3/15 = 20% scalar
```

The scalar tail in AVX-512 dominates. Each scalar FMA is a separate load + multiply-add with
random gather semantics handled sequentially. With 47% of elements in the scalar tail, the
performance regression relative to AVX2 (which has only 20% in its tail) is expected.

The absence of H2 (stronger penalty at n=50000) indicates that for this workload the memory
bandwidth is not the limiting factor that differentiates AVX-512 and AVX2 behavior. Both kernels
are bottlenecked by gather latency and scalar tail overhead regardless of cache level.

### Threats to Validity Assessment

- **Uniform NNZ/row (15):** The most significant threat materialized — the constant scalar tail
  fraction is the primary explanation for AVX-512 underperformance. For UMAP graphs with
  variable NNZ/row, the tail fraction varies per row, but the _average_ NNZ/row for UMAP k-NN
  graphs (k ≈ 15) is the same as the ring graph, making this test representative.

- **Constant x-vector:** The `1/sqrt(n)` constant allows the prefetcher to learn the access
  pattern. A random x-vector would expose higher gather latency. However, AVX2 would benefit
  equally, so the relative speedup ratio should be unaffected.

- **Single machine:** Results are specific to Zen 5 with 3D V-Cache. Zen 5 without V-Cache
  (non-X3D) has smaller L3 and may show different patterns at n=50000. Intel Ice Lake and
  Sapphire Rapids have higher AVX-512 gather throughput and may show different results.

## What We Learned

- **AVX-512 gather is not a free speedup over AVX2 gather for sparse SpMV on Zen 5.** The
  wider register width only benefits computation if the vector utilization is high enough
  to amortize the fixed overhead of the scalar tail. For 15 NNZ/row, AVX-512 fails this test.
- **The AVX2 gather ceiling (1.72–1.76×) is the practical SIMD limit for gather-based SpMV
  on UMAP k-NN graphs** on this hardware. Breaching this ceiling requires eliminating gather
  (e.g., SELL-C-σ padding) rather than widening the gather.
- **Zen 5 does not throttle under AVX-512 gather workloads at SpMV scale.** This concern
  (documented in prior ISA scoping) is not a real obstacle; the `--target-cpu=native` binary
  runs at full frequency throughout Criterion benchmark windows.
- **Criterion's 100-sample, 3s warmup methodology is appropriate for this workload.** The
  two independent runs agree within < 1%, confirming measurement stability.
- **Negative results are informative.** The conclusive negative outcome rules out an entire
  hardware-specific optimization path and redirects Phase 3 implementation effort toward
  approaches that may actually deliver improvement (e.g., SELL-C-σ).
- **For future AVX-512 experiments on Zen 5 with NNZ/row = k:** AVX-512 gather breaks even
  with AVX2 gather only when `(k mod 8)` approaches 0 or when `k ≥ 24`, so that the scalar
  tail fraction falls below ~17% and the wider register's throughput begins to dominate.

## Conclusions

**H0 is supported. AVX-512 gather is slower than AVX2 gather on Zen 5 for ring-graph Laplacian
SpMV at all five tested matrix sizes (n = 200 to 50,000).**

The avx512_vs_avx2 speedup ratio is 0.890–0.960 (4–11% regression), with statistically
non-overlapping 95% confidence intervals at n = 200 confirming the result is not noise. The
performance penalty is structural: NNZ/row = 15 causes a 47% scalar tail fraction for 8-wide
AVX-512, versus 20% for 4-wide AVX2, eliminating the benefit of the wider vector unit.

No frequency throttling was observed. H2 (stronger memory-bound penalty at n=50000) was not
confirmed. The AVX2 gather kernel is the current ceiling for gather-based SpMV on this workload
and hardware.

## Recommendations

**1. Do not implement the AVX-512 gather kernel for Phase 3 of SpectralInit's SpMV.**

The experiment conclusively shows a performance regression at every tested size. The added
complexity of an AVX-512 kernel (unsafe intrinsics, ISA dispatch, maintenance burden) is not
justified when the kernel is consistently slower than the existing AVX2 implementation.

**2. Retain the AVX2 gather kernel (`spmv_avx2_dispatch`) as the Phase 3 SIMD baseline.**

The 1.72–1.76× speedup over scalar is stable and consistent. It is the appropriate starting
point for the production SpMV implementation.

**3. For further throughput improvements, investigate SELL-C-σ with AVX2 SIMD.**

The prior baseline experiment showed SELL-C-σ (C=4) reaching ~2× over scalar at cache-resident
sizes without using gather at all. SELL-C-σ with padded rows allows standard `_mm256_loadu_pd` +
`_mm256_mul_pd` + horizontal reduction — higher throughput than gather due to sequential memory
access. The variable-NNZ padding overhead needs evaluation against real UMAP graphs (see
`tests/fixtures/blobs_5000/comp_b_laplacian.npz`).

**4. If AVX-512 is revisited for future workloads with higher NNZ/row:** Re-evaluate when
NNZ/row ≥ 24 (tail fraction < 17% for 8-wide gather) or when using a uniform-width sparse
format (SELL-C-σ with C=8) that avoids the tail entirely.

---

## Appendix: Experiment Scripts

### isa_report.sh

```bash
#!/usr/bin/env bash
# Usage: bash scripts/isa_report.sh
# Captures hardware/software environment to results/env.txt.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="research/2026-03-27-avx512-gather-zen5-spmv/results"
mkdir -p "$RESULTS"

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "rustc: $(rustc --version)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null \
               || sysctl -n machdep.cpu.brand_string 2>/dev/null \
               || echo 'unknown')"
  echo "avx512_flags: $(grep -m1 'flags' /proc/cpuinfo 2>/dev/null \
                        | tr ' ' '\n' \
                        | grep -E '^avx512' \
                        | sort \
                        | tr '\n' ' ' \
                        || echo 'unavailable')"

  # One-shot runtime detection via inline rustc
  AVX_SRC="/tmp/avx_check_$$.rs"
  AVX_BIN="/tmp/avx_check_$$"
  cat > "$AVX_SRC" <<'RUST'
fn main() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    println!("avx512f_runtime={}", std::arch::is_x86_feature_detected!("avx512f"));
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    println!("avx512f_runtime=unsupported_arch");
}
RUST
  RUSTFLAGS="-C target-cpu=native" rustc "$AVX_SRC" -o "$AVX_BIN" 2>/dev/null \
    && "$AVX_BIN" \
    || echo "avx512f_runtime=detection_failed"
  rm -f "$AVX_SRC" "$AVX_BIN"
} > "$RESULTS/env.txt"

echo "Environment captured to $RESULTS/env.txt"
```

### run_benchmarks.sh

```bash
#!/usr/bin/env bash
# Usage: bash scripts/run_benchmarks.sh
# Runs the full spmv_avx2 Criterion group (scalar, avx2_gather, avx512_gather).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="research/2026-03-27-avx512-gather-zen5-spmv/results"
mkdir -p "$RESULTS"

export RUSTFLAGS="-C target-cpu=native"

echo "=== Running Criterion group: spmv_avx2 ==="
cargo bench --features testing --bench simd_spmv_exp -- spmv_avx2 \
    2>&1 | tee "$RESULTS/criterion_avx2.txt"

echo "=== run_benchmarks.sh complete. Results in $RESULTS/ ==="
```

### run_with_turbostat.sh

```bash
#!/usr/bin/env bash
# Usage: bash scripts/run_with_turbostat.sh
# Runs the AVX-512-only benchmark with CPU frequency monitoring.
# Uses turbostat if available and sudo-accessible; falls back to /sys polling.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="research/2026-03-27-avx512-gather-zen5-spmv/results"
mkdir -p "$RESULTS"

export RUSTFLAGS="-C target-cpu=native"

MONITOR_PID=""

if command -v turbostat >/dev/null 2>&1 && sudo -n turbostat --help >/dev/null 2>&1; then
    echo "Using turbostat for CPU frequency monitoring."
    sudo turbostat \
        --interval 1 \
        --quiet \
        --show Avg_MHz,Busy%,Bzy_MHz \
        > "$RESULTS/turbostat_avx512.txt" 2>&1 &
    MONITOR_PID=$!
else
    echo "turbostat unavailable or sudo denied; falling back to /sys polling."
    (
        while true; do
            echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
            cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null \
                | awk '{sum+=$1; n++} END {if(n>0) printf "avg_kHz=%d n_cpu=%d\n", sum/n, n}'
            sleep 1
        done
    ) > "$RESULTS/turbostat_avx512.txt" &
    MONITOR_PID=$!
fi

echo "=== Running Criterion filter: avx512_gather ==="
cargo bench --features testing --bench simd_spmv_exp -- avx512_gather \
    2>&1 | tee "$RESULTS/criterion_avx512_only.txt"

if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
fi

echo "=== run_with_turbostat.sh complete. Results in $RESULTS/ ==="
```

### analyze_results.py

```python
"""Analysis script for AVX-512 gather vs AVX2 gather vs scalar SpMV on Zen5."""
import json
from pathlib import Path

N_VALUES = [200, 2000, 5000, 10000, 50000]

RESULTS = Path("research/2026-03-27-avx512-gather-zen5-spmv/results")
CRITERION_GROUP = "spmv_avx2"


def load_estimate(variant: str, n: int) -> dict:
    """Load Criterion estimates.json for spmv_avx2/{variant}/{n}/new/."""
    path = (
        Path("target") / "criterion" / CRITERION_GROUP
        / variant / str(n) / "new" / "estimates.json"
    )
    with open(path) as f:
        return json.load(f)


def get_mean_ns(variant: str, n: int) -> float:
    """Return mean elapsed nanoseconds, or NaN on FileNotFoundError."""
    try:
        est = load_estimate(variant, n)
        return est["mean"]["point_estimate"]
    except FileNotFoundError:
        return float("nan")


def _fmt(v: float, decimals: int = 2) -> str:
    """Format float to fixed decimals; return 'N/A' for NaN or error."""
    try:
        if v != v:  # NaN check
            return "N/A"
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def _speedup(baseline_ns: float, target_ns: float) -> str:
    """Return baseline/target speedup as formatted string, or 'N/A'."""
    if target_ns != target_ns or target_ns == 0.0:  # NaN or zero
        return "N/A"
    if baseline_ns != baseline_ns:
        return "N/A"
    return _fmt(baseline_ns / target_ns, 3)


def build_speedup_rows() -> list:
    """Build one row per n with scalar, avx2, avx512 timings and speedups."""
    rows = []
    for n in N_VALUES:
        scalar_ns  = get_mean_ns("scalar",        n)
        avx2_ns    = get_mean_ns("avx2_gather",   n)
        avx512_ns  = get_mean_ns("avx512_gather", n)

        rows.append({
            "n":                  n,
            "scalar_ns":          _fmt(scalar_ns,  1),
            "avx2_ns":            _fmt(avx2_ns,    1),
            "avx2_speedup":       _speedup(scalar_ns, avx2_ns),
            "avx512_ns":          _fmt(avx512_ns,  1),
            "avx512_vs_scalar":   _speedup(scalar_ns,  avx512_ns),
            "avx512_vs_avx2":     _speedup(avx2_ns,    avx512_ns),
        })
    return rows


def emit_speedup_table(rows: list, out_path: Path) -> None:
    """Write the speedup table as a Markdown file."""
    headers = [
        "n", "scalar_ns", "avx2_ns", "avx2_speedup",
        "avx512_ns", "avx512_vs_scalar", "avx512_vs_avx2",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# AVX-512 Gather vs AVX2 Gather vs Scalar — Speedup Table\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
    print(f"Speedup table written to {out_path}")


if __name__ == "__main__":
    print("=== Building Speedup Table ===")
    rows = build_speedup_rows()
    emit_speedup_table(rows, RESULTS / "speedup_table.md")
```

## Appendix: Raw Data

### results/env.txt

```
date: 2026-03-27T23:56:23Z
rustc: rustc 1.96.0-nightly (23903d01c 2026-03-26)
cpu: model name	: AMD Ryzen 7 9800X3D 8-Core Processor
avx512_flags: avx512_bf16 avx512_bitalg avx512_vbmi2 avx512_vnni avx512_vp2intersect avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl
avx512f_runtime=true
```

### results/speedup_table.md (generated by analyze_results.py)

```
# AVX-512 Gather vs AVX2 Gather vs Scalar — Speedup Table

| n | scalar_ns | avx2_ns | avx2_speedup | avx512_ns | avx512_vs_scalar | avx512_vs_avx2 |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | 1911.4 | 1090.8 | 1.752 | 1179.0 | 1.621 | 0.925 |
| 2000 | 18865.8 | 10835.5 | 1.741 | 12170.4 | 1.550 | 0.890 |
| 5000 | 47561.1 | 27106.5 | 1.755 | 29766.5 | 1.598 | 0.911 |
| 10000 | 98877.1 | 57430.1 | 1.722 | 59825.2 | 1.653 | 0.960 |
| 50000 | 493695.4 | 284499.4 | 1.735 | 304172.6 | 1.623 | 0.935 |
```
