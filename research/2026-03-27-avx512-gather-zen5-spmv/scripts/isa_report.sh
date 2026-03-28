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
