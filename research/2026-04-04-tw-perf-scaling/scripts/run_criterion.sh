#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"

cd "$RESEARCH_DIR"

ASM_ONLY=false
if [ "${1:-}" = "--asm-only" ]; then
  ASM_ONLY=true
fi

echo "============================================"
echo "  tw-perf-scaling: Criterion + ASM Inspection"
echo "============================================"

# Verify cargo-show-asm is installed
if ! cargo asm --version >/dev/null 2>&1; then
  echo "ERROR: cargo-show-asm not installed. Run: cargo install cargo-show-asm"
  exit 1
fi

# ASM inspection (always runs)
echo ""
echo "[ASM] Inspecting trustworthiness for AVX2 instructions..."
mkdir -p results/asm
cargo asm --release --manifest-path "$PROJECT_ROOT/Cargo.toml" \
  spectral_init::metrics::trustworthiness 2>/dev/null \
  | grep -E "(ymm|vfmadd|vmovupd|vsubpd|vmulpd)" \
  > results/asm/trustworthiness_asm_avx2.txt || true

if [ -s results/asm/trustworthiness_asm_avx2.txt ]; then
  echo "H3: AUTO-VECTORIZED — NO-GO for manual AVX2" > results/asm/h3_verdict.txt
else
  echo "H3: NOT AUTO-VECTORIZED — IMPLEMENT manual AVX2 kernel" > results/asm/h3_verdict.txt
fi

echo "  Verdict: $(cat results/asm/h3_verdict.txt)"

if [ "$ASM_ONLY" = true ]; then
  echo ""
  echo "ASM-only mode complete."
  exit 0
fi

# Full mode — AVX2 instruction count parity check
echo ""
echo "[PARITY] Checking AVX2 instruction count parity (clean vs testing)..."

CLEAN_COUNT=$(cargo asm --release --manifest-path "$PROJECT_ROOT/Cargo.toml" \
  spectral_init::metrics::trustworthiness 2>/dev/null | grep -c "ymm" || echo 0)

TESTING_COUNT=$(cargo asm --release --features testing --manifest-path "$PROJECT_ROOT/Cargo.toml" \
  spectral_init::metrics::trustworthiness 2>/dev/null | grep -c "ymm" || echo 0)

echo "  Clean build YMM count: $CLEAN_COUNT"
echo "  Testing build YMM count: $TESTING_COUNT"

if [ "$CLEAN_COUNT" != "$TESTING_COUNT" ]; then
  echo "WARNING: AVX2 YMM instruction count differs between clean ($CLEAN_COUNT) and testing ($TESTING_COUNT) builds"
  echo "Aborting Criterion run — testing feature may affect codegen"
  exit 1
fi
echo "  ✓ Parity check passed"

# Full mode — Criterion benchmarks
echo ""
echo "[CRITERION] Running full benchmark suite..."
mkdir -p results/criterion
cargo criterion --bench trustworthiness_bench --manifest-path "$PROJECT_ROOT/Cargo.toml" \
  --message-format=json > results/criterion/criterion_output.json 2>&1 || true

# Copy Criterion HTML reports if present
if [ -d "$PROJECT_ROOT/target/criterion" ]; then
  cp -r "$PROJECT_ROOT/target/criterion/"* results/criterion/
fi

echo "  ✓ Criterion benchmarks complete"
echo ""
echo "============================================"
echo "  Criterion + ASM inspection complete"
echo "============================================"
