#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS="research/2026-03-27-simd-spmv-isa-baseline/results"
mkdir -p "$RESULTS"

# ── Environment capture ──────────────────────────────────────────────────────
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "rustc: $(rustc --version)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
} > "$RESULTS/env.txt"
echo "Environment captured to $RESULTS/env.txt"

# ── opt-level=2 assembly listing ─────────────────────────────────────────────
echo "=== cargo asm opt-level=2 ==="
RUSTFLAGS="-C opt-level=2" \
  cargo asm --features testing --intel \
    "spectral_init::operator::spmv_csr" \
    > "$RESULTS/asm_opt2.txt" 2>&1 || true
echo "ASM (opt-level=2) written to $RESULTS/asm_opt2.txt"

# ── opt-level=2 vectorization remarks ────────────────────────────────────────
echo "=== vectorization remarks opt-level=2 ==="
RUSTFLAGS="-C opt-level=2 -C remark=loop-vectorize" \
  cargo rustc --features testing --lib -- \
    > "$RESULTS/remarks_opt2.txt" 2>&1 || true
echo "Remarks (opt-level=2) written to $RESULTS/remarks_opt2.txt"

# ── opt-level=3 assembly listing ─────────────────────────────────────────────
echo "=== cargo asm opt-level=3 ==="
RUSTFLAGS="-C opt-level=3" \
  cargo asm --features testing --intel \
    "spectral_init::operator::spmv_csr" \
    > "$RESULTS/asm_opt3.txt" 2>&1 || true
echo "ASM (opt-level=3) written to $RESULTS/asm_opt3.txt"

# ── opt-level=3 vectorization remarks ────────────────────────────────────────
echo "=== vectorization remarks opt-level=3 ==="
RUSTFLAGS="-C opt-level=3 -C remark=loop-vectorize" \
  cargo rustc --features testing --lib -- \
    > "$RESULTS/remarks_opt3.txt" 2>&1 || true
echo "Remarks (opt-level=3) written to $RESULTS/remarks_opt3.txt"

echo "=== asm_audit.sh complete. Results in $RESULTS/ ==="
