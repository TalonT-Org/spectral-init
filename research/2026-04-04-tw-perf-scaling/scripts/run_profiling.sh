#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$RESEARCH_DIR/../.." && pwd)"

cd "$RESEARCH_DIR"

echo "============================================"
echo "  tw-perf-scaling: Profiling Suite"
echo "============================================"

NS=(1000 5000 10000 25000 50000 100000)
VARIANTS_ALL_N=(baseline combined)
VARIANTS_100K_ONLY=(thread_local partial_rank avx2_kernel)

mkdir -p results/step_timing

# Baseline and combined at all n-sizes
for variant in "${VARIANTS_ALL_N[@]}"; do
  echo ""
  echo "--- Variant: $variant (all n-sizes) ---"
  for n in "${NS[@]}"; do
    echo "  n=$n ..."
    "$PROJECT_ROOT/target/release/tw_profiler" \
      --x "data/gaussian/gaussian_n${n}_x.npy" \
      --y "data/gaussian/gaussian_n${n}_y.npy" \
      --k 15 \
      --warmup 2 --iters 5 \
      --variant "$variant" \
      --output "results/step_timing/gaussian_n${n}_${variant}.json"
  done
done

# thread_local, partial_rank, avx2_kernel at all n-sizes
for variant in "${VARIANTS_100K_ONLY[@]}"; do
  echo ""
  echo "--- Variant: $variant (all n-sizes) ---"
  for n in "${NS[@]}"; do
    echo "  n=$n ..."
    "$PROJECT_ROOT/target/release/tw_profiler" \
      --x "data/gaussian/gaussian_n${n}_x.npy" \
      --y "data/gaussian/gaussian_n${n}_y.npy" \
      --k 15 \
      --warmup 2 --iters 5 \
      --variant "$variant" \
      --output "results/step_timing/gaussian_n${n}_${variant}.json"
  done
done

# AVX-512 conditional (n=100K only)
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
  echo ""
  echo "--- Variant: avx512_kernel (n=100K, AVX-512 detected) ---"
  "$PROJECT_ROOT/target/release/tw_profiler" \
    --x data/gaussian/gaussian_n100000_x.npy \
    --y data/gaussian/gaussian_n100000_y.npy \
    --k 15 --warmup 2 --iters 5 \
    --variant avx512_kernel \
    --output results/step_timing/gaussian_n100000_avx512_kernel.json
else
  echo ""
  echo "AVX-512 not available — skipping avx512_kernel"
fi

echo ""
echo "============================================"
echo "  Profiling complete. Files written:"
echo "============================================"
ls -1 results/step_timing/gaussian_*.json 2>/dev/null || echo "  (none)"
