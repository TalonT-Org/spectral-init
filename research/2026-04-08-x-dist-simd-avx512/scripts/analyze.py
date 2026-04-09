#!/usr/bin/env python3
"""Analyze x-dist SIMD experiment results.

Reads results/*.json and results/*_criterion.txt, computes:
  - x_dist step speedup per variant
  - total wall-clock speedup per variant
  - Amdahl projection vs measured
  - AVX-512 marginal gain over looped AVX2
  - Tiling marginal gain (if present)

Usage (from repo root):
  python research/2026-04-08-x-dist-simd-avx512/scripts/analyze.py
"""

import json
import pathlib
import re
import sys

RESULTS = pathlib.Path(__file__).parents[1] / "results"

# X_DIST_FRACTION from step-timing baseline (0.589 per experiment plan motivation).
# Will be updated from actual baseline profiler data once available.
AMDAHL_XDIST_FRACTION = 0.589


def load_profiler(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_criterion_median_ms(txt_path: pathlib.Path) -> float | None:
    """Parse Criterion text output for the n=10000 median time in ms."""
    if not txt_path.exists():
        return None
    text = txt_path.read_text()
    # Criterion line format: "trustworthiness_d50/n:10000  time   [X.XXX ms X.XXX ms X.XXX ms]"
    pattern = r"trustworthiness_d50/n:10000\s+time\s+\[[\d.]+ \w+\s+([\d.]+) (\w+)"
    m = re.search(pattern, text)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    if unit == "ms":
        return value
    if unit == "µs" or unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value


def xdist_mean_ns(profiler: dict) -> float | None:
    st = profiler.get("step_timing", {})
    vals = st.get("x_dist", [])
    if not vals:
        return None
    return sum(vals) / len(vals)


def amdahl(xdist_fraction: float, xdist_speedup: float) -> float:
    other = 1.0 - xdist_fraction
    return 1.0 / (other + xdist_fraction / xdist_speedup)


def main() -> None:
    baseline_json = RESULTS / "baseline_profiler.json"
    if not baseline_json.exists():
        print(f"ERROR: baseline profiler not found at {baseline_json}", file=sys.stderr)
        sys.exit(1)

    baseline = load_profiler(baseline_json)
    baseline_total_ms = extract_criterion_median_ms(RESULTS / "baseline_criterion.txt")
    baseline_xdist_ns = xdist_mean_ns(baseline)

    variants = []
    for p in sorted(RESULTS.glob("*_profiler.json")):
        stem = p.stem.replace("_profiler", "")
        if stem == "baseline":
            continue
        variants.append(stem)

    rows = []
    for v in variants:
        prof = load_profiler(RESULTS / f"{v}_profiler.json")
        total_ms = extract_criterion_median_ms(RESULTS / f"{v}_criterion.txt")
        xdist_ns = xdist_mean_ns(prof)

        xdist_speedup = (baseline_xdist_ns / xdist_ns) if (baseline_xdist_ns and xdist_ns) else None
        total_speedup = (baseline_total_ms / total_ms) if (baseline_total_ms and total_ms) else None
        amdahl_pred = amdahl(AMDAHL_XDIST_FRACTION, xdist_speedup) if xdist_speedup else None

        rows.append({
            "variant": v,
            "xdist_speedup": xdist_speedup,
            "total_speedup": total_speedup,
            "amdahl_pred": amdahl_pred,
        })

    # Markdown table
    print("## Speedup Results\n")
    print("| Variant | x_dist speedup | Total speedup | Amdahl predicted | H1 pass (>=1.5x) |")
    print("|---------|---------------|--------------|-----------------|-----------------|")
    for r in rows:
        xs = f"{r['xdist_speedup']:.2f}x" if r['xdist_speedup'] else "n/a"
        ts = f"{r['total_speedup']:.2f}x" if r['total_speedup'] else "n/a"
        ap = f"{r['amdahl_pred']:.2f}x" if r['amdahl_pred'] else "n/a"
        h1 = "Y" if (r['total_speedup'] and r['total_speedup'] >= 1.5) else "N"
        print(f"| {r['variant']} | {xs} | {ts} | {ap} | {h1} |")

    # AVX-512 marginal gain
    avx2_row = next((r for r in rows if "avx2" in r["variant"]), None)
    avx512_row = next((r for r in rows if "avx512" in r["variant"]), None)
    if avx2_row and avx512_row and avx2_row["total_speedup"] and avx512_row["total_speedup"]:
        marginal = avx512_row["total_speedup"] / avx2_row["total_speedup"]
        print(f"\n**AVX-512 marginal gain over looped AVX2:** {marginal:.2f}x "
              f"({'>=1.2x -- ship AVX-512' if marginal >= 1.2 else '<1.2x -- ship AVX2 only'})")

    # Tiling marginal
    tiled_row = next((r for r in rows if "tiled" in r["variant"]), None)
    if avx512_row and tiled_row and avx512_row["total_speedup"] and tiled_row["total_speedup"]:
        tiling_marginal = tiled_row["total_speedup"] / avx512_row["total_speedup"]
        print(f"**Tiling marginal gain:** {tiling_marginal:.2f}x "
              f"({'V-Cache confirms low benefit' if tiling_marginal < 1.05 else '-- tiling worthwhile'})")


if __name__ == "__main__":
    main()
