#!/usr/bin/env python3
"""Analyze x-dist SIMD experiment results.

Reads results/*.json and results/*_criterion.txt, computes:
  - x_dist step speedup per variant
  - total wall-clock speedup per variant
  - Amdahl projection vs measured
  - AVX-512 marginal gain over looped AVX2
  - Tiling marginal gain (if present)

Usage (from repo root):
  python research/2026-04-08-x-dist-simd-avx512/scripts/analyze.py [results_dir]
"""

import json
import pathlib
import re
import sys

# Fallback Amdahl fraction if baseline_timing_summary.json is absent.
AMDAHL_XDIST_FRACTION = 0.589


def _resolve_results(argv: list[str]) -> pathlib.Path:
    if len(argv) > 1:
        return pathlib.Path(argv[1]).resolve()
    return pathlib.Path(__file__).parents[1] / "results"


def load_profiler(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_criterion_median_ms(variant: str, results: pathlib.Path) -> float | None:
    """Try JSON first, then fall back to parsing the .txt file."""
    json_path = results / f"{variant}_criterion.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            return data["trustworthiness_d50"]["10000"]["median_ms"]
        except (KeyError, json.JSONDecodeError):
            pass

    txt_path = results / f"{variant}_criterion.txt"
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


def load_correctness(results: pathlib.Path) -> dict[str, float | None]:
    path = results / "correctness.json"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        out[entry["variant"]] = entry.get("delta")
    return out


def main() -> None:
    RESULTS = _resolve_results(sys.argv)

    # Load actual x_dist fraction from baseline_timing_summary.json if available.
    timing_summary = RESULTS / "baseline_timing_summary.json"
    if timing_summary.exists():
        ts = json.loads(timing_summary.read_text())
        xdist_fraction = ts.get("x_dist_fraction", AMDAHL_XDIST_FRACTION)
    else:
        xdist_fraction = AMDAHL_XDIST_FRACTION

    baseline_json = RESULTS / "baseline_profiler.json"
    if not baseline_json.exists():
        print(f"ERROR: baseline profiler not found at {baseline_json}", file=sys.stderr)
        sys.exit(1)

    baseline = load_profiler(baseline_json)
    baseline_total_ms = extract_criterion_median_ms("baseline", RESULTS)
    baseline_xdist_ns = xdist_mean_ns(baseline)

    correctness = load_correctness(RESULTS)

    variants = []
    for p in sorted(RESULTS.glob("*_profiler.json")):
        stem = p.stem.replace("_profiler", "")
        if stem == "baseline":
            continue
        variants.append(stem)

    rows = []
    for v in variants:
        prof = load_profiler(RESULTS / f"{v}_profiler.json")
        total_ms = extract_criterion_median_ms(v, RESULTS)
        xdist_ns = xdist_mean_ns(prof)

        xdist_speedup = (baseline_xdist_ns / xdist_ns) if (baseline_xdist_ns and xdist_ns) else None
        total_speedup = (baseline_total_ms / total_ms) if (baseline_total_ms and total_ms) else None
        amdahl_pred = amdahl(xdist_fraction, xdist_speedup) if xdist_speedup else None
        delta = correctness.get(v)

        rows.append({
            "variant": v,
            "xdist_speedup": xdist_speedup,
            "total_speedup": total_speedup,
            "amdahl_pred": amdahl_pred,
            "delta": delta,
        })

    # Markdown table
    print("## Speedup Results\n")
    print(f"_Amdahl x_dist fraction: {xdist_fraction:.4f}_\n")
    print("| Variant | x_dist speedup | Total speedup | Amdahl predicted | H1 pass (>=1.5x) | Correctness delta |")
    print("|---------|---------------|--------------|-----------------|-----------------|------------------|")
    for r in rows:
        xs = f"{r['xdist_speedup']:.2f}x" if r['xdist_speedup'] else "n/a"
        ts = f"{r['total_speedup']:.2f}x" if r['total_speedup'] else "n/a"
        ap = f"{r['amdahl_pred']:.2f}x" if r['amdahl_pred'] else "n/a"
        h1 = "Y" if (r['total_speedup'] and r['total_speedup'] >= 1.5) else "N"
        delta_str = f"{r['delta']:.2e}" if r['delta'] is not None else "n/a"
        print(f"| {r['variant']} | {xs} | {ts} | {ap} | {h1} | {delta_str} |")

    # Amdahl validation note
    for r in rows:
        if r["total_speedup"] and r["amdahl_pred"]:
            deviation = abs(r["total_speedup"] - r["amdahl_pred"]) / r["amdahl_pred"]
            if deviation > 0.20:
                print(f"\n> NOTE: {r['variant']} measured speedup ({r['total_speedup']:.2f}x) deviates "
                      f"{deviation*100:.0f}% from Amdahl prediction ({r['amdahl_pred']:.2f}x). "
                      "x_dist fraction estimate may be stale.")

    # AVX-512 marginal gain
    avx2_row = next((r for r in rows if "avx2" in r["variant"] and "tiled" not in r["variant"]), None)
    avx512_row = next((r for r in rows if "avx512" in r["variant"] and "tiled" not in r["variant"]), None)
    if avx2_row and avx512_row and avx2_row["total_speedup"] and avx512_row["total_speedup"]:
        marginal = avx512_row["total_speedup"] / avx2_row["total_speedup"]
        print(f"\n**AVX-512 marginal gain over looped AVX2:** {marginal:.2f}x "
              f"({'>=1.2x -- ship AVX-512' if marginal >= 1.2 else '<1.2x -- ship AVX2 only'})")

    # Tiling marginal
    tiled_rows = [r for r in rows if "tiled" in r["variant"]]
    if tiled_rows and avx512_row and avx512_row["total_speedup"]:
        print("\n## Tiling Marginal Gain\n")
        print("| Tile variant | Total speedup | Marginal vs avx512_looped |")
        print("|-------------|--------------|--------------------------|")
        for tr in tiled_rows:
            if tr["total_speedup"]:
                mg = tr["total_speedup"] / avx512_row["total_speedup"]
                note = "V-Cache confirms low benefit" if mg < 1.05 else "tiling worthwhile"
                print(f"| {tr['variant']} | {tr['total_speedup']:.2f}x | {mg:.2f}x ({note}) |")


if __name__ == "__main__":
    main()
