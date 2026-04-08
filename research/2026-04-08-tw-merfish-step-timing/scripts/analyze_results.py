#!/usr/bin/env python3
"""Analyze tw_profiler step-timing JSON outputs and produce a comparison table."""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

STEPS = ["x_dist", "x_sort", "y_dist", "penalty"]

HISTORICAL_REF = (
    Path(__file__).resolve().parent.parent.parent
    / "2026-04-06-y-heap-bottleneck-optimization"
    / "results"
    / "profiler"
    / "profiler_flat_simd_n10000.json"
)

# Display names for datasets found by filename pattern
DATASET_LABELS = {
    "gaussian_n10k": "Gaussian 10K",
    "merfish_n10k": "MERFISH 10K",
    "merfish_n50k": "MERFISH 50K",
}

# Canonical column order
DATASET_ORDER = ["gaussian_n10k", "merfish_n10k", "merfish_n50k"]


def load_profiler_json(path: Path) -> dict:
    """Load a profiler JSON and validate it has step_timing."""
    with open(path) as f:
        data = json.load(f)
    if "step_timing" not in data:
        print(f"WARNING: {path.name} has no step_timing key", file=sys.stderr)
    return data


def load_historical_reference() -> dict | None:
    """Load the flat_simd historical reference, mapping y_heap -> y_dist."""
    if not HISTORICAL_REF.exists():
        print(f"WARNING: historical reference not found: {HISTORICAL_REF}", file=sys.stderr)
        return None
    data = load_profiler_json(HISTORICAL_REF)
    timing = data.get("step_timing", {})
    if "y_heap" in timing and "y_dist" not in timing:
        timing["y_dist"] = timing.pop("y_heap")
    return data


def warmup_offset(data: dict) -> int:
    """Determine how many warmup entries to skip in step_timing arrays.

    step_timing arrays include all iterations (warmup + measured), while
    data["iters"] only has the measured iterations. The difference gives
    the number of warmup entries to discard.
    """
    n_iters = len(data.get("iters", []))
    sample_key = next(iter(data.get("step_timing", {})), None)
    if sample_key is None:
        return 0
    n_timing = len(data["step_timing"][sample_key])
    return max(0, n_timing - n_iters)


def compute_step_stats(data: dict) -> dict:
    """Compute per-step statistics from a profiler JSON."""
    timing = data.get("step_timing", {})
    offset = warmup_offset(data)
    result = {}
    total_per_iter = None

    for step in STEPS:
        if step not in timing:
            continue
        arr = np.array(timing[step], dtype=float)[offset:]
        ns_to_ms = arr / 1e6
        result[step] = {
            "mean_ms": float(np.mean(ns_to_ms)),
            "std_ms": float(np.std(ns_to_ms, ddof=1)) if len(ns_to_ms) > 1 else 0.0,
            "raw_ns": arr,
        }
        if total_per_iter is None:
            total_per_iter = arr.copy()
        else:
            total_per_iter += arr

    if total_per_iter is not None:
        total_ms = total_per_iter / 1e6
        result["total"] = {
            "mean_ms": float(np.mean(total_ms)),
            "std_ms": float(np.std(total_ms, ddof=1)) if len(total_ms) > 1 else 0.0,
        }
        total_mean = np.mean(total_per_iter)
        for step in STEPS:
            if step in result:
                result[step]["fraction"] = float(np.mean(result[step]["raw_ns"]) / total_mean)

    # Compute x_space_pct with proper per-iteration CI
    x_space_pct, x_space_ci_lo, x_space_ci_hi = compute_x_space_pct_ci(timing, offset)
    result["x_space_pct"] = {
        "mean": x_space_pct,
        "ci_lo": x_space_ci_lo,
        "ci_hi": x_space_ci_hi,
    }

    return result


def compute_x_space_pct_ci(step_timing: dict, warmup_offset: int) -> tuple[float, float, float]:
    """Compute x_space_pct per iteration for a proper CI on the percentage."""
    if "x_dist" not in step_timing or "x_sort" not in step_timing:
        return (0.0, 0.0, 0.0)

    x_dist = np.array(step_timing["x_dist"], dtype=float)[warmup_offset:]
    x_sort = np.array(step_timing["x_sort"], dtype=float)[warmup_offset:]

    total = np.zeros_like(x_dist)
    for step in STEPS:
        if step in step_timing:
            total += np.array(step_timing[step], dtype=float)[warmup_offset:]

    # Avoid division by zero
    mask = total > 0
    x_space = np.zeros_like(x_dist)
    x_space[mask] = (x_dist[mask] + x_sort[mask]) / total[mask] * 100

    n = len(x_space)
    mean = float(np.mean(x_space))
    if n <= 1:
        return (mean, mean, mean)

    std = float(np.std(x_space, ddof=1))
    se = std / sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return (mean, float(ci_lo), float(ci_hi))


def compute_cv(data: dict) -> float:
    """Compute max CV of step fractions across iterations."""
    timing = data.get("step_timing", {})
    offset = warmup_offset(data)

    arrays = {}
    for step in STEPS:
        if step in timing:
            arrays[step] = np.array(timing[step], dtype=float)[offset:]

    if not arrays:
        return 0.0

    total_per_iter = sum(arrays.values())
    max_cv = 0.0
    for step in STEPS:
        if step in arrays:
            fractions = arrays[step] / total_per_iter
            mean_f = np.mean(fractions)
            if mean_f > 0 and len(fractions) > 1:
                cv = float(np.std(fractions, ddof=1) / mean_f)
                max_cv = max(max_cv, cv)

    return max_cv


def fmt_cell(step_stats: dict) -> str:
    """Format a table cell: mean ± std (fraction%)."""
    mean = step_stats["mean_ms"]
    std = step_stats["std_ms"]
    frac = step_stats.get("fraction", 0.0)
    return f"{mean:.1f} \u00b1 {std:.1f} ({frac * 100:.1f}%)"


def build_comparison_table(all_stats: dict, historical_stats: dict | None) -> str:
    """Build a side-by-side markdown comparison table."""
    # Determine columns
    columns = []
    for key in DATASET_ORDER:
        if key in all_stats:
            columns.append((key, DATASET_LABELS.get(key, key)))
    if historical_stats is not None:
        columns.append(("historical", "Historical (flat_simd)"))

    # Header
    header = "| Step |"
    sep = "|------|"
    for _, label in columns:
        header += f" {label} |"
        sep += "------|"

    lines = [header, sep]

    # Step rows
    for step in STEPS:
        row = f"| {step} |"
        for key, _ in columns:
            stats_dict = all_stats.get(key, {}) if key != "historical" else historical_stats
            if stats_dict and step in stats_dict:
                row += f" {fmt_cell(stats_dict[step])} |"
            else:
                row += " - |"
        lines.append(row)

    # Total row
    row = "| **Total** |"
    for key, _ in columns:
        stats_dict = all_stats.get(key, {}) if key != "historical" else historical_stats
        if stats_dict and "total" in stats_dict:
            t = stats_dict["total"]
            row += f" {t['mean_ms']:.1f} \u00b1 {t['std_ms']:.1f} ms |"
        else:
            row += " - |"
    lines.append(row)

    # x_space_pct row
    row = "| **x_space_pct** |"
    for key, _ in columns:
        stats_dict = all_stats.get(key, {}) if key != "historical" else historical_stats
        if stats_dict and "x_space_pct" in stats_dict:
            xs = stats_dict["x_space_pct"]
            row += f" {xs['mean']:.1f}% [{xs['ci_lo']:.1f}, {xs['ci_hi']:.1f}] |"
        else:
            row += " - |"
    lines.append(row)

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze tw_profiler step-timing results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/profiler"),
        help="Directory containing profiler JSON outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Directory for analysis outputs",
    )
    parser.add_argument("--prefix", default="", help="Filename prefix filter")
    parser.add_argument(
        "--cv-only", action="store_true", help="Only print max CV and exit"
    )
    args = parser.parse_args()

    # Resolve paths relative to script location if not absolute
    script_dir = Path(__file__).resolve().parent
    experiment_dir = script_dir.parent
    if not args.results_dir.is_absolute():
        args.results_dir = experiment_dir / args.results_dir
    if not args.output_dir.is_absolute():
        args.output_dir = experiment_dir / args.output_dir

    # Find JSON files
    pattern = f"{args.prefix}*.json"
    json_files = sorted(args.results_dir.glob(pattern))
    if not json_files:
        print(f"ERROR: no JSON files matching '{pattern}' in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load and compute stats
    all_stats = {}
    for jf in json_files:
        # Skip stderr files
        if "stderr" in jf.name:
            continue
        data = load_profiler_json(jf)

        # CV-only mode
        if args.cv_only:
            cv = compute_cv(data)
            print(f"{jf.name}: max_cv={cv:.4f} ({cv * 100:.1f}%)")
            if cv <= 0.15:
                print(f"  -> CV <= 15%: iters=5 is sufficient")
            else:
                print(f"  -> CV > 15%: recommend PROFILER_ITERS=10")
            continue

        # Extract dataset key from filename (strip prefix)
        name = jf.stem
        if args.prefix and name.startswith(args.prefix):
            name = name[len(args.prefix) :]
        step_stats = compute_step_stats(data)
        all_stats[name] = step_stats

    if args.cv_only:
        return

    # Load historical reference
    historical_stats = None
    hist_data = load_historical_reference()
    if hist_data is not None:
        historical_stats = compute_step_stats(hist_data)

    # Build table
    table = build_comparison_table(all_stats, historical_stats)

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.prefix}comparison_table.md"
    out_path.write_text(table)
    print(f"Wrote {out_path}")
    print()
    print(table)

    # Print primary verdict for MERFISH 10K
    for key in ["merfish_n10k", "gaussian_n10k"]:
        if key in all_stats and "x_space_pct" in all_stats[key]:
            xs = all_stats[key]["x_space_pct"]
            label = DATASET_LABELS.get(key, key)
            print(f"Primary verdict ({label}): x_space_pct = {xs['mean']:.1f}% "
                  f"[95% CI: {xs['ci_lo']:.1f}%, {xs['ci_hi']:.1f}%]")
            break


if __name__ == "__main__":
    main()
