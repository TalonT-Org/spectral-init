"""Analysis script for SIMD SpMV ISA baseline experiment."""
import json
import os
import statistics
from pathlib import Path


N_VALUES = [200, 2000, 5000, 10000, 50000]

RESULTS = Path("research/2026-03-27-simd-spmv-isa-baseline/results")


def load_estimate(group: str, bench_name: str) -> dict:
    """Load Criterion estimates.json for a given benchmark group and bench path."""
    estimates_path = (
        Path("target") / "criterion" / group / bench_name / "new" / "estimates.json"
    )
    with open(estimates_path) as f:
        return json.load(f)


def get_mean_ns(group: str, bench_name: str) -> float:
    """Return mean elapsed nanoseconds for a benchmark."""
    est = load_estimate(group, bench_name)
    return est["mean"]["point_estimate"]


def parse_timing_breakdown(timing_file: Path) -> dict:
    """Parse [timing:level_N] lines from timing_breakdown.txt.

    Returns dict mapping 'timing:level_N' -> microseconds (int).
    If the file does not exist, returns an empty dict.
    """
    levels = {}
    if not timing_file.exists():
        return levels
    with open(timing_file) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("[timing:level_"):
                continue
            try:
                bracket_end = line.index("]")
                tag = line[1:bracket_end]                 # e.g. "timing:level_1"
                rest = line[bracket_end + 1:].strip()     # e.g. "1234µs"
                micros_str = rest.replace("\u00b5s", "").replace("us", "").strip()
                levels[tag] = int(micros_str)
            except (ValueError, IndexError):
                continue
    return levels


def compute_level_fractions(levels: dict) -> dict:
    """Compute each level's fraction of total solve time."""
    total = levels.get("timing:level_total", 0)
    if total == 0:
        return {}
    return {
        k: v / total
        for k, v in levels.items()
        if k != "timing:level_total"
    }


def _fmt(v: float, decimals: int = 2) -> str:
    """Format a float to fixed decimals, falling back to 'N/A' on error."""
    try:
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def _speedup(baseline: float, target: float) -> float:
    """Return baseline / target speedup ratio."""
    if target == 0.0:
        return float("nan")
    return baseline / target


def _breakeven(conversion_ns: float, csr_ns: float, sell_c_ns: float) -> str:
    """Compute break-even iteration count.

    = conversion_time / (csr_time - sell_c_time)

    Returns 'inf' if SELL-C is not faster than CSR, 'N/A' on error.
    """
    try:
        delta = csr_ns - sell_c_ns
        if delta <= 0:
            return "inf"
        return _fmt(conversion_ns / delta, 1)
    except (ZeroDivisionError, TypeError):
        return "N/A"


def build_speedup_rows() -> list:
    """Build one row per n_value with CSR, SELL-C4, SELL-C8, AVX2 timings."""
    rows = []
    for n in N_VALUES:
        try:
            csr_ns = get_mean_ns("spmv_csr_scaling", str(n))
        except FileNotFoundError:
            csr_ns = float("nan")

        try:
            c4_ns = get_mean_ns("spmv_sell_c", f"C4/{n}")
        except FileNotFoundError:
            c4_ns = float("nan")

        try:
            c8_ns = get_mean_ns("spmv_sell_c", f"C8/{n}")
        except FileNotFoundError:
            c8_ns = float("nan")

        try:
            avx2_ns = get_mean_ns("spmv_avx2", f"avx2_gather/{n}")
        except FileNotFoundError:
            avx2_ns = float("nan")

        try:
            conv_ns = get_mean_ns("spmv_sell_c_conversion", str(n))
        except FileNotFoundError:
            conv_ns = float("nan")

        rows.append({
            "n": n,
            "csr_ns": _fmt(csr_ns, 1),
            "sell_c4_ns": _fmt(c4_ns, 1),
            "sell_c4_speedup": _fmt(_speedup(csr_ns, c4_ns), 3),
            "sell_c8_ns": _fmt(c8_ns, 1),
            "sell_c8_speedup": _fmt(_speedup(csr_ns, c8_ns), 3),
            "avx2_ns": _fmt(avx2_ns, 1),
            "avx2_speedup": _fmt(_speedup(csr_ns, avx2_ns), 3),
            "sell_c4_breakeven": _breakeven(conv_ns, csr_ns, c4_ns),
            "sell_c8_breakeven": _breakeven(conv_ns, csr_ns, c8_ns),
        })
    return rows


def emit_speedup_table(rows: list, out_path: Path) -> None:
    """Write the speedup table as a Markdown file."""
    headers = [
        "n", "csr_ns", "sell_c4_ns", "sell_c4_speedup",
        "sell_c8_ns", "sell_c8_speedup",
        "avx2_ns", "avx2_speedup",
        "sell_c4_breakeven", "sell_c8_breakeven",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# SpMV ISA Baseline — Speedup Table\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
    print(f"Speedup table written to {out_path}")


if __name__ == "__main__":
    # ── Timing breakdown ───────────────────────────────────────────────────────
    timing_file = RESULTS / "timing_breakdown.txt"
    levels = parse_timing_breakdown(timing_file)
    if levels:
        fractions = compute_level_fractions(levels)
        print("\n=== Solver Level Timing Fractions ===")
        for tag, frac in sorted(fractions.items()):
            level_name = tag.replace("timing:", "")
            us = levels.get(tag, 0)
            print(f"  {level_name:30s}: {frac*100:6.1f}%  ({us} µs)")
        total_us = levels.get("timing:level_total", 0)
        print(f"  {'total':30s}: {total_us} µs")
    else:
        print(f"No timing data found in {timing_file} (run pipeline_blobs5000 bench first)")

    # ── Speedup table ──────────────────────────────────────────────────────────
    print("\n=== Building Speedup Table ===")
    rows = build_speedup_rows()
    out_path = RESULTS / "speedup_table.md"
    emit_speedup_table(rows, out_path)
