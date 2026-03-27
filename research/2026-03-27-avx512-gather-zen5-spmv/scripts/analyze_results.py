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
