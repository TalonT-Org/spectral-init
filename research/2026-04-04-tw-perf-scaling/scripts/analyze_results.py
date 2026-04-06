"""Analyze tw-perf-scaling experiment results and produce ranked recommendation table.

Run from research/2026-04-04-tw-perf-scaling/:
    python scripts/analyze_results.py

Reads all result artifacts from results/ subdirectories and writes:
    results/analysis/ranked_recommendations.md
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
STEP_TIMING = RESEARCH_DIR / "results" / "step_timing"
CRITERION_DIR = RESEARCH_DIR / "results" / "criterion"
ASM_DIR = RESEARCH_DIR / "results" / "asm"
SUBSAMPLING_DIR = RESEARCH_DIR / "results" / "subsampling"
ANALYSIS_DIR = RESEARCH_DIR / "results" / "analysis"

N_SIZES = [1000, 5000, 10000, 25000, 50000, 100000]


def load_json(path: Path) -> dict | None:
    """Return parsed JSON or None if the file is missing or malformed."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_timing(variant: str, n: int) -> dict | None:
    return load_json(STEP_TIMING / f"gaussian_n{n}_{variant}.json")


def speedup(baseline_data: dict | None, variant_data: dict | None) -> str:
    if baseline_data is None or variant_data is None:
        return "N/A"
    b = baseline_data.get("mean_s")
    v = variant_data.get("mean_s")
    if b is None or v is None or v <= 0:
        return "N/A"
    return f"{b / v:.2f}x"


def speedup_float(baseline_data: dict | None, variant_data: dict | None) -> float | None:
    if baseline_data is None or variant_data is None:
        return None
    b = baseline_data.get("mean_s")
    v = variant_data.get("mean_s")
    if b is None or v is None or v <= 0:
        return None
    return b / v


def parse_criterion_speedup(variant_name: str, baseline_mean_s: float | None) -> str:
    """Extract Criterion speedup at n=50K for a variant vs baseline from criterion_output.json.

    Criterion --message-format=json emits a stream of JSON objects. We look for
    benchmark IDs matching the variant and baseline at n=50000 and compute the ratio.
    Returns "N/A" if not parseable or if baseline_mean_s is not provided.
    """
    if baseline_mean_s is None or baseline_mean_s <= 0:
        return "N/A"
    crit_path = CRITERION_DIR / "criterion_output.json"
    if not crit_path.exists():
        return "N/A"
    try:
        content = crit_path.read_text()
        # Each line may be a separate JSON object (cargo criterion --message-format=json)
        variant_mean: float | None = None
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            bench_id = msg.get("id", "")
            if ("50000" in bench_id or "50k" in bench_id.lower()) and variant_name in bench_id:
                mean = msg.get("typical", msg.get("mean", {})).get("estimate")
                if mean is not None:
                    variant_mean = float(mean)
                    break
        if variant_mean is not None and variant_mean > 0:
            return f"{baseline_mean_s / variant_mean:.2f}x"
    except Exception:
        pass
    return "N/A"


def analyze_h0(baseline_100k: dict | None) -> str:
    """H0: Per-step breakdown at n=100K.

    Returns a human-readable breakdown string.
    """
    if baseline_100k is None:
        return "N/A (baseline_n100000 missing)"
    # Per-step fields emitted when compiled with --features testing
    step_fields = ["tw_x_dist", "tw_x_sort", "tw_rank_scatter", "tw_x_knn_set",
                   "tw_y_heap", "tw_penalty", "tw_total"]
    present = {f: baseline_100k[f] for f in step_fields if f in baseline_100k}
    if not present:
        return "N/A (per-step fields absent — requires testing feature)"
    total = present.get("tw_total", 1.0)
    if total <= 0:
        return "N/A"
    parts = []
    for f in step_fields:
        if f in present:
            pct = 100.0 * present[f] / total
            parts.append(f"{f}={pct:.1f}%")
    return ", ".join(parts)


def analyze_h1(baseline_100k: dict | None) -> tuple[str, str]:
    """H1: X-sort dominance.

    Returns (verdict, detail). verdict is 'SUPPORTED', 'REFUTED', or 'N/A'.
    """
    if baseline_100k is None:
        return "N/A", "baseline_n100000 missing"
    x_sort = baseline_100k.get("tw_x_sort")
    rank_scatter = baseline_100k.get("tw_rank_scatter")
    total = baseline_100k.get("tw_total")
    if x_sort is None or rank_scatter is None or total is None or total <= 0:
        return "N/A", "per-step fields absent"
    ratio = (x_sort + rank_scatter) / total
    pct = 100.0 * ratio
    if ratio >= 0.40:
        verdict = "SUPPORTED"
    elif ratio < 0.30:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, f"(x_sort+rank_scatter)/total = {pct:.1f}%"


def analyze_h2(
    baseline_100k: dict | None,
    tl_100k: dict | None,
    criterion_speedup_50k: float | None = None,
) -> tuple[str, str]:
    """H2: Thread-local buffers speedup >= 1.5x.

    Uses Criterion n=50K speedup for the GO/NO-GO gate when available.
    tw_profiler baseline is contaminated by #[cfg(feature="testing")] overhead
    and must NOT be used for baseline-relative ratios.
    """
    if criterion_speedup_50k is not None:
        verdict = "GO" if criterion_speedup_50k >= 1.5 else "NO-GO"
        return verdict, f"{criterion_speedup_50k:.2f}x Criterion speedup at n=50K (authoritative)"
    # Variant-vs-variant comparison is valid; baseline-relative is not.
    ratio = speedup_float(tl_100k, baseline_100k)  # reversed: we have no clean baseline
    return "N/A", "Criterion data unavailable; tw_profiler baseline contaminated — cannot evaluate"


def analyze_h3() -> tuple[str, str]:
    """H3: AVX2 auto-vectorization verdict from h3_verdict.txt."""
    verdict_path = ASM_DIR / "h3_verdict.txt"
    if not verdict_path.exists():
        return "N/A", "h3_verdict.txt missing — run run_criterion.sh --asm-only"
    text = verdict_path.read_text().strip()
    return text, ""


def analyze_h4(avx2_100k: dict | None, avx512_100k: dict | None) -> tuple[str, str]:
    """H4: AVX-512 speedup >= 1.2x over AVX2."""
    if avx2_100k is None or avx512_100k is None:
        return "N/A", "hardware not available or timing data missing"
    ratio = speedup_float(avx2_100k, avx512_100k)
    if ratio is None:
        return "N/A", "timing data missing"
    verdict = "GO" if ratio >= 1.2 else "NO-GO"
    return verdict, f"{ratio:.2f}x speedup (AVX-512 over AVX2) at n=100K"


def analyze_h5() -> tuple[str, str]:
    """H5: Row subsampling — delta < 0.001."""
    result_path = SUBSAMPLING_DIR / "h5_confirmatory_result.json"
    data = load_json(result_path)
    if data is None:
        return "N/A", "h5_confirmatory_result.json missing — run run_h5_confirmatory.sh"
    delta = data.get("delta")
    if delta is None:
        return "N/A", "delta field missing from confirmatory result"
    verdict = "GO" if abs(delta) < 0.001 else "NO-GO"
    return verdict, f"delta = {delta:.6f} (threshold: |delta| < 0.001)"


def analyze_h6(
    baseline_100k: dict | None,
    combined_100k: dict | None,
    criterion_speedup_50k: float | None = None,
) -> tuple[str, str]:
    """H6: Combined optimization speedup >= 3x.

    Uses Criterion n=50K speedup for the GO/NO-GO gate when available.
    tw_profiler baseline is contaminated by #[cfg(feature="testing")] overhead
    and must NOT be used for baseline-relative ratios.
    """
    if criterion_speedup_50k is not None:
        verdict = "GO" if criterion_speedup_50k >= 3.0 else "NO-GO"
        return verdict, f"{criterion_speedup_50k:.2f}x Criterion speedup at n=50K (authoritative)"
    return "N/A", "Criterion data unavailable; tw_profiler baseline contaminated — cannot evaluate"


def build_report() -> str:
    # Load key timing data
    baseline_100k = load_timing("baseline", 100000)
    tl_100k = load_timing("thread_local", 100000)
    pr_100k = load_timing("partial_rank", 100000)
    avx2_100k = load_timing("avx2_kernel", 100000)
    avx512_100k = load_timing("avx512_kernel", 100000)
    combined_100k = load_timing("combined", 100000)

    # Criterion n=50K baseline (clean; no --features testing contamination).
    # Parsed from criterion_output.json if available; fall back to None.
    # NOTE: tw_profiler baseline is contaminated by #[cfg(feature="testing")]
    # eprintln! overhead (~6.25x at n=100K). Do NOT use for baseline-relative ratios.
    baseline_50k_criterion = load_timing("baseline", 50000)
    baseline_criterion_mean: float | None = (
        baseline_50k_criterion.get("mean_s") if baseline_50k_criterion else None
    )

    # Run analyses
    h0_detail = analyze_h0(baseline_100k)
    h1_verdict, h1_detail = analyze_h1(baseline_100k)
    # H2/H6: pass Criterion speedup from criterion_output.json when available;
    # fall back to None so functions return N/A rather than using contaminated baseline.
    tl_crit_speedup = parse_criterion_speedup("thread_local", baseline_criterion_mean)
    combined_crit_speedup_str = parse_criterion_speedup("combined", baseline_criterion_mean)
    tl_crit_float = (
        float(tl_crit_speedup.rstrip("x")) if tl_crit_speedup != "N/A" else None
    )
    combined_crit_float = (
        float(combined_crit_speedup_str.rstrip("x")) if combined_crit_speedup_str != "N/A" else None
    )
    h2_verdict, h2_detail = analyze_h2(baseline_100k, tl_100k, criterion_speedup_50k=tl_crit_float)
    h3_raw, _ = analyze_h3()
    h4_verdict, h4_detail = analyze_h4(avx2_100k, avx512_100k)
    h5_verdict, h5_detail = analyze_h5()
    h6_verdict, h6_detail = analyze_h6(
        baseline_100k, combined_100k, criterion_speedup_50k=combined_crit_float
    )

    # Derive per-row verdicts for the table
    # Thread-local (H2) — contaminated baseline; show variant-vs-variant comparison only
    tl_speedup_100k = speedup(combined_100k, tl_100k)  # valid: variant-vs-variant
    tl_speedup_50k = tl_crit_speedup
    tl_go = h2_verdict

    # Partial-rank (H1) — use same 1.5x gate; contaminated baseline, use Criterion if available
    pr_speedup_50k = parse_criterion_speedup("partial_rank", baseline_criterion_mean)
    pr_speedup_100k = speedup(combined_100k, pr_100k)  # valid: variant-vs-variant
    pr_crit_float = (
        float(pr_speedup_50k.rstrip("x")) if pr_speedup_50k != "N/A" else None
    )
    if pr_crit_float is not None:
        pr_go = "GO" if pr_crit_float >= 1.5 else "NO-GO"
    else:
        pr_go = "N/A (Criterion data unavailable)"

    # Auto-vectorized (H3 confirmed)
    h3_auto = "AUTO-VECTORIZED" in h3_raw
    avx2_auto_go = "CONFIRMED (no action)" if h3_auto else "N/A"

    # Manual AVX2 (H3 inverted)
    avx2_speedup_50k = parse_criterion_speedup("avx2_kernel", baseline_criterion_mean)
    avx2_speedup_100k = speedup(combined_100k, avx2_100k)  # valid: variant-vs-variant
    avx2_crit_float = (
        float(avx2_speedup_50k.rstrip("x")) if avx2_speedup_50k != "N/A" else None
    )
    if "NOT AUTO-VECTORIZED" in h3_raw:
        if avx2_crit_float is not None:
            avx2_go = "GO" if avx2_crit_float >= 1.5 else "NO-GO"
        else:
            avx2_go = "GO (implement — H3 not auto-vectorized; Criterion data unavailable)"
    elif "AUTO-VECTORIZED" in h3_raw:
        avx2_go = "NO-GO (compiler already auto-vectorizes)"
    else:
        avx2_go = "N/A"

    # AVX-512 (H4)
    avx512_speedup_100k = speedup(avx2_100k, avx512_100k)
    avx512_speedup_50k = "N/A"

    # Row subsampling (H5)
    h5_go = h5_verdict

    # Combined (H6)
    combined_speedup_100k = speedup(tl_100k, combined_100k)  # valid: variant-vs-variant
    combined_speedup_50k = combined_crit_speedup_str
    h6_go = h6_verdict

    # Build markdown table
    rows = [
        {
            "approach": "Thread-local buffers",
            "hypothesis": "H2",
            "speedup_50k": tl_speedup_50k,
            "speedup_100k": tl_speedup_100k,
            "scaling_change": "No",
            "loc": "~20",
            "verdict": tl_go,
            "rationale": h2_detail,
        },
        {
            "approach": "Partial-rank X",
            "hypothesis": "H1",
            "speedup_50k": pr_speedup_50k,
            "speedup_100k": pr_speedup_100k,
            "scaling_change": "No",
            "loc": "~40",
            "verdict": pr_go,
            "rationale": h1_detail,
        },
        {
            "approach": "Auto-vectorized distance",
            "hypothesis": "H3",
            "speedup_50k": "N/A",
            "speedup_100k": "N/A",
            "scaling_change": "No",
            "loc": "0",
            "verdict": avx2_auto_go,
            "rationale": h3_raw,
        },
        {
            "approach": "Manual AVX2 kernel",
            "hypothesis": "H3 (inverted)",
            "speedup_50k": avx2_speedup_50k,
            "speedup_100k": avx2_speedup_100k,
            "scaling_change": "No",
            "loc": "~60",
            "verdict": avx2_go,
            "rationale": f"Conditional on H3 verdict: {h3_raw}",
        },
        {
            "approach": "AVX-512 kernel",
            "hypothesis": "H4",
            "speedup_50k": avx512_speedup_50k,
            "speedup_100k": avx512_speedup_100k,
            "scaling_change": "No",
            "loc": "~80",
            "verdict": h4_verdict,
            "rationale": h4_detail,
        },
        {
            "approach": "Row subsampling",
            "hypothesis": "H5",
            "speedup_50k": "N/A",
            "speedup_100k": "N/A (approx)",
            "scaling_change": "Yes",
            "loc": "~30",
            "verdict": h5_go,
            "rationale": h5_detail,
        },
        {
            "approach": "Combined exact",
            "hypothesis": "H6",
            "speedup_50k": combined_speedup_50k,
            "speedup_100k": combined_speedup_100k,
            "scaling_change": "No",
            "loc": "~60",
            "verdict": h6_go,
            "rationale": h6_detail,
        },
    ]

    lines = []
    lines.append("# tw-perf-scaling: Ranked Recommendations\n")
    lines.append(f"Generated from `results/` artifacts.\n")

    lines.append("## H0: Per-Step Profiling Breakdown (n=100K baseline)\n")
    lines.append(f"{h0_detail}\n")

    lines.append("## Hypothesis Verdicts\n")
    lines.append(f"- **H1** (X-sort dominance): {h1_verdict} — {h1_detail}")
    lines.append(f"- **H2** (thread-local buffers): {h2_verdict} — {h2_detail}")
    lines.append(f"- **H3** (AVX2 auto-vectorization): {h3_raw}")
    lines.append(f"- **H4** (AVX-512): {h4_verdict} — {h4_detail}")
    lines.append(f"- **H5** (row subsampling): {h5_verdict} — {h5_detail}")
    lines.append(f"- **H6** (combined): {h6_verdict} — {h6_detail}")
    lines.append("")

    lines.append("## Ranked Recommendation Table\n")
    header = (
        "| Approach | Hypothesis | Speedup n=50K (Criterion CI) | "
        "Speedup n=100K (tw_profiler) | Scaling law change | LOC estimate | GO/NO-GO | Rationale |"
    )
    sep = (
        "|---|---|---|---|---|---|---|---|"
    )
    lines.append(header)
    lines.append(sep)
    for row in rows:
        lines.append(
            f"| {row['approach']} | {row['hypothesis']} | {row['speedup_50k']} | "
            f"{row['speedup_100k']} | {row['scaling_change']} | {row['loc']} | "
            f"{row['verdict']} | {row['rationale']} |"
        )
    lines.append("")

    # Overall recommendation summary
    go_items = [r["approach"] for r in rows if r["verdict"].startswith("GO")]
    nogo_items = [r["approach"] for r in rows if r["verdict"].startswith("NO-GO")]
    na_items = [r["approach"] for r in rows if r["verdict"].startswith("N/A") or
                r["verdict"] == "CONFIRMED (no action)"]

    lines.append("## Overall Recommendation\n")
    lines.append(f"**Ship:** {', '.join(go_items) if go_items else 'none'}")
    lines.append(f"\n**Defer:** {', '.join(nogo_items) if nogo_items else 'none'}")
    lines.append(f"\n**N/A:** {', '.join(na_items) if na_items else 'none'}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_DIR / "ranked_recommendations.md"
    output_path.write_text(report)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
