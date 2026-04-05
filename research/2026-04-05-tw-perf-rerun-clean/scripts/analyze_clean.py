#!/usr/bin/env python3
"""
analyze_clean.py — Hypothesis verdict report for tw-perf-rerun-clean.
Usage: python analyze_clean.py [--dry-run]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
H5_DIR = os.path.join(EXP_DIR, "results", "h5")
CRIT_DIR = os.path.join(EXP_DIR, "results", "criterion")
STEP_DIR = os.path.join(EXP_DIR, "results", "step_timing")
ANA_DIR = os.path.join(EXP_DIR, "results", "analysis")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hypothesis verdict report")
    p.add_argument("--dry-run", action="store_true",
                   help="Run without requiring result files; write placeholder report")
    return p.parse_args()


def load_jsonlines(path):
    """Parse a JSON-lines file; return list of parsed dicts (skip blank lines)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_criterion_records(path):
    """Parse Criterion JSON-lines; return dict: benchmark_id -> {point, ci, samples}."""
    try:
        records = load_jsonlines(path)
    except FileNotFoundError:
        return {}

    # Accumulate raw sample arrays from "sample" reason records
    sample_map = {}
    for rec in records:
        if rec.get("reason") == "sample" and "id" in rec:
            vals = rec.get("sample", {}).get("measured_values")
            if vals:
                sample_map[rec["id"]] = [float(v) for v in vals]

    # Accumulate point estimates and CIs from "benchmark-complete" records
    result = {}
    for rec in records:
        if rec.get("reason") != "benchmark-complete":
            continue
        bid = rec.get("id", "")
        typical = rec.get("typical", {})
        point = typical.get("estimate")
        if point is None:
            point = rec.get("mean", {}).get("point_estimate")
        ci_lower = typical.get("lower_bound")
        ci_upper = typical.get("upper_bound")
        result[bid] = {
            "point": float(point) if point is not None else None,
            "ci": (
                float(ci_lower) if ci_lower is not None else None,
                float(ci_upper) if ci_upper is not None else None,
            ),
            "samples": sample_map.get(bid),
        }
    return result


def find_record(data, bench_prefix, n_tag):
    """Find a benchmark record where bench_prefix and n_tag are both substrings of the ID."""
    for bid, v in data.items():
        if bench_prefix in bid and n_tag in bid:
            return bid, v
    return None, None


# ---------------------------------------------------------------------------
# Section H5
# ---------------------------------------------------------------------------

def section_h5(dry_run):
    """H5: approximate trustworthiness accuracy — 10 seeds at m=5000."""
    lines = ["## H5: Approximate Trustworthiness Accuracy"]

    if dry_run:
        lines += [
            "  DRY RUN — no data loaded.",
            "  Expected: results/h5/h5_trial_seed{42..51}.json",
            "  Verdict: INCONCLUSIVE (dry run)",
        ]
        return "\n".join(lines)

    seed_files = sorted(glob.glob(os.path.join(H5_DIR, "h5_trial_seed*.json")))
    if not seed_files:
        lines.append("  ERROR: no h5_trial_seed*.json files found in results/h5/")
        return "\n".join(lines)

    speedup_ratios = []
    deltas = []
    for fpath in seed_files:
        with open(fpath) as f:
            d = json.load(f)
        speedup_ratios.append(d["wall_approx_s"] / d["wall_exact_s"])
        deltas.append(abs(d["delta"]))

    median_speedup = float(np.median(speedup_ratios))
    range_speedup = (float(np.min(speedup_ratios)), float(np.max(speedup_ratios)))

    # 95% t-CI for mean |delta| with df=9 (R9 — not z=1.96)
    n = len(deltas)
    mean_delta = float(np.mean(deltas))
    se_delta = float(np.std(deltas, ddof=1) / math.sqrt(n))
    t_crit_r9 = float(t_dist.ppf(0.975, df=9))
    ci_half = t_crit_r9 * se_delta
    ci_lower = mean_delta - ci_half
    ci_upper = mean_delta + ci_half

    THRESHOLD = 0.01
    near_threshold = (ci_half > 0.5 * mean_delta) if mean_delta > 0 else False
    if ci_upper < THRESHOLD:
        verdict = "POSITIVE (approx accurate: CI below threshold)"
    elif ci_lower > THRESHOLD:
        verdict = "NEGATIVE (approx inaccurate: CI above threshold)"
    elif near_threshold:
        verdict = "INCONCLUSIVE (near-threshold: wide CI spans threshold)"
    else:
        verdict = "INCONCLUSIVE (CI spans threshold)"

    lines += [
        f"  Seeds loaded: {len(seed_files)}",
        f"  Speedup ratio (wall_approx/wall_exact): "
        f"median={median_speedup:.4f}, range=[{range_speedup[0]:.4f}, {range_speedup[1]:.4f}]",
        f"  |delta| mean={mean_delta:.6f}, 95% t-CI (df=9, t={t_crit_r9:.4f}): "
        f"[{ci_lower:.6f}, {ci_upper:.6f}]",
        f"  Verdict: **{verdict}**",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section H-100K
# ---------------------------------------------------------------------------

def section_h100k(dry_run):
    """H-100K: Criterion variant speedups at n=100K — bootstrap + Holm-Bonferroni."""
    lines = [
        "## H-100K: Criterion Variant Speedups at n=100K",
        "  W6 Note: Criterion CIs are non-deterministic across runs due to "
        "bootstrapped estimation; point estimates are the stable quantity.",
    ]

    if dry_run:
        lines += ["  DRY RUN — no data loaded.", "  Verdict: INCONCLUSIVE (dry run)"]
        return "\n".join(lines)

    forward_data = extract_criterion_records(os.path.join(CRIT_DIR, "criterion_output.json"))
    reversed_data = extract_criterion_records(os.path.join(CRIT_DIR, "criterion_reversed_output.json"))

    if not forward_data:
        lines.append("  ERROR: criterion_output.json not found or empty.")
        return "\n".join(lines)

    VARIANTS = ["thread_local", "partial_rank", "avx2", "combined"]
    BENCH_PREFIX = {
        "baseline": "tw_baseline",
        "thread_local": "tw_thread_local",
        "partial_rank": "tw_partial_rank",
        "avx2": "tw_avx2",
        "combined": "tw_combined",
    }

    baseline_id, baseline_rec = find_record(forward_data, "tw_baseline", "100000")
    if baseline_rec is None:
        lines.append("  ERROR: baseline n=100K record not found in criterion_output.json.")
        return "\n".join(lines)

    rng = np.random.default_rng(seed=42)
    N_BOOTSTRAP = 10_000
    pvals = []
    variant_results = []

    for v in VARIANTS:
        vid, vrec = find_record(forward_data, BENCH_PREFIX[v], "100000")
        if vrec is None:
            lines.append(f"  WARN: {v} n=100K record not found; skipping.")
            pvals.append(1.0)
            variant_results.append((v, None, None, None, "MISSING"))
            continue

        b_samp = baseline_rec["samples"]
        v_samp = vrec["samples"]

        if b_samp and v_samp:
            b_arr = np.array(b_samp)
            v_arr = np.array(v_samp)
            ratios = np.array([
                np.mean(rng.choice(v_arr, size=len(v_arr))) /
                np.mean(rng.choice(b_arr, size=len(b_arr)))
                for _ in range(N_BOOTSTRAP)
            ])
            p = float(np.mean(ratios >= 1.0))
            ci_boot = (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))
            mean_r = float(np.mean(ratios))
            fallback = False
        else:
            # W5 fallback: aggregate CI bounds unavailable; use point estimates
            lines.append(f"  W5 FALLBACK CI: raw samples unavailable for {v} or baseline.")
            b_point = baseline_rec["point"] or 1.0
            v_point = vrec["point"] or 1.0
            mean_r = v_point / b_point if b_point else float("nan")
            p = 0.5 if abs(mean_r - 1.0) < 0.01 else (1.0 if mean_r >= 1.0 else 0.0)
            ci_boot = (mean_r * 0.95, mean_r * 1.05)
            fallback = True

        pvals.append(p)
        variant_results.append((v, mean_r, ci_boot, p, "FALLBACK" if fallback else "BOOTSTRAP"))

    reject_arr, pvals_adj, _, _ = multipletests(pvals, method="holm")

    lines += [
        f"  Baseline n=100K id: {baseline_id}",
        f"  Bootstrap samples: {N_BOOTSTRAP}",
        "",
        "  | Variant | Mean Ratio | 95% Boot CI | p-value | Holm adj p | Reject H0 | Method |",
        "  |---------|------------|-------------|---------|------------|-----------|--------|",
    ]
    for i, (v, mean_r, ci_b, p, method) in enumerate(variant_results):
        if mean_r is None:
            lines.append(f"  | {v} | N/A | N/A | N/A | N/A | N/A | MISSING |")
        else:
            lines.append(
                f"  | {v} | {mean_r:.4f} | [{ci_b[0]:.4f}, {ci_b[1]:.4f}] "
                f"| {p:.4f} | {pvals_adj[i]:.4f} | {reject_arr[i]} | {method} |"
            )

    # W4: cache warm-state check
    if reversed_data:
        w4_flags = []
        for bprefix in ["tw_combined", "tw_baseline"]:
            _, frec = find_record(forward_data, bprefix, "100000")
            _, rrec = find_record(reversed_data, bprefix, "100000")
            if frec and rrec and frec["point"] and rrec["point"]:
                diff_frac = abs(frec["point"] - rrec["point"]) / frec["point"]
                if diff_frac > 0.05:
                    w4_flags.append(f"{bprefix}: {diff_frac*100:.1f}% difference")
        if w4_flags:
            lines.append(f"  W4 ANOMALY: Cache warm-state bias >5%: {'; '.join(w4_flags)}")
        else:
            lines.append("  W4: No cache warm-state anomaly (forward vs reversed <5% for all checked benchmarks)")
    else:
        lines.append("  W4: criterion_reversed_output.json not found; W4 check skipped.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section H-partial-MERFISH
# ---------------------------------------------------------------------------

def section_h_partial_merfish(dry_run):
    """H-partial-MERFISH: CI half-width Gaussian n=50K vs MERFISH n=50K."""
    lines = ["## H-partial-MERFISH: Partial Rank MERFISH vs Gaussian CI Width (n=50K)"]

    if dry_run:
        lines += ["  DRY RUN — no data loaded.", "  Verdict: INCONCLUSIVE (dry run)"]
        return "\n".join(lines)

    forward_data = extract_criterion_records(os.path.join(CRIT_DIR, "criterion_output.json"))
    merfish_data = extract_criterion_records(os.path.join(CRIT_DIR, "criterion_merfish_output.json"))

    gauss_id, gauss_rec = find_record(forward_data, "tw_partial_rank", "50000")
    mer_id, mer_rec = find_record(merfish_data, "tw_partial_rank_merfish", "50000")

    if gauss_rec is None:
        lines.append("  ERROR: tw_partial_rank n=50K not found in criterion_output.json.")
        return "\n".join(lines)
    if mer_rec is None:
        lines.append("  ERROR: tw_partial_rank_merfish n=50K not found in criterion_merfish_output.json.")
        return "\n".join(lines)

    def ci_half_width(rec):
        lo, hi = rec["ci"]
        if lo is not None and hi is not None:
            return (hi - lo) / 2.0
        return None

    gauss_hw = ci_half_width(gauss_rec)
    mer_hw = ci_half_width(mer_rec)

    if gauss_hw is None or mer_hw is None:
        lines.append("  WARN: CI bounds unavailable; cannot compute half-widths.")
        return "\n".join(lines)

    ratio_hw = mer_hw / gauss_hw if gauss_hw > 0 else float("nan")

    if ratio_hw <= 2.0:
        verdict = "NO ANOMALY (MERFISH CI width ≤ 2× Gaussian)"
    else:
        verdict = (f"ELEVATED VARIANCE (MERFISH CI width {ratio_hw:.1f}× Gaussian; "
                   "investigate data heterogeneity)")

    lines += [
        f"  Gaussian n=50K CI half-width: {gauss_hw:.2f} ns  (id: {gauss_id})",
        f"  MERFISH  n=50K CI half-width: {mer_hw:.2f} ns  (id: {mer_id})",
        f"  MERFISH/Gaussian ratio: {ratio_hw:.2f}",
        f"  Verdict: **{verdict}**",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section H0/H1-clean
# ---------------------------------------------------------------------------

def section_h0_h1_clean(dry_run):
    """H0/H1-clean: step CPU-time fractions — baseline, n=100K, 30 iterations."""
    STEP_KEYS = ["x_dist", "x_sort", "rank_scatter", "x_knn_set", "y_heap", "penalty"]
    # Expected ordering by mean fraction (most expensive first) — W3 descriptive only
    EXPECTED_ORDER = ["x_dist", "x_sort", "rank_scatter", "y_heap", "x_knn_set", "penalty"]

    lines = ["## H0/H1-clean: Step CPU-Time Fractions (Baseline, n=100K)"]

    if dry_run:
        lines += [
            "  DRY RUN — no data loaded.",
            f"  Expected step keys: {STEP_KEYS}",
            "  Verdict: INCONCLUSIVE (dry run)",
        ]
        return "\n".join(lines)

    baseline_path = os.path.join(STEP_DIR, "gaussian_n100000_baseline.json")
    if not os.path.exists(baseline_path):
        lines.append(f"  ERROR: {baseline_path} not found.")
        return "\n".join(lines)

    with open(baseline_path) as f:
        data = json.load(f)

    step_times_ns = data.get("step_times_ns")
    if not step_times_ns:
        lines.append("  ERROR: step_times_ns field missing. Was tw_profiler built with --features profiling?")
        return "\n".join(lines)

    n_iters = len(step_times_ns)
    df = n_iters - 1
    # t(0.975, df=29) ≈ 2.045
    t_crit = float(t_dist.ppf(0.975, df=df))

    fracs_by_step = {k: [] for k in STEP_KEYS}
    for iter_rec in step_times_ns:
        total = sum(iter_rec.get(k, 0) for k in STEP_KEYS)
        if total == 0:
            continue
        for k in STEP_KEYS:
            fracs_by_step[k].append(iter_rec.get(k, 0) / total)

    lines += [
        f"  Iterations: {n_iters}, df={df}, t_crit(0.975,df={df})={t_crit:.4f}",
        "",
        "  | Step | Mean Fraction | 95% t-CI |",
        "  |------|--------------|----------|",
    ]

    mean_fracs = {}
    for k in STEP_KEYS:
        arr = np.array(fracs_by_step[k])
        mean_f = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        hw = t_crit * se
        mean_fracs[k] = mean_f
        lines.append(
            f"  | {k:<14} | {mean_f:.4f} ({mean_f*100:.1f}%) "
            f"| [{mean_f-hw:.4f}, {mean_f+hw:.4f}] |"
        )

    # W3 CI-ordering check (descriptive only — no p-value)
    observed_order = sorted(STEP_KEYS, key=lambda k: mean_fracs[k], reverse=True)
    if observed_order == EXPECTED_ORDER:
        lines.append(f"  W3: Step ordering matches expected: {' > '.join(observed_order)}")
    else:
        lines += [
            "  W3: Step ordering anomaly (descriptive only — no p-value):",
            f"       Expected: {' > '.join(EXPECTED_ORDER)}",
            f"       Observed: {' > '.join(observed_order)}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(ANA_DIR, exist_ok=True)

    if args.dry_run:
        print("=== DRY RUN MODE ===")
        print("Running all sections without requiring result files.")
        print()

    sections = [
        section_h5(args.dry_run),
        section_h100k(args.dry_run),
        section_h_partial_merfish(args.dry_run),
        section_h0_h1_clean(args.dry_run),
    ]

    report_lines = ["# tw-perf-rerun-clean Analysis Report", ""]
    if args.dry_run:
        report_lines += ["> **DRY RUN** — placeholder report; no real data loaded.", ""]
    for section in sections:
        report_lines.append(section)
        report_lines.append("")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(ANA_DIR, "analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
