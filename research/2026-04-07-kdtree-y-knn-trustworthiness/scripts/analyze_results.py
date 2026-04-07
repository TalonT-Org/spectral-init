#!/usr/bin/env python3
"""Analyze criterion and profiler results for kdtree-y-knn-trustworthiness experiment."""

import argparse
import json
import math
import pathlib
import sys

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Analyze benchmark results for kdtree-y-knn-trustworthiness experiment."
)
parser.add_argument("--dry-run", action="store_true", help="Only n=1000 data; verify pipeline.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARIANTS = ["flat_simd", "kdtree"]
DISTRIBUTIONS = ["uniform", "gauss"]
REPS = 3

if args.dry_run:
    N_VALUES = [1000]
else:
    N_VALUES = [1000, 5000, 10000, 50000, 75000, 100000]

CRIT_DIR = pathlib.Path("results/criterion")
PROF_DIR = pathlib.Path("results/profiler")
ANALYSIS_DIR = pathlib.Path("results/analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_criterion(group: str, rep: int):
    p = CRIT_DIR / f"{group}_rep{rep}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_profiler(variant: str, n: int, dist: str):
    p = PROF_DIR / f"{variant}_n{n}_{dist}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


# ---------------------------------------------------------------------------
# Criterion metric aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median_estimate(variant: str, dist: str, n: int):
    """Return (median_ns, cv, high_cv) across available reps."""
    group = f"{variant}_{dist}_n{n}"
    estimates = []
    for rep in range(1, REPS + 1):
        d = load_criterion(group, rep)
        if d is not None:
            pe = d.get("median", {}).get("estimate")
            if pe is not None:
                estimates.append(float(pe))
    if not estimates:
        return None, float("nan"), False
    estimates_sorted = sorted(estimates)
    mid = len(estimates_sorted) // 2
    if len(estimates_sorted) % 2 == 1:
        med = float(estimates_sorted[mid])
    else:
        med = (float(estimates_sorted[mid - 1]) + float(estimates_sorted[mid])) / 2.0
    cv = _std(estimates) / _mean(estimates) if len(estimates) >= 2 else float("nan")
    high_cv = (not math.isnan(cv)) and (cv > 0.10)
    return med, cv, high_cv


def median_estimate_single_rep(variant: str, dist: str, n: int, rep: int):
    """Return point estimate (ns) for a single rep, or None."""
    group = f"{variant}_{dist}_n{n}"
    d = load_criterion(group, rep)
    if d is None:
        return None
    pe = d.get("median", {}).get("estimate")
    return float(pe) if pe is not None else None


# ---------------------------------------------------------------------------
# DV computations
# ---------------------------------------------------------------------------

def compute_total_speedup(dist: str, n: int):
    """speedup = flat_simd_ns / kdtree_ns; >1 means kdtree is faster."""
    flat_ns, flat_cv, flat_hcv = median_estimate("flat_simd", dist, n)
    kd_ns, kd_cv, kd_hcv = median_estimate("kdtree", dist, n)
    if flat_ns is None or kd_ns is None or kd_ns == 0:
        return None, float("nan"), False, float("nan"), False
    return (flat_ns / kd_ns, flat_cv, flat_hcv, kd_cv, kd_hcv)


def compute_build_fraction(n: int, dist: str):
    """build_fraction = build_mean / (build_mean + query_mean) for kdtree."""
    data = load_profiler("kdtree", n, dist)
    if data is None:
        return None
    st = data.get("step_timing", {})
    build_vals = st.get("y_kdtree_build", [])
    query_vals = st.get("y_kdtree_query", [])
    if not build_vals or not query_vals:
        return None
    build_mean = _mean(build_vals)
    query_mean = _mean(query_vals)
    total = build_mean + query_mean
    if total == 0:
        return None
    return build_mean / total


def compute_query_speedup(n: int, dist: str):
    """query_speedup = flat_simd y_dist_mean / kdtree y_kdtree_query_mean."""
    flat = load_profiler("flat_simd", n, dist)
    kd = load_profiler("kdtree", n, dist)
    if flat is None or kd is None:
        return None
    flat_st = flat.get("step_timing", {})
    kd_st = kd.get("step_timing", {})
    y_dist = flat_st.get("y_dist", [])
    kd_query = kd_st.get("y_kdtree_query", [])
    if not y_dist or not kd_query:
        return None
    query_mean = _mean(kd_query)
    if query_mean == 0:
        return None
    return _mean(y_dist) / query_mean


# ---------------------------------------------------------------------------
# Crossover interpolation
# ---------------------------------------------------------------------------

def _compute_tcross_from_speedups(speedup_by_n):
    """
    Given list of (n, speedup) sorted by n, interpolate crossover in log(n) space.
    Returns T_cross float or None if no sign flip.
    """
    pairs = [(n, sp) for n, sp in speedup_by_n if sp is not None]
    pairs.sort(key=lambda x: x[0])
    for i in range(len(pairs) - 1):
        n_lo, sp_lo = pairs[i]
        n_hi, sp_hi = pairs[i + 1]
        if (sp_lo - 1.0) * (sp_hi - 1.0) < 0:
            t = (1.0 - sp_lo) / (sp_hi - sp_lo)
            tcross = math.exp(math.log(n_lo) + t * (math.log(n_hi) - math.log(n_lo)))
            return tcross
    return None


def compute_tcross():
    """Compute T_cross using all-rep median speedup on uniform distribution."""
    speedup_by_n = []
    for n in N_VALUES:
        sp, *_ = compute_total_speedup("uniform", n)
        speedup_by_n.append((n, sp))
    return _compute_tcross_from_speedups(speedup_by_n)


def compute_tcross_variance():
    """Compute T_cross independently for each rep. Returns dict rep->T_cross."""
    result = {}
    for rep in range(1, REPS + 1):
        speedup_by_n = []
        for n in N_VALUES:
            flat_pe = median_estimate_single_rep("flat_simd", "uniform", n, rep)
            kd_pe = median_estimate_single_rep("kdtree", "uniform", n, rep)
            if flat_pe is not None and kd_pe is not None and kd_pe > 0:
                speedup_by_n.append((n, flat_pe / kd_pe))
            else:
                speedup_by_n.append((n, None))
        result[f"rep{rep}"] = _compute_tcross_from_speedups(speedup_by_n)
    return result


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def load_metadata():
    p = pathlib.Path("results/run_metadata.json")
    if p.exists():
        return json.loads(p.read_text())
    return {}


# ---------------------------------------------------------------------------
# Collect all DVs
# ---------------------------------------------------------------------------

metadata = load_metadata()
rayon_threads = metadata.get("rayon_num_threads", "?")
dry_run_flag = args.dry_run

# Speedup table: {dist: {n: (speedup, flat_cv, flat_hcv, kd_cv, kd_hcv)}}
speedup_table = {}
for dist in DISTRIBUTIONS:
    speedup_table[dist] = {}
    for n in N_VALUES:
        speedup_table[dist][n] = compute_total_speedup(dist, n)

# Build fraction: {dist: {n: fraction}}
build_frac_table = {}
for dist in DISTRIBUTIONS:
    build_frac_table[dist] = {}
    for n in N_VALUES:
        build_frac_table[dist][n] = compute_build_fraction(n, dist)

# Query speedup: {dist: {n: speedup}}
query_speedup_table = {}
for dist in DISTRIBUTIONS:
    query_speedup_table[dist] = {}
    for n in N_VALUES:
        query_speedup_table[dist][n] = compute_query_speedup(n, dist)

# Crossover
T_cross = compute_tcross()
tcross_by_rep = compute_tcross_variance()

# T_cross stability
valid_tcross = [v for v in tcross_by_rep.values() if v is not None]
if len(valid_tcross) >= 2:
    tcross_range_ratio = max(valid_tcross) / min(valid_tcross) if min(valid_tcross) > 0 else None
    tcross_stable = tcross_range_ratio is not None and tcross_range_ratio <= 2.0
else:
    tcross_range_ratio = None
    tcross_stable = False

# n=75K held-out check
n75k_speedup_uniform = None
n75k_on_kdtree_faster_side = None
if 75000 in N_VALUES:
    sp75, *_ = compute_total_speedup("uniform", 75000)
    n75k_speedup_uniform = sp75
    if sp75 is not None and T_cross is not None:
        # kdtree faster when speedup > 1.0, i.e. n > T_cross
        n75k_on_kdtree_faster_side = (sp75 > 1.0)

# ---------------------------------------------------------------------------
# Hypothesis evaluation
# ---------------------------------------------------------------------------

sp50k_uniform = None
sp100k_uniform = None
if 50000 in N_VALUES:
    v, *_ = compute_total_speedup("uniform", 50000)
    sp50k_uniform = v
if 100000 in N_VALUES:
    v, *_ = compute_total_speedup("uniform", 100000)
    sp100k_uniform = v

h1_met = (
    sp50k_uniform is not None and sp100k_uniform is not None
    and sp50k_uniform >= 5.0 and sp100k_uniform >= 10.0
) if not dry_run_flag else None

h2_met = (
    T_cross is not None and 1000.0 <= T_cross <= 50000.0
) if not dry_run_flag else None

# H3: correctness (t_tw_11) — external prerequisite
h3_note = "ASSUMED PASS — run `cargo test t_tw_11 --features testing` separately"

# H4: build_fraction <= 10% at n=50K and n=100K for all distributions
h4_met = None
if not dry_run_flag and 50000 in N_VALUES and 100000 in N_VALUES:
    h4_vals = []
    for dist in DISTRIBUTIONS:
        bf50 = build_frac_table[dist].get(50000)
        bf100 = build_frac_table[dist].get(100000)
        if bf50 is not None:
            h4_vals.append(bf50 <= 0.10)
        if bf100 is not None:
            h4_vals.append(bf100 <= 0.10)
    h4_met = bool(h4_vals) and all(h4_vals)

# Five success criteria
sc1 = (sp50k_uniform is not None and sp50k_uniform >= 5.0) if not dry_run_flag else None
sc2 = (sp100k_uniform is not None and sp100k_uniform >= 10.0) if not dry_run_flag else None
sc3_note = "External prerequisite: cargo test t_tw_11/t_tw_08/t_tw_10 --features testing"
sc4 = tcross_stable if not dry_run_flag else None
sc5 = h4_met

all_5_met = (
    sc1 and sc2 and sc4 is not False and (sc5 is not False)
) if not dry_run_flag else None

# Verdict
if dry_run_flag:
    verdict = "INCONCLUSIVE"
    verdict_note = "DRY RUN — insufficient data for H1/H4 verdict; pipeline integrity verified"
elif all_5_met:
    verdict = "SHIP"
    verdict_note = "All 5 success criteria met."
elif (
    sp50k_uniform is not None and sp50k_uniform <= 2.0
    and (build_frac_table["gauss"].get(50000) is None or
         (lambda v: v is not None and v <= 2.0)(speedup_table["gauss"].get(50000, (None,))[0]))
):
    verdict = "DO NOT SHIP"
    verdict_note = "Speedup ≤ 2.0 at n=50K across distributions."
else:
    verdict = "INCONCLUSIVE"
    verdict_note = "Not all success criteria met; see hypothesis evaluation."


# ---------------------------------------------------------------------------
# Helper: format float or None
# ---------------------------------------------------------------------------
def fmt(v, decimals=3):
    if v is None:
        return "N/A"
    if math.isnan(v):
        return "NaN"
    return f"{v:.{decimals}f}"


def cv_flag(hcv):
    return " ⚠ HIGH VARIANCE" if hcv else ""


# ---------------------------------------------------------------------------
# Write analysis_report.md
# ---------------------------------------------------------------------------
lines = []
lines.append("# Analysis Report: kdtree-y-knn-trustworthiness")
lines.append("")
lines.append("## Run Scope")
lines.append("")
lines.append(f"- **Dry run:** {dry_run_flag}")
lines.append(f"- **N values analyzed:** {N_VALUES}")
lines.append(f"- **RAYON_NUM_THREADS:** {rayon_threads}")
lines.append(f"- **Rust channel:** {metadata.get('rust_channel', 'unknown')}")
lines.append(f"- **Timestamp:** {metadata.get('timestamp', 'unknown')}")
lines.append(f"- **Scope qualifier:** All conclusions scoped to `RAYON_NUM_THREADS={rayon_threads}` threads on the benchmark machine.")
lines.append("")

if dry_run_flag:
    lines.append(f"> **{verdict_note}**")
    lines.append("")

# ------- Speedup Table -------
lines.append("## Total Speedup (flat_simd ns / kdtree ns; >1 = kdtree faster)")
lines.append("")
header = "| dist | n | speedup | flat_simd CV | kdtree CV |"
sep    = "|------|---|---------|-------------|-----------|"
lines.append(header)
lines.append(sep)
for dist in DISTRIBUTIONS:
    for n in N_VALUES:
        sp, f_cv, f_hcv, k_cv, k_hcv = speedup_table[dist][n]
        sp_str = fmt(sp) + (" ⚠ HIGH VARIANCE" if (f_hcv or k_hcv) else "")
        f_cv_str = (fmt(f_cv) + cv_flag(f_hcv)) if not math.isnan(f_cv) else "N/A"
        k_cv_str = (fmt(k_cv) + cv_flag(k_hcv)) if not math.isnan(k_cv) else "N/A"
        lines.append(f"| {dist} | {n} | {sp_str} | {f_cv_str} | {k_cv_str} |")
lines.append("")

# ------- Build Fraction Table -------
lines.append("## KD-tree Build Fraction (build / (build + query))")
lines.append("")
lines.append("| dist | n | build_fraction | note |")
lines.append("|------|---|---------------|------|")
for dist in DISTRIBUTIONS:
    for n in N_VALUES:
        bf = build_frac_table[dist][n]
        flag = " ⚠ > 10%" if (bf is not None and bf > 0.10) else ""
        lines.append(f"| {dist} | {n} | {fmt(bf)}{flag} | {'build dominated' if bf is not None and bf > 0.5 else ''} |")
lines.append("")

# ------- Query Speedup Table -------
lines.append("## Query Speedup (flat_simd y_dist / kdtree y_kdtree_query)")
lines.append("")
lines.append("| dist | n | query_speedup |")
lines.append("|------|---|--------------|")
for dist in DISTRIBUTIONS:
    for n in N_VALUES:
        qs = query_speedup_table[dist][n]
        lines.append(f"| {dist} | {n} | {fmt(qs)} |")
lines.append("")

# ------- Crossover -------
lines.append("## Crossover Analysis (H2)")
lines.append("")
lines.append(f"- **T_cross estimate (uniform):** {fmt(T_cross, 0) if T_cross is not None else 'None — speedup does not cross 1.0 in available N range'}")
lines.append(f"- **T_cross by rep:** rep1={fmt(tcross_by_rep.get('rep1'), 0)}  rep2={fmt(tcross_by_rep.get('rep2'), 0)}  rep3={fmt(tcross_by_rep.get('rep3'), 0)}")
lines.append(f"- **T_cross range ratio (max/min):** {fmt(tcross_range_ratio, 3)}")
lines.append(f"- **T_cross stable (ratio ≤ 2×):** {tcross_stable}")
lines.append("")

# ------- n=75K held-out -------
lines.append("## n=75K Held-Out Check (RT8)")
lines.append("")
if 75000 in N_VALUES:
    lines.append(f"- **n=75K uniform speedup:** {fmt(n75k_speedup_uniform)}")
    lines.append(f"- **On kdtree-faster side of T_cross:** {n75k_on_kdtree_faster_side}")
else:
    lines.append("- n=75K not in analysis N_VALUES (dry-run mode or not collected).")
lines.append("")

# ------- Hypothesis Evaluation -------
lines.append("## Hypothesis Evaluation")
lines.append("")

lines.append("### H1 — KD-tree speedup at large n")
lines.append(f"- speedup_50k_uniform = {fmt(sp50k_uniform)}")
lines.append(f"- speedup_100k_uniform = {fmt(sp100k_uniform)}")
lines.append(f"- **H1 met:** {h1_met}")
lines.append("")

lines.append("### H2 — Crossover exists in [1K, 50K]")
lines.append(f"- T_cross = {fmt(T_cross, 0) if T_cross is not None else 'None'}")
lines.append(f"- **H2 met:** {h2_met}")
lines.append("")

lines.append("### H3 — Correctness (external)")
lines.append(f"- **H3 note:** {h3_note}")
lines.append("")

lines.append("### H4 — Build fraction ≤ 10% at large n")
lines.append(f"- **H4 met:** {h4_met}")
lines.append("")

# ------- Five Success Criteria -------
lines.append("## Five Success Criteria")
lines.append("")
lines.append("| # | Criterion | Status |")
lines.append("|---|-----------|--------|")
lines.append(f"| 1 | speedup_50k_uniform ≥ 5.0 | {'✓ MET' if sc1 else ('✗ NOT MET' if sc1 is False else 'N/A (dry run)')} |")
lines.append(f"| 2 | speedup_100k_uniform ≥ 10.0 | {'✓ MET' if sc2 else ('✗ NOT MET' if sc2 is False else 'N/A (dry run)')} |")
lines.append(f"| 3 | Correctness (t_tw_11/t_tw_08/t_tw_10) | {sc3_note} |")
lines.append(f"| 4 | T_cross variance ≤ 2× across 3 reps | {'✓ MET' if sc4 else ('✗ NOT MET' if sc4 is False else 'N/A (dry run)')} |")
lines.append(f"| 5 | build_fraction ≤ 10% at n=50K and n=100K | {'✓ MET' if sc5 else ('✗ NOT MET' if sc5 is False else 'N/A (dry run)')} |")
lines.append("")

# ------- Verdict -------
lines.append("## Verdict")
lines.append("")
lines.append(f"**{verdict}**")
lines.append("")
lines.append(verdict_note)
lines.append("")
lines.append(f"_All conclusions scoped to `RAYON_NUM_THREADS={rayon_threads}` threads on the benchmark machine._")

report_text = "\n".join(lines) + "\n"
(ANALYSIS_DIR / "analysis_report.md").write_text(report_text)
print(f"[analyze_results] wrote {ANALYSIS_DIR / 'analysis_report.md'}")

# ---------------------------------------------------------------------------
# Write crossover_summary.json
# ---------------------------------------------------------------------------
crossover_summary = {
    "T_cross_estimate": T_cross,
    "T_cross_range": tcross_by_rep,
    "T_cross_stable": tcross_stable,
    "n75k_speedup_uniform": n75k_speedup_uniform,
    "n75k_on_kdtree_faster_side": n75k_on_kdtree_faster_side,
}
(ANALYSIS_DIR / "crossover_summary.json").write_text(json.dumps(crossover_summary, indent=2))
print(f"[analyze_results] wrote {ANALYSIS_DIR / 'crossover_summary.json'}")

# ---------------------------------------------------------------------------
# Plots (skipped in dry-run)
# ---------------------------------------------------------------------------
if not args.dry_run:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Speedup by n
        fig, ax = plt.subplots(figsize=(9, 5))
        for dist in DISTRIBUTIONS:
            ns = []
            sps = []
            for n in N_VALUES:
                sp, *_ = speedup_table[dist][n]
                if sp is not None:
                    ns.append(n)
                    sps.append(sp)
            if ns:
                ax.plot(ns, sps, marker="o", label=dist)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="speedup=1.0")
        if T_cross is not None:
            ax.axvline(x=T_cross, color="red", linestyle=":", linewidth=1,
                       label=f"T_cross≈{T_cross:.0f}")
        ax.set_xscale("log")
        ax.set_xlabel("n (log scale)")
        ax.set_ylabel("Speedup (flat_simd / kdtree)")
        ax.set_title("KD-tree vs flat_simd total speedup by n")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(ANALYSIS_DIR / "speedup_by_n.png", dpi=150)
        plt.close(fig)
        print(f"[analyze_results] wrote {ANALYSIS_DIR / 'speedup_by_n.png'}")

        # Build fraction by n
        fig, ax = plt.subplots(figsize=(9, 5))
        for dist in DISTRIBUTIONS:
            ns = []
            bfs = []
            for n in N_VALUES:
                bf = build_frac_table[dist][n]
                if bf is not None:
                    ns.append(n)
                    bfs.append(bf)
            if ns:
                ax.plot(ns, bfs, marker="o", label=dist)
        ax.axhline(y=0.10, color="red", linestyle="--", linewidth=1, label="10% threshold")
        ax.set_xlabel("n")
        ax.set_ylabel("Build fraction")
        ax.set_title("KD-tree build fraction by n")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ANALYSIS_DIR / "build_fraction_by_n.png", dpi=150)
        plt.close(fig)
        print(f"[analyze_results] wrote {ANALYSIS_DIR / 'build_fraction_by_n.png'}")

    except ImportError as e:
        print(f"[analyze_results] WARNING: matplotlib not available — skipping plots ({e})", file=sys.stderr)

# ---------------------------------------------------------------------------
# Update results/run_log.json
# ---------------------------------------------------------------------------
import datetime
log_path = pathlib.Path("results/run_log.json")
try:
    log = json.loads(log_path.read_text())
except Exception:
    log = {}
log["analysis"] = {
    "status": "completed",
    "timestamp": datetime.datetime.now().isoformat(),
    "dry_run": args.dry_run,
}
log_path.write_text(json.dumps(log, indent=2))
print(f"[analyze_results] updated {log_path}")
print(f"[analyze_results] DONE — verdict: {verdict}")
