#!/usr/bin/env python3
"""Analyze Criterion benchmark and profiler outputs for the y_heap bottleneck experiment.

Usage:
    python analyze_results.py [--results-dir PATH] [--output-dir PATH] [--stage1-only]
"""

# CRITICAL: set backend before any other matplotlib import
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

VARIANTS = ["baseline", "heap_reuse", "flat_partial", "flat_simd"]
N_VALUES = [1000, 5000, 10000]
STEPS = ["x_dist", "x_sort", "y_heap", "penalty"]


# ---------------------------------------------------------------------------
# Step 2 — load_criterion_results
# ---------------------------------------------------------------------------

def load_criterion_results(results_dir: Path) -> dict:
    """Parse Criterion estimates JSONs for all variant/n combinations.

    Returns:
        {variant: {n: {mean_ns, speedup, ci_lb, ci_ub, ci_synthetic,
                       ratio_ci_lb, ratio_ci_ub}}}
    Also loads _escalated variants and stores under key "{n}_escalated".
    """
    criterion_dir = results_dir / "criterion"
    raw: dict = {}  # variant -> n -> {mean_ns, ci_lb, ci_ub, ci_synthetic}

    for variant in VARIANTS:
        for n in N_VALUES:
            for suffix, key in [("", n), ("_escalated", f"{n}_escalated")]:
                path = criterion_dir / f"y_heap_{variant}_n{n}{suffix}.json"
                if not path.exists():
                    if suffix == "_escalated":
                        continue  # escalated files are optional, no warning
                    warnings.warn(f"Missing: {path}")
                    continue
                try:
                    with open(path) as f:
                        doc = json.load(f)
                except Exception as exc:
                    warnings.warn(f"Failed to parse {path}: {exc}")
                    continue

                mean_ns = doc["mean"]["point_estimate"]
                ci_obj = doc["mean"].get("confidence_interval")
                if ci_obj and "lower_bound" in ci_obj and "upper_bound" in ci_obj:
                    ci_lb = ci_obj["lower_bound"]
                    ci_ub = ci_obj["upper_bound"]
                    ci_synthetic = False
                else:
                    ci_lb = mean_ns * 0.95
                    ci_ub = mean_ns * 1.05
                    ci_synthetic = True

                raw.setdefault(variant, {})[key] = {
                    "mean_ns": mean_ns,
                    "ci_lb": ci_lb,
                    "ci_ub": ci_ub,
                    "ci_synthetic": ci_synthetic,
                }

    # Second pass: compute speedup and ratio CI bounds
    data: dict = {}
    baseline_raw = raw.get("baseline", {})

    for variant, n_map in raw.items():
        data.setdefault(variant, {})
        for key, entry in n_map.items():
            # Determine the n-value integer for baseline lookup
            n_int = key if isinstance(key, int) else int(str(key).replace("_escalated", ""))
            base_entry = baseline_raw.get(n_int) if not str(key).endswith("_escalated") \
                else baseline_raw.get(f"{n_int}_escalated") or baseline_raw.get(n_int)

            mean_ns = entry["mean_ns"]
            ci_lb_ns = entry["ci_lb"]
            ci_ub_ns = entry["ci_ub"]

            if variant == "baseline":
                speedup = 1.0
                ratio_ci_lb = 1.0
                ratio_ci_ub = 1.0
            elif base_entry is not None:
                base_mean = base_entry["mean_ns"]
                base_ci_lb = base_entry["ci_lb"]
                base_ci_ub = base_entry["ci_ub"]
                speedup = base_mean / mean_ns if mean_ns > 0 else None
                # Conservative ratio CI: base_ci_lb/variant_ci_ub .. base_ci_ub/variant_ci_lb
                ratio_ci_lb = base_ci_lb / ci_ub_ns if ci_ub_ns > 0 else None
                ratio_ci_ub = base_ci_ub / ci_lb_ns if ci_lb_ns > 0 else None
            else:
                speedup = None
                ratio_ci_lb = None
                ratio_ci_ub = None

            data[variant][key] = {
                "mean_ns": mean_ns,
                "ci_lb": ci_lb_ns,
                "ci_ub": ci_ub_ns,
                "ci_synthetic": entry["ci_synthetic"],
                "speedup": speedup,
                "ratio_ci_lb": ratio_ci_lb,
                "ratio_ci_ub": ratio_ci_ub,
            }

    return data


# ---------------------------------------------------------------------------
# Step 3 — compute_hypothesis
# ---------------------------------------------------------------------------

def compute_hypothesis(data: dict, results_dir: Path) -> tuple:
    """Evaluate the primary hypothesis.

    Returns:
        (decision_code, details_dict)

    decision_code: "POSITIVE" | "WEAK_POSITIVE" | "ESCALATE" |
                   "NEGATIVE" | "INCONCLUSIVE" | "NO_DATA"
    """
    entry = data.get("flat_simd", {}).get(10000)
    if entry is None:
        return ("NO_DATA", {"primary_text": "flat_simd n=10000 data absent"})

    point = entry.get("speedup")
    ratio_ci_lb = entry.get("ratio_ci_lb")

    if point is None:
        return ("NO_DATA", {"primary_text": "flat_simd n=10000 speedup could not be computed"})

    details: dict = {
        "speedup": point,
        "ratio_ci_lb": ratio_ci_lb,
        "ratio_ci_ub": entry.get("ratio_ci_ub"),
        "ci_synthetic": entry.get("ci_synthetic", False),
    }

    if entry.get("ci_synthetic", False):
        return ("INCONCLUSIVE", {
            **details,
            "primary_text": (
                "INCONCLUSIVE — CI data is synthetic (±5% fallback); "
                "cannot declare statistical significance from heuristic bounds"
            ),
        })

    if ratio_ci_lb is not None and ratio_ci_lb > 1.0:
        primary = "POSITIVE"
        details["primary_text"] = (
            f"POSITIVE — flat_simd n=10000 speedup {point:.4f}× "
            f"(ratio 95% CI lb {ratio_ci_lb:.4f} > 1.0)"
        )
    elif point >= 1.1:
        primary = "ESCALATE"
        ci_lb_str = f"{ratio_ci_lb:.4f}" if ratio_ci_lb is not None else "N/A"
        details["primary_text"] = (
            f"ESCALATE — point estimate {point:.4f}× ≥ 1.1 but CI lb "
            f"{ci_lb_str} ≤ 1.0; Stage 2 needed"
        )
    else:
        primary = "NEGATIVE"
        details["primary_text"] = (
            f"NEGATIVE — flat_simd n=10000 speedup {point:.4f}× "
            f"(ratio 95% CI lb {ratio_ci_lb})"
        )

    if primary != "ESCALATE":
        return (primary, details)

    # Stage 2: check escalated data
    esc_entry = data.get("flat_simd", {}).get("10000_escalated")
    if esc_entry is None:
        details["stage2"] = "not_run"
        return (primary, details)

    esc_ratio_ci_lb = esc_entry.get("ratio_ci_lb")
    if esc_ratio_ci_lb is not None and esc_ratio_ci_lb > 1.05:
        decision = "WEAK_POSITIVE"
        details["primary_text"] += f"; Stage 2 WEAK_POSITIVE (esc CI lb {esc_ratio_ci_lb:.4f} > 1.05)"
    else:
        decision = "INCONCLUSIVE"
        details["primary_text"] += f"; Stage 2 INCONCLUSIVE (esc CI lb {esc_ratio_ci_lb})"

    details["stage2_ratio_ci_lb"] = esc_ratio_ci_lb
    return (decision, details)


# ---------------------------------------------------------------------------
# Step 4 — build_speedup_table_md
# ---------------------------------------------------------------------------

def build_speedup_table_md(data: dict) -> str:
    """Build a Markdown speedup table for all variants × n combinations."""
    header = (
        "| Variant | n | Mean (ms) | Speedup | CI lb | CI ub | Sig |\n"
        "|---------|---|-----------|---------|-------|-------|-----|"
    )
    rows = [header]
    for variant in VARIANTS:
        for n in N_VALUES:
            entry = data.get(variant, {}).get(n)
            if entry is None:
                rows.append(f"| {variant} | {n} | — | — | — | — | — |")
                continue

            mean_ms = f"{entry['mean_ns'] / 1_000_000:.3f}"
            speedup_str = f"{entry['speedup']:.4f}" if entry.get("speedup") is not None else "—"

            if variant == "baseline":
                ci_lb_str = "1.0000"
                ci_ub_str = "1.0000"
            else:
                rlb = entry.get("ratio_ci_lb")
                rub = entry.get("ratio_ci_ub")
                synth = " (estimated)" if entry.get("ci_synthetic") else ""
                ci_lb_str = (f"{rlb:.4f}{synth}" if rlb is not None else "—")
                ci_ub_str = (f"{rub:.4f}{synth}" if rub is not None else "—")

            rlb = entry.get("ratio_ci_lb")
            sig = "*" if (rlb is not None and rlb > 1.0) else ""

            rows.append(
                f"| {variant} | {n} | {mean_ms} | {speedup_str} | {ci_lb_str} | {ci_ub_str} | {sig} |"
            )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Step 5 — build_causal_table_md
# ---------------------------------------------------------------------------

def build_causal_table_md(data: dict, n: int = 10000) -> str:
    """Compute bundle attribution at n=10000 and return as Markdown table."""

    def _speedup(variant: str):
        return data.get(variant, {}).get(n, {}).get("speedup")

    s_hr = _speedup("heap_reuse")
    s_fp = _speedup("flat_partial")
    s_fs = _speedup("flat_simd")

    alloc_frac = (1 - 1 / s_hr) if s_hr else None
    ds_frac = (1 - s_hr / s_fp) if (s_hr and s_fp) else None
    simd_frac = (1 - s_fp / s_fs) if (s_fp and s_fs) else None

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    caption = "Bundle attribution (W2: conflated bundles, not single-cause isolation)"
    header = (
        f"_{caption}_\n\n"
        "| Bundle | Attribution fraction | n |\n"
        "|--------|----------------------|---|"
    )
    rows = [
        header,
        f"| Allocation (malloc elim.) | {_fmt(alloc_frac)} | {n} |",
        f"| DS change (BTreeMap→Vec) | {_fmt(ds_frac)} | {n} |",
        f"| SIMD (flat layout)       | {_fmt(simd_frac)} | {n} |",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Step 6 — load_profiler_results
# ---------------------------------------------------------------------------

def load_profiler_results(results_dir: Path) -> dict:
    """Load profiler JSON files for all variants at n=10000.

    Returns:
        {variant: raw_dict}  — absent if file missing or unparseable
    """
    profiler_dir = results_dir / "profiler"
    result: dict = {}
    for variant in VARIANTS:
        path = profiler_dir / f"profiler_{variant}_n10000.json"
        if not path.exists():
            warnings.warn(f"Missing profiler: {path}")
            continue
        try:
            with open(path) as f:
                result[variant] = json.load(f)
        except Exception as exc:
            warnings.warn(f"Failed to parse {path}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Step 7 — compute_step_fractions
# ---------------------------------------------------------------------------

def compute_step_fractions(profiler_data: dict) -> dict:
    """Compute per-step timing fractions from profiler data.

    Returns:
        {variant: {step: {mean_ns, std_ns, ci_lb_ns, ci_ub_ns},
                   "y_heap_fraction": float}}
        or {variant: None} if step_timing missing for that variant.
    """
    result: dict = {}
    for variant, raw in profiler_data.items():
        if "step_timing" not in raw:
            result[variant] = None
            continue

        timing = raw["step_timing"]
        step_stats: dict = {}
        step_means: list = []

        for step in STEPS:
            if step not in timing:
                continue
            arr = np.array(timing[step], dtype=float)
            if len(arr) < 2:
                mean = float(arr.mean()) if len(arr) == 1 else 0.0
                step_stats[step] = {
                    "mean_ns": mean,
                    "std_ns": 0.0,
                    "ci_lb_ns": mean,
                    "ci_ub_ns": mean,
                }
                step_means.append(mean)
                continue

            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            se = std / np.sqrt(len(arr))
            ci_lb, ci_ub = stats.t.interval(0.95, df=len(arr) - 1, loc=mean, scale=se)
            step_stats[step] = {
                "mean_ns": mean,
                "std_ns": std,
                "ci_lb_ns": float(ci_lb),
                "ci_ub_ns": float(ci_ub),
            }
            step_means.append(mean)

        if len(step_means) < len(STEPS):
            missing = [s for s in STEPS if s not in step_stats]
            warnings.warn(
                f"{variant}: missing steps {missing} — "
                f"y_heap_fraction may be inflated ({len(step_means)}/{len(STEPS)} steps present)"
            )

        total = sum(step_means)
        y_heap_mean = step_stats.get("y_heap", {}).get("mean_ns", 0.0)
        y_heap_fraction = (y_heap_mean / total) if total > 0 else None

        variant_result = dict(step_stats)
        variant_result["y_heap_fraction"] = y_heap_fraction
        result[variant] = variant_result

    return result


# ---------------------------------------------------------------------------
# Step 8 — build_step_fractions_md
# ---------------------------------------------------------------------------

def build_step_fractions_md(step_fracs: dict, criterion_data: dict) -> str:
    """Build a Markdown table of per-step timing fractions."""
    note = "_per-call wall-clock step fraction (profiling feature enabled)_"
    header = (
        f"{note}\n\n"
        "| Variant | x_dist (ms) | x_sort (ms) | y_heap (ms) | penalty (ms) | y_heap % |\n"
        "|---------|-------------|-------------|-------------|--------------|----------|"
    )
    rows = [header]
    warnings_list: list = []

    for variant in VARIANTS:
        fracs = step_fracs.get(variant)
        if fracs is None:
            rows.append(f"| {variant} | n/a | n/a | n/a | n/a | n/a |")
            continue

        def _ms(step: str) -> str:
            s = fracs.get(step)
            if s is None:
                return "—"
            return f"{s['mean_ns'] / 1_000_000:.3f}"

        y_frac = fracs.get("y_heap_fraction")
        y_pct = f"{y_frac * 100:.1f}" if y_frac is not None else "—"

        rows.append(
            f"| {variant} | {_ms('x_dist')} | {_ms('x_sort')} | {_ms('y_heap')} "
            f"| {_ms('penalty')} | {y_pct} |"
        )

        if fracs.get("warning"):
            warnings_list.append(f"> **{variant}:** {fracs['warning']}")

    result = "\n".join(rows)
    if warnings_list:
        result += "\n\n" + "\n".join(warnings_list)
    return result


# ---------------------------------------------------------------------------
# Step 9 — build_correctness_block
# ---------------------------------------------------------------------------

def build_correctness_block(results_dir: Path) -> str:
    """Return correctness checklist, prepending data_verification.txt if present."""
    fixed = (
        "Run `cargo test --features testing` and confirm t_tw_01–t_tw_07 pass for all variants "
        "with |ΔT| < 1e-12."
    )
    dv_path = results_dir / "data_verification.txt"
    if dv_path.exists():
        try:
            dv_text = dv_path.read_text()
            return f"```\n{dv_text.strip()}\n```\n\n{fixed}"
        except Exception:
            pass
    return fixed


# ---------------------------------------------------------------------------
# Step 10 — build_shipping_block
# ---------------------------------------------------------------------------

def build_shipping_block(decision: str, details: dict, data: dict = None) -> str:
    """Return a shipping recommendation based on the hypothesis decision."""
    if data is None:
        data = {}

    if decision == "POSITIVE":
        # Check if heap_reuse also shows significant speedup and overlapping CIs
        hr_entry = data.get("heap_reuse", {}).get(10000, {})
        fs_entry = data.get("flat_simd", {}).get(10000, {})
        hr_ci_lb = hr_entry.get("ratio_ci_lb")
        fs_ci_ub = fs_entry.get("ratio_ci_ub")
        hr_ci_ub = hr_entry.get("ratio_ci_ub")
        fs_ci_lb = fs_entry.get("ratio_ci_lb")

        if (hr_ci_lb is not None and hr_ci_lb > 1.0
                and fs_ci_lb is not None and hr_ci_ub is not None
                and fs_ci_lb <= hr_ci_ub):
            return (
                "**SHIP `heap_reuse`** — both `heap_reuse` and `flat_simd` show significant "
                "speedup and their CIs overlap; prefer `heap_reuse` (simpler implementation, "
                "no SIMD dependency).\n\n"
                "Fallback: `flat_simd` if `heap_reuse` regresses in future profiling."
            )
        return (
            "**SHIP `flat_simd`** — CI lower bound > 1.0 confirms statistically significant "
            "speedup at n=10000. Apply `flat_simd` variant to production."
        )

    if decision == "WEAK_POSITIVE":
        return (
            "**SHIP with caveat** — `flat_simd` shows WEAK_POSITIVE in two-stage analysis. "
            "Be aware of elevated Type I error risk from the two-stage design (W8). "
            "Consider additional profiling before committing."
        )

    if decision == "ESCALATE":
        return (
            "**DEFER** — Stage 2 (escalated benchmark) not yet run. "
            "Execute Stage 2 before making a shipping decision."
        )

    # INCONCLUSIVE, NEGATIVE, NO_DATA
    return (
        "**RECOMMEND H3 (KD-tree)** — `flat_simd` does not show a statistically significant "
        "speedup. The dominant root cause from causal decomposition should guide the next "
        "optimization target. Consider switching to a KD-tree–based neighbor search."
    )


# ---------------------------------------------------------------------------
# Step 11 — plot_speedup_chart
# ---------------------------------------------------------------------------

def plot_speedup_chart(data: dict, output_dir: Path) -> None:
    variants_plot = ["heap_reuse", "flat_partial", "flat_simd"]
    n_labels = ["1K", "5K", "10K"]
    n_values = [1000, 5000, 10000]

    x = np.arange(len(n_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="baseline (1.0×)")

    for i, variant in enumerate(variants_plot):
        means, err_lo, err_hi = [], [], []
        for n in n_values:
            entry = data.get(variant, {}).get(n)
            if entry and entry.get("speedup") is not None:
                sp = entry["speedup"]
                lb = entry.get("ratio_ci_lb", sp * 0.95)
                ub = entry.get("ratio_ci_ub", sp * 1.05)
                means.append(sp)
                err_lo.append(sp - lb)
                err_hi.append(ub - sp)
            else:
                means.append(np.nan)
                err_lo.append(0.0)
                err_hi.append(0.0)

        mask = ~np.isnan(means)
        positions = x[mask] + (i - 1) * width
        m_arr = np.array(means)[mask]
        el = np.array(err_lo)[mask]
        eu = np.array(err_hi)[mask]

        if m_arr.size > 0:
            ax.bar(
                positions, m_arr, width,
                yerr=[el, eu] if el.any() or eu.any() else None,
                label=variant, capsize=4, error_kw={"linewidth": 1.2},
            )

    ax.set_xticks(x)
    ax.set_xticklabels(n_labels)
    ax.set_xlabel("n")
    ax.set_ylabel("Speedup ratio vs baseline")
    ax.set_title("y_heap Variant Speedup vs Baseline")
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "speedup_ratios.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 12 — write_report
# ---------------------------------------------------------------------------

def write_report(sections: dict, output_dir: Path, metadata: dict) -> None:
    """Assemble and write analysis_report.md."""
    hw = metadata.get("hardware", "n/a")
    n_vals = metadata.get("n_values", N_VALUES)
    k = metadata.get("k", 15)
    dt = metadata.get("date", _today())

    body = f"""# y_heap Bottleneck Optimization — Analysis Report

**Date:** {dt}  **n:** {n_vals}  **k:** {k}  **Hardware:** {hw}

## Primary Result

{sections.get("primary_result", "—")}

## Speedup Table

{sections.get("speedup_table", "—")}

## Causal Decomposition

{sections.get("causal_table", "—")}

## Step Fractions

{sections.get("step_fractions", "—")}

## Correctness

{sections.get("correctness", "—")}

## Shipping Decision

{sections.get("shipping_decision", "—")}

## Threats to Validity

See experiment plan Analysis Plan section (W1–W8).
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_report.md").write_text(body)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_hardware_profile(results_dir: Path) -> str:
    path = results_dir / "hardware_profile.txt"
    if path.exists():
        try:
            return path.read_text().strip()
        except Exception:
            pass
    return "n/a"


def _format_primary_result(decision: str, details: dict) -> str:
    text = details.get("primary_text", decision)
    return f"**Decision:** `{decision}`\n\n{text}"


def _today() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Step 13 — main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/", type=Path)
    parser.add_argument("--output-dir",  default="results/analysis/", type=Path)
    parser.add_argument("--stage1-only", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir  = args.output_dir

    data = load_criterion_results(results_dir)
    decision, details = compute_hypothesis(data, results_dir)

    if args.stage1_only:
        print(details.get("primary_text", decision))
        sys.exit(0)

    profiler_data = load_profiler_results(results_dir)
    step_fracs    = compute_step_fractions(profiler_data)

    hardware = _read_hardware_profile(results_dir)

    sections = {
        "primary_result":    _format_primary_result(decision, details),
        "speedup_table":     build_speedup_table_md(data),
        "causal_table":      build_causal_table_md(data, n=10000),
        "step_fractions":    build_step_fractions_md(step_fracs, data),
        "correctness":       build_correctness_block(results_dir),
        "shipping_decision": build_shipping_block(decision, details, data),
    }
    metadata = {"date": _today(), "n_values": N_VALUES, "k": 15, "hardware": hardware}

    output_dir.mkdir(parents=True, exist_ok=True)
    write_report(sections, output_dir, metadata)
    plot_speedup_chart(data, output_dir)

    print(f"Report written to: {output_dir / 'analysis_report.md'}")
    print(f"Chart written to:  {output_dir / 'speedup_ratios.png'}")


if __name__ == "__main__":
    main()
