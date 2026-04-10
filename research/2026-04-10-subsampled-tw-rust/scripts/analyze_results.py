"""Analyze subsampled trustworthiness experiment results.

Loads trial JSON from results/raw/, computes per-cell statistics,
evaluates hypotheses H1-H6 with statistical tests, and writes
verdicts.json, summary.md, and three PNG plots.

Usage:
    micromamba run -n subsampled-tw-rust python scripts/analyze_results.py
"""

import datetime
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    EXPROOT,
    M_VALUES,
    PYTHON_SPEEDUP_10K,
)


# -- Data loading -------------------------------------------------------------


def load_trials(raw_dir):
    """Glob *.json from raw_dir, parse, group by mode."""
    grouped = {"exact": [], "subsample": [], "sanity": []}
    files = sorted(raw_dir.glob("*.json"))
    for f in files:
        try:
            trial = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: skipping {f.name}: {e}", file=sys.stderr)
            continue
        mode = trial.get("mode")
        if mode in grouped:
            grouped[mode].append(trial)
        else:
            print(f"WARNING: unknown mode '{mode}' in {f.name}", file=sys.stderr)

    for mode, trials in grouped.items():
        print(f"  Loaded {len(trials)} {mode} trials", file=sys.stderr)
    return grouped


# -- Per-cell statistics ------------------------------------------------------


def compute_cell_stats(subsample_trials):
    """Per-(n, m) cell: mean|dT|, max|dT|, std(T_sub), count, speedup."""
    cells = {}
    for t in subsample_trials:
        key = (t["n"], t["m"])
        cells.setdefault(key, []).append(t)

    result = {}
    for key, trials in cells.items():
        abs_deltas = np.array([t["abs_delta_t"] for t in trials])
        t_subs = np.array([t["t_sub"] for t in trials])

        # Speedup: median wall time within each trial, then ratio
        wall_exacts = np.array([np.median(t["wall_exact_ms"]) for t in trials])
        wall_subs = np.array([np.median(t["wall_sub_ms"]) for t in trials])
        median_wall_exact = np.median(wall_exacts)
        median_wall_sub = np.median(wall_subs)
        speedup = median_wall_exact / median_wall_sub if median_wall_sub > 0 else float("inf")

        mean_abs = float(np.mean(abs_deltas))
        std_abs = float(np.std(abs_deltas, ddof=1)) if len(abs_deltas) > 1 else 0.0

        # Outlier reporting (seed protocol): flag but include
        if len(abs_deltas) > 2 and std_abs > 0:
            threshold = mean_abs + 3 * std_abs
            outliers = [i for i, v in enumerate(abs_deltas) if v > threshold]
            if outliers:
                n, m = key
                print(
                    f"  WARNING: outliers at (n={n}, m={m}), seed indices {outliers}: "
                    f"values {abs_deltas[outliers].tolist()}",
                    file=sys.stderr,
                )

        result[key] = {
            "mean_abs_delta_t": mean_abs,
            "max_abs_delta_t": float(np.max(abs_deltas)),
            "std_t_sub": float(np.std(t_subs, ddof=1)) if len(t_subs) > 1 else 0.0,
            "count": len(trials),
            "median_wall_exact_ms": float(median_wall_exact),
            "median_wall_sub_ms": float(median_wall_sub),
            "speedup_ratio": float(speedup),
        }
    return result


# -- Hypothesis tests ---------------------------------------------------------


def test_h1(cell_stats, subsample_trials):
    """One-sample t-test: mean|dT| at (n=10K, m=2000) < 0.01, one-sided a=0.025."""
    target_trials = [t for t in subsample_trials if t["n"] == 10000 and t["m"] == 2000]
    if len(target_trials) < 3:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": f"Only {len(target_trials)} trials at (n=10000, m=2000), need >= 3",
        }

    abs_deltas = np.array([t["abs_delta_t"] for t in target_trials])
    mean_val = float(np.mean(abs_deltas))
    sem = float(np.std(abs_deltas, ddof=1) / np.sqrt(len(abs_deltas)))

    result = stats.ttest_1samp(abs_deltas, popmean=0.01, alternative="less")
    t_crit = stats.t.ppf(0.975, df=len(abs_deltas) - 1)
    ci_upper = mean_val + t_crit * sem

    return {
        "verdict": "PASS" if result.pvalue < 0.025 else "FAIL",
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_upper_97_5": float(ci_upper),
        "mean_abs_delta_T": mean_val,
        "max_abs_delta_T": float(np.max(abs_deltas)),
        "n_seeds": len(target_trials),
        "secondary_threshold_0.003": bool(mean_val < 0.003),
    }


def test_h2(cell_stats):
    """Per-stratum OLS: speedup_ratio ~ n/m, bootstrap R-squared."""
    strata_results = {}
    all_pass = True

    for n_val in [10000, 50000]:
        m_values = M_VALUES.get(n_val, [])
        x_vals = []
        y_vals = []
        for m_val in m_values:
            key = (n_val, m_val)
            if key in cell_stats:
                x_vals.append(n_val / m_val)
                y_vals.append(cell_stats[key]["speedup_ratio"])

        if len(x_vals) < 3:
            strata_results[str(n_val)] = {
                "verdict": "INSUFFICIENT_DATA",
                "reason": f"Only {len(x_vals)} m-values with data for n={n_val}, need >= 3",
            }
            all_pass = False
            continue

        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)

        # Linear OLS: speedup ~ n/m
        slope_lin, intercept_lin, r_lin, _, _ = stats.linregress(x_arr, y_arr)
        r2_lin = r_lin ** 2
        y_pred_lin = slope_lin * x_arr + intercept_lin
        rmse_lin = float(np.sqrt(np.mean((y_arr - y_pred_lin) ** 2)))

        # Log-linear OLS: log(speedup) ~ log(n/m)
        log_x = np.log(x_arr)
        log_y = np.log(np.maximum(y_arr, 1e-10))
        slope_log, intercept_log, r_log, _, _ = stats.linregress(log_x, log_y)
        r2_log = r_log ** 2
        y_pred_log = np.exp(slope_log * log_x + intercept_log)
        rmse_log = float(np.sqrt(np.mean((y_arr - y_pred_log) ** 2)))

        # Linearity determination
        rmse_reduction = (rmse_lin - rmse_log) / rmse_lin if rmse_lin > 0 else 0
        linearity = "log-linear" if rmse_reduction > 0.20 else "linear"

        # Bootstrap R-squared (1000 iterations) on the chosen model
        rng = np.random.default_rng(42)
        boot_r2 = []
        for _ in range(1000):
            idx = rng.integers(0, len(x_arr), size=len(x_arr))
            bx, by = x_arr[idx], y_arr[idx]
            if linearity == "log-linear":
                bx_fit, by_fit = np.log(bx), np.log(np.maximum(by, 1e-10))
            else:
                bx_fit, by_fit = bx, by
            if np.std(bx_fit) < 1e-12:
                continue
            _, _, r_boot, _, _ = stats.linregress(bx_fit, by_fit)
            boot_r2.append(r_boot ** 2)

        if len(boot_r2) > 0:
            ci_lower = float(np.percentile(boot_r2, 2.5))
            ci_upper = float(np.percentile(boot_r2, 97.5))
        else:
            ci_lower = 0.0
            ci_upper = 0.0

        stratum_pass = ci_lower > 0.90
        if not stratum_pass:
            all_pass = False

        strata_results[str(n_val)] = {
            "verdict": "PASS" if stratum_pass else "FAIL",
            "r2_linear": float(r2_lin),
            "r2_log_linear": float(r2_log),
            "rmse_linear": rmse_lin,
            "rmse_log_linear": rmse_log,
            "linearity": linearity,
            "r2_ci_lower_95": ci_lower,
            "r2_ci_upper_95": ci_upper,
            "slope": float(slope_lin if linearity == "linear" else slope_log),
            "n_m_values": len(x_vals),
        }

    overall = all(
        s.get("verdict") == "PASS"
        for s in strata_results.values()
    )
    insufficient = any(
        s.get("verdict") == "INSUFFICIENT_DATA"
        for s in strata_results.values()
    )

    if insufficient and not any(s.get("verdict") == "PASS" for s in strata_results.values()):
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": "Not enough data in any stratum",
            "strata": strata_results,
        }

    return {
        "verdict": "PASS" if overall else "FAIL",
        "strata": strata_results,
    }


def test_h3(cell_stats):
    """Log-log OLS: std(T_sub) ~ m, one-sided t-test on slope vs -0.3."""
    log_m = []
    log_std = []
    for (n_val, m_val), cs in cell_stats.items():
        if cs["std_t_sub"] > 0:
            log_m.append(math.log(m_val))
            log_std.append(math.log(cs["std_t_sub"]))

    if len(log_m) < 3:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": f"Only {len(log_m)} cells with std > 0, need >= 3",
        }

    log_m_arr = np.array(log_m)
    log_std_arr = np.array(log_std)

    slope, intercept, r_value, p_value_2sided, std_err = stats.linregress(
        log_m_arr, log_std_arr
    )

    # One-sided t-test: H0: slope >= -0.3, H1: slope < -0.3
    t_stat = (slope - (-0.3)) / std_err if std_err > 0 else 0.0
    df = len(log_m) - 2
    p_onesided = float(stats.t.cdf(t_stat, df))

    return {
        "verdict": "PASS" if (slope <= -0.3 and p_onesided < 0.025) else "FAIL",
        "slope": float(slope),
        "slope_se": float(std_err),
        "t_statistic": float(t_stat),
        "p_value_onesided": p_onesided,
        "r_squared": float(r_value ** 2),
        "n_cells": len(log_m),
    }


def test_h4(cell_stats):
    """Compare Rust speedup to Python reference at overlapping (n=10K, m) points."""
    comparisons = {}
    for m_val, py_speedup in PYTHON_SPEEDUP_10K.items():
        key = (10000, m_val)
        if key not in cell_stats:
            comparisons[str(m_val)] = {"status": "NOT_EVALUATED", "reason": "no Rust data"}
            continue
        rust_speedup = cell_stats[key]["speedup_ratio"]
        if rust_speedup <= 0 or py_speedup <= 0:
            comparisons[str(m_val)] = {"status": "NOT_EVALUATED", "reason": "zero speedup"}
            continue
        log2_ratio = math.log2(rust_speedup / py_speedup)
        comparisons[str(m_val)] = {
            "rust_speedup": rust_speedup,
            "python_speedup": py_speedup,
            "log2_ratio": float(log2_ratio),
            "within_2x": abs(log2_ratio) < 1.0,
        }

    evaluated = [c for c in comparisons.values() if "within_2x" in c]
    if not evaluated:
        return {
            "verdict": "NOT_EVALUATED",
            "reason": "no overlapping data points",
            "comparisons": comparisons,
        }

    all_within = all(c["within_2x"] for c in evaluated)
    return {
        "verdict": "PASS" if all_within else "FAIL",
        "comparisons": comparisons,
        "n_evaluated": len(evaluated),
    }


def test_h5(cell_stats, subsample_trials):
    """Same as H1 at (n=50K, m=2000). Exploratory."""
    target_trials = [t for t in subsample_trials if t["n"] == 50000 and t["m"] == 2000]
    if len(target_trials) < 3:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": f"Only {len(target_trials)} trials at (n=50000, m=2000), need >= 3",
        }

    abs_deltas = np.array([t["abs_delta_t"] for t in target_trials])
    mean_val = float(np.mean(abs_deltas))
    sem = float(np.std(abs_deltas, ddof=1) / np.sqrt(len(abs_deltas)))

    result = stats.ttest_1samp(abs_deltas, popmean=0.01, alternative="less")
    t_crit = stats.t.ppf(0.975, df=len(abs_deltas) - 1)
    ci_upper = mean_val + t_crit * sem

    return {
        "verdict": "PASS" if result.pvalue < 0.025 else "FAIL",
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_upper_97_5": float(ci_upper),
        "mean_abs_delta_T": mean_val,
        "max_abs_delta_T": float(np.max(abs_deltas)),
        "n_seeds": len(target_trials),
        "secondary_threshold_0.003": bool(mean_val < 0.003),
    }


def test_h6(sanity_trials):
    """Sanity: abs_delta_t < 1e-10 for both n=10K and n=50K."""
    if not sanity_trials:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reason": "No sanity trials found",
        }

    results = []
    all_pass = True
    for t in sanity_trials:
        ok = t["abs_delta_t"] < 1e-10
        results.append({
            "n": t["n"],
            "m": t["m"],
            "abs_delta_t": t["abs_delta_t"],
            "pass": ok,
        })
        if not ok:
            all_pass = False

    return {
        "verdict": "PASS" if all_pass else "FAIL",
        "trials": results,
    }


# -- Plotting -----------------------------------------------------------------


def generate_plots(cell_stats, output_dir):
    """Create error_vs_m.png, speedup_vs_m.png, variance_decay.png."""
    _plot_error_vs_m(cell_stats, output_dir)
    _plot_speedup_vs_m(cell_stats, output_dir)
    _plot_variance_decay(cell_stats, output_dir)


def _plot_error_vs_m(cell_stats, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_val in [10000, 50000]:
        m_vals = sorted(m for (n, m) in cell_stats if n == n_val)
        means = [cell_stats[(n_val, m)]["mean_abs_delta_t"] for m in m_vals]
        # Use std of abs_delta_t as error bars (approximate from cell stats)
        ax.plot(m_vals, means, "o-", label=f"n={n_val:,}")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.7, label="threshold=0.01")
    ax.set_xlabel("m (subsample size)")
    ax.set_ylabel("mean |delta T|")
    ax.set_title("Approximation Error vs Subsample Size")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_m.png", dpi=150)
    plt.close(fig)


def _plot_speedup_vs_m(cell_stats, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_val in [10000, 50000]:
        m_vals = sorted(m for (n, m) in cell_stats if n == n_val)
        speedups = [cell_stats[(n_val, m)]["speedup_ratio"] for m in m_vals]
        ax.plot(m_vals, speedups, "o-", label=f"n={n_val:,} (Rust)")

    # Overlay Python reference
    py_ms = sorted(PYTHON_SPEEDUP_10K.keys())
    py_speedups = [PYTHON_SPEEDUP_10K[m] for m in py_ms]
    ax.plot(py_ms, py_speedups, "s--", color="gray", alpha=0.7, label="n=10K (Python ref)")

    ax.set_xlabel("m (subsample size)")
    ax.set_ylabel("Speedup ratio (exact / subsample)")
    ax.set_title("Speedup vs Subsample Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "speedup_vs_m.png", dpi=150)
    plt.close(fig)


def _plot_variance_decay(cell_stats, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_val in [10000, 50000]:
        m_vals = sorted(m for (n, m) in cell_stats if n == n_val)
        stds = [cell_stats[(n_val, m)]["std_t_sub"] for m in m_vals]
        # Filter out zero-std cells for log-log
        valid = [(m, s) for m, s in zip(m_vals, stds) if s > 0]
        if valid:
            vm, vs = zip(*valid)
            ax.plot(vm, vs, "o-", label=f"n={n_val:,}")

    # O(1/sqrt(m)) reference line
    if cell_stats:
        all_m = sorted({m for (_, m) in cell_stats})
        if len(all_m) >= 2:
            m_ref = np.array(all_m, dtype=float)
            # Scale reference to first valid point
            first_std = None
            first_m = None
            for (n_val, m_val), cs in sorted(cell_stats.items()):
                if cs["std_t_sub"] > 0:
                    first_std = cs["std_t_sub"]
                    first_m = m_val
                    break
            if first_std is not None:
                ref_line = first_std * np.sqrt(first_m / m_ref)
                ax.plot(m_ref, ref_line, "k--", alpha=0.4, label=r"$O(1/\sqrt{m})$ ref")

    ax.set_xlabel("m (subsample size)")
    ax.set_ylabel("std(T_sub)")
    ax.set_title("Variance Decay (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "variance_decay.png", dpi=150)
    plt.close(fig)


# -- Summary report -----------------------------------------------------------


def write_summary(cell_stats, verdicts, output_dir):
    """Write results/analysis/summary.md with tables and narrative."""
    lines = []
    lines.append("# Subsampled Trustworthiness Experiment — Summary")
    lines.append("")
    lines.append(f"Generated: {verdicts['timestamp']}")
    lines.append("")

    # Overall verdict
    lines.append(f"## Overall Verdict: **{verdicts['overall']}**")
    lines.append("")

    # Hypothesis table
    lines.append("## Hypothesis Results")
    lines.append("")
    lines.append("| Hypothesis | Verdict | Key Metric |")
    lines.append("|------------|---------|------------|")
    for hname in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        h = verdicts["hypotheses"][hname]
        verdict = h["verdict"]
        if verdict == "PASS":
            badge = "PASS"
        elif verdict == "FAIL":
            badge = "FAIL"
        elif verdict == "INSUFFICIENT_DATA":
            badge = "INSUFFICIENT_DATA"
        else:
            badge = verdict

        key_metric = _extract_key_metric(hname, h)
        lines.append(f"| {hname} | {badge} | {key_metric} |")
    lines.append("")

    # Cell stats table
    if cell_stats:
        lines.append("## Per-Cell Statistics")
        lines.append("")
        lines.append("| n | m | count | mean|dT| | max|dT| | std(T_sub) | speedup |")
        lines.append("|---|---|-------|---------|---------|------------|---------|")
        for (n_val, m_val) in sorted(cell_stats.keys()):
            cs = cell_stats[(n_val, m_val)]
            lines.append(
                f"| {n_val:,} | {m_val:,} | {cs['count']} "
                f"| {cs['mean_abs_delta_t']:.6f} "
                f"| {cs['max_abs_delta_t']:.6f} "
                f"| {cs['std_t_sub']:.6f} "
                f"| {cs['speedup_ratio']:.2f}x |"
            )
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines))
    print(f"  Wrote {output_dir / 'summary.md'}", file=sys.stderr)


def _extract_key_metric(hname, h):
    """Extract a short description of the key metric for summary table."""
    if h["verdict"] == "INSUFFICIENT_DATA":
        return h.get("reason", "insufficient data")
    if hname == "H1":
        return f"mean|dT|={h.get('mean_abs_delta_T', '?'):.6f}, p={h.get('p_value', '?'):.4f}"
    if hname == "H2":
        strata = h.get("strata", {})
        parts = []
        for k, v in strata.items():
            if "r2_ci_lower_95" in v:
                parts.append(f"n={k}: R2_CI_lo={v['r2_ci_lower_95']:.3f}")
        return "; ".join(parts) if parts else str(h.get("verdict", "?"))
    if hname == "H3":
        return f"slope={h.get('slope', '?'):.4f}, p={h.get('p_value_onesided', '?'):.4f}"
    if hname == "H4":
        return f"{h.get('n_evaluated', 0)} points evaluated"
    if hname == "H5":
        return f"mean|dT|={h.get('mean_abs_delta_T', '?'):.6f}, p={h.get('p_value', '?'):.4f}"
    if hname == "H6":
        trials = h.get("trials", [])
        return f"{len(trials)} sanity trials checked"
    return ""


# -- Main ---------------------------------------------------------------------


def main(exproot=None):
    """Run analysis pipeline. exproot overrides EXPROOT for testing."""
    root = exproot or EXPROOT
    raw_dir = root / "results" / "raw"
    output_dir = root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trials...", file=sys.stderr)
    trials = load_trials(raw_dir)

    print("Computing cell statistics...", file=sys.stderr)
    cell_stats = compute_cell_stats(trials["subsample"])

    print("Testing hypotheses...", file=sys.stderr)
    verdicts = {
        "experiment": "subsampled-tw-rust-tradeoff",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "hypotheses": {
            "H1": test_h1(cell_stats, trials["subsample"]),
            "H2": test_h2(cell_stats),
            "H3": test_h3(cell_stats),
            "H4": test_h4(cell_stats),
            "H5": test_h5(cell_stats, trials["subsample"]),
            "H6": test_h6(trials["sanity"]),
        },
    }

    # Overall: PASS only if H1,H2,H3,H5,H6 are PASS and H4 is PASS|SKIPPED|NOT_EVALUATED
    h = verdicts["hypotheses"]
    required_pass = all(h[k]["verdict"] == "PASS" for k in ["H1", "H2", "H3", "H5", "H6"])
    h4_ok = h["H4"]["verdict"] in ("PASS", "SKIPPED", "NOT_EVALUATED")
    verdicts["overall"] = "PASS" if (required_pass and h4_ok) else "FAIL"

    verdicts_path = output_dir / "verdicts.json"
    verdicts_path.write_text(json.dumps(verdicts, indent=2))
    print(f"  Wrote {verdicts_path}", file=sys.stderr)

    print("Generating plots...", file=sys.stderr)
    generate_plots(cell_stats, output_dir)

    print("Writing summary...", file=sys.stderr)
    write_summary(cell_stats, verdicts, output_dir)

    print(f"Overall verdict: {verdicts['overall']}", file=sys.stderr)
    return verdicts


if __name__ == "__main__":
    main()
