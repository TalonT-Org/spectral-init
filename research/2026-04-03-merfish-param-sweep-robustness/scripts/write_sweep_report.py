#!/usr/bin/env python3
"""write_sweep_report.py — Evaluate H1–H5 and write report.md (Phase 5)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "report.md"

sys.path.insert(0, str(SCRIPTS_DIR))
from analyze_sweep import compute_cv, INIT_METHODS  # noqa: E402


def _procrustes_status(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value < 0.05:
        return "PASS"
    if value <= 0.10:
        return "WARNING"
    return "FAIL"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without requiring tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = " | ".join(str(row[c]) for c in cols)
        rows.append(f"| {cells} |")
    return "\n".join([header, sep] + rows)


def compute_h1_verdict(df: pd.DataFrame, compute_cv_fn) -> dict:
    results = {}
    for metric in ("trustworthiness", "triplet_accuracy"):
        cv_df = compute_cv_fn(df, metric, "n_neighbors", INIT_METHODS)
        cv_map = dict(zip(cv_df["init_method"], cv_df["cv"]))
        rust_cv = cv_map.get("rust_spectral", float("nan"))
        random_cv = cv_map.get("random", float("nan"))
        ratio = (
            rust_cv / random_cv
            if (pd.notna(rust_cv) and pd.notna(random_cv) and random_cv > 0)
            else float("nan")
        )
        results[f"cv_ratio_{metric}"] = ratio
    ratios = [v for v in results.values() if pd.notna(v)]
    if not ratios:
        verdict = "INCONCLUSIVE"
    elif all(r < 0.8 for r in ratios):
        verdict = "SUPPORTED"
    elif any(r > 1.2 for r in ratios):
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, **results}


def compute_h2_table(df: pd.DataFrame, solver_levels_path: Path) -> pd.DataFrame:
    solver_levels = {}
    if solver_levels_path.exists():
        solver_levels = json.loads(solver_levels_path.read_text())

    rust = df[df["init_method"] == "rust_spectral"].copy()
    rows = []
    for _, row in rust.iterrows():
        param_swept = row["param_swept"]
        param_val = row["param_value"]
        metric_val = str(param_val) if param_swept == "metric" else "euclidean"
        config_key = f"{param_swept}_{param_val}_{metric_val}"
        procrustes = row["procrustes_rust_vs_python"]
        sl = solver_levels.get(config_key)
        rows.append(
            {
                "config_key": config_key,
                "param_swept": param_swept,
                "param_value": param_val,
                "procrustes_rust_vs_python": procrustes,
                "solver_level": sl,
                "status": _procrustes_status(procrustes),
            }
        )
    return pd.DataFrame(rows)


def compute_h3_verdict(df: pd.DataFrame) -> dict:
    pv_num = pd.to_numeric(df["param_value"], errors="coerce")
    sub = df[(df["param_swept"] == "n_neighbors") & (pv_num <= 15)].copy()
    sub["_pv_num"] = pd.to_numeric(sub["param_value"], errors="coerce")
    if len(sub) < 2:
        return {
            "verdict": "INCONCLUSIVE",
            "slope_rust_spectral": None,
            "slope_random": None,
        }
    slopes = {}
    for method in ("rust_spectral", "random"):
        mdf = sub[sub["init_method"] == method].sort_values("_pv_num")
        if len(mdf) >= 2:
            result = stats.linregress(mdf["_pv_num"], mdf["trustworthiness"])
            slopes[method] = result.slope
        else:
            slopes[method] = None
    rust_slope = slopes.get("rust_spectral")
    random_slope = slopes.get("random")
    if rust_slope is None or random_slope is None:
        verdict = "INCONCLUSIVE"
    elif abs(random_slope) > abs(rust_slope):
        verdict = "SUPPORTED"
    else:
        verdict = "REFUTED"
    return {
        "verdict": verdict,
        "slope_rust_spectral": rust_slope,
        "slope_random": random_slope,
    }


def compute_h4_verdict(df: pd.DataFrame) -> dict:
    all_cv = {}
    for method in INIT_METHODS:
        vals = df.loc[df["init_method"] == method, "sna"].dropna()
        mean = vals.mean()
        cv = (
            float(vals.std() / mean)
            if (len(vals) >= 2 and mean > 0)
            else float("nan")
        )
        all_cv[method] = cv
    valid = [v for v in all_cv.values() if pd.notna(v) and v > 0]
    if len(valid) >= 2 and max(valid) / min(valid) < 2.0:
        verdict = "STABLE"
    elif valid:
        verdict = "VARIABLE"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "sna_cv_by_method": all_cv}


def compute_h5_verdict(df: pd.DataFrame) -> dict:
    sub = df[
        (df["param_swept"] == "min_dist") & (df["init_method"] == "rust_spectral")
    ]
    vals = sub["procrustes_rust_vs_python"].dropna()
    if len(vals) < 2:
        return {"verdict": "INCONCLUSIVE", "procrustes_std": None}
    std = float(vals.std())
    verdict = "SUPPORTED" if std < 0.01 else "REFUTED"
    return {"verdict": verdict, "procrustes_std": std}


def write_report_md(
    df: pd.DataFrame,
    tsne_df: pd.DataFrame | None,
    h1: dict,
    h2_table: pd.DataFrame,
    h3: dict,
    h4: dict,
    h5: dict,
    results_dir: Path,
) -> str:
    lines = []
    lines.append("# MERFISH Param-Sweep Robustness — Results Report\n")

    # -- Hypothesis Verdicts --
    lines.append("## Hypothesis Verdicts\n")
    lines.append("| Hypothesis | Verdict | Rationale |")
    lines.append("|------------|---------|-----------|")
    cv_trust = h1.get("cv_ratio_trustworthiness", float("nan"))
    cv_trip = h1.get("cv_ratio_triplet_accuracy", float("nan"))
    cv_trust_str = f"{cv_trust:.3f}" if pd.notna(cv_trust) else "N/A"
    cv_trip_str = f"{cv_trip:.3f}" if pd.notna(cv_trip) else "N/A"
    lines.append(
        f"| H1: Spectral init CV stability | {h1['verdict']} | CV_ratio trust={cv_trust_str}, triplet={cv_trip_str} |"
    )
    h2_status_summary = (
        h2_table["status"].value_counts().to_dict() if not h2_table.empty else {}
    )
    lines.append(
        f"| H2: Rust–Python Procrustes | (see table) | PASS={h2_status_summary.get('PASS', 0)}, WARNING={h2_status_summary.get('WARNING', 0)}, FAIL={h2_status_summary.get('FAIL', 0)} |"
    )
    lines.append(
        f"| H3: Random degrades faster at low N | {h3['verdict']} | slope_rust={h3['slope_rust_spectral']}, slope_random={h3['slope_random']} |"
    )
    lines.append(
        f"| H4: SNA metric stability | {h4['verdict']} | CV range: {h4['sna_cv_by_method']} |"
    )
    lines.append(
        f"| H5: Procrustes stable across min_dist | {h5['verdict']} | std={h5['procrustes_std']} |"
    )
    lines.append("")

    # -- Quantitative CV Table --
    lines.append("## Quantitative CV Table\n")
    lines.append("CV computed across n_neighbors sweep dimension.\n")
    lines.append("| init_method | trustworthiness CV | triplet_accuracy CV |")
    lines.append("|-------------|-------------------|---------------------|")
    for method in INIT_METHODS:
        cv_t_df = compute_cv(df, "trustworthiness", "n_neighbors", [method])
        cv_a_df = compute_cv(df, "triplet_accuracy", "n_neighbors", [method])
        cv_t = cv_t_df["cv"].iloc[0] if not cv_t_df.empty else float("nan")
        cv_a = cv_a_df["cv"].iloc[0] if not cv_a_df.empty else float("nan")
        cv_t_str = f"{cv_t:.4f}" if pd.notna(cv_t) else "nan"
        cv_a_str = f"{cv_a:.4f}" if pd.notna(cv_a) else "nan"
        lines.append(f"| {method} | {cv_t_str} | {cv_a_str} |")
    lines.append("")

    # -- Procrustes Alignment (H2) --
    lines.append("## Procrustes Alignment (H2)\n")
    if h2_table.empty:
        lines.append("_No rust_spectral rows in results_sweep.csv._\n")
    else:
        lines.append(_df_to_markdown(h2_table))
    lines.append("")

    # -- t-SNE Reference Comparison --
    lines.append("## t-SNE Reference Comparison\n")
    if tsne_df is not None:
        best_row = tsne_df.loc[tsne_df["trustworthiness"].idxmax()]
        lines.append(
            f"Best t-SNE trustworthiness: {best_row['trustworthiness']:.4f} at perplexity={int(best_row['perplexity'])}."
        )
        rust_best = df[df["init_method"] == "rust_spectral"]["trustworthiness"].max()
        lines.append(f"Best rust_spectral trustworthiness: {rust_best:.4f}.")
    else:
        lines.append(
            "_results_tsne.csv not found — t-SNE comparison unavailable._"
        )
    lines.append("")

    # -- Solver Level Diagnostics --
    lines.append("## Solver Level Diagnostics\n")
    sl_path = results_dir / "solver_levels.json"
    if sl_path.exists():
        sl_data = json.loads(sl_path.read_text())
        lines.append("| config_key | solver_level |")
        lines.append("|------------|-------------|")
        for k, v in sl_data.items():
            lines.append(f"| {k} | {v if v is not None else 'null'} |")
    else:
        lines.append("_solver_levels.json not found._")
    lines.append("")

    # -- Threats to Validity --
    lines.append("## Threats to Validity\n")
    lines.append(
        "- **Incomplete sweep:** results_sweep.csv may contain partial data if the sweep was interrupted. Verdicts derived from fewer than all 56 configs should be treated as provisional."
    )
    lines.append(
        "- **Single dataset:** All results are from the MERFISH 10k cell dataset. Generalization to other datasets is unverified."
    )
    lines.append(
        "- **Single random seed:** `RANDOM_STATE=42` throughout. Variance estimates may not reflect true stochasticity."
    )
    lines.append("")

    # -- Success Criteria Checklist --
    lines.append("## Success Criteria Checklist\n")
    plots_dir = results_dir / "plots"
    expected_files = [
        plots_dir / "trustworthiness_vs_n_neighbors.png",
        plots_dir / "triplet_accuracy_vs_n_neighbors.png",
        plots_dir / "trustworthiness_vs_min_dist.png",
        plots_dir / "cv_comparison_bar.png",
        plots_dir / "procrustes_rust_vs_python_heatmap.png",
        plots_dir / "tsne_reference.png",
        results_dir / "solver_levels.json",
    ]
    for f in expected_files:
        mark = "x" if f.exists() else " "
        lines.append(f"- [{mark}] `{f.name}`")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write MERFISH sweep report")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.results_dir / "results_sweep.csv")
    tsne_path = args.results_dir / "results_tsne.csv"
    tsne_df = pd.read_csv(tsne_path) if tsne_path.exists() else None
    solver_levels_path = args.results_dir / "solver_levels.json"

    h1 = compute_h1_verdict(df, compute_cv)
    h2_table = compute_h2_table(df, solver_levels_path)
    h3 = compute_h3_verdict(df)
    h4 = compute_h4_verdict(df)
    h5 = compute_h5_verdict(df)

    report_text = write_report_md(
        df, tsne_df, h1, h2_table, h3, h4, h5, args.results_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text)
    print(f"[write_sweep_report] Saved → {args.output}")


if __name__ == "__main__":
    main()
