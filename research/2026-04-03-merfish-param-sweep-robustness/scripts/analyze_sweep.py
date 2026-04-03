#!/usr/bin/env python3
"""analyze_sweep.py — Compute CV, plots, and solver_levels.json (Phase 5)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPTS_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SWEEP_CSV = RESULTS_DIR / "results_sweep.csv"
TSNE_CSV = RESULTS_DIR / "results_tsne.csv"
SOLVER_LEVELS_JSON = RESULTS_DIR / "solver_levels.json"

INIT_METHODS = ["rust_spectral", "python_spectral", "pca", "random"]
INIT_COLORS = {
    "rust_spectral": "#e65100",
    "python_spectral": "#1a237e",
    "pca": "#004d40",
    "random": "#6a1b9a",
}
LINE_PLOTS = [
    ("trustworthiness", "n_neighbors"),
    ("triplet_accuracy", "n_neighbors"),
    ("trustworthiness", "min_dist"),
]


def compute_cv(
    df: pd.DataFrame,
    metric: str,
    sweep_dim: str,
    init_methods: list[str],
) -> pd.DataFrame:
    sub = df[df["param_swept"] == sweep_dim]
    rows = []
    for method in init_methods:
        vals = sub.loc[sub["init_method"] == method, metric].dropna()
        mean = vals.mean()
        cv = float(vals.std() / mean) if (len(vals) >= 2 and mean != 0) else float("nan")
        rows.append({"init_method": method, "metric": metric, "cv": cv})
    return pd.DataFrame(rows)


def plot_line_charts(
    df: pd.DataFrame,
    tsne_df: pd.DataFrame | None,
    plots_dir: Path,
) -> None:
    for metric, sweep_dim in LINE_PLOTS:
        sub = df[df["param_swept"] == sweep_dim].copy()
        fig, ax = plt.subplots(figsize=(8, 5))

        for method in INIT_METHODS:
            mdf = sub[sub["init_method"] == method].sort_values("param_value")
            ax.plot(
                mdf["param_value"], mdf[metric],
                label=method, color=INIT_COLORS[method], marker="o",
            )

        if sweep_dim == "n_neighbors":
            ax.set_xscale("log")

        # t-SNE overlay only on trustworthiness × n_neighbors
        if metric == "trustworthiness" and sweep_dim == "n_neighbors" and tsne_df is not None:
            ax.plot(
                tsne_df["perplexity"], tsne_df["trustworthiness"],
                linestyle="--", color="gray", marker="s", label="t-SNE (perplexity)",
            )

        ax.set_xlabel(sweep_dim)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {sweep_dim}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_vs_{sweep_dim}.png", dpi=150)
        plt.close(fig)


def plot_cv_bar(df: pd.DataFrame, plots_dir: Path) -> None:
    cv_rows = []
    for metric in ("trustworthiness", "triplet_accuracy"):
        cv_df = compute_cv(df, metric, "n_neighbors", INIT_METHODS)
        cv_rows.append(cv_df)
    combined = pd.concat(cv_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=combined, x="metric", y="cv", hue="init_method",
                palette=INIT_COLORS, ax=ax)
    ax.set_title("CV across n_neighbors sweep")
    ax.set_ylabel("Coefficient of Variation")
    ax.legend(title="init_method")
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_comparison_bar.png", dpi=150)
    plt.close(fig)


def plot_procrustes_heatmap(df: pd.DataFrame, plots_dir: Path) -> None:
    rust = df[df["init_method"] == "rust_spectral"].copy()
    rust["param_value_str"] = rust["param_value"].astype(str)

    pivot = rust.pivot_table(
        index="param_swept", columns="param_value_str",
        values="procrustes_rust_vs_python", aggfunc="first",
    )
    sl_pivot = rust.pivot_table(
        index="param_swept", columns="param_value_str",
        values="solver_level", aggfunc="first",
    )
    # Build annotation: solver_level int string where available
    annot = sl_pivot.reindex_like(pivot).map(
        lambda x: str(int(x)) if pd.notna(x) else ""
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)), 4))
    sns.heatmap(
        pivot, annot=annot, fmt="", cmap="RdYlGn_r",
        center=0.05, vmin=0, vmax=0.2, ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Procrustes rust vs python (annotated with solver_level)")
    fig.tight_layout()
    fig.savefig(plots_dir / "procrustes_rust_vs_python_heatmap.png", dpi=150)
    plt.close(fig)


def _format_param_val(val) -> str:
    """Format param_value for use in JSON keys; whole-number floats become ints."""
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def write_solver_levels(df: pd.DataFrame, output_path: Path) -> None:
    rust = df[df["init_method"] == "rust_spectral"].copy()
    result: dict[str, int | None] = {}
    for _, row in rust.iterrows():
        param_swept = row["param_swept"]
        param_val = row["param_value"]
        pv_str = _format_param_val(param_val)
        metric_val = pv_str if param_swept == "metric" else "euclidean"
        key = f"{param_swept}_{pv_str}_{metric_val}"
        sl = row["solver_level"]
        result[key] = int(sl) if pd.notna(sl) else None
    output_path.write_text(json.dumps(result, indent=2))


def plot_tsne_reference(tsne_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tsne_df["perplexity"], tsne_df["trustworthiness"],
            marker="o", label="trustworthiness")
    ax.plot(tsne_df["perplexity"], tsne_df["triplet_accuracy"],
            marker="s", linestyle="--", label="triplet_accuracy")
    ax.set_xlabel("perplexity")
    ax.set_ylabel("metric value")
    ax.set_title("t-SNE metrics vs perplexity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "tsne_reference.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze MERFISH sweep results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "results_sweep.csv")
    tsne_path = results_dir / "results_tsne.csv"
    tsne_df = pd.read_csv(tsne_path) if tsne_path.exists() else None

    plot_line_charts(df, tsne_df, plots_dir)
    plot_cv_bar(df, plots_dir)
    plot_procrustes_heatmap(df, plots_dir)
    write_solver_levels(df, results_dir / "solver_levels.json")
    if tsne_df is not None:
        plot_tsne_reference(tsne_df, plots_dir)
    print(f"[analyze_sweep] Done. Plots → {plots_dir}")


if __name__ == "__main__":
    main()
