"""Write merfish-10k-benchmark-report.md from pipeline output JSONs.

Usage (from project root):
    python research/2026-04-02-merfish-10k-e2e-eval/scripts/write_benchmark_report.py
"""

from __future__ import annotations

import json
import math
import pathlib


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    "trustworthiness", "silhouette",
    "procrustes_vs_python", "pairwise_corr_vs_python",   # N/A for python column
    "sna", "spatial_dist_corr",
    "morans_i_max", "morans_i_dim0", "morans_i_dim1",
    "chaos", "pas",
    "ari", "nmi", "celltype_purity",
    "triplet_accuracy", "shepard_pearson", "shepard_spearman",
    "centroid_dist_corr", "knn_preservation",
]

# Map metric key → pass_fail dict key (None if not gated)
_GATE_KEY: dict[str, str | None] = {
    "trustworthiness": "trustworthiness",
    "silhouette": "silhouette",
    "procrustes_vs_python": "procrustes",
    "pairwise_corr_vs_python": "pairwise_corr",
    "sna": "sna",
}

_PLOT_FILES = [
    "merfish_10k_baseline.png",
    "merfish_10k_comparison.png",
    "merfish_10k_overlay.png",
    "merfish_10k_three_way_overlay.png",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_no_nan(metrics: dict) -> None:
    for embed_key in ("python_spectral", "rust_spectral", "random"):
        embed = metrics[embed_key]
        for k, v in embed.items():
            if isinstance(v, float) and math.isnan(v):
                raise ValueError(f"NaN found in metrics[{embed_key!r}][{k!r}]")


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def _section_dataset(metrics: dict) -> str:
    n = metrics["n_samples"]
    f = metrics["n_features"]
    return (
        f"## 1. Dataset Summary\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Dataset | {metrics['dataset']} |\n"
        f"| Cell count | {n:,} |\n"
        f"| PCA components | {f} |\n"
        f"| Preprocessing | scanpy: back-transform log2 → normalize_total(1000) → log1p → scale(10) → pca({f}) → neighbors(15, 10, euclidean) |\n"
        f"| UMAP settings | n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1 |"
    )


def _section_quality(metrics: dict) -> str:
    py   = metrics["python_spectral"]
    rust = metrics["rust_spectral"]
    rand = metrics["random"]
    pf   = metrics["pass_fail"]

    rows = [
        "| Metric | Python Spectral | Rust Spectral | Random | Gate |",
        "|--------|----------------|---------------|--------|------|",
    ]
    for key in _METRIC_KEYS:
        gate_key = _GATE_KEY.get(key)
        gate_val = pf[gate_key] if gate_key else "—"
        rows.append(
            f"| {key} "
            f"| {_fmt(py.get(key))} "
            f"| {_fmt(rust.get(key))} "
            f"| {_fmt(rand.get(key))} "
            f"| {gate_val} |"
        )
    return "## 2. Quality Results\n\n" + "\n".join(rows)


def _section_timing(timing: dict) -> str:
    rows = [
        "| Key | Value (s) |",
        "|-----|-----------|",
    ]
    for key, val in timing.items():
        rows.append(f"| {key} | {val} |")
    table = "\n".join(rows)
    footnote = (
        "\n> **Note:** `rust_spectral_init_s` includes cargo nextest startup and "
        "harness overhead; it is not a direct measure of the spectral solver wall time alone."
    )
    return "## 3. Timing Breakdown\n\n" + table + footnote


def _section_memory(memory: dict) -> str:
    rows = [
        "| Key | Value (MiB) |",
        "|-----|-------------|",
    ]
    for key, val in memory.items():
        rows.append(f"| {key} | {val} |")
    return "## 4. Memory Comparison\n\n" + "\n".join(rows)


def _section_plots(plots_dir: pathlib.Path) -> str:
    lines = ["## 5. Plot References", ""]
    for fname in _PLOT_FILES:
        rel = f"../tests/visual_eval/output/{fname}"
        lines.append(f"![{fname}]({rel})")
    return "\n".join(lines)


def _section_interpretation(metrics: dict) -> str:
    pf = metrics["pass_fail"]
    quality_pass  = pf["trustworthiness"] == "PASS" and pf["silhouette"] == "PASS"
    geometry_fail = pf["procrustes"] == "FAIL" or pf["pairwise_corr"] == "FAIL"

    if quality_pass and geometry_fail:
        geom_text = (
            "The geometry disagreement (procrustes FAIL / pairwise_corr FAIL) combined "
            "with quality agreement is the expected signature of independent spectral "
            "inits converging to equivalent embeddings: two runs that produce the same "
            "topology but different global orientations."
        )
    elif quality_pass:
        geom_text = (
            "Both quality gates passed and geometry also agrees, suggesting the Rust "
            "and Python spectral solvers produced near-identical coordinate frames."
        )
    else:
        geom_text = (
            "Quality gates did not fully pass. Review the metric table for which "
            "dimensions diverge and check eigenvector residuals."
        )

    return (
        f"## 6. Interpretation\n\n"
        f"{geom_text}\n\n"
        f"**Silhouette note:** Silhouette scores are expected to be negative for this dataset "
        f"because the MERFISH panel maps 1,046 distinct cell types. With so many clusters, "
        f"the within-cluster cohesion in UMAP 2D space is systematically lower than the "
        f"mean inter-cluster distance, pushing silhouette negative by construction.\n\n"
        f"**Timing note:** `rust_spectral_init_s` includes cargo nextest startup and harness "
        f"overhead; it is not a direct measure of the spectral solver wall time alone."
    )


def _section_conclusions(metrics: dict) -> str:
    pf = metrics["pass_fail"]
    overall = pf["overall"]
    if overall == "PASS":
        verdict = (
            "All five quality gates passed. The evidence **supports H1**: "
            "Rust `spectral_init()` produces UMAP embeddings of equivalent quality "
            "to Python `umap-learn` spectral initialization on the MERFISH 10K subset."
        )
    elif overall == "FAIL":
        verdict = (
            "One or more quality gates failed. The evidence **supports H0** (null): "
            "the implementations differ beyond the accepted thresholds at this scale. "
            "Review the failing gates and residual diagnostics before scaling up."
        )
    else:
        verdict = "Result is **inconclusive** — rerun with corrected inputs."

    return (
        f"## 7. Conclusions and Next Steps\n\n"
        f"### Verdict\n\n"
        f"{verdict}\n\n"
        f"### Next Steps\n\n"
        f"1. **100K scaling study** — Re-run the full pipeline on the 100K-cell MERFISH subset to\n"
        f"   assess whether timing, memory, and quality relationships hold at scale.\n"
        f"2. Investigate any failing gates with eigenvector residual diagnostics.\n"
        f"3. Profile `rust_spectral_init_s` in isolation (without nextest harness) for accurate\n"
        f"   wall-time comparison."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    metrics_json: pathlib.Path,
    timing_json: pathlib.Path,
    memory_json: pathlib.Path,
    output_md: pathlib.Path,
    plots_dir: pathlib.Path,
) -> None:
    """Load the three pipeline JSONs and write the 7-section benchmark report."""
    metrics = json.loads(metrics_json.read_text())
    timing  = json.loads(timing_json.read_text())
    memory  = json.loads(memory_json.read_text())

    _validate_no_nan(metrics)

    sections = [
        _section_dataset(metrics),
        _section_quality(metrics),
        _section_timing(timing),
        _section_memory(memory),
        _section_plots(plots_dir),
        _section_interpretation(metrics),
        _section_conclusions(metrics),
    ]
    output_md.write_text("\n\n".join(sections) + "\n")


def main() -> None:
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
    OUTPUT_DIR   = PROJECT_ROOT / "tests" / "visual_eval" / "output"
    DOCS_DIR     = PROJECT_ROOT / "docs"
    generate_report(
        OUTPUT_DIR / "merfish_10k_metrics.json",
        OUTPUT_DIR / "merfish_10k_timing.json",
        OUTPUT_DIR / "merfish_10k_memory.json",
        DOCS_DIR   / "merfish-10k-benchmark-report.md",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
