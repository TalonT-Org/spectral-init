#!/usr/bin/env python3
"""
generate_umap_comparisons.py — Visual evaluation pipeline for spectral-init.

Phase 1 (--phase baseline): Generates Python UMAP reference outputs for Tier 1
synthetic datasets: spectral init coords, final embeddings, graphs, and plots.

Phase 2 (--phase compare): Loads Phase 1 artifacts and Rust spectral init
coordinates, runs three-way UMAP SGD comparison, produces plots and metrics.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

TIER1_DATASETS = [
    "blobs_1000",
    "circles_1000",
    "swiss_roll_2000",
    "two_moons_1000",
    "blobs_hd_2000",
]


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Load a Tier 1 synthetic dataset by name.

    Returns (X, labels, name) where X is float64 and labels is int32.
    """
    from sklearn.datasets import (
        make_blobs,
        make_circles,
        make_moons,
        make_swiss_roll,
    )

    generators = {
        "blobs_1000": lambda: make_blobs(
            n_samples=1000, n_features=10, centers=5, random_state=42
        ),
        "circles_1000": lambda: make_circles(
            n_samples=1000, noise=0.05, factor=0.5, random_state=42
        ),
        "swiss_roll_2000": lambda: (
            lambda X, t: (X, np.digitize(t, np.percentile(t, [20, 40, 60, 80])))
        )(*make_swiss_roll(n_samples=2000, noise=0.0, random_state=42)),
        "two_moons_1000": lambda: make_moons(
            n_samples=1000, noise=0.05, random_state=42
        ),
        "blobs_hd_2000": lambda: make_blobs(
            n_samples=2000, n_features=50, centers=10, random_state=42
        ),
    }

    if name not in generators:
        raise ValueError(f"Unknown dataset: {name!r}. Valid names: {TIER1_DATASETS}")

    X, labels = generators[name]()
    return X.astype(np.float64), labels.astype(np.int32), name


def export_graph(graph: scipy.sparse.csr_matrix, path: Path) -> None:
    """Export a fuzzy k-NN graph in Rust-compatible CSR npz format."""
    g = graph.tocsr()
    np.savez(
        path,
        data=g.data.astype(np.float32),
        indices=g.indices.astype(np.int32),
        indptr=g.indptr.astype(np.int32),
        shape=np.array(g.shape, dtype=np.int32),
        format=b"csr",
    )


def run_baseline(name: str, output_dir: Path) -> None:
    """Run Phase 1 baseline generation for a single dataset."""
    import umap as umap_lib
    from umap.spectral import spectral_layout
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score
    from scipy.sparse import eye, diags
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.csgraph import connected_components

    X, labels, _ = load_dataset(name)

    # Fit UMAP
    mapper = umap_lib.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    ).fit(X)

    # Extract spectral init coordinates (pre-SGD)
    init_coords = spectral_layout(
        data=X,
        graph=mapper.graph_,
        dim=2,
        random_state=np.random.RandomState(42),
    )

    # Get final embedding
    final_embedding = mapper.embedding_.astype(np.float32)

    # Build symmetric normalized Laplacian and compute eigenvalue spectrum
    graph = mapper.graph_
    degree = np.array(graph.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
    L = eye(graph.shape[0]) - D_inv_sqrt @ graph @ D_inv_sqrt
    try:
        eigenvalues, _ = eigsh(L, k=10, which="SM")
    except scipy.sparse.linalg.ArpackNoConvergence as exc:
        raise RuntimeError(
            f"eigsh failed to converge for dataset {name!r} "
            f"(matrix shape {L.shape})"
        ) from exc
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))

    # Compute quality metrics
    tw = trustworthiness(X, final_embedding, n_neighbors=15)
    sil = silhouette_score(final_embedding, labels)
    n_conn, _ = connected_components(graph, directed=False)
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) >= 2 else 0.0
    condition_number = (
        float(eigenvalues[-1] / eigenvalues[1]) if len(eigenvalues) >= 2 and eigenvalues[1] > 0 else float("inf")
    )
    metrics = {
        "trustworthiness": tw,
        "silhouette": sil,
        "n_components": n_conn,
        "spectral_gap": spectral_gap,
        "condition_number": condition_number,
    }

    # Export artifacts
    export_graph(mapper.graph_, output_dir / f"{name}_graph.npz")
    np.save(output_dir / f"{name}_py_spectral.npy", init_coords.astype(np.float64))
    np.save(output_dir / f"{name}_py_final.npy", final_embedding)
    np.save(output_dir / f"{name}_labels.npy", labels.astype(np.int32))

    # Generate baseline plot
    _make_baseline_plot(
        name, init_coords, final_embedding, labels, eigenvalues, metrics, output_dir
    )


def _make_baseline_plot(
    name: str,
    init_coords: np.ndarray,
    final_embedding: np.ndarray,
    labels: np.ndarray,
    eigenvalues: np.ndarray,
    metrics: dict,
    output_dir: Path,
) -> None:
    """Generate a 2×2 baseline comparison plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Style fallback: seaborn-v0_8-whitegrid → seaborn-whitegrid
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Use default style if neither is available

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(name, fontsize=14, fontweight="bold")

    scatter_kw = dict(c=labels, cmap="tab10", s=2, alpha=0.5)

    # [0,0] Python Spectral Init
    axes[0, 0].scatter(init_coords[:, 0], init_coords[:, 1], **scatter_kw)
    axes[0, 0].set_title(f"{name} — Python Spectral Init")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # [0,1] Python UMAP Final
    axes[0, 1].scatter(final_embedding[:, 0], final_embedding[:, 1], **scatter_kw)
    axes[0, 1].set_title(f"{name} — Python UMAP (final)")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # [1,0] Eigenvalue Spectrum
    ax_eig = axes[1, 0]
    ax_eig.bar(range(len(eigenvalues)), eigenvalues)
    ax_eig.text(
        0.98,
        0.95,
        f"λ₂ - λ₁ = {metrics['spectral_gap']:.4f}",
        transform=ax_eig.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax_eig.set_title("Eigenvalue Spectrum (first 10)")
    ax_eig.set_xlabel("k")
    ax_eig.set_ylabel("λ_k")

    # [1,1] Metrics text panel
    ax_txt = axes[1, 1]
    ax_txt.set_axis_off()
    text_content = (
        f"Trustworthiness:  {metrics['trustworthiness']:.4f}\n"
        f"Silhouette score: {metrics['silhouette']:.4f}\n"
        f"Connected comps:  {metrics['n_components']}\n"
        f"Spectral gap:     {metrics['spectral_gap']:.4f}\n"
        f"Condition number: {metrics['condition_number']:.2f}"
    )
    ax_txt.text(
        0.05,
        0.95,
        text_content,
        transform=ax_txt.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )
    ax_txt.set_title("Metrics")

    plt.tight_layout()
    out_path = output_dir / f"{name}_baseline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def _compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    embed_py: np.ndarray,
    embed_rust: np.ndarray,
    embed_rand: np.ndarray,
) -> dict:
    """Compute comparison metrics for three UMAP strategies."""
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score
    from scipy.spatial import procrustes
    from scipy.spatial.distance import pdist

    n = X.shape[0]
    for name, arr in [("labels", labels), ("embed_py", embed_py), ("embed_rust", embed_rust), ("embed_rand", embed_rand)]:
        if arr.shape[0] != n:
            raise ValueError(f"_compute_metrics: {name} has {arr.shape[0]} rows but X has {n}")

    def _tw(emb):
        return float(trustworthiness(X, emb, n_neighbors=15))

    def _sil(emb):
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(emb, labels))

    def _procrustes_disp(ref, other):
        _, _, disp = procrustes(ref, other)
        return float(disp)

    def _pairwise_corr(ref, other):
        if n > 2000:
            idx = np.random.RandomState(42).choice(n, 2000, replace=False)
            ref = ref[idx]
            other = other[idx]
        d_ref = pdist(ref)
        d_other = pdist(other)
        return float(np.corrcoef(d_ref, d_other)[0, 1])

    tw_py = _tw(embed_py)
    tw_rust = _tw(embed_rust)
    tw_rand = _tw(embed_rand)

    sil_py = _sil(embed_py)
    sil_rust = _sil(embed_rust)
    sil_rand = _sil(embed_rand)

    proc_rust = _procrustes_disp(embed_py, embed_rust)
    proc_rand = _procrustes_disp(embed_py, embed_rand)

    corr_rust = _pairwise_corr(embed_py, embed_rust)
    corr_rand = _pairwise_corr(embed_py, embed_rand)

    pf_proc = "PASS" if proc_rust < 0.05 else "FAIL"
    pf_corr = "PASS" if corr_rust > 0.99 else "FAIL"
    pf_tw = "PASS" if abs(tw_rust - tw_py) < 0.01 else "FAIL"
    pf_sil = "PASS" if abs(sil_rust - sil_py) < 0.05 else "FAIL"
    overall = "PASS" if all(v == "PASS" for v in [pf_proc, pf_corr, pf_tw, pf_sil]) else "FAIL"

    return {
        "python_spectral": {
            "trustworthiness": tw_py,
            "silhouette": sil_py,
        },
        "rust_spectral": {
            "trustworthiness": tw_rust,
            "silhouette": sil_rust,
            "procrustes_vs_python": proc_rust,
            "pairwise_corr_vs_python": corr_rust,
        },
        "random": {
            "trustworthiness": tw_rand,
            "silhouette": sil_rand,
            "procrustes_vs_python": proc_rand,
            "pairwise_corr_vs_python": corr_rand,
        },
        "pass_fail": {
            "procrustes": pf_proc,
            "pairwise_corr": pf_corr,
            "trustworthiness": pf_tw,
            "silhouette": pf_sil,
            "overall": overall,
        },
    }


def _make_comparison_plot(
    name: str,
    py_spectral: np.ndarray,
    rust_init: np.ndarray,
    embed_py: np.ndarray,
    embed_rust: np.ndarray,
    embed_rand: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate a 2×3 comparison plot (pre-SGD inits and post-SGD embeddings)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = py_spectral.shape[0]
    rand_init = np.random.RandomState(42).uniform(-10, 10, (n, 2))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    scatter_kw = dict(c=labels, cmap="tab10", s=2, alpha=0.5)

    # Top row — pre-SGD inits
    axes[0, 0].scatter(py_spectral[:, 0], py_spectral[:, 1], **scatter_kw)
    axes[0, 0].set_title(f"{name} — Python Spectral Init (pre-SGD)")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    axes[0, 1].scatter(rust_init[:, 0], rust_init[:, 1], **scatter_kw)
    axes[0, 1].set_title(f"{name} — Rust Spectral Init (pre-SGD)")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    axes[0, 2].scatter(rand_init[:, 0], rand_init[:, 1], **scatter_kw)
    axes[0, 2].set_title(f"{name} — Random Init (pre-SGD)")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # Bottom row — post-SGD embeddings
    axes[1, 0].scatter(embed_py[:, 0], embed_py[:, 1], **scatter_kw)
    axes[1, 0].set_title(f"{name} — Python Init → SGD")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    axes[1, 1].scatter(embed_rust[:, 0], embed_rust[:, 1], **scatter_kw)
    axes[1, 1].set_title(f"{name} — Rust Init → SGD")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    axes[1, 2].scatter(embed_rand[:, 0], embed_rand[:, 1], **scatter_kw)
    axes[1, 2].set_title(f"{name} — Random Init → SGD")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    plt.tight_layout()
    out_path = output_dir / f"{name}_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def _make_overlay_plot(
    name: str,
    embed_py: np.ndarray,
    embed_rust: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate an overlay plot of Python vs Rust SGD embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(embed_py[:, 0], embed_py[:, 1], c="blue", marker="o", alpha=0.3, s=3, label="Python Init → SGD")
    ax.scatter(embed_rust[:, 0], embed_rust[:, 1], c="red", marker="x", alpha=0.3, s=3, label="Rust Init → SGD")
    ax.legend()
    ax.set_title(f"{name}: Python vs Rust SGD Overlay")
    ax.set_xticks([])
    ax.set_yticks([])

    out_path = output_dir / f"{name}_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def run_compare(name: str, output_dir: Path) -> dict | None:
    """Run Phase 2 comparison for a single dataset."""
    import json
    import umap as umap_lib

    rust_init_path = output_dir / f"{name}_rust_init.npy"
    if not rust_init_path.exists():
        print(f"  [WARN] {name}: rust_init.npy not found — skipping")
        return None

    py_spectral = np.load(output_dir / f"{name}_py_spectral.npy")
    py_final = np.load(output_dir / f"{name}_py_final.npy")
    rust_init = np.load(rust_init_path)
    labels = np.load(output_dir / f"{name}_labels.npy")
    X, _, _ = load_dataset(name)

    umap_kw = dict(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="euclidean", random_state=42, n_jobs=1,
    )

    embed_py = py_final.astype(np.float64)
    embed_rust = umap_lib.UMAP(init=rust_init, **umap_kw).fit_transform(X)
    embed_rand = umap_lib.UMAP(init="random", **umap_kw).fit_transform(X)

    metrics = _compute_metrics(X, labels, embed_py, embed_rust, embed_rand)

    pf = metrics["pass_fail"]
    rand_m = metrics["random"]
    py_m = metrics["python_spectral"]

    rand_proc_pass = rand_m["procrustes_vs_python"] < 0.05
    rand_corr_pass = rand_m["pairwise_corr_vs_python"] > 0.99
    rand_tw_pass = abs(rand_m["trustworthiness"] - py_m["trustworthiness"]) < 0.01
    rand_sil_pass = abs(rand_m["silhouette"] - py_m["silhouette"]) < 0.05
    if pf["overall"] == "PASS" and all([rand_proc_pass, rand_corr_pass, rand_tw_pass, rand_sil_pass]):
        print(f"  [WARN] {name}: random init also passes all thresholds — dataset may be too easy")

    _make_comparison_plot(name, py_spectral, rust_init, embed_py, embed_rust, embed_rand, labels, output_dir)
    _make_overlay_plot(name, embed_py, embed_rust, output_dir)

    result = dict(metrics)
    result["dataset"] = name
    result["n_samples"] = int(X.shape[0])
    result["n_features"] = int(X.shape[1])

    json_path = output_dir / f"{name}_metrics.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved metrics: {json_path}")

    print(f"  {name:25s} {pf['overall']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual evaluation pipeline for spectral-init."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["baseline", "compare"],
        help="Pipeline phase to run.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Run a single named dataset instead of all Tier 1 datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/visual_eval/output",
        help="Output directory (default: tests/visual_eval/output).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset and args.dataset not in TIER1_DATASETS:
        parser.error(f"unknown dataset {args.dataset!r}. Valid names: {TIER1_DATASETS}")
    datasets = [args.dataset] if args.dataset else TIER1_DATASETS

    for name in datasets:
        t0 = time.time()
        print(f"[{name}] phase={args.phase} ...")
        if args.phase == "baseline":
            run_baseline(name, output_dir)
        else:
            run_compare(name, output_dir)
        print(f"[{name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
