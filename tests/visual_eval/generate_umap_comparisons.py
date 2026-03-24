#!/usr/bin/env python3
"""
generate_umap_comparisons.py — Visual evaluation pipeline for spectral-init.

Phase 1 (--phase baseline): Generates Python UMAP reference outputs for Tier 1
synthetic datasets: spectral init coords, final embeddings, graphs, and plots.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual evaluation pipeline for spectral-init."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["baseline"],
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
        run_baseline(name, output_dir)
        print(f"[{name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
