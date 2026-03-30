#!/usr/bin/env python3
"""
generate_merfish_comparisons.py — MERFISH-specific UMAP comparison pipeline.

Phase 1 (--phase baseline): Loads the committed 10K MERFISH subset, preprocesses
with scanpy, fits Python UMAP, exports the graph for Rust, and generates a baseline plot.

Phase 2 (--phase compare): Loads Phase 1 artifacts and Rust spectral init coordinates,
runs three-way UMAP SGD, produces plots and metrics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# Shared helpers live in generate_umap_comparisons.py (same directory).
# sys.path[0] is this script's directory when run directly; also set by conftest.py.
sys.path.insert(0, str(Path(__file__).parent))
from generate_umap_comparisons import (  # noqa: E402
    export_graph,
    _compute_metrics,
    _make_baseline_plot,
    _make_comparison_plot,
    _make_overlay_plot,
    _make_three_way_overlay,
)

DATASET_NAME = "merfish_10k"
_DATA_DIR = Path(__file__).parent / "merfish_data"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def load_merfish_data(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 10K MERFISH subset from compressed .npz files.

    Returns:
        expression  : float32 (10000, 1122)
        spatial     : float32 (10000, 2)
        labels      : int32   (10000,)
        section_ids : int32   (10000,)
    """
    for fname in [
        "merfish_10k_expression.npz",
        "merfish_10k_spatial.npz",
        "merfish_10k_labels.npz",
        "merfish_10k_section_ids.npz",
    ]:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"merfish_10k: data file not found — {data_dir / fname}"
            )
    expression = np.load(data_dir / "merfish_10k_expression.npz")["arr_0"].astype(np.float32)
    spatial = np.load(data_dir / "merfish_10k_spatial.npz")["arr_0"].astype(np.float32)
    labels = np.load(data_dir / "merfish_10k_labels.npz")["arr_0"].astype(np.int32)
    section_ids = np.load(data_dir / "merfish_10k_section_ids.npz")["arr_0"].astype(np.int32)
    return expression, spatial, labels, section_ids


def preprocess_merfish(expression: np.ndarray) -> tuple[anndata.AnnData, np.ndarray]:
    """Run scanpy preprocessing pipeline on expression matrix.

    Pipeline: AnnData wrap → normalize_total(1000) → log1p → scale(10)
              → pca(30) → neighbors(15, 30, euclidean)

    Returns:
        adata  : AnnData with .obsm['X_pca'] populated
        X_pca  : float64 ndarray (n, 30)
    """
    import anndata
    import scanpy as sc

    adata = anndata.AnnData(X=expression.astype(np.float32))
    sc.pp.normalize_total(adata, target_sum=1000)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, metric="euclidean")
    X_pca = adata.obsm["X_pca"].astype(np.float64)
    return adata, X_pca


def run_baseline(output_dir: Path, data_dir: Path = _DATA_DIR) -> None:
    """Run Phase 1 baseline generation for the MERFISH 10K subset."""
    import umap as umap_lib
    from umap.spectral import spectral_layout
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score
    from scipy.sparse import eye, diags
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.csgraph import connected_components

    print(f"  Loading MERFISH 10K data from {data_dir}...")
    expression, spatial, labels, _ = load_merfish_data(data_dir)

    print("  Preprocessing (scanpy: normalize → log1p → scale → PCA → neighbors)...")
    _, X_pca = preprocess_merfish(expression)

    print("  Fitting UMAP on PCA features...")
    mapper = umap_lib.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    ).fit(X_pca)

    # Python spectral init (pre-SGD)
    init_coords = spectral_layout(
        data=X_pca,
        graph=mapper.graph_,
        dim=2,
        random_state=np.random.RandomState(42),
    )

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
            "eigsh failed to converge for merfish_10k "
            f"(matrix shape {L.shape})"
        ) from exc
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
    if len(eigenvalues) != 10:
        raise RuntimeError(
            f"eigsh returned {len(eigenvalues)} eigenvalues; expected 10"
        )

    # Compute baseline metrics for plot
    tw = trustworthiness(X_pca, final_embedding, n_neighbors=15)
    sil = silhouette_score(final_embedding, labels)
    n_conn, _ = connected_components(graph, directed=False)
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) >= 2 else 0.0
    condition_number = (
        float(eigenvalues[-1] / eigenvalues[1])
        if len(eigenvalues) >= 2 and eigenvalues[1] > 0
        else float("inf")
    )
    metrics = {
        "trustworthiness": tw,
        "silhouette": sil,
        "n_components": n_conn,
        "spectral_gap": spectral_gap,
        "condition_number": condition_number,
    }

    # Export artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    export_graph(mapper.graph_, output_dir / "merfish_10k_graph.npz")
    np.save(output_dir / "merfish_10k_py_spectral.npy", init_coords.astype(np.float64))
    np.save(output_dir / "merfish_10k_py_final.npy", final_embedding)
    np.save(output_dir / "merfish_10k_pca.npy", X_pca)
    np.save(output_dir / "merfish_10k_labels.npy", labels.astype(np.int32))
    np.savez_compressed(output_dir / "merfish_10k_spatial.npz", arr_0=spatial)

    _make_baseline_plot(
        DATASET_NAME, init_coords, final_embedding, labels, eigenvalues, metrics, output_dir
    )


def run_compare(output_dir: Path) -> dict | None:
    """Run Phase 2 three-way comparison for the MERFISH 10K dataset.

    Unlike run_baseline, this function does not accept data_dir because Phase 2
    reads all inputs (embeddings, labels, PCA) from Phase 1 artifacts in output_dir.
    """
    import json
    import umap as umap_lib

    rust_init_path = output_dir / "merfish_10k_rust_init.npy"
    if not rust_init_path.exists():
        print(f"  [WARN] merfish_10k: rust_init.npy not found — skipping")
        return None

    for baseline_file in [
        output_dir / "merfish_10k_py_spectral.npy",
        output_dir / "merfish_10k_py_final.npy",
        output_dir / "merfish_10k_labels.npy",
        output_dir / "merfish_10k_pca.npy",
    ]:
        if not baseline_file.exists():
            raise FileNotFoundError(
                f"  [ERROR] merfish_10k: baseline file not found — {baseline_file}"
            )

    py_spectral = np.load(output_dir / "merfish_10k_py_spectral.npy")
    py_final = np.load(output_dir / "merfish_10k_py_final.npy")
    rust_init = np.load(rust_init_path)
    labels = np.load(output_dir / "merfish_10k_labels.npy")
    X_pca = np.load(output_dir / "merfish_10k_pca.npy")

    umap_kw = dict(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )

    embed_py = py_final.astype(np.float64)
    embed_rust = umap_lib.UMAP(init=rust_init, **umap_kw).fit_transform(X_pca)
    embed_rand = umap_lib.UMAP(init="random", **umap_kw).fit_transform(X_pca)

    metrics = _compute_metrics(X_pca, labels, embed_py, embed_rust, embed_rand)

    pf = metrics["pass_fail"]
    rand_m = metrics["random"]
    py_m = metrics["python_spectral"]
    rand_proc_pass = rand_m["procrustes_vs_python"] < 0.05
    rand_corr_pass = rand_m["pairwise_corr_vs_python"] > 0.99
    rand_tw_pass = abs(rand_m["trustworthiness"] - py_m["trustworthiness"]) < 0.01
    rand_sil_pass = abs(rand_m["silhouette"] - py_m["silhouette"]) < 0.05
    if pf["overall"] == "PASS" and all([rand_proc_pass, rand_corr_pass, rand_tw_pass, rand_sil_pass]):
        print(f"  [WARN] merfish_10k: random init also passes all thresholds")

    _make_comparison_plot(
        DATASET_NAME, py_spectral, rust_init, embed_py, embed_rust, embed_rand, labels, output_dir
    )
    _make_overlay_plot(DATASET_NAME, embed_py, embed_rust, output_dir)
    _make_three_way_overlay(DATASET_NAME, embed_py, embed_rust, embed_rand, output_dir)

    result = dict(metrics)
    result["dataset"] = DATASET_NAME
    result["n_samples"] = int(X_pca.shape[0])
    result["n_features"] = int(X_pca.shape[1])

    json_path = output_dir / "merfish_10k_metrics.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved metrics: {json_path}")
    print(f"  {'merfish_10k':25s} {pf['overall']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MERFISH-specific UMAP comparison pipeline."
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["baseline", "compare"],
        help="Pipeline phase to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory (default: tests/visual_eval/output).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[merfish_10k] phase={args.phase} ...")
    if args.phase == "baseline":
        run_baseline(output_dir)
    else:
        run_compare(output_dir)
    print(f"[merfish_10k] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
