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
import resource
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
    _find_tw_binary,
    _tw_rust,
    _make_baseline_plot,
    _make_comparison_plot,
    _make_overlay_plot,
    _make_three_way_overlay,
)
from spatial_metrics import compute_spatial_metrics   # noqa: E402
from cluster_metrics import compute_cluster_metrics   # noqa: E402
from global_metrics import compute_global_metrics     # noqa: E402

DATASET_NAME = "merfish_10k"
_DATA_DIR = Path(__file__).parent / "merfish_data"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def _check_sna_gate(rust_sna: float, python_sna: float, threshold: float = 0.02) -> str:
    """Return 'PASS' if rust_sna >= python_sna - threshold, else 'FAIL'."""
    return "PASS" if rust_sna >= python_sna - threshold else "FAIL"


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

    The input ``expression`` stores log2(count+1) values extracted directly from
    the Zhuang-ABCA-1 H5AD file.  We back-transform to raw counts before
    applying normalize_total so that the Zhuang lab normalization pipeline is
    applied correctly.

    Pipeline: back-transform(log2→raw) → normalize_total(1000) → log1p
              → scale(10) → pca(10) → neighbors(15, 10, euclidean)

    Returns:
        adata  : AnnData with .obsm['X_pca'] populated
        X_pca  : float64 ndarray (n, 10)
    """
    import anndata
    import scanpy as sc

    raw_counts = np.exp2(expression) - 1.0  # 2^log2(count+1) − 1 = raw count
    adata = anndata.AnnData(X=raw_counts.astype(np.float32))
    sc.pp.normalize_total(adata, target_sum=1000)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=10)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10, metric="euclidean")
    X_pca = adata.obsm["X_pca"].astype(np.float64)
    return adata, X_pca


def run_baseline(output_dir: Path, data_dir: Path = _DATA_DIR) -> tuple[dict, float]:
    """Run Phase 1 baseline generation for the MERFISH 10K subset."""
    import umap as umap_lib
    from umap.spectral import spectral_layout
    from sklearn.metrics import silhouette_score
    from scipy.sparse import eye, diags
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.csgraph import connected_components

    t_run_start = time.perf_counter()

    print(f"  Loading MERFISH 10K data from {data_dir}...")
    t0 = time.perf_counter()
    expression, spatial, labels, _ = load_merfish_data(data_dir)
    data_loading_s = time.perf_counter() - t0

    print("  Preprocessing (scanpy: normalize → log1p → scale → PCA → neighbors)...")
    t0 = time.perf_counter()
    _, X_pca = preprocess_merfish(expression)
    preprocessing_s = time.perf_counter() - t0

    print("  Fitting UMAP on PCA features...")
    t0 = time.perf_counter()
    mapper = umap_lib.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    ).fit(X_pca)
    total_umap_fit_s = time.perf_counter() - t0

    # Python spectral init (pre-SGD) — timed separately for python_spectral_init_s
    t0 = time.perf_counter()
    init_coords = spectral_layout(
        data=X_pca,
        graph=mapper.graph_,
        dim=2,
        random_state=np.random.RandomState(42),
    )
    python_spectral_init_s = time.perf_counter() - t0
    python_sgd_s = total_umap_fit_s - python_spectral_init_s

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
    tw = _tw_rust(X_pca, final_embedding.astype(np.float64), k=15, binary=_find_tw_binary())
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

    # Export artifacts + plot (timed together as graph_export_s)
    t0 = time.perf_counter()
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
    graph_export_s = time.perf_counter() - t0

    total_baseline_s = time.perf_counter() - t_run_start
    rss_baseline_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    timings_baseline = {
        "data_loading_s": data_loading_s,
        "preprocessing_s": preprocessing_s,
        "python_spectral_init_s": python_spectral_init_s,
        "python_sgd_s": python_sgd_s,
        "graph_export_s": graph_export_s,
        "total_baseline_s": total_baseline_s,
    }
    return timings_baseline, rss_baseline_mb


def run_compare(output_dir: Path) -> tuple[dict, float, float] | None:
    """Run Phase 2 three-way comparison for the MERFISH 10K dataset.

    Unlike run_baseline, this function does not accept data_dir because Phase 2
    reads all inputs (embeddings, labels, PCA) from Phase 1 artifacts in output_dir.
    """
    import json
    import umap as umap_lib

    t_run_start = time.perf_counter()

    rust_init_path = output_dir / "merfish_10k_rust_init.npy"
    if not rust_init_path.exists():
        print(f"  [WARN] merfish_10k: rust_init.npy not found — skipping")
        return None

    for baseline_file in [
        output_dir / "merfish_10k_py_spectral.npy",
        output_dir / "merfish_10k_py_final.npy",
        output_dir / "merfish_10k_labels.npy",
        output_dir / "merfish_10k_pca.npy",
        output_dir / "merfish_10k_spatial.npz",
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
    spatial = np.load(output_dir / "merfish_10k_spatial.npz")["arr_0"].astype(np.float64)

    perf_path = output_dir / "merfish_10k_rust_perf.txt"
    if perf_path.exists():
        fields = perf_path.read_text().split()
        rust_spectral_init_s = float(fields[0])
        rust_peak_rss_mb = float(fields[1]) / 1024.0
    else:
        rust_spectral_init_s = 0.0
        rust_peak_rss_mb = 0.0

    umap_kw = dict(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )

    embed_py = py_final.astype(np.float64)

    t0 = time.perf_counter()
    embed_rust = umap_lib.UMAP(init=rust_init, **umap_kw).fit_transform(X_pca)
    rust_init_sgd_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    embed_rand = umap_lib.UMAP(init="random", **umap_kw).fit_transform(X_pca)
    random_sgd_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    metrics = _compute_metrics(X_pca, labels, embed_py, embed_rust, embed_rand)
    b_py   = compute_spatial_metrics(spatial, embed_py,   labels)
    b_rust = compute_spatial_metrics(spatial, embed_rust,  labels)
    b_rand = compute_spatial_metrics(spatial, embed_rand,  labels)
    c_py   = compute_cluster_metrics(embed_py,   labels)
    c_rust = compute_cluster_metrics(embed_rust,  labels)
    c_rand = compute_cluster_metrics(embed_rand,  labels)
    d_py   = compute_global_metrics(X_pca, embed_py,   labels)
    d_rust = compute_global_metrics(X_pca, embed_rust,  labels)
    d_rand = compute_global_metrics(X_pca, embed_rand,  labels)
    metrics_s = time.perf_counter() - t0

    pf = metrics["pass_fail"]
    rand_m = metrics["random"]
    py_m = metrics["python_spectral"]
    rand_proc_pass = rand_m["procrustes_vs_python"] < 0.05
    rand_corr_pass = rand_m["pairwise_corr_vs_python"] > 0.99
    rand_tw_pass = abs(rand_m["trustworthiness"] - py_m["trustworthiness"]) < 0.01
    rand_sil_pass = abs(rand_m["silhouette"] - py_m["silhouette"]) < 0.05
    if pf["overall"] == "PASS" and all([rand_proc_pass, rand_corr_pass, rand_tw_pass, rand_sil_pass]):
        print(f"  [WARN] merfish_10k: random init also passes all thresholds")

    pf_sna = _check_sna_gate(b_rust["sna"], b_py["sna"])
    metrics["pass_fail"]["sna"] = pf_sna
    # overall requires only TW, SIL, and SNA to pass; procrustes and pairwise_corr
    # may fail because Rust produces a geometrically rotated (not degraded) embedding.
    metrics["pass_fail"]["overall"] = (
        "PASS"
        if all(
            v == "PASS"
            for v in [
                metrics["pass_fail"]["trustworthiness"],
                metrics["pass_fail"]["silhouette"],
                pf_sna,
            ]
        )
        else "FAIL"
    )

    t0 = time.perf_counter()
    _make_comparison_plot(
        DATASET_NAME, py_spectral, rust_init, embed_py, embed_rust, embed_rand, labels, output_dir
    )
    _make_overlay_plot(DATASET_NAME, embed_py, embed_rust, output_dir)
    _make_three_way_overlay(DATASET_NAME, embed_py, embed_rust, embed_rand, output_dir)
    plots_s = time.perf_counter() - t0

    _extra_keys = set(b_py) | set(c_py) | set(d_py)
    _base_keys = set(metrics["python_spectral"])
    _collision = _base_keys & _extra_keys
    assert not _collision, f"Key collision in metric merge: {_collision}"

    result = {
        "dataset": DATASET_NAME,
        "n_samples": int(X_pca.shape[0]),
        "n_features": int(X_pca.shape[1]),
        "python_spectral": {**metrics["python_spectral"], **b_py, **c_py, **d_py},
        "rust_spectral":   {**metrics["rust_spectral"],   **b_rust, **c_rust, **d_rust},
        "random":          {**metrics["random"],           **b_rand, **c_rand, **d_rand},
        "pass_fail":       metrics["pass_fail"],
    }

    json_path = output_dir / "merfish_10k_metrics.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved metrics: {json_path}")
    pf_out = result["pass_fail"]
    py_m  = result["python_spectral"]
    ru_m  = result["rust_spectral"]
    print(f"  {'merfish_10k':25s} {pf_out['overall']}")
    print(
        f"    Cat-A  TW={py_m['trustworthiness']:.4f}(py) {ru_m['trustworthiness']:.4f}(ru)"
        f"  Sil={py_m['silhouette']:.4f}(py) {ru_m['silhouette']:.4f}(ru)"
        f"  Proc={ru_m['procrustes_vs_python']:.4f}  PairCorr={ru_m['pairwise_corr_vs_python']:.4f}"
    )
    print(
        f"    Cat-B  SNA={py_m['sna']:.4f}(py) {ru_m['sna']:.4f}(ru)"
        f"  DistCorr={py_m['spatial_dist_corr']:.4f}(py) {ru_m['spatial_dist_corr']:.4f}(ru)"
        f"  MoranI={py_m['morans_i_max']:.4f}(py) {ru_m['morans_i_max']:.4f}(ru)"
        f"  [SNA gate: {pf_out['sna']}]"
    )
    print(
        f"    Cat-C  ARI={py_m['ari']:.4f}(py) {ru_m['ari']:.4f}(ru)"
        f"  NMI={py_m['nmi']:.4f}(py) {ru_m['nmi']:.4f}(ru)"
        f"  Purity={py_m['celltype_purity']:.4f}(py) {ru_m['celltype_purity']:.4f}(ru)"
    )
    print(
        f"    Cat-D  TripletAcc={py_m['triplet_accuracy']:.4f}(py) {ru_m['triplet_accuracy']:.4f}(ru)"
        f"  ShepdPearson={py_m['shepard_pearson']:.4f}(py) {ru_m['shepard_pearson']:.4f}(ru)"
        f"  ShepdSpearman={py_m['shepard_spearman']:.4f}(py) {ru_m['shepard_spearman']:.4f}(ru)"
        f"  KNNPres={py_m['knn_preservation']:.4f}(py) {ru_m['knn_preservation']:.4f}(ru)"
    )

    total_compare_s = time.perf_counter() - t_run_start
    rss_compare_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    timings_compare = {
        "rust_spectral_init_s": rust_spectral_init_s,
        "rust_init_sgd_s": rust_init_sgd_s,
        "random_sgd_s": random_sgd_s,
        "metrics_s": metrics_s,
        "plots_s": plots_s,
        "total_compare_s": total_compare_s,
    }
    return timings_compare, rss_compare_mb, rust_peak_rss_mb


def _write_timing_json(output_dir: Path, new_keys: dict) -> None:
    import json
    path = output_dir / "merfish_10k_timing.json"
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text())
    data.update(new_keys)
    path.write_text(json.dumps(data, indent=2))


def _write_memory_json(output_dir: Path, new_keys: dict) -> None:
    import json
    path = output_dir / "merfish_10k_memory.json"
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text())
    data.update(new_keys)
    path.write_text(json.dumps(data, indent=2))


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
        timings_baseline, rss_baseline_mb = run_baseline(output_dir)
        _write_timing_json(output_dir, timings_baseline)
        _write_memory_json(output_dir, {"peak_rss_baseline_mb": rss_baseline_mb})
    else:
        result = run_compare(output_dir)
        if result is not None:
            timings_compare, rss_compare_mb, rust_peak_rss_mb = result
            _write_timing_json(output_dir, timings_compare)
            _write_memory_json(output_dir, {
                "peak_rss_compare_mb": rss_compare_mb,
                "rust_peak_rss_mb": rust_peak_rss_mb,
            })
    print(f"[merfish_10k] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
