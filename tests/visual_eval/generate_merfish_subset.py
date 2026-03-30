"""
Generate a spatially-stratified 10K-cell subset of the MERFISH Zhuang-ABCA-1 dataset.

Usage:
    python generate_merfish_subset.py          # generate and commit artifacts
    python generate_merfish_subset.py --check  # validate existing artifacts only
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import anndata
import numpy as np
import polars as pl
from scipy.stats import spearmanr

DATA_DIR = Path(os.environ.get("MERFISH_DATA_DIR", "/home/talon/projects/spectral-init/data/merfish-abca1"))
OUTPUT_DIR = Path(__file__).parent / "merfish_data"
N_TARGET = 10_000
N_GENES = 1122
GRID_SIZE = 50
SEED = 42


def load_metadata() -> pl.DataFrame:
    """Load cell metadata using Polars lazy scan (never eagerly loads 630 MB)."""
    df = (
        pl.scan_csv(
            DATA_DIR / "cell_metadata.csv",
            schema_overrides={"cell_label": pl.Utf8},
        )
        .select(["cell_label", "x", "y", "brain_section_label", "cluster_alias"])
        .drop_nulls(subset=["x", "y", "cluster_alias"])
        .collect()
    )
    return df


def spatial_stratified_subsample(
    coords: np.ndarray, n_target: int, grid_size: int = 50, seed: int = 42
) -> np.ndarray:
    """
    Spatially-stratified subsampling using a grid_size x grid_size grid over the (x,y) bounding box.

    Returns array of shape (n_target,) with dtype int64 — indices into coords.
    """
    rng = np.random.default_rng(seed)
    x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), grid_size + 1)
    y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), grid_size + 1)

    # clip to [0, grid_size-1] to handle boundary points
    x_idx = np.clip(np.digitize(coords[:, 0], x_bins) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(coords[:, 1], y_bins) - 1, 0, grid_size - 1)
    bin_ids = x_idx * grid_size + y_idx

    unique_bins, bin_counts = np.unique(bin_ids, return_counts=True)
    fractions = bin_counts / bin_counts.sum()
    per_bin_n = np.maximum(1, np.round(fractions * n_target).astype(int))

    # Exact count correction: add/remove from largest bins until sum == n_target
    total = int(per_bin_n.sum())
    diff = n_target - total
    if diff != 0:
        sort_idx = np.argsort(-bin_counts)
        sign = 1 if diff > 0 else -1
        for i in range(abs(diff)):
            per_bin_n[sort_idx[i % len(sort_idx)]] += sign
        per_bin_n = np.maximum(0, per_bin_n)

    indices = []
    for bid, n_sample in zip(unique_bins, per_bin_n):
        mask = np.where(bin_ids == bid)[0]
        n_sample = min(int(n_sample), len(mask))
        chosen = rng.choice(mask, n_sample, replace=False)
        indices.extend(chosen.tolist())

    arr = np.array(indices, dtype=np.int64)
    # Trim/top-up safeguard (handles edge case where per_bin_n was clipped by bin size)
    if len(arr) > n_target:
        arr = arr[:n_target]
    elif len(arr) < n_target:
        remaining = n_target - len(arr)
        used = set(arr.tolist())
        pool = np.array([i for i in range(len(bin_ids)) if i not in used])
        extra = rng.choice(pool, remaining, replace=False)
        arr = np.concatenate([arr, extra])
    return arr


def encode_section_ids(section_labels_np: np.ndarray) -> np.ndarray:
    """Map brain_section_label strings to 0-based sorted integer IDs."""
    unique_labels = sorted(set(section_labels_np.tolist()))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    return np.array([label_to_id[s] for s in section_labels_np], dtype=np.int32)


def build_obs_index(adata) -> dict:
    """Build dict mapping cell_label string → integer row index in H5AD."""
    return {name: i for i, name in enumerate(adata.obs_names)}


def extract_expression(adata, h5ad_row_indices: np.ndarray) -> np.ndarray:
    """
    Extract expression matrix for the given H5AD row indices.

    Sorts indices for efficient backed='r' access, then restores original order.
    Returns shape (N, n_genes) float32.
    """
    sort_order = np.argsort(h5ad_row_indices)
    sorted_indices = h5ad_row_indices[sort_order]
    inverse_order = np.argsort(sort_order)

    expr = adata[sorted_indices].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    else:
        expr = np.array(expr)

    expr = expr.astype(np.float32)
    return expr[inverse_order]


def compute_spearman_freq_corr(full_df: pl.DataFrame, subset_df: pl.DataFrame) -> float:
    """Compute Spearman correlation of cluster_alias frequency distributions."""
    full_counts = (
        full_df.group_by("cluster_alias")
        .agg(pl.len().alias("full_count"))
    )
    subset_counts = (
        subset_df.group_by("cluster_alias")
        .agg(pl.len().alias("subset_count"))
    )
    joined = full_counts.join(subset_counts, on="cluster_alias", how="left").fill_null(0)
    full_freq = joined["full_count"].to_numpy().astype(float)
    subset_freq = joined["subset_count"].to_numpy().astype(float)
    result = spearmanr(full_freq, subset_freq)
    return float(result.statistic)


def _print_validation_summary(
    n_cells: int,
    n_genes: int,
    n_sections: int,
    spearman_r: float,
    expr_shape: tuple,
    spatial_arr: np.ndarray,
    checksum: str,
) -> bool:
    """Print validation summary and return True if all checks pass."""
    # Note: with 5168 unique cluster types and only 10K cells (~1046 types representable),
    # ~80% of types have 0 in the subset, creating massive rank ties. The achievable
    # Spearman (even with random sampling) is ~0.56. Threshold is 0.50.
    spearman_pass = spearman_r > 0.50
    expr_pass = expr_shape == (N_TARGET, N_GENES)
    all_pass = spearman_pass and expr_pass and n_cells == N_TARGET and n_genes == N_GENES

    print("MERFISH 10K Subset Validation")
    print(f"  n_cells:           {n_cells}")
    print(f"  n_genes:           {n_genes}")
    print(f"  n_sections:        {n_sections}")
    print(f"  freq_spearman_r:   {spearman_r:.4f}   {'[PASS]' if spearman_pass else '[FAIL]'} (threshold: >0.50)")
    print(f"  expression shape:  {expr_shape} {'[PASS]' if expr_pass else '[FAIL]'}")
    print(
        f"  spatial extent:    x=[{spatial_arr[:, 0].min():.2f}, {spatial_arr[:, 0].max():.2f}]"
        f" y=[{spatial_arr[:, 1].min():.2f}, {spatial_arr[:, 1].max():.2f}]"
    )
    print(f"  checksum:          sha256:{checksum}")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def generate_subset() -> None:
    """Main generation logic — load, subsample, extract, save, validate."""
    print("=" * 60)
    print("MERFISH 10K Subset Generator")
    print(f"  Data dir:  {DATA_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata
    print("Loading cell metadata (Polars lazy scan)...")
    meta_df = load_metadata()
    print(f"  Loaded {len(meta_df):,} cells after dropping nulls")

    # 2. Extract spatial coordinates
    coords = meta_df.select(["x", "y"]).to_numpy().astype(np.float32)

    # 3. Spatially-stratified subsample
    print(f"Running spatial stratified subsampling ({GRID_SIZE}x{GRID_SIZE} grid, seed={SEED})...")
    subset_row_indices = spatial_stratified_subsample(coords, N_TARGET, GRID_SIZE, SEED)
    assert len(subset_row_indices) == N_TARGET, f"Expected {N_TARGET}, got {len(subset_row_indices)}"
    print(f"  Selected {len(subset_row_indices):,} cells")

    # 4. Extract subset metadata rows
    subset_df = meta_df[subset_row_indices]

    # 5. Open H5AD (memory-mapped)
    h5ad_path = DATA_DIR / "Zhuang-ABCA-1-log2.h5ad"
    print(f"Opening H5AD: {h5ad_path}")
    adata = anndata.read_h5ad(str(h5ad_path), backed="r")
    print(f"  H5AD shape: {adata.shape}")
    try:
        # 6. Map cell_labels to H5AD row indices
        obs_index = build_obs_index(adata)
        cell_labels = subset_df["cell_label"].to_list()
        h5ad_indices = np.array([obs_index[lbl] for lbl in cell_labels], dtype=np.int64)

        # 7. Extract expression matrix
        print("Extracting expression matrix (backed='r')...")
        expr_matrix = extract_expression(adata, h5ad_indices)
        print(f"  Expression shape: {expr_matrix.shape}, dtype: {expr_matrix.dtype}")
    finally:
        # 8. Close backed file
        adata.file.close()

    # 9. Build output arrays
    spatial_arr = subset_df.select(["x", "y"]).to_numpy().astype(np.float32)
    labels_arr = subset_df["cluster_alias"].to_numpy().astype(np.int32)
    section_ids_arr = encode_section_ids(subset_df["brain_section_label"].to_numpy())

    # 10. Compute Spearman frequency correlation
    print("Computing Spearman freq correlation vs full dataset...")
    spearman_r = compute_spearman_freq_corr(meta_df, subset_df)

    # 11. Compute checksum
    sorted_indices = np.sort(subset_row_indices)
    checksum_hex = hashlib.sha256(sorted_indices.tobytes()).hexdigest()

    # 12. Build meta dict
    cluster_alias_counts = (
        subset_df.group_by("cluster_alias")
        .agg(pl.len().alias("count"))
        .sort("cluster_alias")
    )
    cell_type_counts = {
        str(row["cluster_alias"]): int(row["count"])
        for row in cluster_alias_counts.iter_rows(named=True)
    }
    n_sections = len(set(subset_df["brain_section_label"].to_list()))
    meta = {
        "n_cells": N_TARGET,
        "n_genes": N_GENES,
        "n_sections_sampled": n_sections,
        "spatial_extent": {
            "x_min": float(spatial_arr[:, 0].min()),
            "x_max": float(spatial_arr[:, 0].max()),
            "y_min": float(spatial_arr[:, 1].min()),
            "y_max": float(spatial_arr[:, 1].max()),
        },
        "cell_type_counts": cell_type_counts,
        "freq_spearman_r": float(spearman_r),
        "subset_indices_checksum": f"sha256:{checksum_hex}",
    }

    # 13. Save artifacts
    print("Saving artifacts...")
    np.savez_compressed(OUTPUT_DIR / "merfish_10k_expression.npz", arr_0=expr_matrix)
    np.savez_compressed(OUTPUT_DIR / "merfish_10k_spatial.npz", arr_0=spatial_arr)
    np.savez_compressed(OUTPUT_DIR / "merfish_10k_labels.npz", arr_0=labels_arr)
    np.savez_compressed(OUTPUT_DIR / "merfish_10k_section_ids.npz", arr_0=section_ids_arr)
    with open(OUTPUT_DIR / "merfish_10k_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Artifacts saved.")

    # 14. Print validation summary
    all_pass = _print_validation_summary(
        n_cells=N_TARGET,
        n_genes=N_GENES,
        n_sections=n_sections,
        spearman_r=spearman_r,
        expr_shape=expr_matrix.shape,
        spatial_arr=spatial_arr,
        checksum=checksum_hex,
    )

    if spearman_r <= 0.50:
        raise ValueError(f"Validation FAILED — freq_spearman_r {spearman_r:.4f} <= 0.50. See summary above.")


def validate_subset() -> None:
    """--check mode: validate existing artifacts without regenerating."""
    artifacts = [
        "merfish_10k_expression.npz",
        "merfish_10k_spatial.npz",
        "merfish_10k_labels.npz",
        "merfish_10k_section_ids.npz",
        "merfish_10k_meta.json",
    ]

    # Check all files exist
    missing = [f for f in artifacts if not (OUTPUT_DIR / f).exists()]
    if missing:
        print(f"FAIL: Missing artifacts: {missing}")
        sys.exit(1)

    failures = []

    # Load and check .npz files
    expr = np.load(OUTPUT_DIR / "merfish_10k_expression.npz")["arr_0"]
    if expr.shape != (N_TARGET, N_GENES):
        failures.append(f"expression shape {expr.shape} != ({N_TARGET}, {N_GENES})")
    if expr.dtype != np.float32:
        failures.append(f"expression dtype {expr.dtype} != float32")

    spatial = np.load(OUTPUT_DIR / "merfish_10k_spatial.npz")["arr_0"]
    if spatial.shape != (N_TARGET, 2):
        failures.append(f"spatial shape {spatial.shape} != ({N_TARGET}, 2)")
    if spatial.dtype != np.float32:
        failures.append(f"spatial dtype {spatial.dtype} != float32")

    labels = np.load(OUTPUT_DIR / "merfish_10k_labels.npz")["arr_0"]
    if labels.shape != (N_TARGET,):
        failures.append(f"labels shape {labels.shape} != ({N_TARGET},)")
    if labels.dtype != np.int32:
        failures.append(f"labels dtype {labels.dtype} != int32")

    section_ids = np.load(OUTPUT_DIR / "merfish_10k_section_ids.npz")["arr_0"]
    if section_ids.shape != (N_TARGET,):
        failures.append(f"section_ids shape {section_ids.shape} != ({N_TARGET},)")
    if section_ids.dtype != np.int32:
        failures.append(f"section_ids dtype {section_ids.dtype} != int32")

    # Load and check meta.json
    with open(OUTPUT_DIR / "merfish_10k_meta.json") as f:
        meta = json.load(f)

    if meta["n_cells"] != N_TARGET:
        failures.append(f"n_cells {meta['n_cells']} != {N_TARGET}")
    if meta["n_genes"] != N_GENES:
        failures.append(f"n_genes {meta['n_genes']} != {N_GENES}")
    if meta["freq_spearman_r"] <= 0.50:
        failures.append(f"freq_spearman_r {meta['freq_spearman_r']:.4f} <= 0.50")
    extent = meta["spatial_extent"]
    if extent["x_min"] >= extent["x_max"]:
        failures.append("spatial x_min >= x_max")
    if extent["y_min"] >= extent["y_max"]:
        failures.append("spatial y_min >= y_max")

    checksum = meta["subset_indices_checksum"].replace("sha256:", "")
    n_sections = meta["n_sections_sampled"]
    spearman_r = meta["freq_spearman_r"]

    _print_validation_summary(
        n_cells=meta["n_cells"],
        n_genes=meta["n_genes"],
        n_sections=n_sections,
        spearman_r=spearman_r,
        expr_shape=expr.shape,
        spatial_arr=spatial,
        checksum=checksum,
    )

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or validate the spatially-stratified 10K MERFISH subset."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing subset artifacts, no regen",
    )
    args = parser.parse_args()

    if args.check:
        validate_subset()
    else:
        generate_subset()


if __name__ == "__main__":
    main()
