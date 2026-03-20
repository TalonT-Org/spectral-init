"""
test_pipeline_references.py — Tests for full-pipeline reference fixtures.

Tests full_spectral.npz and full_umap_e2e.npz generation, manifest completeness,
--verify flag integration, verify_fixtures.py standalone invocation, and
byte-identical reruns.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parent / "generate_fixtures.py"
VERIFY_SCRIPT = Path(__file__).parent / "verify_fixtures.py"

EXPECTED_STEP_FILES = [
    "step0_raw_data.npz",
    "step1_knn.npz",
    "step2_smooth_knn.npz",
    "step3_membership.npz",
    "step4_symmetrized.npz",
    "step5a_pruned.npz",
    "comp_a_degrees.npz",
    "comp_b_laplacian.npz",
    "comp_c_components.npz",
    "comp_d_eigensolver.npz",
    "comp_e_selection.npz",
    "comp_f_scaling.npz",
    "full_spectral.npz",
    "full_umap_e2e.npz",
]


def _run(datasets, outdir, extra_args=None):
    cmd = [sys.executable, str(SCRIPT), "--output-dir", outdir, "--datasets"] + datasets
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"CMD: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.fixture(scope="session")
def pipeline_refs_outdir(tmp_path_factory):
    td = tmp_path_factory.mktemp("pipeline_refs")
    _run(["blobs_50", "disconnected_200"], str(td))
    return td


@pytest.mark.parametrize("dataset,n", [("blobs_50", 50), ("disconnected_200", 200)])
def test_full_spectral_shape_dtype(pipeline_refs_outdir, dataset, n):
    d = np.load(pipeline_refs_outdir / dataset / "full_spectral.npz", allow_pickle=False)
    embedding = d["embedding"]
    assert embedding.shape == (n, 2), f"Expected ({n}, 2), got {embedding.shape}"
    assert embedding.dtype == np.float64, f"Expected float64, got {embedding.dtype}"


@pytest.mark.parametrize("dataset", ["blobs_50", "disconnected_200"])
def test_full_spectral_finite(pipeline_refs_outdir, dataset):
    d = np.load(pipeline_refs_outdir / dataset / "full_spectral.npz", allow_pickle=False)
    embedding = d["embedding"]
    assert np.all(np.isfinite(embedding)), "embedding contains non-finite values"
    assert not np.all(embedding == 0), "embedding is all zeros"


@pytest.mark.parametrize("dataset,n", [("blobs_50", 50), ("disconnected_200", 200)])
def test_full_umap_e2e_shape_dtype(pipeline_refs_outdir, dataset, n):
    d = np.load(pipeline_refs_outdir / dataset / "full_umap_e2e.npz", allow_pickle=False)
    embedding = d["embedding"]
    assert embedding.shape == (n, 2), f"Expected ({n}, 2), got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
    assert np.all(np.isfinite(embedding)), "embedding contains non-finite values"


def test_full_spectral_matches_comp_e_connected(pipeline_refs_outdir):
    """For blobs_50 (connected): full_spectral embedding matches comp_e_selection up to
    per-column sign normalization (largest-abs element positive)."""
    dataset_dir = pipeline_refs_outdir / "blobs_50"
    full_emb = np.load(dataset_dir / "full_spectral.npz", allow_pickle=False)["embedding"]
    comp_e = np.load(dataset_dir / "comp_e_selection.npz", allow_pickle=False)["embedding"]

    for col in range(2):
        v_full = full_emb[:, col].copy()
        v_comp = comp_e[:, col].copy()
        if v_full[np.argmax(np.abs(v_full))] < 0:
            v_full = -v_full
        if v_comp[np.argmax(np.abs(v_comp))] < 0:
            v_comp = -v_comp
        assert np.allclose(v_full, v_comp, atol=1e-3), (
            f"Column {col}: max_diff={float(np.abs(v_full - v_comp).max()):.4f} > atol=1e-3"
        )


def test_cross_check_uses_comp_e_not_comp_f(pipeline_refs_outdir):
    """Cross-check compares full_spectral vs comp_e_selection (f64), not comp_f pre_noise."""
    dataset_dir = pipeline_refs_outdir / "blobs_50"
    full_emb = np.load(dataset_dir / "full_spectral.npz", allow_pickle=False)["embedding"]  # f64
    comp_e = np.load(dataset_dir / "comp_e_selection.npz", allow_pickle=False)["embedding"]  # f64

    assert full_emb.dtype == np.float64
    assert comp_e.dtype == np.float64

    for col in range(2):
        v_full = full_emb[:, col].copy()
        v_comp = comp_e[:, col].copy()
        # Sign normalize: flip so largest-abs element is positive
        if v_full[np.argmax(np.abs(v_full))] < 0:
            v_full = -v_full
        if v_comp[np.argmax(np.abs(v_comp))] < 0:
            v_comp = -v_comp
        assert np.allclose(v_full, v_comp, atol=1e-3), (
            f"Column {col}: max_diff={float(np.abs(v_full - v_comp).max()):.4f}"
        )


def test_manifest_has_step_files(pipeline_refs_outdir):
    with open(pipeline_refs_outdir / "manifest.json") as f:
        manifest = json.load(f)

    entry = next(d for d in manifest["datasets"] if d["name"] == "blobs_50")
    step_files = entry.get("step_files")
    assert step_files is not None, "manifest entry missing 'step_files'"
    assert isinstance(step_files, list), f"step_files must be a list, got {type(step_files)}"
    assert len(step_files) == 14, f"Expected 14 step_files, got {len(step_files)}"
    assert set(step_files) == set(EXPECTED_STEP_FILES), (
        f"step_files mismatch.\nExpected: {sorted(EXPECTED_STEP_FILES)}\nGot: {sorted(step_files)}"
    )


def test_verify_flag_passes(tmp_path):
    _run(["blobs_50"], str(tmp_path), extra_args=["--verify"])


def test_verify_script_standalone(pipeline_refs_outdir):
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--output-dir",
            str(pipeline_refs_outdir),
            "--datasets",
            "blobs_50",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"verify_fixtures.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_full_pipeline_deterministic(tmp_path_factory):
    run1 = tmp_path_factory.mktemp("run1")
    run2 = tmp_path_factory.mktemp("run2")
    _run(["blobs_50"], str(run1))
    _run(["blobs_50"], str(run2))

    for fname, key, rtol in [
        ("full_spectral.npz", "embedding", 1e-10),
        ("full_umap_e2e.npz", "embedding", 1e-6),
    ]:
        a1 = np.load(run1 / "blobs_50" / fname, allow_pickle=False)[key]
        a2 = np.load(run2 / "blobs_50" / fname, allow_pickle=False)[key]
        assert np.allclose(a1, a2, rtol=rtol), (
            f"{fname}[{key!r}] differs between runs: max_diff={float(np.abs(a1 - a2).max()):.2e}"
        )
