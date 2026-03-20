"""
test_pipeline_references.py — Tests for full-pipeline reference fixtures.

Tests full_spectral.npz and full_umap_e2e.npz generation, manifest completeness,
--verify flag integration, verify_fixtures.py standalone invocation, and
byte-identical reruns.
"""
from __future__ import annotations

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


def test_full_spectral_matches_comp_f_connected(pipeline_refs_outdir):
    """For blobs_50 (connected): full_spectral embedding should match comp_f pre_noise up to sign."""
    dataset_dir = pipeline_refs_outdir / "blobs_50"
    full_emb = np.load(dataset_dir / "full_spectral.npz", allow_pickle=False)["embedding"]
    pre_noise = np.load(dataset_dir / "comp_f_scaling.npz", allow_pickle=False)["pre_noise"]

    full_f32 = full_emb.astype(np.float32)
    for col in range(2):
        v_full = full_f32[:, col]
        v_comp = pre_noise[:, col]
        if np.dot(v_full.astype(np.float64), v_comp.astype(np.float64)) < 0:
            v_full = -v_full
        assert np.allclose(v_full, v_comp, atol=0.01), (
            f"Column {col}: max_diff={float(np.abs(v_full - v_comp).max()):.4f} > atol=0.01"
        )


def test_manifest_has_step_files(tmp_path):
    import json

    _run(["blobs_50"], str(tmp_path))
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)

    assert len(manifest["datasets"]) == 1
    entry = manifest["datasets"][0]
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


def test_full_pipeline_byte_identical(tmp_path_factory):
    run1 = tmp_path_factory.mktemp("run1")
    run2 = tmp_path_factory.mktemp("run2")
    _run(["blobs_50"], str(run1))
    _run(["blobs_50"], str(run2))

    for fname in ["full_spectral.npz", "full_umap_e2e.npz"]:
        b1 = (run1 / "blobs_50" / fname).read_bytes()
        b2 = (run2 / "blobs_50" / fname).read_bytes()
        assert b1 == b2, f"{fname} is not byte-identical between runs"
