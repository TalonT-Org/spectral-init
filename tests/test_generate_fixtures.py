# tests/test_generate_fixtures.py
"""Tests for the KNN pipeline steps in generate_fixtures.py."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parent / "generate_fixtures.py"


def _run(datasets: list[str], outdir: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPT), "--output-dir", outdir, "--datasets"] + datasets
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_step0_raw_data_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step0_raw_data.npz")
        assert set(d.files) >= {"X", "n_samples", "n_features"}
        assert d["X"].dtype == np.float64
        assert d["X"].shape == (50, 2)
        assert d["n_samples"] == np.int32(50)
        assert d["n_features"] == np.int32(2)


def test_step1_knn_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step1_knn.npz")
        assert set(d.files) >= {"knn_indices", "knn_dists", "n_neighbors"}
        assert d["knn_indices"].dtype == np.int32
        assert d["knn_indices"].shape == (50, 15)
        assert d["knn_dists"].dtype == np.float32
        assert d["knn_dists"].shape == (50, 15)
        assert (d["knn_indices"] >= 0).all() and (d["knn_indices"] < 50).all()
        assert (d["knn_dists"] >= 0).all()
        assert d["n_neighbors"] == np.int32(15)


def test_step2_smooth_knn_output():
    with tempfile.TemporaryDirectory() as td:
        _run(["blobs_50"], td)
        d = np.load(Path(td) / "blobs_50" / "step2_smooth_knn.npz")
        assert set(d.files) >= {"sigmas", "rhos", "n_neighbors"}
        assert d["sigmas"].dtype == np.float32
        assert d["sigmas"].shape == (50,)
        assert d["rhos"].dtype == np.float32
        assert d["rhos"].shape == (50,)
        assert (d["sigmas"] > 0).all()
        assert (d["rhos"] >= 0).all()
        assert d["n_neighbors"] == np.int32(15)


def test_pipeline_is_deterministic():
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        _run(["blobs_50"], td_a)
        _run(["blobs_50"], td_b)
        for fname in ("step0_raw_data.npz", "step1_knn.npz", "step2_smooth_knn.npz"):
            file_a = Path(td_a) / "blobs_50" / fname
            file_b = Path(td_b) / "blobs_50" / fname
            assert file_a.read_bytes() == file_b.read_bytes(), f"{fname} is not byte-identical"


def test_all_7_datasets_generate():
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(SCRIPT), "--output-dir", td]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        expected_datasets = [
            "blobs_50", "blobs_500", "moons_200", "blobs_5000",
            "circles_300", "near_dupes_100", "disconnected_200",
        ]
        for name in expected_datasets:
            for fname in ("step0_raw_data.npz", "step1_knn.npz", "step2_smooth_knn.npz"):
                assert (Path(td) / name / fname).exists(), f"Missing {name}/{fname}"
