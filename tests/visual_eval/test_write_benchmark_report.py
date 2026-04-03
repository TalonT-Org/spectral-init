"""Tests for write_benchmark_report.py — groupC benchmark report generator."""

import json
import math
import sys
from pathlib import Path

import pytest

# Add the research scripts dir to sys.path so we can import the module directly.
_SCRIPTS_DIR = (
    Path(__file__).parent.parent.parent
    / "research"
    / "2026-04-02-merfish-10k-e2e-eval"
    / "scripts"
)
sys.path.insert(0, str(_SCRIPTS_DIR))
import write_benchmark_report as wbr  # noqa: E402


def _make_metrics(overall="PASS"):
    base = {
        "trustworthiness": 0.850, "silhouette": -0.05,
        "sna": 0.42, "spatial_dist_corr": 0.31,
        "morans_i_max": 0.55, "morans_i_dim0": 0.50, "morans_i_dim1": 0.55,
        "chaos": 0.12, "pas": 0.08,
        "ari": 0.72, "nmi": 0.68, "celltype_purity": 0.79,
        "triplet_accuracy": 0.88, "shepard_pearson": 0.90,
        "shepard_spearman": 0.89, "centroid_dist_corr": 0.87,
        "knn_preservation": 0.76,
    }
    rust = {**base, "procrustes_vs_python": 0.02, "pairwise_corr_vs_python": 0.999}
    random = {**base, "procrustes_vs_python": 0.15, "pairwise_corr_vs_python": 0.72}
    return {
        "dataset": "merfish_10k",
        "n_samples": 10000,
        "n_features": 10,
        "python_spectral": base.copy(),
        "rust_spectral": rust,
        "random": random,
        "pass_fail": {
            "procrustes": "PASS", "pairwise_corr": "PASS",
            "trustworthiness": "PASS", "silhouette": "PASS",
            "sna": "PASS", "overall": overall,
        },
    }


def _make_timing():
    return {
        "data_loading_s": 1.2, "preprocessing_s": 8.4,
        "python_spectral_init_s": 0.73, "python_sgd_s": 12.1,
        "graph_export_s": 0.05, "total_baseline_s": 22.5,
        "rust_spectral_init_s": 14.3, "rust_init_sgd_s": 11.8,
        "random_sgd_s": 11.6, "metrics_s": 45.2,
        "plots_s": 3.1, "total_compare_s": 86.0,
    }


def _make_memory():
    return {
        "peak_rss_baseline_mb": 812.4,
        "peak_rss_compare_mb": 1540.7,
        "rust_peak_rss_mb": 221.6,
    }


@pytest.fixture()
def json_files(tmp_path):
    m = tmp_path / "merfish_10k_metrics.json"
    t = tmp_path / "merfish_10k_timing.json"
    mem = tmp_path / "merfish_10k_memory.json"
    m.write_text(json.dumps(_make_metrics()))
    t.write_text(json.dumps(_make_timing()))
    mem.write_text(json.dumps(_make_memory()))
    return m, t, mem


@pytest.fixture()
def report_text(json_files, tmp_path):
    m, t, mem = json_files
    out = tmp_path / "report.md"
    wbr.generate_report(m, t, mem, out, tmp_path)
    return out.read_text()


def test_report_has_all_seven_section_headers(report_text):
    for i in range(1, 8):
        assert f"## {i}." in report_text, f"Section {i} header missing"


def test_quality_table_contains_all_metric_names(report_text):
    metric_keys = [
        "trustworthiness", "silhouette", "procrustes_vs_python",
        "pairwise_corr_vs_python", "sna", "spatial_dist_corr",
        "morans_i_max", "morans_i_dim0", "morans_i_dim1", "chaos", "pas",
        "ari", "nmi", "celltype_purity", "triplet_accuracy", "shepard_pearson",
        "shepard_spearman", "centroid_dist_corr", "knn_preservation",
    ]
    for key in metric_keys:
        assert key in report_text, f"Metric key '{key}' missing from report"


def test_quality_table_shows_na_for_python_procrustes(report_text):
    assert "N/A" in report_text


def test_pass_fail_column_shows_values(report_text):
    assert "PASS" in report_text


def test_timing_table_has_all_twelve_keys(report_text):
    timing_keys = [
        "data_loading_s", "preprocessing_s", "python_spectral_init_s", "python_sgd_s",
        "graph_export_s", "total_baseline_s", "rust_spectral_init_s", "rust_init_sgd_s",
        "random_sgd_s", "metrics_s", "plots_s", "total_compare_s",
    ]
    for key in timing_keys:
        assert key in report_text, f"Timing key '{key}' missing from report"


def test_memory_table_has_all_three_keys(report_text):
    memory_keys = [
        "peak_rss_baseline_mb",
        "peak_rss_compare_mb",
        "rust_peak_rss_mb",
    ]
    for key in memory_keys:
        assert key in report_text, f"Memory key '{key}' missing from report"


def test_plot_references_in_report(report_text):
    plot_files = [
        "merfish_10k_baseline.png",
        "merfish_10k_comparison.png",
        "merfish_10k_overlay.png",
        "merfish_10k_three_way_overlay.png",
    ]
    for fname in plot_files:
        assert fname in report_text, f"Plot reference '{fname}' missing from report"


def test_no_none_or_nan_literals(report_text):
    assert "None" not in report_text
    assert "nan" not in report_text


def test_nan_validation_raises(tmp_path):
    bad_metrics = _make_metrics()
    bad_metrics["rust_spectral"]["trustworthiness"] = float("nan")
    m = tmp_path / "merfish_10k_metrics.json"
    t = tmp_path / "merfish_10k_timing.json"
    mem = tmp_path / "merfish_10k_memory.json"
    m.write_text(json.dumps(bad_metrics))
    t.write_text(json.dumps(_make_timing()))
    mem.write_text(json.dumps(_make_memory()))
    out = tmp_path / "report.md"
    with pytest.raises(ValueError, match="NaN"):
        wbr.generate_report(m, t, mem, out, tmp_path)
    assert not out.exists()


def test_interpretation_expected_signature_text(tmp_path):
    metrics = _make_metrics()
    # quality PASS + geometry FAIL → expected signature
    metrics["pass_fail"]["procrustes"] = "FAIL"
    m = tmp_path / "merfish_10k_metrics.json"
    t = tmp_path / "merfish_10k_timing.json"
    mem = tmp_path / "merfish_10k_memory.json"
    m.write_text(json.dumps(metrics))
    t.write_text(json.dumps(_make_timing()))
    mem.write_text(json.dumps(_make_memory()))
    out = tmp_path / "report.md"
    wbr.generate_report(m, t, mem, out, tmp_path)
    text = out.read_text()
    assert "expected signature" in text


def test_interpretation_includes_silhouette_note(report_text):
    assert "1,046" in report_text or "1046" in report_text or "cell types" in report_text


def test_interpretation_includes_nextest_note(report_text):
    assert "rust_spectral_init_s" in report_text
    assert "nextest" in report_text


def test_conclusions_mention_100k(report_text):
    assert "100K" in report_text


def test_conclusions_verdict_pass(json_files, tmp_path):
    m, t, mem = json_files
    out = tmp_path / "report_pass.md"
    wbr.generate_report(m, t, mem, out, tmp_path)
    assert "H1" in out.read_text()


def test_conclusions_verdict_fail(tmp_path):
    metrics = _make_metrics(overall="FAIL")
    m = tmp_path / "merfish_10k_metrics.json"
    t = tmp_path / "merfish_10k_timing.json"
    mem = tmp_path / "merfish_10k_memory.json"
    m.write_text(json.dumps(metrics))
    t.write_text(json.dumps(_make_timing()))
    mem.write_text(json.dumps(_make_memory()))
    out = tmp_path / "report_fail.md"
    wbr.generate_report(m, t, mem, out, tmp_path)
    text = out.read_text()
    assert "H0" in text or "inconclusive" in text


def test_dataset_summary_shows_cell_count(report_text):
    assert "10,000" in report_text or "10000" in report_text


def test_numeric_values_appear_verbatim(report_text):
    # trustworthiness = 0.850 from the mock JSON
    assert "0.8500" in report_text or "0.85" in report_text
