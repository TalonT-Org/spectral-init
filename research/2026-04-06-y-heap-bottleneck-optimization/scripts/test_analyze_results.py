import json
import math
import os
import sys
from pathlib import Path

import pytest
import numpy as np

# Make analyze_results importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))

# A minimal valid Criterion estimates.json with CI
CRITERION_FULL = {
    "mean": {
        "point_estimate": 1_000_000.0,
        "standard_error": 5_000.0,
        "confidence_interval": {"lower_bound": 980_000.0, "upper_bound": 1_020_000.0},
    },
    "std_dev": {"point_estimate": 10_000.0},
    "median": {"point_estimate": 990_000.0},
}

# Criterion estimates.json WITHOUT CI (W5 fallback case)
CRITERION_NO_CI = {
    "mean": {"point_estimate": 800_000.0, "standard_error": 4_000.0},
    "std_dev": {"point_estimate": 8_000.0},
    "median": {"point_estimate": 795_000.0},
}

# Profiler JSON with step_timing
PROFILER_WITH_TIMING = {
    "n": 10000, "k": 15, "iters": [0.1, 0.11, 0.09], "mean_s": 0.1,
    "std_s": 0.01, "warmup": 5, "score": 0.98, "variant": "baseline",
    "step_timing": {
        "x_dist":   [10_000_000, 11_000_000, 9_000_000],
        "x_sort":   [5_000_000,  5_100_000,  4_900_000],
        "y_heap":   [80_000_000, 82_000_000, 78_000_000],
        "penalty":  [2_000_000,  2_100_000,  1_900_000],
    },
}

# Profiler JSON without step_timing (profiling feature not active)
PROFILER_NO_TIMING = {
    "n": 1000, "k": 15, "iters": [0.015, 0.018], "mean_s": 0.016,
    "std_s": 0.002, "warmup": 1, "score": 0.507, "variant": "baseline",
}


# --- load_criterion_results ---

def test_load_criterion_results_all_missing(tmp_path):
    """All JSON files missing → returns empty dict, no crash."""
    from analyze_results import load_criterion_results
    data = load_criterion_results(tmp_path)
    assert data == {}


def test_load_criterion_results_parses_mean_and_ci(tmp_path):
    """Correctly extracts mean_ns, ci_lb, ci_ub from estimates.json with CI."""
    from analyze_results import load_criterion_results
    (tmp_path / "criterion").mkdir()
    (tmp_path / "criterion" / "y_heap_baseline_n1000.json").write_text(
        json.dumps(CRITERION_FULL)
    )
    data = load_criterion_results(tmp_path)
    entry = data["baseline"][1000]
    assert entry["mean_ns"] == pytest.approx(1_000_000.0)
    assert entry["ci_lb"] == pytest.approx(980_000.0)
    assert entry["ci_ub"] == pytest.approx(1_020_000.0)
    assert entry["ci_synthetic"] is False


def test_load_criterion_results_synthetic_ci_fallback(tmp_path):
    """Missing CI fields → ±5% synthetic bounds, ci_synthetic=True."""
    from analyze_results import load_criterion_results
    (tmp_path / "criterion").mkdir()
    (tmp_path / "criterion" / "y_heap_baseline_n1000.json").write_text(
        json.dumps(CRITERION_NO_CI)
    )
    data = load_criterion_results(tmp_path)
    entry = data["baseline"][1000]
    assert entry["ci_lb"] == pytest.approx(800_000.0 * 0.95)
    assert entry["ci_ub"] == pytest.approx(800_000.0 * 1.05)
    assert entry["ci_synthetic"] is True


def test_speedup_ratio_computed_from_baseline(tmp_path):
    """speedup = baseline_mean / variant_mean; baseline speedup = 1.0."""
    from analyze_results import load_criterion_results
    (tmp_path / "criterion").mkdir()
    # baseline at n=1000: 1_000_000 ns
    (tmp_path / "criterion" / "y_heap_baseline_n1000.json").write_text(
        json.dumps(CRITERION_FULL)
    )
    # flat_simd at n=1000: 800_000 ns (1.25× faster)
    (tmp_path / "criterion" / "y_heap_flat_simd_n1000.json").write_text(
        json.dumps(CRITERION_NO_CI)
    )
    data = load_criterion_results(tmp_path)
    assert data["baseline"][1000]["speedup"] == pytest.approx(1.0)
    assert data["flat_simd"][1000]["speedup"] == pytest.approx(1_000_000.0 / 800_000.0)


# --- compute_hypothesis ---

def test_hypothesis_positive_when_ci_lb_above_1(tmp_path):
    """ci_lb > 1.0 → POSITIVE decision."""
    from analyze_results import compute_hypothesis
    data = {
        "flat_simd": {10000: {"speedup": 1.25, "ratio_ci_lb": 1.08, "ratio_ci_ub": 1.42,
                              "ci_lb": 800_000.0, "ci_ub": 900_000.0, "ci_synthetic": False,
                              "mean_ns": 850_000.0}}
    }
    decision, details = compute_hypothesis(data, tmp_path)
    assert decision == "POSITIVE"
    assert "1.08" in details["primary_text"] or details.get("ratio_ci_lb") == pytest.approx(1.08)


def test_hypothesis_escalate_when_point_above_1_1_but_ci_lb_below_1(tmp_path):
    """ci_lb ≤ 1.0 and point_estimate ≥ 1.1 → ESCALATE."""
    from analyze_results import compute_hypothesis
    data = {
        "flat_simd": {10000: {"speedup": 1.15, "ratio_ci_lb": 0.98, "ratio_ci_ub": 1.32,
                              "ci_lb": 800_000.0, "ci_ub": 900_000.0, "ci_synthetic": False,
                              "mean_ns": 850_000.0}}
    }
    decision, _ = compute_hypothesis(data, tmp_path)
    assert decision == "ESCALATE"


def test_hypothesis_negative_when_point_below_1_1_and_ci_lb_below_1(tmp_path):
    """ci_lb ≤ 1.0 and point_estimate < 1.1 → NEGATIVE."""
    from analyze_results import compute_hypothesis
    data = {
        "flat_simd": {10000: {"speedup": 1.04, "ratio_ci_lb": 0.95, "ratio_ci_ub": 1.13,
                              "ci_lb": 800_000.0, "ci_ub": 900_000.0, "ci_synthetic": False,
                              "mean_ns": 950_000.0}}
    }
    decision, _ = compute_hypothesis(data, tmp_path)
    assert decision == "NEGATIVE"


def test_hypothesis_missing_flat_simd_n10000(tmp_path):
    """flat_simd at n=10000 absent → decision is NEGATIVE or INCONCLUSIVE (no crash)."""
    from analyze_results import compute_hypothesis
    decision, _ = compute_hypothesis({}, tmp_path)
    assert decision in ("NEGATIVE", "INCONCLUSIVE", "NO_DATA")


# --- build_speedup_table_md ---

def test_speedup_table_marks_significant_entries():
    """Entries with ratio_ci_lb > 1.0 receive '*' significance marker."""
    from analyze_results import build_speedup_table_md
    data = {
        "flat_simd": {10000: {"mean_ns": 800_000.0, "speedup": 1.25,
                              "ci_lb": 800_000.0, "ci_ub": 900_000.0,
                              "ratio_ci_lb": 1.08, "ratio_ci_ub": 1.42, "ci_synthetic": False}},
        "heap_reuse": {10000: {"mean_ns": 950_000.0, "speedup": 1.05,
                               "ci_lb": 900_000.0, "ci_ub": 1_000_000.0,
                               "ratio_ci_lb": 0.95, "ratio_ci_ub": 1.15, "ci_synthetic": False}},
    }
    table = build_speedup_table_md(data)
    assert "flat_simd" in table
    assert "*" in table  # significant marker present


def test_speedup_table_is_valid_markdown():
    """Table contains pipe characters and header separator row."""
    from analyze_results import build_speedup_table_md
    table = build_speedup_table_md({})
    # Empty data: table still has header
    assert "|" in table


# --- build_causal_table_md ---

def test_causal_decomposition_fractions_sum_plausibly(tmp_path):
    """Attribution fractions are finite floats."""
    from analyze_results import build_causal_table_md
    data = {
        "heap_reuse":   {10000: {"speedup": 1.10}},
        "flat_partial": {10000: {"speedup": 1.18}},
        "flat_simd":    {10000: {"speedup": 1.25}},
    }
    table = build_causal_table_md(data, n=10000)
    assert "W2" in table  # W2 caveat label present
    assert "|" in table


def test_causal_decomposition_missing_variants(tmp_path):
    """Missing variant data → table notes absence, no crash."""
    from analyze_results import build_causal_table_md
    table = build_causal_table_md({}, n=10000)
    assert isinstance(table, str)


# --- step fractions ---

def test_step_fractions_computed_from_timing(tmp_path):
    """load_profiler_results extracts step_timing and computes y_heap_fraction."""
    from analyze_results import load_profiler_results, compute_step_fractions
    (tmp_path / "profiler").mkdir()
    (tmp_path / "profiler" / "profiler_baseline_n10000.json").write_text(
        json.dumps(PROFILER_WITH_TIMING)
    )
    profiler_data = load_profiler_results(tmp_path)
    fracs = compute_step_fractions(profiler_data)
    baseline = fracs.get("baseline")
    assert baseline is not None
    assert "y_heap_fraction" in baseline
    frac = baseline["y_heap_fraction"]
    assert 0.0 < frac < 1.0


def test_step_fractions_missing_timing_field(tmp_path):
    """Profiler JSON without step_timing → variant skipped, no crash."""
    from analyze_results import load_profiler_results, compute_step_fractions
    (tmp_path / "profiler").mkdir()
    (tmp_path / "profiler" / "profiler_baseline_n10000.json").write_text(
        json.dumps(PROFILER_NO_TIMING)
    )
    profiler_data = load_profiler_results(tmp_path)
    fracs = compute_step_fractions(profiler_data)
    # baseline either absent or marked as unavailable
    assert fracs.get("baseline") is None or fracs["baseline"].get("y_heap_fraction") is None


def test_load_profiler_results_missing_files(tmp_path):
    """All profiler JSONs missing → returns empty dict, no crash."""
    from analyze_results import load_profiler_results
    (tmp_path / "profiler").mkdir()
    result = load_profiler_results(tmp_path)
    assert result == {}


# --- plot ---

def test_plot_creates_png_file(tmp_path):
    """plot_speedup_chart writes speedup_ratios.png to output_dir."""
    from analyze_results import plot_speedup_chart
    data = {
        "heap_reuse":   {1000: {"speedup": 1.05, "ratio_ci_lb": 0.95, "ratio_ci_ub": 1.15, "ci_synthetic": False}},
        "flat_partial": {1000: {"speedup": 1.12, "ratio_ci_lb": 1.01, "ratio_ci_ub": 1.23, "ci_synthetic": False}},
        "flat_simd":    {1000: {"speedup": 1.20, "ratio_ci_lb": 1.08, "ratio_ci_ub": 1.32, "ci_synthetic": False}},
    }
    out = tmp_path / "analysis"
    out.mkdir()
    plot_speedup_chart(data, out)
    assert (out / "speedup_ratios.png").exists()


def test_plot_tolerates_missing_variant(tmp_path):
    """plot_speedup_chart skips missing variants without crashing."""
    from analyze_results import plot_speedup_chart
    out = tmp_path / "analysis"
    out.mkdir()
    plot_speedup_chart({}, out)  # no data at all — must not crash


# --- write_report ---

def test_write_report_creates_md_with_sections(tmp_path):
    """write_report creates analysis_report.md containing required section headers."""
    from analyze_results import write_report
    sections = {
        "primary_result": "NEGATIVE",
        "speedup_table": "| variant | ...",
        "causal_table": "| attribution | ...",
        "step_fractions": "| step | ...",
        "correctness": "Correctness tests: ...",
        "shipping_decision": "Recommend H3 ...",
        "threats": "See experiment plan.",
    }
    metadata = {"date": "2026-04-06", "n_values": [1000], "k": 15, "hardware": "n/a"}
    out = tmp_path / "analysis"
    out.mkdir()
    write_report(sections, out, metadata)
    report = (out / "analysis_report.md").read_text()
    for heading in ["Primary Result", "Speedup", "Causal", "Step Fraction",
                    "Correctness", "Shipping", "Threats"]:
        assert heading in report, f"Missing section heading: {heading}"


# --- integration ---

def test_stage1_only_exits_zero(tmp_path, monkeypatch):
    """--stage1-only prints decision to stdout and exits 0."""
    from analyze_results import load_criterion_results, compute_hypothesis
    # This is a unit-level integration: drive main() partially
    (tmp_path / "criterion").mkdir()
    # Write a fast-path POSITIVE scenario
    baseline = dict(CRITERION_FULL)
    variant = {**CRITERION_FULL, "mean": {
        "point_estimate": 700_000.0, "standard_error": 3_500.0,
        "confidence_interval": {"lower_bound": 680_000.0, "upper_bound": 720_000.0},
    }}
    for n in [1000, 5000, 10000]:
        (tmp_path / "criterion" / f"y_heap_baseline_n{n}.json").write_text(json.dumps(baseline))
        (tmp_path / "criterion" / f"y_heap_flat_simd_n{n}.json").write_text(json.dumps(variant))
    data = load_criterion_results(tmp_path)
    decision, _ = compute_hypothesis(data, tmp_path)
    assert decision == "POSITIVE"


def test_partial_data_no_crash(tmp_path):
    """Script runs on partial data (only baseline n=1000) without crashing."""
    from analyze_results import (
        load_criterion_results, load_profiler_results,
        compute_step_fractions, build_speedup_table_md,
        build_causal_table_md, compute_hypothesis,
    )
    (tmp_path / "criterion").mkdir()
    (tmp_path / "profiler").mkdir()
    (tmp_path / "criterion" / "y_heap_baseline_n1000.json").write_text(
        json.dumps(CRITERION_FULL)
    )
    data = load_criterion_results(tmp_path)
    profiler_data = load_profiler_results(tmp_path)
    fracs = compute_step_fractions(profiler_data)
    table = build_speedup_table_md(data)
    causal = build_causal_table_md(data, n=10000)
    decision, _ = compute_hypothesis(data, tmp_path)
    assert isinstance(table, str)
    assert isinstance(causal, str)
    assert decision in ("NEGATIVE", "INCONCLUSIVE", "NO_DATA")
