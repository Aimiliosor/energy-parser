"""Tests for energy_parser.data_validator module."""

import pandas as pd
import numpy as np
import pytest

from energy_parser.data_validator import (
    check_record_count,
    check_valid_value_preservation,
    check_unexpected_nans,
    check_sum_preservation,
    classify_outliers_by_severity,
    calculate_completeness,
    check_impossible_spikes,
    calculate_temporal_summary,
    calculate_statistics,
    calculate_quality_score,
    run_validation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_standard_df(rows: int = 100, freq: str = "15min",
                      consumption_base: float = 10.0) -> pd.DataFrame:
    """Build a clean transformed DataFrame with Date & Time + Consumption (kW)."""
    dates = pd.date_range("2024-01-01", freq=freq, periods=rows)
    np.random.seed(42)
    values = consumption_base + np.random.normal(0, 0.5, rows)
    return pd.DataFrame({
        "Date & Time": dates,
        "Consumption (kW)": values,
    })


def _make_quality_report(total_rows: int = 100,
                         expected_timestamps: int = 100,
                         gaps: list | None = None,
                         missing_timestamps: int = 0,
                         missing_values: dict | None = None,
                         duplicates: list | None = None,
                         outliers: list | None = None,
                         negatives: list | None = None) -> dict:
    """Build a mock quality report dict matching run_quality_check output."""
    return {
        "total_rows": total_rows,
        "expected_timestamps": expected_timestamps,
        "date_range": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        "granularity": "15 min",
        "gaps": gaps or [],
        "missing_timestamps": missing_timestamps,
        "missing_values": missing_values or {},
        "duplicates": duplicates or [],
        "outliers": outliers or [],
        "negatives": negatives or [],
    }


# ---------------------------------------------------------------------------
# TestCheckRecordCount
# ---------------------------------------------------------------------------

class TestCheckRecordCount:
    def test_equal_rows_pass(self):
        orig = pd.DataFrame({"a": range(100)})
        proc = pd.DataFrame({"a": range(100)})
        result = check_record_count(orig, proc)
        assert result["status"] == "PASS"

    def test_nat_drop_within_5pct_pass(self):
        orig = pd.DataFrame({"a": range(100)})
        proc = pd.DataFrame({"a": range(96)})
        result = check_record_count(orig, proc)
        assert result["status"] == "PASS"
        assert "4.0% loss" in result["details"]

    def test_more_than_5pct_loss_warn(self):
        orig = pd.DataFrame({"a": range(100)})
        proc = pd.DataFrame({"a": range(90)})
        result = check_record_count(orig, proc)
        assert result["status"] == "WARN"
        assert ">5%" in result["details"]

    def test_extra_rows_fail(self):
        orig = pd.DataFrame({"a": range(50)})
        proc = pd.DataFrame({"a": range(60)})
        result = check_record_count(orig, proc)
        assert result["status"] == "FAIL"

    def test_empty_original_warn(self):
        orig = pd.DataFrame({"a": []})
        proc = pd.DataFrame({"a": range(5)})
        result = check_record_count(orig, proc)
        assert result["status"] == "WARN"


# ---------------------------------------------------------------------------
# TestCheckValidValuePreservation
# ---------------------------------------------------------------------------

class TestCheckValidValuePreservation:
    def test_100pct_pass(self):
        orig = pd.DataFrame({"col0": [1, 2, 3], "col1": [10, 20, 30]})
        proc = pd.DataFrame({"Date & Time": [1, 2, 3],
                              "Consumption (kW)": [10.0, 20.0, 30.0]})
        result = check_valid_value_preservation(orig, proc, consumption_col=1, production_col=None)
        assert result["status"] == "PASS"

    def test_above_95pct_pass(self):
        orig = pd.DataFrame({"vals": list(range(100))})
        proc_vals = list(range(96)) + [np.nan] * 4
        proc = pd.DataFrame({"Date & Time": range(100),
                              "Consumption (kW)": proc_vals})
        result = check_valid_value_preservation(orig, proc, consumption_col=0, production_col=None)
        assert result["status"] == "PASS"

    def test_80_to_95pct_warn(self):
        orig = pd.DataFrame({"vals": list(range(100))})
        proc_vals = list(range(85)) + [np.nan] * 15
        proc = pd.DataFrame({"Date & Time": range(100),
                              "Consumption (kW)": proc_vals})
        result = check_valid_value_preservation(orig, proc, consumption_col=0, production_col=None)
        assert result["status"] == "WARN"

    def test_below_80pct_fail(self):
        orig = pd.DataFrame({"vals": list(range(100))})
        proc_vals = list(range(70)) + [np.nan] * 30
        proc = pd.DataFrame({"Date & Time": range(100),
                              "Consumption (kW)": proc_vals})
        result = check_valid_value_preservation(orig, proc, consumption_col=0, production_col=None)
        assert result["status"] == "FAIL"


# ---------------------------------------------------------------------------
# TestCheckUnexpectedNans
# ---------------------------------------------------------------------------

class TestCheckUnexpectedNans:
    def test_no_new_nan_pass(self):
        orig = pd.DataFrame({"vals": [1, 2, np.nan, 4]})
        proc = pd.DataFrame({"Date & Time": [1, 2, 3, 4],
                              "Consumption (kW)": [1.0, 2.0, np.nan, 4.0]})
        result = check_unexpected_nans(orig, proc, consumption_col=0, production_col=None)
        assert result["status"] == "PASS"

    def test_new_nan_warn(self):
        orig = pd.DataFrame({"vals": [1, 2, 3, 4]})
        proc = pd.DataFrame({"Date & Time": [1, 2, 3, 4],
                              "Consumption (kW)": [1.0, np.nan, 3.0, 4.0]})
        result = check_unexpected_nans(orig, proc, consumption_col=0, production_col=None)
        assert result["status"] == "WARN"
        assert "1 new NaN" in result["details"]


# ---------------------------------------------------------------------------
# TestCheckSumPreservation
# ---------------------------------------------------------------------------

class TestCheckSumPreservation:
    def test_less_than_1pct_pass(self):
        before = pd.DataFrame({"Date & Time": [1, 2], "v": [100.0, 200.0]})
        after = pd.DataFrame({"Date & Time": [1, 2], "v": [100.5, 200.0]})
        result = check_sum_preservation(before, after)
        assert result["status"] == "PASS"

    def test_1_to_5pct_warn(self):
        before = pd.DataFrame({"Date & Time": [1, 2], "v": [100.0, 100.0]})
        after = pd.DataFrame({"Date & Time": [1, 2], "v": [100.0, 106.0]})
        result = check_sum_preservation(before, after)
        assert result["status"] == "WARN"

    def test_above_5pct_fail(self):
        before = pd.DataFrame({"Date & Time": [1, 2], "v": [100.0, 100.0]})
        after = pd.DataFrame({"Date & Time": [1, 2], "v": [100.0, 120.0]})
        result = check_sum_preservation(before, after)
        assert result["status"] == "FAIL"

    def test_zero_sum_edge_case(self):
        before = pd.DataFrame({"Date & Time": [1], "v": [0.0]})
        after = pd.DataFrame({"Date & Time": [1], "v": [0.0]})
        result = check_sum_preservation(before, after)
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# TestClassifyOutliersBySeverity
# ---------------------------------------------------------------------------

class TestClassifyOutliersBySeverity:
    def test_empty_outliers(self):
        report = _make_quality_report(outliers=[])
        result = classify_outliers_by_severity(report)
        assert result == {"low": 0, "medium": 0, "high": 0, "total": 0}

    def test_low_severity(self):
        outliers = [{"value": 30, "median": 10, "type": "high", "row": 0, "column": "c"}]
        report = _make_quality_report(outliers=outliers)
        result = classify_outliers_by_severity(report)
        assert result["low"] == 1  # ratio = 3 <= 5
        assert result["total"] == 1

    def test_medium_severity(self):
        outliers = [{"value": 80, "median": 10, "type": "high", "row": 0, "column": "c"}]
        report = _make_quality_report(outliers=outliers)
        result = classify_outliers_by_severity(report)
        assert result["medium"] == 1  # ratio = 8 (5 < 8 <= 10)

    def test_high_severity(self):
        outliers = [{"value": 150, "median": 10, "type": "high", "row": 0, "column": "c"}]
        report = _make_quality_report(outliers=outliers)
        result = classify_outliers_by_severity(report)
        assert result["high"] == 1  # ratio = 15 > 10

    def test_mixed_severities(self):
        outliers = [
            {"value": 20, "median": 10, "type": "high", "row": 0, "column": "c"},   # low (2)
            {"value": 70, "median": 10, "type": "high", "row": 1, "column": "c"},   # medium (7)
            {"value": 200, "median": 10, "type": "high", "row": 2, "column": "c"},  # high (20)
        ]
        report = _make_quality_report(outliers=outliers)
        result = classify_outliers_by_severity(report)
        assert result["low"] == 1
        assert result["medium"] == 1
        assert result["high"] == 1
        assert result["total"] == 3


# ---------------------------------------------------------------------------
# TestCalculateCompleteness
# ---------------------------------------------------------------------------

class TestCalculateCompleteness:
    def test_100_percent(self):
        report = _make_quality_report(total_rows=100, expected_timestamps=100)
        assert calculate_completeness(report) == 100.0

    def test_partial(self):
        report = _make_quality_report(total_rows=80, expected_timestamps=100)
        assert calculate_completeness(report) == 80.0

    def test_zero_expected_edge_case(self):
        report = _make_quality_report(total_rows=0, expected_timestamps=0)
        assert calculate_completeness(report) == 0.0

    def test_zero_expected_with_rows(self):
        report = _make_quality_report(total_rows=10, expected_timestamps=0)
        assert calculate_completeness(report) == 100.0

    def test_clamped_over_100(self):
        report = _make_quality_report(total_rows=110, expected_timestamps=100)
        assert calculate_completeness(report) == 100.0


# ---------------------------------------------------------------------------
# TestCheckImpossibleSpikes
# ---------------------------------------------------------------------------

class TestCheckImpossibleSpikes:
    def test_smooth_data_no_spikes(self):
        df = _make_standard_df(rows=50, consumption_base=10.0)
        spikes = check_impossible_spikes(df)
        assert len(spikes) == 0

    def test_spike_detected(self):
        df = _make_standard_df(rows=50, consumption_base=10.0)
        # Inject a spike: value 500 when mean is ~10
        df.loc[30, "Consumption (kW)"] = 500.0
        spikes = check_impossible_spikes(df)
        assert len(spikes) >= 1
        assert any(s["row"] == 30 for s in spikes)

    def test_gradual_increase_ok(self):
        dates = pd.date_range("2024-01-01", freq="15min", periods=50)
        values = list(range(1, 51))  # gradually increasing
        df = pd.DataFrame({"Date & Time": dates, "Consumption (kW)": values})
        spikes = check_impossible_spikes(df)
        assert len(spikes) == 0


# ---------------------------------------------------------------------------
# TestCalculateStatistics
# ---------------------------------------------------------------------------

class TestCalculateStatistics:
    def test_basic_stats(self):
        df = _make_standard_df(rows=100, consumption_base=10.0)
        stats = calculate_statistics(df)
        assert "Consumption (kW)" in stats
        s = stats["Consumption (kW)"]
        assert "mean" in s
        assert "median" in s
        assert "std" in s
        assert "min" in s
        assert "max" in s
        assert "skewness" in s
        assert "kurtosis" in s

    def test_nan_handling(self):
        dates = pd.date_range("2024-01-01", freq="15min", periods=10)
        values = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        df = pd.DataFrame({"Date & Time": dates, "Consumption (kW)": values})
        stats = calculate_statistics(df)
        s = stats["Consumption (kW)"]
        # Mean should be computed from non-NaN values only
        assert s["mean"] == pytest.approx(np.nanmean([1, 2, 4, 5, 7, 8, 9, 10]))

    def test_single_column(self):
        dates = pd.date_range("2024-01-01", freq="15min", periods=5)
        df = pd.DataFrame({"Date & Time": dates, "Values": [10, 20, 30, 40, 50]})
        stats = calculate_statistics(df)
        assert len(stats) == 1
        assert "Values" in stats


# ---------------------------------------------------------------------------
# TestCalculateQualityScore
# ---------------------------------------------------------------------------

class TestCalculateQualityScore:
    def test_perfect_100(self):
        score, breakdown = calculate_quality_score(
            completeness_pct=100.0,
            integrity_results=[],
            temporal_summary={"gaps_count": 0, "missing_timestamps": 0, "duplicates_count": 0},
            consistency_issues={"negatives_count": 0, "spikes_count": 0},
            outlier_classification={"low": 0, "medium": 0, "high": 0, "total": 0},
        )
        assert score == 100
        assert breakdown["completeness"] == 30.0
        assert breakdown["integrity"] == 25.0
        assert breakdown["temporal"] == 20.0
        assert breakdown["consistency"] == 15.0
        assert breakdown["outlier_impact"] == 10.0

    def test_completeness_scaling(self):
        score, breakdown = calculate_quality_score(
            completeness_pct=50.0,
            integrity_results=[],
            temporal_summary={"gaps_count": 0, "missing_timestamps": 0, "duplicates_count": 0},
            consistency_issues={"negatives_count": 0, "spikes_count": 0},
            outlier_classification={"low": 0, "medium": 0, "high": 0, "total": 0},
        )
        assert breakdown["completeness"] == 15.0  # 50% of 30

    def test_integrity_fail_deduction(self):
        results = [{"name": "test", "status": "FAIL", "details": "failed"}]
        score, breakdown = calculate_quality_score(
            completeness_pct=100.0,
            integrity_results=results,
            temporal_summary={"gaps_count": 0, "missing_timestamps": 0, "duplicates_count": 0},
            consistency_issues={"negatives_count": 0, "spikes_count": 0},
            outlier_classification={"low": 0, "medium": 0, "high": 0, "total": 0},
        )
        assert breakdown["integrity"] == 17.0  # 25 - 8

    def test_capped_deductions(self):
        """Many gaps should cap at 15 deduction for temporal."""
        score, breakdown = calculate_quality_score(
            completeness_pct=100.0,
            integrity_results=[],
            temporal_summary={"gaps_count": 50, "missing_timestamps": 0, "duplicates_count": 20},
            consistency_issues={"negatives_count": 0, "spikes_count": 0},
            outlier_classification={"low": 0, "medium": 0, "high": 0, "total": 0},
        )
        assert breakdown["temporal"] == 0  # fully deducted (15+5 = 20)

    def test_outlier_deductions(self):
        score, breakdown = calculate_quality_score(
            completeness_pct=100.0,
            integrity_results=[],
            temporal_summary={"gaps_count": 0, "missing_timestamps": 0, "duplicates_count": 0},
            consistency_issues={"negatives_count": 0, "spikes_count": 0},
            outlier_classification={"low": 4, "medium": 2, "high": 1, "total": 7},
        )
        # Deduction: 4*0.5 + 2*1.0 + 1*2.0 = 2 + 2 + 2 = 6
        assert breakdown["outlier_impact"] == 4.0  # 10 - 6


# ---------------------------------------------------------------------------
# TestRunValidation
# ---------------------------------------------------------------------------

class TestRunValidation:
    def test_without_original_df(self):
        df = _make_standard_df(rows=100)
        report = _make_quality_report(total_rows=100, expected_timestamps=100)
        kpi = run_validation(df=df, quality_report=report)
        assert kpi["integrity"]["status"] == "N/A"
        assert "completeness_pct" in kpi
        assert "quality_score" in kpi
        assert "quality_score_breakdown" in kpi
        assert "missing_values" in kpi
        assert "outliers" in kpi
        assert "timestamp_issues" in kpi
        assert "value_range" in kpi
        assert "processing_accuracy_pct" in kpi
        assert "statistics" in kpi
        assert "detailed_results" in kpi

    def test_with_original_df(self):
        orig = pd.DataFrame({
            "date": pd.date_range("2024-01-01", freq="15min", periods=100),
            "consumption": np.random.normal(10, 0.5, 100),
        })
        df = _make_standard_df(rows=100)
        report = _make_quality_report(total_rows=100, expected_timestamps=100)
        kpi = run_validation(df=df, quality_report=report,
                             original_df=orig, consumption_col=1)
        assert kpi["integrity"]["status"] != "N/A"

    def test_kpi_dict_structure(self):
        df = _make_standard_df(rows=50)
        report = _make_quality_report(total_rows=50, expected_timestamps=50)
        kpi = run_validation(df=df, quality_report=report)

        assert isinstance(kpi["completeness_pct"], float)
        assert isinstance(kpi["quality_score"], int)
        assert isinstance(kpi["quality_score_breakdown"], dict)
        assert isinstance(kpi["missing_values"], int)
        assert isinstance(kpi["outliers"], dict)
        assert isinstance(kpi["timestamp_issues"], int)
        assert isinstance(kpi["value_range"], dict)
        assert isinstance(kpi["integrity"], dict)
        assert isinstance(kpi["processing_accuracy_pct"], float)
        assert isinstance(kpi["statistics"], dict)
        assert isinstance(kpi["detailed_results"], list)

    def test_return_types_detailed_results(self):
        df = _make_standard_df(rows=50)
        report = _make_quality_report(total_rows=50, expected_timestamps=50)
        kpi = run_validation(df=df, quality_report=report)
        for r in kpi["detailed_results"]:
            assert "name" in r
            assert "status" in r
            assert r["status"] in ("PASS", "WARN", "FAIL")
            assert "details" in r

    def test_quality_score_in_range(self):
        df = _make_standard_df(rows=100)
        report = _make_quality_report(total_rows=100, expected_timestamps=100)
        kpi = run_validation(df=df, quality_report=report)
        assert 0 <= kpi["quality_score"] <= 100
