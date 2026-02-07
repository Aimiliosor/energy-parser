"""Tests for energy_parser.report_generator module."""

import os
import tempfile

import pandas as pd
import numpy as np
import pytest

from energy_parser.statistics import (
    compute_yearly_stats,
    compute_seasonal_weekly_profile,
    run_statistical_analysis,
    DAY_ORDER,
)
from energy_parser.report_generator import (
    generate_seasonal_chart,
    generate_monthly_bar_chart,
    generate_pdf_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yearly_df(days=60, freq="1h", consumption_base=10.0,
                    with_production=False):
    """Build a DataFrame spanning `days` with hourly data."""
    periods = int(days * 24)
    dates = pd.date_range("2024-01-01", freq=freq, periods=periods)
    np.random.seed(42)
    consumption = consumption_base + np.random.normal(0, 1, len(dates))
    consumption = np.clip(consumption, 0.1, None)

    data = {
        "Date & Time": dates,
        "Consumption (kW)": consumption,
    }
    if with_production:
        production = consumption_base * 0.5 + np.random.normal(0, 0.5, len(dates))
        production = np.clip(production, 0.0, None)
        data["Production (kW)"] = production

    return pd.DataFrame(data)


def _make_stats_result(days=60, with_production=False):
    """Build a full stats_result dict via run_statistical_analysis."""
    df = _make_yearly_df(days=days, with_production=with_production)
    return run_statistical_analysis(df, hours_per_interval=1.0), df


# ---------------------------------------------------------------------------
# TestGenerateSeasonalChart
# ---------------------------------------------------------------------------

class TestGenerateSeasonalChart:
    def test_returns_bytes(self):
        df = _make_yearly_df(days=90)
        profiles = compute_seasonal_weekly_profile(df, 1.0)
        winter_df = profiles["Consumption (kW)"]["Winter"]
        result = generate_seasonal_chart(winter_df, "Consumption (kW)", "Winter")
        assert isinstance(result, bytes)

    def test_png_header(self):
        df = _make_yearly_df(days=90)
        profiles = compute_seasonal_weekly_profile(df, 1.0)
        winter_df = profiles["Consumption (kW)"]["Winter"]
        result = generate_seasonal_chart(winter_df, "Consumption (kW)", "Winter")
        # PNG files start with \x89PNG
        assert result[:4] == b"\x89PNG"

    def test_non_empty(self):
        df = _make_yearly_df(days=90)
        profiles = compute_seasonal_weekly_profile(df, 1.0)
        winter_df = profiles["Consumption (kW)"]["Winter"]
        result = generate_seasonal_chart(winter_df, "Consumption (kW)", "Winter")
        assert len(result) > 1000  # A real PNG chart is at least a few KB


# ---------------------------------------------------------------------------
# TestGenerateMonthlyBarChart
# ---------------------------------------------------------------------------

class TestGenerateMonthlyBarChart:
    def test_returns_bytes(self):
        monthly = {m: float(m * 100) for m in range(1, 13)}
        result = generate_monthly_bar_chart(monthly, "Consumption (kW)")
        assert isinstance(result, bytes)

    def test_valid_png(self):
        monthly = {m: float(m * 100) for m in range(1, 13)}
        result = generate_monthly_bar_chart(monthly, "Consumption (kW)")
        assert result[:4] == b"\x89PNG"

    def test_non_empty(self):
        monthly = {m: float(m * 100) for m in range(1, 13)}
        result = generate_monthly_bar_chart(monthly, "Consumption (kW)")
        assert len(result) > 1000


# ---------------------------------------------------------------------------
# TestGeneratePdfReport
# ---------------------------------------------------------------------------

class TestGeneratePdfReport:
    def test_creates_file(self, tmp_path):
        stats_result, df = _make_stats_result()
        output = str(tmp_path / "report.pdf")
        result = generate_pdf_report(output, stats_result)
        assert os.path.isfile(result)

    def test_valid_pdf_header(self, tmp_path):
        stats_result, df = _make_stats_result()
        output = str(tmp_path / "report.pdf")
        generate_pdf_report(output, stats_result)
        with open(output, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_pdf_has_content(self, tmp_path):
        stats_result, df = _make_stats_result()
        output = str(tmp_path / "report.pdf")
        generate_pdf_report(output, stats_result)
        size = os.path.getsize(output)
        assert size > 5000  # A real PDF should be at least a few KB

    def test_with_kpi_data(self, tmp_path):
        stats_result, df = _make_stats_result()
        kpi = {
            "quality_score": 85,
            "completeness_pct": 98.5,
            "missing_values": 3,
            "timestamp_issues": 1,
            "processing_accuracy_pct": 99.2,
            "untrustworthiness": {"pct": 2.1, "rating": "Excellent"},
        }
        output = str(tmp_path / "report_kpi.pdf")
        result = generate_pdf_report(output, stats_result, kpi_data=kpi)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 5000

    def test_without_kpi_data(self, tmp_path):
        stats_result, df = _make_stats_result()
        output = str(tmp_path / "report_no_kpi.pdf")
        result = generate_pdf_report(output, stats_result, kpi_data=None)
        assert os.path.isfile(result)

    def test_with_logo_path(self, tmp_path):
        stats_result, df = _make_stats_result()
        # Use a non-existent logo — should not crash
        output = str(tmp_path / "report_logo.pdf")
        result = generate_pdf_report(output, stats_result,
                                      logo_path="nonexistent_logo.png")
        assert os.path.isfile(result)

    def test_with_real_logo_if_exists(self, tmp_path):
        stats_result, df = _make_stats_result()
        logo = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "Logo_new_white.png")
        output = str(tmp_path / "report_real_logo.pdf")
        result = generate_pdf_report(output, stats_result, logo_path=logo)
        assert os.path.isfile(result)

    def test_with_production_column(self, tmp_path):
        stats_result, df = _make_stats_result(with_production=True)
        output = str(tmp_path / "report_prod.pdf")
        result = generate_pdf_report(output, stats_result)
        assert os.path.isfile(result)
        # Should be larger than single-column report (more charts)
        assert os.path.getsize(result) > 5000

    def test_returns_output_path(self, tmp_path):
        stats_result, df = _make_stats_result()
        output = str(tmp_path / "report.pdf")
        result = generate_pdf_report(output, stats_result)
        assert result == output
