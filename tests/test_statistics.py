"""Tests for energy_parser.statistics module."""

import pandas as pd
import numpy as np
import pytest

from energy_parser.statistics import (
    compute_yearly_stats,
    compute_seasonal_weekly_profile,
    run_statistical_analysis,
    SEASON_MAP,
    DAY_ORDER,
    ALL_METRIC_KEYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yearly_df(days=365, freq="1h", consumption_base=10.0,
                    with_production=False):
    """Build a DataFrame spanning `days` with hourly data."""
    periods = int(days * 24 / (pd.Timedelta(freq).total_seconds() / 3600))
    dates = pd.date_range("2024-01-01", freq=freq, periods=periods)
    np.random.seed(42)
    consumption = consumption_base + np.random.normal(0, 1, len(dates))
    consumption = np.clip(consumption, 0.1, None)  # no negatives

    data = {
        "Date & Time": dates,
        "Consumption (kW)": consumption,
    }
    if with_production:
        production = consumption_base * 0.5 + np.random.normal(0, 0.5, len(dates))
        production = np.clip(production, 0.0, None)
        data["Production (kW)"] = production

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# TestComputeYearlyStats
# ---------------------------------------------------------------------------

class TestComputeYearlyStats:
    def test_total_kwh_calculation(self):
        df = _make_yearly_df(days=365, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        # total_kwh = sum * hours_per_interval
        expected = df["Consumption (kW)"].sum() * 1.0
        assert abs(col_stats["total_kwh"] - expected) < 0.01

    def test_total_kwh_with_different_interval(self):
        df = _make_yearly_df(days=30, freq="15min", consumption_base=20.0)
        stats = compute_yearly_stats(df, hours_per_interval=0.25)
        col_stats = stats["Consumption (kW)"]
        expected = df["Consumption (kW)"].sum() * 0.25
        assert abs(col_stats["total_kwh"] - expected) < 0.01

    def test_monthly_totals_sum_equals_total(self):
        df = _make_yearly_df(days=365, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        monthly_sum = sum(col_stats["monthly_totals"].values())
        assert abs(monthly_sum - col_stats["total_kwh"]) < 1.0

    def test_monthly_totals_has_12_months(self):
        df = _make_yearly_df(days=365, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        assert len(col_stats["monthly_totals"]) == 12
        for month in range(1, 13):
            assert month in col_stats["monthly_totals"]

    def test_peak_timestamp_is_at_max_value(self):
        df = _make_yearly_df(days=30, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        max_idx = df["Consumption (kW)"].idxmax()
        expected_ts = str(df.loc[max_idx, "Date & Time"])
        assert col_stats["peak_timestamp"] == expected_ts

    def test_off_peak_timestamp_is_min_positive(self):
        df = _make_yearly_df(days=30, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        positive = df[df["Consumption (kW)"] > 0]["Consumption (kW)"]
        min_idx = positive.idxmin()
        expected_ts = str(df.loc[min_idx, "Date & Time"])
        assert col_stats["off_peak_timestamp"] == expected_ts

    def test_dual_column(self):
        df = _make_yearly_df(days=30, freq="1h", with_production=True)
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        assert "Consumption (kW)" in stats
        assert "Production (kW)" in stats
        assert stats["Production (kW)"]["total_kwh"] > 0

    def test_single_column(self):
        df = _make_yearly_df(days=30, freq="1h", with_production=False)
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        assert "Consumption (kW)" in stats
        assert "Production (kW)" not in stats

    def test_mean_median_std(self):
        df = _make_yearly_df(days=30, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        assert abs(col_stats["mean_kw"] - df["Consumption (kW)"].mean()) < 0.01
        assert abs(col_stats["median_kw"] - df["Consumption (kW)"].median()) < 0.01
        assert abs(col_stats["std_kw"] - df["Consumption (kW)"].std()) < 0.01

    def test_min_max(self):
        df = _make_yearly_df(days=30, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        assert abs(col_stats["min_kw"] - df["Consumption (kW)"].min()) < 0.01
        assert abs(col_stats["max_kw"] - df["Consumption (kW)"].max()) < 0.01

    def test_handles_nan_values(self):
        df = _make_yearly_df(days=30, freq="1h")
        df.loc[5:10, "Consumption (kW)"] = np.nan
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        assert col_stats["total_kwh"] > 0
        assert not np.isnan(col_stats["mean_kw"])

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Date & Time": pd.Series(dtype="datetime64[ns]"),
                           "Consumption (kW)": pd.Series(dtype=float)})
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        assert col_stats["total_kwh"] == 0.0
        assert col_stats["peak_timestamp"] == "N/A"

    def test_daily_avg_kwh(self):
        df = _make_yearly_df(days=365, freq="1h")
        stats = compute_yearly_stats(df, hours_per_interval=1.0)
        col_stats = stats["Consumption (kW)"]
        assert col_stats["daily_avg_kwh"] > 0
        # daily avg ~ total / 365
        assert abs(col_stats["daily_avg_kwh"] - col_stats["total_kwh"] / 365) < 1.0


# ---------------------------------------------------------------------------
# TestComputeSeasonalWeeklyProfile
# ---------------------------------------------------------------------------

class TestComputeSeasonalWeeklyProfile:
    def test_all_four_seasons_present(self):
        df = _make_yearly_df(days=365, freq="1h")
        profiles = compute_seasonal_weekly_profile(df, hours_per_interval=1.0)
        col_profiles = profiles["Consumption (kW)"]
        assert set(col_profiles.keys()) == {"Winter", "Spring", "Summer", "Autumn"}

    def test_profile_shape(self):
        df = _make_yearly_df(days=365, freq="1h")
        profiles = compute_seasonal_weekly_profile(df, hours_per_interval=1.0)
        for season_name, season_df in profiles["Consumption (kW)"].items():
            assert season_df.shape == (24, 7), f"{season_name} shape is {season_df.shape}"
            assert list(season_df.columns) == DAY_ORDER
            assert list(season_df.index) == list(range(24))

    def test_correct_season_assignment(self):
        # Create data only in January (should be Winter)
        dates = pd.date_range("2024-01-01", "2024-01-31 23:00", freq="1h")
        df = pd.DataFrame({
            "Date & Time": dates,
            "Consumption (kW)": 10.0,
        })
        profiles = compute_seasonal_weekly_profile(df, hours_per_interval=1.0)
        winter = profiles["Consumption (kW)"]["Winter"]
        spring = profiles["Consumption (kW)"]["Spring"]
        # Winter should have non-zero values, Spring should be all zeros
        assert winter.values.sum() > 0
        assert spring.values.sum() == 0

    def test_handles_missing_season(self):
        # Data only covers 3 months (one season)
        dates = pd.date_range("2024-06-01", "2024-08-31 23:00", freq="1h")
        df = pd.DataFrame({
            "Date & Time": dates,
            "Consumption (kW)": 10.0,
        })
        profiles = compute_seasonal_weekly_profile(df, hours_per_interval=1.0)
        summer = profiles["Consumption (kW)"]["Summer"]
        winter = profiles["Consumption (kW)"]["Winter"]
        assert summer.values.sum() > 0
        assert winter.values.sum() == 0

    def test_dual_column_profiles(self):
        df = _make_yearly_df(days=365, freq="1h", with_production=True)
        profiles = compute_seasonal_weekly_profile(df, hours_per_interval=1.0)
        assert "Consumption (kW)" in profiles
        assert "Production (kW)" in profiles

    def test_profile_values_are_means(self):
        # All consumption = 10.0, so profile should be ~10.0 everywhere
        dates = pd.date_range("2024-01-01", periods=24*7*4, freq="1h")
        df = pd.DataFrame({
            "Date & Time": dates,
            "Consumption (kW)": 10.0,
        })
        profiles = compute_seasonal_weekly_profile(df, hours_per_interval=1.0)
        winter = profiles["Consumption (kW)"]["Winter"]
        # All values in winter should be 10.0 (or 0 for days not present)
        non_zero = winter.values[winter.values > 0]
        if len(non_zero) > 0:
            assert np.allclose(non_zero, 10.0, atol=0.01)


# ---------------------------------------------------------------------------
# TestRunStatisticalAnalysis
# ---------------------------------------------------------------------------

class TestRunStatisticalAnalysis:
    def test_full_run_returns_all_keys(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = run_statistical_analysis(df, hours_per_interval=1.0)
        assert "yearly" in result
        assert "seasonal" in result
        assert "selected_metrics" in result

    def test_selected_metrics_filtering(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = run_statistical_analysis(
            df, hours_per_interval=1.0,
            selected_metrics=["total_kwh", "mean_kw"])
        assert result["selected_metrics"] == ["total_kwh", "mean_kw"]
        # seasonal should be empty since "seasonal_profile" not selected
        assert result["seasonal"] == {}

    def test_seasonal_profile_included_when_selected(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = run_statistical_analysis(
            df, hours_per_interval=1.0,
            selected_metrics=["seasonal_profile"])
        assert "Consumption (kW)" in result["seasonal"]

    def test_none_selected_metrics_returns_all(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = run_statistical_analysis(df, hours_per_interval=1.0,
                                          selected_metrics=None)
        assert result["selected_metrics"] == ALL_METRIC_KEYS

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Date & Time": pd.Series(dtype="datetime64[ns]"),
                           "Consumption (kW)": pd.Series(dtype=float)})
        result = run_statistical_analysis(df, hours_per_interval=1.0)
        assert result["yearly"]["Consumption (kW)"]["total_kwh"] == 0.0
