"""Tests for energy_parser.statistics module."""

import pandas as pd
import numpy as np
import pytest

from energy_parser.statistics import (
    compute_yearly_stats,
    compute_seasonal_weekly_profile,
    compute_peak_analysis,
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

    def test_peak_analysis_included_when_selected(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = run_statistical_analysis(
            df, hours_per_interval=1.0,
            selected_metrics=["peak_analysis"])
        assert "Consumption (kW)" in result["peaks"]

    def test_peak_analysis_excluded_when_not_selected(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = run_statistical_analysis(
            df, hours_per_interval=1.0,
            selected_metrics=["total_kwh"])
        assert result["peaks"] == {}

    def test_peak_analysis_in_all_metric_keys(self):
        assert "peak_analysis" in ALL_METRIC_KEYS


# ---------------------------------------------------------------------------
# TestComputePeakAnalysis
# ---------------------------------------------------------------------------

class TestComputePeakAnalysis:
    def test_returns_top_peaks(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0, top_n=5)
        peaks = result["Consumption (kW)"]["top_peaks"]
        assert len(peaks) <= 5
        assert len(peaks) > 0

    def test_top_peaks_sorted_by_value(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0, top_n=5)
        peaks = result["Consumption (kW)"]["top_peaks"]
        values = [p["value"] for p in peaks]
        assert values == sorted(values, reverse=True)

    def test_top_peak_fields(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        peak = result["Consumption (kW)"]["top_peaks"][0]
        expected_keys = {"rank", "value", "timestamp", "day_of_week",
                         "month", "duration_hours", "rise_rate", "fall_rate"}
        assert expected_keys == set(peak.keys())

    def test_rank_starts_at_one(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0, top_n=3)
        peaks = result["Consumption (kW)"]["top_peaks"]
        ranks = [p["rank"] for p in peaks]
        assert ranks == [1, 2, 3]

    def test_patterns_hourly_distribution(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        hourly = result["Consumption (kW)"]["patterns"]["hourly_distribution"]
        assert len(hourly) == 24
        assert all(h in hourly for h in range(24))
        assert all(v >= 0 for v in hourly.values())

    def test_patterns_daily_distribution(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        daily = result["Consumption (kW)"]["patterns"]["daily_distribution"]
        assert len(daily) == 7
        assert all(d in daily for d in DAY_ORDER)

    def test_patterns_monthly_distribution(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        monthly = result["Consumption (kW)"]["patterns"]["monthly_distribution"]
        assert len(monthly) == 12
        assert all(m in monthly for m in range(1, 13))

    def test_peak_frequency(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        freq = result["Consumption (kW)"]["patterns"]["peak_frequency"]
        assert freq["above_90th"] >= freq["above_95th"] >= freq["above_99th"]
        assert freq["above_90th"] > 0

    def test_total_peaks_detected(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        total = result["Consumption (kW)"]["patterns"]["total_peaks_detected"]
        assert total > 0

    def test_characteristics_keys(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        chars = result["Consumption (kW)"]["characteristics"]
        assert "avg_duration_hours" in chars
        assert "avg_rise_rate" in chars
        assert "avg_fall_rate" in chars
        assert "clustering" in chars

    def test_duration_positive(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        chars = result["Consumption (kW)"]["characteristics"]
        assert chars["avg_duration_hours"] > 0

    def test_clustering_counts(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        clustering = result["Consumption (kW)"]["characteristics"]["clustering"]
        total = clustering["clustered_count"] + clustering["isolated_count"]
        detected = result["Consumption (kW)"]["patterns"]["total_peaks_detected"]
        assert total == detected

    def test_thresholds_keys(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        thresholds = result["Consumption (kW)"]["thresholds"]
        assert "p90_value" in thresholds
        assert "p95_value" in thresholds
        assert "time_above_p90_hours" in thresholds
        assert "time_above_p95_hours" in thresholds
        assert "peak_to_avg_ratio" in thresholds

    def test_p95_greater_than_p90(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        thresholds = result["Consumption (kW)"]["thresholds"]
        assert thresholds["p95_value"] >= thresholds["p90_value"]

    def test_peak_to_avg_ratio_above_one(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        ratio = result["Consumption (kW)"]["thresholds"]["peak_to_avg_ratio"]
        assert ratio > 1.0

    def test_time_above_p90_greater_than_p95(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        thresholds = result["Consumption (kW)"]["thresholds"]
        assert thresholds["time_above_p90_hours"] >= thresholds["time_above_p95_hours"]

    def test_peak_timeline(self):
        df = _make_yearly_df(days=90, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        timeline = result["Consumption (kW)"]["peak_timeline"]
        assert len(timeline) > 0
        assert "timestamp" in timeline[0]
        assert "value" in timeline[0]

    def test_dual_column(self):
        df = _make_yearly_df(days=90, freq="1h", with_production=True)
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        assert "Consumption (kW)" in result
        assert "Production (kW)" in result

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Date & Time": pd.Series(dtype="datetime64[ns]"),
                           "Consumption (kW)": pd.Series(dtype=float)})
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        col = result["Consumption (kW)"]
        assert col["top_peaks"] == []
        assert col["patterns"]["total_peaks_detected"] == 0
        assert col["thresholds"]["p90_value"] == 0.0

    def test_very_short_series(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({
            "Date & Time": dates,
            "Consumption (kW)": [5.0, 10.0, 5.0],
        })
        result = compute_peak_analysis(df, hours_per_interval=1.0)
        col = result["Consumption (kW)"]
        assert col["top_peaks"] == []
        assert col["patterns"]["total_peaks_detected"] == 0

    def test_custom_top_n(self):
        df = _make_yearly_df(days=365, freq="1h")
        result = compute_peak_analysis(df, hours_per_interval=1.0, top_n=3)
        peaks = result["Consumption (kW)"]["top_peaks"]
        assert len(peaks) <= 3

    def test_fifteen_min_interval(self):
        df = _make_yearly_df(days=30, freq="15min", consumption_base=20.0)
        result = compute_peak_analysis(df, hours_per_interval=0.25)
        chars = result["Consumption (kW)"]["characteristics"]
        # Duration should be in 0.25h increments
        for peak in result["Consumption (kW)"]["top_peaks"]:
            assert peak["duration_hours"] >= 0.25
