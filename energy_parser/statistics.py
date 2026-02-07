"""Statistical analysis for energy data.

Pure-function module. Operates on the transformed DataFrame
(columns: "Date & Time", "Consumption (kW)", optionally "Production (kW)").
"""

import pandas as pd
import numpy as np


SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

ALL_METRIC_KEYS = [
    "total_kwh", "mean_kw", "median_kw", "std_kw",
    "min_max_kw", "peak_times", "monthly_totals",
    "daily_avg_kwh", "seasonal_profile",
]


def _value_columns(df: pd.DataFrame) -> list[str]:
    """Return value column names (everything except Date & Time)."""
    return [c for c in df.columns if c != "Date & Time"]


def compute_yearly_stats(df: pd.DataFrame, hours_per_interval: float) -> dict:
    """Compute yearly consumption/production metrics.

    Returns dict with keys for each value column:
    {
        "Consumption (kW)": {
            "total_kwh": float,
            "mean_kw": float,
            "median_kw": float,
            "std_kw": float,
            "min_kw": float,
            "max_kw": float,
            "peak_timestamp": str,
            "off_peak_timestamp": str,
            "monthly_totals": {1: float, 2: float, ...},
            "daily_avg_kwh": float,
        },
        "Production (kW)": { ... }  # if column exists
    }
    """
    result = {}
    value_cols = _value_columns(df)

    for col in value_cols:
        series = df[col].dropna()
        if series.empty:
            result[col] = {
                "total_kwh": 0.0,
                "mean_kw": 0.0,
                "median_kw": 0.0,
                "std_kw": 0.0,
                "min_kw": 0.0,
                "max_kw": 0.0,
                "peak_timestamp": "N/A",
                "off_peak_timestamp": "N/A",
                "monthly_totals": {},
                "daily_avg_kwh": 0.0,
            }
            continue

        total_kwh = float(series.sum() * hours_per_interval)
        mean_kw = float(series.mean())
        median_kw = float(series.median())
        std_kw = float(series.std()) if len(series) > 1 else 0.0
        min_kw = float(series.min())
        max_kw = float(series.max())

        # Peak timestamp (max value)
        max_idx = series.idxmax()
        peak_timestamp = str(df.loc[max_idx, "Date & Time"])

        # Off-peak timestamp (min value > 0, or absolute min if all <= 0)
        positive_series = series[series > 0]
        if not positive_series.empty:
            min_idx = positive_series.idxmin()
        else:
            min_idx = series.idxmin()
        off_peak_timestamp = str(df.loc[min_idx, "Date & Time"])

        # Monthly totals (kWh)
        dt_col = pd.to_datetime(df["Date & Time"])
        monthly_totals = {}
        for month in range(1, 13):
            mask = dt_col.dt.month == month
            month_sum = df.loc[mask, col].dropna().sum()
            monthly_totals[month] = float(month_sum * hours_per_interval)

        # Daily average kWh
        num_days = max(1, (dt_col.max() - dt_col.min()).days)
        daily_avg_kwh = total_kwh / num_days

        result[col] = {
            "total_kwh": total_kwh,
            "mean_kw": mean_kw,
            "median_kw": median_kw,
            "std_kw": std_kw,
            "min_kw": min_kw,
            "max_kw": max_kw,
            "peak_timestamp": peak_timestamp,
            "off_peak_timestamp": off_peak_timestamp,
            "monthly_totals": monthly_totals,
            "daily_avg_kwh": daily_avg_kwh,
        }

    return result


def compute_seasonal_weekly_profile(df: pd.DataFrame,
                                     hours_per_interval: float) -> dict:
    """Compute typical weekly load profile per season.

    Groups by (season, day_of_week, hour) and aggregates mean kW.

    Returns:
    {
        "Consumption (kW)": {
            "Winter": DataFrame(index=hour 0-23, columns=Mon-Sun),
            "Spring": DataFrame(...),
            "Summer": DataFrame(...),
            "Autumn": DataFrame(...)
        },
        "Production (kW)": { ... }  # if column exists
    }
    """
    result = {}
    value_cols = _value_columns(df)

    dt_col = pd.to_datetime(df["Date & Time"])
    seasons = dt_col.dt.month.map(SEASON_MAP)
    day_names = dt_col.dt.day_name()
    hours = dt_col.dt.hour

    for col in value_cols:
        work_df = pd.DataFrame({
            "season": seasons,
            "day": day_names,
            "hour": hours,
            "value": df[col].values,
        })

        col_result = {}
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            season_data = work_df[work_df["season"] == season]
            if season_data.empty:
                # Empty DataFrame with correct shape
                col_result[season] = pd.DataFrame(
                    0.0, index=range(24), columns=DAY_ORDER)
                continue

            pivot = season_data.groupby(["hour", "day"])["value"].mean().unstack(fill_value=0)

            # Ensure all hours 0-23 and all days present
            pivot = pivot.reindex(index=range(24), fill_value=0)
            for day in DAY_ORDER:
                if day not in pivot.columns:
                    pivot[day] = 0.0
            pivot = pivot[DAY_ORDER]

            col_result[season] = pivot

        result[col] = col_result

    return result


def run_statistical_analysis(df: pd.DataFrame,
                              hours_per_interval: float,
                              selected_metrics: list[str] | None = None) -> dict:
    """Orchestrator. Calls compute_yearly_stats + compute_seasonal_weekly_profile.

    selected_metrics: list of metric keys to include, or None for all.
    Metric keys: "total_kwh", "mean_kw", "median_kw", "std_kw",
                 "min_max_kw", "peak_times", "monthly_totals",
                 "daily_avg_kwh", "seasonal_profile"

    Returns:
    {
        "yearly": <compute_yearly_stats result>,
        "seasonal": <compute_seasonal_weekly_profile result>,
        "selected_metrics": list[str]
    }
    """
    if selected_metrics is None:
        selected_metrics = list(ALL_METRIC_KEYS)

    yearly = compute_yearly_stats(df, hours_per_interval)

    # Only compute seasonal profiles if requested
    if "seasonal_profile" in selected_metrics:
        seasonal = compute_seasonal_weekly_profile(df, hours_per_interval)
    else:
        seasonal = {}

    return {
        "yearly": yearly,
        "seasonal": seasonal,
        "selected_metrics": selected_metrics,
    }
