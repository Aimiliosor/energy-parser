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
    "daily_avg_kwh", "seasonal_profile", "peak_analysis",
    "frequency_histogram", "cumulative_distribution",
]

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _value_columns(df: pd.DataFrame) -> list[str]:
    """Return value column names (everything except Date & Time and data_source)."""
    return [c for c in df.columns if c not in ("Date & Time", "data_source")]


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


def _find_peak_duration(series: pd.Series, peak_idx: int,
                         threshold_ratio: float,
                         hours_per_interval: float) -> float:
    """Find how long consumption stays above threshold_ratio * peak_value.

    Walks outward from peak_idx in both directions until value drops
    below the threshold. Returns duration in hours.
    """
    peak_val = series.iloc[peak_idx]
    threshold = peak_val * threshold_ratio
    count = 1  # the peak itself

    # Walk left
    i = peak_idx - 1
    while i >= 0 and series.iloc[i] >= threshold:
        count += 1
        i -= 1

    # Walk right
    i = peak_idx + 1
    while i < len(series) and series.iloc[i] >= threshold:
        count += 1
        i += 1

    return count * hours_per_interval


def _compute_rate(series: pd.Series, peak_idx: int,
                  hours_per_interval: float, direction: str) -> float:
    """Compute rise rate (direction='rise') or fall rate (direction='fall').

    Rise rate: average kW/hour increase over the 3 intervals before peak.
    Fall rate: average kW/hour decrease over the 3 intervals after peak.
    Returns absolute kW per hour.
    """
    window = 3
    peak_val = series.iloc[peak_idx]

    if direction == "rise":
        start = max(0, peak_idx - window)
        if start == peak_idx:
            return 0.0
        base_val = series.iloc[start]
        intervals = peak_idx - start
    else:  # fall
        end = min(len(series) - 1, peak_idx + window)
        if end == peak_idx:
            return 0.0
        base_val = series.iloc[end]
        intervals = end - peak_idx

    if intervals == 0:
        return 0.0

    delta = abs(peak_val - base_val)
    return float(delta / (intervals * hours_per_interval))


def compute_peak_analysis(df: pd.DataFrame,
                           hours_per_interval: float,
                           top_n: int = 5,
                           duration_threshold: float = 0.9) -> dict:
    """Comprehensive peak consumption analysis.

    Identifies significant peaks (local maxima above the 90th percentile),
    analyses patterns, characteristics, and threshold behaviour.

    Parameters
    ----------
    df : DataFrame with "Date & Time" and value columns.
    hours_per_interval : hours between successive readings.
    top_n : number of top peaks to return (default 5).
    duration_threshold : fraction of peak value used for duration calc (0.9 = 90%).

    Returns dict keyed by column name, each containing:
      top_peaks, patterns, characteristics, thresholds.
    """
    result = {}
    value_cols = _value_columns(df)
    dt_col = pd.to_datetime(df["Date & Time"])

    # Determine if data_source filtering should be applied
    has_data_source = "data_source" in df.columns

    for col in value_cols:
        # Filter to original data only for peak detection
        if has_data_source:
            original_mask = df["data_source"] == "original"
            total_points = len(df[col].dropna())
            original_points = len(df.loc[original_mask, col].dropna())
            excluded_points = total_points - original_points
            excluded_pct = (excluded_points / total_points * 100) if total_points > 0 else 0.0
            filter_applied = excluded_points > 0

            # Use original data for peak detection
            analysis_df = df.loc[original_mask].copy()
            analysis_dt = pd.to_datetime(analysis_df["Date & Time"])
        else:
            total_points = len(df[col].dropna())
            original_points = total_points
            excluded_points = 0
            excluded_pct = 0.0
            filter_applied = False
            analysis_df = df
            analysis_dt = dt_col

        series = analysis_df[col].dropna()
        if len(series) < 5:
            peak_result = _empty_peak_result()
            peak_result["data_filtering"] = {
                "total_points": total_points,
                "original_points": original_points,
                "excluded_points": excluded_points,
                "excluded_pct": round(excluded_pct, 1),
                "filter_applied": filter_applied,
                "excluded_peaks_in_corrected": 0,
            }
            result[col] = peak_result
            continue

        values = series.values
        p90 = float(np.percentile(values, 90))
        p95 = float(np.percentile(values, 95))
        mean_val = float(np.mean(values))

        # --- Detect peaks: local maxima above 90th percentile ---
        # A peak is a point higher than both its neighbours and above p90.
        peak_indices = []
        for i in range(1, len(series) - 1):
            if (values[i] > values[i - 1] and values[i] >= values[i + 1]
                    and values[i] >= p90):
                peak_indices.append(series.index[i])

        # Also check first / last if they are above p90
        if len(values) > 0 and values[0] >= p90 and (len(values) < 2 or values[0] >= values[1]):
            peak_indices.insert(0, series.index[0])
        if len(values) > 1 and values[-1] >= p90 and values[-1] >= values[-2]:
            peak_indices.append(series.index[-1])

        # Remove duplicates, sort by value descending
        peak_indices = sorted(set(peak_indices),
                              key=lambda idx: analysis_df.loc[idx, col], reverse=True)

        # --- Top N peaks ---
        top_peaks = []
        for rank, idx in enumerate(peak_indices[:top_n], 1):
            ts = analysis_dt.loc[idx]
            val = float(analysis_df.loc[idx, col])

            # Position in the reset series for duration / rate calc
            pos = series.index.get_loc(idx)
            duration = _find_peak_duration(
                series, pos, duration_threshold, hours_per_interval)
            rise_rate = _compute_rate(series, pos, hours_per_interval, "rise")
            fall_rate = _compute_rate(series, pos, hours_per_interval, "fall")

            top_peaks.append({
                "rank": rank,
                "value": val,
                "timestamp": ts.strftime("%d/%m/%Y %H:%M"),
                "day_of_week": ts.day_name(),
                "month": MONTH_NAMES[ts.month - 1],
                "duration_hours": round(duration, 2),
                "rise_rate": round(rise_rate, 2),
                "fall_rate": round(fall_rate, 2),
            })

        # --- Patterns ---
        peak_timestamps = [analysis_dt.loc[i] for i in peak_indices]

        # Hourly distribution
        hourly_dist = {h: 0 for h in range(24)}
        for ts in peak_timestamps:
            hourly_dist[ts.hour] += 1

        # Daily distribution
        daily_dist = {d: 0 for d in DAY_ORDER}
        for ts in peak_timestamps:
            daily_dist[ts.day_name()] += 1

        # Monthly distribution
        monthly_dist = {m: 0 for m in range(1, 13)}
        for ts in peak_timestamps:
            monthly_dist[ts.month] += 1

        # Peak frequency above thresholds
        p99 = float(np.percentile(values, 99)) if len(values) > 0 else 0.0
        above_90 = int(np.sum(values >= p90))
        above_95 = int(np.sum(values >= p95))
        above_99 = int(np.sum(values >= p99))

        patterns = {
            "hourly_distribution": hourly_dist,
            "daily_distribution": daily_dist,
            "monthly_distribution": monthly_dist,
            "peak_frequency": {
                "above_90th": above_90,
                "above_95th": above_95,
                "above_99th": above_99,
            },
            "total_peaks_detected": len(peak_indices),
        }

        # --- Characteristics ---
        durations = []
        rise_rates = []
        fall_rates = []
        for idx in peak_indices:
            pos = series.index.get_loc(idx)
            durations.append(
                _find_peak_duration(series, pos, duration_threshold,
                                    hours_per_interval))
            rise_rates.append(
                _compute_rate(series, pos, hours_per_interval, "rise"))
            fall_rates.append(
                _compute_rate(series, pos, hours_per_interval, "fall"))

        avg_duration = float(np.mean(durations)) if durations else 0.0
        avg_rise = float(np.mean(rise_rates)) if rise_rates else 0.0
        avg_fall = float(np.mean(fall_rates)) if fall_rates else 0.0

        # Clustering: peaks within 24 h of another peak
        clustered = 0
        isolated = 0
        if peak_timestamps:
            sorted_pts = sorted(peak_timestamps)
            cluster_sizes = []
            current_cluster = [sorted_pts[0]]
            for i in range(1, len(sorted_pts)):
                if (sorted_pts[i] - sorted_pts[i - 1]).total_seconds() <= 86400:
                    current_cluster.append(sorted_pts[i])
                else:
                    cluster_sizes.append(len(current_cluster))
                    if len(current_cluster) == 1:
                        isolated += 1
                    else:
                        clustered += len(current_cluster)
                    current_cluster = [sorted_pts[i]]
            # Last cluster
            cluster_sizes.append(len(current_cluster))
            if len(current_cluster) == 1:
                isolated += 1
            else:
                clustered += len(current_cluster)
            avg_cluster = float(np.mean(cluster_sizes))
        else:
            avg_cluster = 0.0

        characteristics = {
            "avg_duration_hours": round(avg_duration, 2),
            "avg_rise_rate": round(avg_rise, 2),
            "avg_fall_rate": round(avg_fall, 2),
            "clustering": {
                "clustered_count": clustered,
                "isolated_count": isolated,
                "avg_cluster_size": round(avg_cluster, 2),
            },
        }

        # --- Thresholds ---
        time_above_p90 = float(np.sum(values >= p90)) * hours_per_interval
        time_above_p95 = float(np.sum(values >= p95)) * hours_per_interval
        peak_to_avg = float(np.max(values) / mean_val) if mean_val > 0 else 0.0

        thresholds = {
            "p90_value": round(p90, 2),
            "p95_value": round(p95, 2),
            "time_above_p90_hours": round(time_above_p90, 2),
            "time_above_p95_hours": round(time_above_p95, 2),
            "peak_to_avg_ratio": round(peak_to_avg, 2),
        }

        # --- Peak timeline data (for charts) ---
        peak_timeline = []
        for idx in peak_indices:
            ts = analysis_dt.loc[idx]
            peak_timeline.append({
                "timestamp": ts,
                "value": float(analysis_df.loc[idx, col]),
            })

        # --- Count excluded peaks in corrected data ---
        excluded_peaks_in_corrected = 0
        if has_data_source and filter_applied:
            corrected_mask = df["data_source"] != "original"
            corrected_series = df.loc[corrected_mask, col].dropna()
            if len(corrected_series) >= 3:
                corr_values = corrected_series.values
                for i in range(1, len(corrected_series) - 1):
                    if (corr_values[i] > corr_values[i - 1]
                            and corr_values[i] >= corr_values[i + 1]
                            and corr_values[i] >= p90):
                        excluded_peaks_in_corrected += 1

        # --- Data filtering transparency ---
        data_filtering = {
            "total_points": total_points,
            "original_points": original_points,
            "excluded_points": excluded_points,
            "excluded_pct": round(excluded_pct, 1),
            "filter_applied": filter_applied,
            "excluded_peaks_in_corrected": excluded_peaks_in_corrected,
        }

        result[col] = {
            "top_peaks": top_peaks,
            "patterns": patterns,
            "characteristics": characteristics,
            "thresholds": thresholds,
            "peak_timeline": peak_timeline,
            "data_filtering": data_filtering,
        }

    return result


def _empty_peak_result() -> dict:
    """Return an empty peak analysis result for columns with insufficient data."""
    return {
        "top_peaks": [],
        "patterns": {
            "hourly_distribution": {h: 0 for h in range(24)},
            "daily_distribution": {d: 0 for d in DAY_ORDER},
            "monthly_distribution": {m: 0 for m in range(1, 13)},
            "peak_frequency": {"above_90th": 0, "above_95th": 0, "above_99th": 0},
            "total_peaks_detected": 0,
        },
        "characteristics": {
            "avg_duration_hours": 0.0,
            "avg_rise_rate": 0.0,
            "avg_fall_rate": 0.0,
            "clustering": {"clustered_count": 0, "isolated_count": 0,
                           "avg_cluster_size": 0.0},
        },
        "thresholds": {
            "p90_value": 0.0, "p95_value": 0.0,
            "time_above_p90_hours": 0.0, "time_above_p95_hours": 0.0,
            "peak_to_avg_ratio": 0.0,
        },
        "peak_timeline": [],
    }


def compute_frequency_histogram(df: pd.DataFrame) -> dict:
    """Compute frequency histogram data for each value column.

    Uses Sturges' rule for bin count: k = ceil(1 + log2(n)).

    Returns dict keyed by column name:
    {
        "Consumption (kW)": {
            "bin_edges": list[float],
            "counts": list[int],
            "bin_width": float,
            "mean": float,
            "median": float,
            "std": float,
            "n_bins": int,
        }
    }
    """
    result = {}
    value_cols = _value_columns(df)

    for col in value_cols:
        series = df[col].dropna()
        if len(series) < 2:
            result[col] = {
                "bin_edges": [], "counts": [], "bin_width": 0.0,
                "mean": 0.0, "median": 0.0, "std": 0.0, "n_bins": 0,
            }
            continue

        # Sturges' rule
        n_bins = max(5, int(np.ceil(1 + np.log2(len(series)))))
        n_bins = min(n_bins, 100)  # cap for very large datasets

        counts, bin_edges = np.histogram(series.values, bins=n_bins)
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0.0

        result[col] = {
            "bin_edges": [float(e) for e in bin_edges],
            "counts": [int(c) for c in counts],
            "bin_width": round(bin_width, 4),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "n_bins": n_bins,
        }

    return result


def compute_cumulative_distribution(df: pd.DataFrame) -> dict:
    """Compute cumulative distribution data for each value column.

    Returns dict keyed by column name:
    {
        "Consumption (kW)": {
            "values": list[float],       # sorted values
            "cdf": list[float],          # cumulative probabilities 0-100
            "percentiles": {
                "p50": float,
                "p90": float,
                "p95": float,
            },
            "count": int,
        }
    }
    """
    result = {}
    value_cols = _value_columns(df)

    for col in value_cols:
        series = df[col].dropna().sort_values().reset_index(drop=True)
        if len(series) < 2:
            result[col] = {
                "values": [], "cdf": [],
                "percentiles": {"p50": 0.0, "p90": 0.0, "p95": 0.0},
                "count": 0,
            }
            continue

        values = series.values
        n = len(values)
        cdf = np.arange(1, n + 1) / n * 100  # percentage 0-100

        result[col] = {
            "values": [float(v) for v in values],
            "cdf": [float(c) for c in cdf],
            "percentiles": {
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
            },
            "count": n,
        }

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

    # Only compute peak analysis if requested
    if "peak_analysis" in selected_metrics:
        peaks = compute_peak_analysis(df, hours_per_interval)
    else:
        peaks = {}

    # Frequency histogram
    if "frequency_histogram" in selected_metrics:
        histogram = compute_frequency_histogram(df)
    else:
        histogram = {}

    # Cumulative distribution
    if "cumulative_distribution" in selected_metrics:
        cdf = compute_cumulative_distribution(df)
    else:
        cdf = {}

    return {
        "yearly": yearly,
        "seasonal": seasonal,
        "peaks": peaks,
        "histogram": histogram,
        "cdf": cdf,
        "selected_metrics": selected_metrics,
    }
