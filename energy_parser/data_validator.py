"""Data Quality Validation & KPI computation.

Pure-function module. Consumes the quality_report dict from
run_quality_check() — does NOT duplicate its detection logic.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Input / Output Integrity Functions
# ---------------------------------------------------------------------------

def check_record_count(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    """Compare row counts between original and processed DataFrames.

    PASS  – nat_dropped >= 0 and <= 5 % loss
    WARN  – > 5 % loss
    FAIL  – processed has more rows than original
    """
    orig_count = len(original_df)
    proc_count = len(processed_df)

    if orig_count == 0:
        return {"name": "Record Count", "status": "WARN",
                "details": "Original DataFrame is empty"}

    if proc_count > orig_count:
        return {"name": "Record Count", "status": "FAIL",
                "details": f"Processed ({proc_count}) > original ({orig_count})"}

    loss_pct = (orig_count - proc_count) / orig_count * 100
    if loss_pct <= 5:
        return {"name": "Record Count", "status": "PASS",
                "details": f"{proc_count}/{orig_count} rows retained ({loss_pct:.1f}% loss)"}
    return {"name": "Record Count", "status": "WARN",
            "details": f"{proc_count}/{orig_count} rows retained ({loss_pct:.1f}% loss, >5%)"}


def check_valid_value_preservation(original_df: pd.DataFrame,
                                   processed_df: pd.DataFrame,
                                   consumption_col: int,
                                   production_col: int | None) -> dict:
    """Count non-NaN values in raw consumption col vs processed 'Consumption (kW)'.

    PASS >= 95 %, WARN >= 80 %, FAIL < 80 %
    """
    orig_series = original_df.iloc[:, consumption_col]
    orig_valid = int(orig_series.notna().sum())

    if "Consumption (kW)" in processed_df.columns:
        proc_valid = int(processed_df["Consumption (kW)"].notna().sum())
    else:
        # Fallback: first value column
        value_cols = [c for c in processed_df.columns if c != "Date & Time"]
        if not value_cols:
            return {"name": "Value Preservation", "status": "WARN",
                    "details": "No value columns found in processed data"}
        proc_valid = int(processed_df[value_cols[0]].notna().sum())

    if orig_valid == 0:
        pct = 100.0
    else:
        pct = proc_valid / orig_valid * 100

    if pct >= 95:
        status = "PASS"
    elif pct >= 80:
        status = "WARN"
    else:
        status = "FAIL"

    return {"name": "Value Preservation", "status": status,
            "details": f"{proc_valid}/{orig_valid} valid values preserved ({pct:.1f}%)"}


def check_unexpected_nans(original_df: pd.DataFrame,
                          processed_df: pd.DataFrame,
                          consumption_col: int,
                          production_col: int | None) -> dict:
    """Detect NaN introduced during processing (new NaNs = processed - original)."""
    orig_nans = int(original_df.iloc[:, consumption_col].isna().sum())

    if "Consumption (kW)" in processed_df.columns:
        proc_nans = int(processed_df["Consumption (kW)"].isna().sum())
    else:
        value_cols = [c for c in processed_df.columns if c != "Date & Time"]
        proc_nans = int(processed_df[value_cols[0]].isna().sum()) if value_cols else 0

    new_nans = max(0, proc_nans - orig_nans)

    if new_nans == 0:
        return {"name": "Unexpected NaNs", "status": "PASS",
                "details": "No new NaN values introduced"}
    return {"name": "Unexpected NaNs", "status": "WARN",
            "details": f"{new_nans} new NaN values introduced during processing"}


def check_sum_preservation(before_df: pd.DataFrame, after_df: pd.DataFrame) -> dict:
    """Compare value column sums pre/post correction.

    PASS < 1 % drift, WARN < 5 %, FAIL >= 5 %
    """
    def _col_sum(df):
        value_cols = [c for c in df.columns if c != "Date & Time"]
        return sum(df[c].sum() for c in value_cols)

    before_sum = _col_sum(before_df)
    after_sum = _col_sum(after_df)

    if before_sum == 0 and after_sum == 0:
        return {"name": "Sum Preservation", "status": "PASS",
                "details": "Both sums are zero — no drift"}
    if before_sum == 0:
        return {"name": "Sum Preservation", "status": "WARN",
                "details": "Before-sum is zero, cannot compute drift percentage"}

    drift_pct = abs(after_sum - before_sum) / abs(before_sum) * 100

    if drift_pct < 1:
        status = "PASS"
    elif drift_pct < 5:
        status = "WARN"
    else:
        status = "FAIL"

    return {"name": "Sum Preservation", "status": status,
            "details": f"Sum drift: {drift_pct:.2f}% (before={before_sum:.2f}, after={after_sum:.2f})"}


# ---------------------------------------------------------------------------
# Enhanced Quality Analysis (on top of existing quality_report)
# ---------------------------------------------------------------------------

def classify_outliers_by_severity(quality_report: dict) -> dict:
    """Classify each outlier by severity based on ratio to median.

    low: ratio <= 5, medium: 5-10, high: > 10
    """
    outliers = quality_report.get("outliers", [])
    counts = {"low": 0, "medium": 0, "high": 0, "total": len(outliers)}

    for o in outliers:
        value = o["value"]
        median = o["median"]
        if median <= 0:
            counts["low"] += 1
            continue

        if o["type"] == "high":
            ratio = value / median
        else:
            ratio = median / value if value != 0 else float("inf")

        if ratio <= 5:
            counts["low"] += 1
        elif ratio <= 10:
            counts["medium"] += 1
        else:
            counts["high"] += 1

    return counts


def calculate_completeness(quality_report: dict) -> float:
    """(total_rows / expected_timestamps) * 100, clamped [0, 100]."""
    total_rows = quality_report.get("total_rows", 0)
    expected = quality_report.get("expected_timestamps", 0)

    if expected <= 0:
        return 100.0 if total_rows > 0 else 0.0

    pct = (total_rows / expected) * 100
    return max(0.0, min(100.0, pct))


def check_impossible_spikes(df: pd.DataFrame, window: int = 10) -> list[dict]:
    """Values exceeding 10x rolling average -> spike detected."""
    spikes = []
    value_cols = [c for c in df.columns if c != "Date & Time"]

    for col in value_cols:
        series = df[col].dropna()
        if len(series) < window + 1:
            continue
        rolling_avg = series.rolling(window=window, min_periods=1).mean().shift(1)
        for idx in series.index:
            avg = rolling_avg.get(idx)
            val = series.get(idx)
            if avg is not None and not pd.isna(avg) and avg > 0 and val > avg * 10:
                spikes.append({"row": idx, "column": col,
                               "value": float(val), "rolling_avg": float(avg)})
    return spikes


def calculate_temporal_summary(quality_report: dict) -> dict:
    """Aggregate: gaps count, missing timestamps, duplicates count."""
    return {
        "gaps_count": len(quality_report.get("gaps", [])),
        "missing_timestamps": quality_report.get("missing_timestamps", 0),
        "duplicates_count": len(quality_report.get("duplicates", [])),
    }


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Per value column: mean, median, std, min, max, skewness, kurtosis."""
    stats = {}
    value_cols = [c for c in df.columns if c != "Date & Time"]

    for col in value_cols:
        series = df[col].dropna()
        if series.empty:
            stats[col] = {"mean": 0, "median": 0, "std": 0,
                          "min": 0, "max": 0, "skewness": 0, "kurtosis": 0}
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "skewness": float(series.skew()) if len(series) > 2 else 0.0,
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0.0,
        }
    return stats


# ---------------------------------------------------------------------------
# Composite Quality Score (0-100)
# ---------------------------------------------------------------------------

def calculate_quality_score(completeness_pct: float,
                            integrity_results: list[dict],
                            temporal_summary: dict,
                            consistency_issues: dict,
                            outlier_classification: dict) -> tuple[int, dict]:
    """Calculate composite quality score from 0-100.

    Weights:
      Completeness (30 pts): linear scale from completeness %
      Integrity    (25 pts): deductions per FAIL/WARN check
      Temporal     (20 pts): -1 per gap (cap 15), -1 per duplicate group (cap 5)
      Consistency  (15 pts): -3 per negative issue (cap 10), -2 per spike (cap 5)
      Outlier      (10 pts): -0.5 low, -1.0 medium, -2.0 high (cap 10)
    """
    breakdown = {}

    # Completeness (30 pts)
    completeness_pts = max(0, min(30, completeness_pct / 100 * 30))
    breakdown["completeness"] = round(completeness_pts, 1)

    # Integrity (25 pts)
    integrity_pts = 25.0
    for result in integrity_results:
        if result["status"] == "FAIL":
            integrity_pts -= 8
        elif result["status"] == "WARN":
            integrity_pts -= 3
    integrity_pts = max(0, integrity_pts)
    breakdown["integrity"] = round(integrity_pts, 1)

    # Temporal (20 pts)
    gap_deduction = min(15, temporal_summary.get("gaps_count", 0) * 1)
    dup_deduction = min(5, temporal_summary.get("duplicates_count", 0) * 1)
    temporal_pts = max(0, 20 - gap_deduction - dup_deduction)
    breakdown["temporal"] = round(temporal_pts, 1)

    # Consistency (15 pts)
    neg_deduction = min(10, consistency_issues.get("negatives_count", 0) * 3)
    spike_deduction = min(5, consistency_issues.get("spikes_count", 0) * 2)
    consistency_pts = max(0, 15 - neg_deduction - spike_deduction)
    breakdown["consistency"] = round(consistency_pts, 1)

    # Outlier impact (10 pts)
    outlier_deduction = (
        outlier_classification.get("low", 0) * 0.5
        + outlier_classification.get("medium", 0) * 1.0
        + outlier_classification.get("high", 0) * 2.0
    )
    outlier_deduction = min(10, outlier_deduction)
    outlier_pts = max(0, 10 - outlier_deduction)
    breakdown["outlier_impact"] = round(outlier_pts, 1)

    total = completeness_pts + integrity_pts + temporal_pts + consistency_pts + outlier_pts
    score = max(0, min(100, int(round(total))))

    return score, breakdown


# ---------------------------------------------------------------------------
# Untrustworthiness Metric
# ---------------------------------------------------------------------------

def calculate_untrustworthiness(df: pd.DataFrame,
                                quality_report: dict,
                                outlier_classification: dict,
                                spikes: list[dict]) -> dict:
    """Calculate what percentage of data points are flagged as unreliable.

    Counts unique row indices affected by: missing values, outliers,
    timestamp gaps, negative values, duplicates, and impossible spikes.

    Returns dict with pct, flagged_count, total_count, rating, and breakdown.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {"pct": 0.0, "flagged": 0, "total": 0,
                "rating": "Excellent", "color_tier": "green",
                "breakdown": {}}

    flagged_rows = set()
    breakdown = {}

    # Missing values (NaN in value columns)
    value_cols = [c for c in df.columns if c != "Date & Time"]
    missing_rows = set()
    for col in value_cols:
        nan_indices = df[df[col].isna()].index.tolist()
        missing_rows.update(nan_indices)
    flagged_rows.update(missing_rows)
    breakdown["missing_values"] = len(missing_rows)

    # Outliers
    outlier_rows = set()
    for o in quality_report.get("outliers", []):
        outlier_rows.add(o["row"])
    flagged_rows.update(outlier_rows)
    breakdown["outliers"] = len(outlier_rows)

    # Timestamp gaps — count rows adjacent to gaps
    gap_rows = set()
    for g in quality_report.get("gaps", []):
        gap_rows.add(g["after_row"])
        if g["after_row"] + 1 < total_rows:
            gap_rows.add(g["after_row"] + 1)
    # Also add missing timestamp count (virtual rows that don't exist)
    missing_ts = quality_report.get("missing_timestamps", 0)
    flagged_rows.update(gap_rows)
    breakdown["timestamp_gaps"] = len(gap_rows) + missing_ts

    # Negative values
    neg_rows = set()
    for n in quality_report.get("negatives", []):
        neg_rows.add(n["row"])
    flagged_rows.update(neg_rows)
    breakdown["negatives"] = len(neg_rows)

    # Duplicates
    dup_rows = set()
    for d in quality_report.get("duplicates", []):
        for idx in d["row_indices"]:
            dup_rows.add(idx)
    flagged_rows.update(dup_rows)
    breakdown["duplicates"] = len(dup_rows)

    # Impossible spikes
    spike_rows = set()
    for s in spikes:
        spike_rows.add(s["row"])
    flagged_rows.update(spike_rows)
    breakdown["spikes"] = len(spike_rows)

    # Total flagged includes actual flagged rows + virtual missing timestamps
    total_flagged = len(flagged_rows) + missing_ts
    # Denominator includes actual rows + missing timestamps (the full expected dataset)
    total_considered = total_rows + missing_ts

    pct = (total_flagged / total_considered * 100) if total_considered > 0 else 0.0
    pct = min(100.0, pct)

    # Rating tiers
    if pct <= 5:
        rating, color_tier = "Excellent", "green"
    elif pct <= 15:
        rating, color_tier = "Acceptable", "yellow"
    elif pct <= 30:
        rating, color_tier = "Poor", "orange"
    else:
        rating, color_tier = "Critical", "red"

    return {
        "pct": round(pct, 1),
        "flagged": total_flagged,
        "total": total_considered,
        "rating": rating,
        "color_tier": color_tier,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Actionable Recommendations
# ---------------------------------------------------------------------------

def generate_recommendations(quality_report: dict,
                             outlier_classification: dict,
                             temporal_summary: dict,
                             spikes: list[dict],
                             untrustworthiness: dict) -> list[dict]:
    """Generate prioritized, actionable recommendations based on detected issues.

    Each recommendation has: priority (1=critical), category, message.
    Sorted by priority (most critical first).
    """
    recs = []

    # --- Priority 1 (Critical) ---

    # Negative values
    negatives = quality_report.get("negatives", [])
    if negatives:
        cols = sorted(set(n["column"] for n in negatives))
        recs.append({
            "priority": 1,
            "category": "Negative Values",
            "message": (f"Verify meter readings — {len(negatives)} negative consumption "
                        f"value(s) detected in {', '.join(cols)}. Negative readings typically "
                        f"indicate meter errors or incorrect data export configuration."),
        })

    # High-severity outliers
    if outlier_classification.get("high", 0) > 0:
        high_count = outlier_classification["high"]
        outlier_dates = []
        for o in quality_report.get("outliers", []):
            if o["median"] > 0 and (o["value"] / o["median"] if o["type"] == "high"
                                     else o["median"] / o["value"] if o["value"] != 0
                                     else float("inf")) > 10:
                outlier_dates.append(str(o["row"]))
        date_hint = f" at rows {', '.join(outlier_dates[:5])}" if outlier_dates else ""
        if len(outlier_dates) > 5:
            date_hint += f" and {len(outlier_dates) - 5} more"
        recs.append({
            "priority": 1,
            "category": "Severe Outliers",
            "message": (f"Review {high_count} extreme outlier(s){date_hint} — "
                        f"values exceed 10x the median. Possible meter malfunction "
                        f"or data corruption. Consider replacing with interpolated values."),
        })

    # --- Priority 2 (Important) ---

    # Timestamp gaps
    gaps = quality_report.get("gaps", [])
    missing_ts = quality_report.get("missing_timestamps", 0)
    if gaps:
        granularity = quality_report.get("granularity", "unknown")
        recs.append({
            "priority": 2,
            "category": "Timestamp Gaps",
            "message": (f"Check data export settings — found {len(gaps)} gap(s) with "
                        f"{missing_ts} missing timestamp(s). Expecting {granularity} intervals "
                        f"but found discontinuities. Fill gaps using previous-day values or "
                        f"interpolation to ensure continuous time series."),
        })

    # Missing values
    missing_values = quality_report.get("missing_values", {})
    total_missing = sum(v["count"] for v in missing_values.values())
    if total_missing > 0:
        cols = list(missing_values.keys())
        recs.append({
            "priority": 2,
            "category": "Missing Values",
            "message": (f"Fill {total_missing} missing value(s) in {', '.join(cols)}. "
                        f"Use previous-day fill for periodic data or linear interpolation "
                        f"for short gaps. Remove incomplete periods at the start/end of the "
                        f"dataset if they cannot be reliably filled."),
        })

    # Duplicates
    duplicates = quality_report.get("duplicates", [])
    if duplicates:
        total_dup_rows = sum(len(d["row_indices"]) for d in duplicates)
        recs.append({
            "priority": 2,
            "category": "Duplicate Timestamps",
            "message": (f"Resolve {len(duplicates)} duplicate timestamp group(s) "
                        f"affecting {total_dup_rows} rows. Keep the first occurrence, "
                        f"average the values, or investigate if they represent different "
                        f"meter channels that need separate processing."),
        })

    # --- Priority 3 (Advisory) ---

    # Medium/low outliers
    med_count = outlier_classification.get("medium", 0)
    low_count = outlier_classification.get("low", 0)
    if med_count > 0 or low_count > 0:
        parts = []
        if med_count > 0:
            parts.append(f"{med_count} moderate")
        if low_count > 0:
            parts.append(f"{low_count} mild")
        recs.append({
            "priority": 3,
            "category": "Outliers",
            "message": (f"Review {' and '.join(parts)} outlier(s) — these may represent "
                        f"unusual but valid events (holidays, maintenance, weather extremes). "
                        f"Cap at threshold values if they skew analysis, or keep if explainable."),
        })

    # Impossible spikes
    if spikes:
        recs.append({
            "priority": 3,
            "category": "Consumption Spikes",
            "message": (f"Investigate {len(spikes)} sudden spike(s) exceeding 10x the "
                        f"rolling average. These often indicate meter resets, data "
                        f"accumulation errors, or brief equipment faults."),
        })

    # --- Priority 4 (General) ---

    # Overall quality general advice
    untrust_pct = untrustworthiness.get("pct", 0)
    if untrust_pct > 30:
        recs.append({
            "priority": 4,
            "category": "General",
            "message": ("Data quality is critical — consider re-exporting from the source "
                        "system or contacting the data provider. Over 30% of records have "
                        "quality issues that may significantly impact analysis accuracy."),
        })
    elif untrust_pct > 15:
        recs.append({
            "priority": 4,
            "category": "General",
            "message": ("Data quality is below acceptable thresholds. Apply automated "
                        "corrections (gap filling, outlier capping) and verify results "
                        "against known billing or metering data before using in production."),
        })

    # If no issues at all
    if not recs:
        recs.append({
            "priority": 4,
            "category": "General",
            "message": ("Data quality is excellent — no significant issues detected. "
                        "The dataset is ready for analysis and export."),
        })

    # Sort by priority
    recs.sort(key=lambda r: r["priority"])
    return recs


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_validation(df: pd.DataFrame,
                   quality_report: dict,
                   original_df: pd.DataFrame | None = None,
                   pre_correction_df: pd.DataFrame | None = None,
                   consumption_col: int | None = None,
                   production_col: int | None = None) -> dict:
    """Run full validation and return KPI dict.

    When original_df is None, integrity checks are skipped (status="N/A").
    """
    # --- Enhanced quality analysis ---
    outlier_classification = classify_outliers_by_severity(quality_report)
    completeness_pct = calculate_completeness(quality_report)
    spikes = check_impossible_spikes(df)
    temporal_summary = calculate_temporal_summary(quality_report)
    statistics = calculate_statistics(df)

    # --- Integrity checks ---
    integrity_results = []
    if original_df is not None and consumption_col is not None:
        integrity_results.append(check_record_count(original_df, df))
        integrity_results.append(
            check_valid_value_preservation(original_df, df,
                                          consumption_col, production_col))
        integrity_results.append(
            check_unexpected_nans(original_df, df,
                                 consumption_col, production_col))

    # Sum preservation (pre-correction vs post-correction)
    if pre_correction_df is not None:
        integrity_results.append(check_sum_preservation(pre_correction_df, df))

    # Determine overall integrity status
    if not integrity_results:
        integrity_status = "N/A"
        integrity_details = "No original data provided for comparison"
    else:
        statuses = [r["status"] for r in integrity_results]
        if "FAIL" in statuses:
            integrity_status = "FAIL"
        elif "WARN" in statuses:
            integrity_status = "WARN"
        else:
            integrity_status = "PASS"
        integrity_details = "; ".join(r["details"] for r in integrity_results)

    # --- Consistency issues ---
    negatives_count = len(quality_report.get("negatives", []))
    consistency_issues = {
        "negatives_count": negatives_count,
        "spikes_count": len(spikes),
    }

    # --- Composite score ---
    quality_score, breakdown = calculate_quality_score(
        completeness_pct, integrity_results, temporal_summary,
        consistency_issues, outlier_classification,
    )

    # --- Processing accuracy ---
    if original_df is not None and consumption_col is not None:
        orig_valid = int(original_df.iloc[:, consumption_col].notna().sum())
        if "Consumption (kW)" in df.columns:
            proc_valid = int(df["Consumption (kW)"].notna().sum())
        else:
            value_cols = [c for c in df.columns if c != "Date & Time"]
            proc_valid = int(df[value_cols[0]].notna().sum()) if value_cols else 0
        processing_accuracy = (proc_valid / orig_valid * 100) if orig_valid > 0 else 100.0
    else:
        processing_accuracy = 100.0

    # --- Missing values total ---
    missing_values_total = sum(
        v["count"] for v in quality_report.get("missing_values", {}).values()
    )

    # --- Value range ---
    value_cols = [c for c in df.columns if c != "Date & Time"]
    all_values = pd.concat([df[c] for c in value_cols]).dropna() if value_cols else pd.Series(dtype=float)
    if all_values.empty:
        value_range = {"min": 0.0, "max": 0.0, "avg": 0.0}
    else:
        value_range = {
            "min": float(all_values.min()),
            "max": float(all_values.max()),
            "avg": float(all_values.mean()),
        }

    # --- Timestamp issues ---
    timestamp_issues = (temporal_summary["gaps_count"]
                        + temporal_summary["missing_timestamps"]
                        + temporal_summary["duplicates_count"])

    # --- Build detailed results list ---
    detailed_results = list(integrity_results)  # copy
    # Add completeness result
    if completeness_pct >= 95:
        c_status = "PASS"
    elif completeness_pct >= 80:
        c_status = "WARN"
    else:
        c_status = "FAIL"
    detailed_results.append({
        "name": "Completeness",
        "status": c_status,
        "details": f"{completeness_pct:.1f}% of expected timestamps present",
    })
    # Add spike result
    if spikes:
        detailed_results.append({
            "name": "Impossible Spikes",
            "status": "WARN",
            "details": f"{len(spikes)} spike(s) detected (>10x rolling average)",
        })
    else:
        detailed_results.append({
            "name": "Impossible Spikes",
            "status": "PASS",
            "details": "No impossible spikes detected",
        })
    # Add negatives result
    if negatives_count > 0:
        detailed_results.append({
            "name": "Negative Values",
            "status": "WARN",
            "details": f"{negatives_count} negative value(s) found",
        })
    else:
        detailed_results.append({
            "name": "Negative Values",
            "status": "PASS",
            "details": "No negative values",
        })

    # --- Untrustworthiness ---
    untrustworthiness = calculate_untrustworthiness(
        df, quality_report, outlier_classification, spikes)

    # --- Recommendations ---
    recommendations = generate_recommendations(
        quality_report, outlier_classification, temporal_summary,
        spikes, untrustworthiness)

    return {
        "completeness_pct": round(completeness_pct, 1),
        "quality_score": quality_score,
        "quality_score_breakdown": breakdown,
        "missing_values": missing_values_total,
        "outliers": outlier_classification,
        "timestamp_issues": timestamp_issues,
        "value_range": value_range,
        "integrity": {"status": integrity_status, "details": integrity_details},
        "processing_accuracy_pct": round(processing_accuracy, 1),
        "statistics": statistics,
        "detailed_results": detailed_results,
        "untrustworthiness": untrustworthiness,
        "recommendations": recommendations,
    }
