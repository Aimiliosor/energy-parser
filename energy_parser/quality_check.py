import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def find_time_gaps(dates: pd.Series, granularity_minutes: float) -> list[dict]:
    """Find gaps in the time series where timestamps are missing."""
    gaps = []
    freq = pd.Timedelta(minutes=granularity_minutes)
    sorted_dates = dates.dropna().sort_values().reset_index(drop=True)

    for i in range(1, len(sorted_dates)):
        diff = sorted_dates.iloc[i] - sorted_dates.iloc[i - 1]
        if diff > freq * 1.5:  # Allow 50% tolerance
            n_missing = int(round(diff / freq)) - 1
            gaps.append({
                "after_row": i - 1,
                "from": sorted_dates.iloc[i - 1],
                "to": sorted_dates.iloc[i],
                "missing_count": n_missing,
            })
    return gaps


def find_duplicates(dates: pd.Series) -> list[dict]:
    """Find duplicate timestamps."""
    dupes = dates[dates.duplicated(keep=False)]
    if dupes.empty:
        return []

    result = []
    for ts in dupes.unique():
        indices = dates[dates == ts].index.tolist()
        result.append({"timestamp": ts, "row_indices": indices})
    return result


def find_outliers(series: pd.Series, col_name: str) -> list[dict]:
    """Find outliers: values > 3x median or < median/10."""
    clean = series.dropna()
    if clean.empty:
        return []

    median = clean.median()
    if median <= 0:
        return []

    outliers = []
    high_threshold = median * 3
    low_threshold = median / 10

    for idx, val in series.items():
        if pd.isna(val):
            continue
        if val > high_threshold or (val > 0 and val < low_threshold):
            outliers.append({
                "row": idx,
                "value": val,
                "median": median,
                "column": col_name,
                "type": "high" if val > high_threshold else "low",
            })
    return outliers


def find_negative_values(series: pd.Series, col_name: str) -> list[dict]:
    """Find negative values."""
    negatives = []
    for idx, val in series.items():
        if pd.notna(val) and val < 0:
            negatives.append({"row": idx, "value": val, "column": col_name})
    return negatives


def generate_expected_timestamps(start: pd.Timestamp, end: pd.Timestamp, freq_minutes: float) -> pd.DatetimeIndex:
    """Generate the full expected timestamp range."""
    freq = pd.Timedelta(minutes=freq_minutes)
    return pd.date_range(start=start, end=end, freq=freq)


def run_quality_check(df: pd.DataFrame, granularity_label: str, hours_per_interval: float,
                      silent: bool = False) -> dict:
    """Run all quality checks and return a report dict."""
    if not silent:
        console.print("\n[bold cyan]Phase 4: Quality Check[/bold cyan]")

    granularity_minutes = hours_per_interval * 60
    dates = df["Date & Time"]
    report = {}

    # --- Completeness ---
    valid_dates = dates.dropna().sort_values()
    if valid_dates.empty:
        if not silent:
            console.print("[red]No valid dates found. Cannot perform quality check.[/red]")
        return {}

    start = valid_dates.iloc[0]
    end = valid_dates.iloc[-1]
    expected = generate_expected_timestamps(start, end, granularity_minutes)
    report["total_rows"] = len(df)
    report["expected_timestamps"] = len(expected)
    report["date_range"] = (start, end)
    report["granularity"] = granularity_label

    # Missing timestamps (gaps)
    gaps = find_time_gaps(dates, granularity_minutes)
    total_missing_timestamps = sum(g["missing_count"] for g in gaps)
    report["gaps"] = gaps
    report["missing_timestamps"] = total_missing_timestamps

    # Missing values (NaN in value columns)
    value_cols = [c for c in df.columns if c not in ("Date & Time", "data_source")]
    missing_values = {}
    for col in value_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_rows = df[df[col].isna()].index.tolist()
            missing_values[col] = {"count": nan_count, "rows": nan_rows}
    report["missing_values"] = missing_values

    # --- Validity ---
    # Duplicates
    duplicates = find_duplicates(dates)
    report["duplicates"] = duplicates

    # Negative values
    all_negatives = []
    for col in value_cols:
        all_negatives.extend(find_negative_values(df[col], col))
    report["negatives"] = all_negatives

    # Outliers
    all_outliers = []
    for col in value_cols:
        all_outliers.extend(find_outliers(df[col], col))
    report["outliers"] = all_outliers

    # --- Display Report ---
    if not silent:
        display_report(report)
    return report


def display_report(report: dict):
    """Display the quality report using rich tables."""
    # Summary table
    summary = Table(title="Quality Check Summary", show_lines=True)
    summary.add_column("Metric", style="bold", width=30)
    summary.add_column("Value", width=20)

    start, end = report["date_range"]
    summary.add_row("Date range", f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}")
    summary.add_row("Granularity", report["granularity"])
    summary.add_row("Total rows in file", str(report["total_rows"]))
    summary.add_row("Expected timestamps", str(report["expected_timestamps"]))
    summary.add_row("Missing timestamps (gaps)", str(report["missing_timestamps"]))

    total_nan = sum(v["count"] for v in report["missing_values"].values())
    summary.add_row("Rows with missing values", str(total_nan))
    summary.add_row("Duplicate timestamps", str(len(report["duplicates"])))
    summary.add_row("Negative values", str(len(report["negatives"])))
    summary.add_row("Outliers detected", str(len(report["outliers"])))

    # Color-coded status
    issues = report["missing_timestamps"] + total_nan + len(report["duplicates"]) + len(report["outliers"])
    if issues == 0:
        summary.add_row("Overall", "[bold green]CLEAN - No issues found[/bold green]")
    else:
        summary.add_row("Overall", f"[bold yellow]{issues} issue(s) found[/bold yellow]")

    console.print(summary)

    # Detail: gaps
    if report["gaps"]:
        gap_table = Table(title="Time Gaps", show_lines=True)
        gap_table.add_column("From", width=20)
        gap_table.add_column("To", width=20)
        gap_table.add_column("Missing Timestamps", width=18)
        for g in report["gaps"][:20]:
            gap_table.add_row(
                f"{g['from']:%Y-%m-%d %H:%M}",
                f"{g['to']:%Y-%m-%d %H:%M}",
                str(g["missing_count"]),
            )
        if len(report["gaps"]) > 20:
            console.print(f"  ... and {len(report['gaps']) - 20} more gaps")
        console.print(gap_table)

    # Detail: missing values
    if report["missing_values"]:
        for col, info in report["missing_values"].items():
            console.print(f"\n  [yellow]Missing values in '{col}': {info['count']}[/yellow]")
            sample_rows = info["rows"][:10]
            console.print(f"    Sample rows: {sample_rows}")
            if len(info["rows"]) > 10:
                console.print(f"    ... and {len(info['rows']) - 10} more")

    # Detail: duplicates
    if report["duplicates"]:
        console.print(f"\n  [yellow]Duplicate timestamps: {len(report['duplicates'])}[/yellow]")
        for d in report["duplicates"][:10]:
            console.print(f"    {d['timestamp']:%Y-%m-%d %H:%M} at rows: {d['row_indices']}")

    # Detail: outliers
    if report["outliers"]:
        outlier_table = Table(title="Outliers", show_lines=True)
        outlier_table.add_column("Row", width=8)
        outlier_table.add_column("Column", width=20)
        outlier_table.add_column("Value", width=15)
        outlier_table.add_column("Median", width=15)
        outlier_table.add_column("Type", width=8)
        for o in report["outliers"][:20]:
            outlier_table.add_row(
                str(o["row"]),
                o["column"],
                f"{o['value']:.2f}",
                f"{o['median']:.2f}",
                o["type"],
            )
        if len(report["outliers"]) > 20:
            console.print(f"  ... and {len(report['outliers']) - 20} more outliers")
        console.print(outlier_table)

    # Detail: negatives
    if report["negatives"]:
        console.print(f"\n  [yellow]Negative values: {len(report['negatives'])}[/yellow]")
        for n in report["negatives"][:10]:
            console.print(f"    Row {n['row']}: {n['column']} = {n['value']:.2f}")
