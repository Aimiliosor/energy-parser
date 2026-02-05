import re
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Common date format patterns to try
DATE_FORMATS = [
    ("%Y-%m-%dT%H:%M:%S%z", "ISO 8601 with timezone"),
    ("%Y-%m-%dT%H:%M:%S", "ISO 8601"),
    ("%Y-%m-%dT%H:%M", "ISO 8601 short"),
    ("%Y-%m-%d %H:%M:%S", "YYYY-MM-DD HH:MM:SS"),
    ("%Y-%m-%d %H:%M", "YYYY-MM-DD HH:MM"),
    ("%d/%m/%Y %H:%M:%S", "DD/MM/YYYY HH:MM:SS"),
    ("%d/%m/%Y %H:%M", "DD/MM/YYYY HH:MM"),
    ("%d.%m.%Y %H:%M:%S", "DD.MM.YYYY HH:MM:SS"),
    ("%d.%m.%Y %H:%M", "DD.MM.YYYY HH:MM"),
    ("%m/%d/%Y %H:%M:%S", "MM/DD/YYYY HH:MM:SS"),
    ("%m/%d/%Y %H:%M", "MM/DD/YYYY HH:MM"),
    ("%Y/%m/%d %H:%M:%S", "YYYY/MM/DD HH:MM:SS"),
    ("%Y/%m/%d %H:%M", "YYYY/MM/DD HH:MM"),
    ("%d-%m-%Y %H:%M:%S", "DD-MM-YYYY HH:MM:SS"),
    ("%d-%m-%Y %H:%M", "DD-MM-YYYY HH:MM"),
]


def parse_number_for_analysis(value: str) -> float | None:
    """Try to parse a string as a number (handles European format)."""
    if not value or not value.strip():
        return None
    value = value.strip()
    # European format: 1.234,56 or 1234,56
    if "," in value and "." in value:
        # Determine which is the decimal separator
        last_comma = value.rfind(",")
        last_dot = value.rfind(".")
        if last_comma > last_dot:
            # European: 1.234,56
            value = value.replace(".", "").replace(",", ".")
        else:
            # US: 1,234.56
            value = value.replace(",", "")
    elif "," in value:
        # Could be European decimal: 1234,56
        value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def detect_date_format(series: pd.Series) -> tuple[str | None, str | None]:
    """Detect the date format of a column. Returns (format_string, description)."""
    sample = series.dropna().head(50).astype(str)
    if sample.empty:
        return None, None

    for fmt, desc in DATE_FORMATS:
        success = 0
        for val in sample:
            val = val.strip()
            if not val:
                continue
            try:
                pd.to_datetime(val, format=fmt)
                success += 1
            except (ValueError, TypeError):
                pass
        if success >= len(sample) * 0.8:
            return fmt, desc

    # Fallback: try pandas auto-parsing
    try:
        pd.to_datetime(sample, dayfirst=True)
        return "auto", "Auto-detected (dayfirst)"
    except Exception:
        pass

    return None, None


def detect_granularity(dates: pd.Series) -> tuple[str, float]:
    """Detect time granularity from a datetime series.
    Returns (label, hours_per_interval)."""
    if len(dates) < 2:
        return "unknown", 1.0

    diffs = dates.diff().dropna()
    if diffs.empty:
        return "unknown", 1.0

    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60

    if minutes <= 12:
        return "10 min", 10 / 60
    elif minutes <= 17:
        return "15 min", 15 / 60
    elif minutes <= 35:
        return "30 min", 30 / 60
    elif minutes <= 75:
        return "1 hour", 1.0
    elif minutes <= 1500:
        return f"{int(round(minutes))} min", minutes / 60
    else:
        hours = minutes / 60
        return f"{hours:.1f} hours", hours


def detect_unit(series: pd.Series) -> str:
    """Detect likely unit based on value magnitudes."""
    numeric_values = []
    for val in series:
        parsed = parse_number_for_analysis(str(val))
        if parsed is not None and parsed >= 0:
            numeric_values.append(parsed)

    if not numeric_values:
        return "kW"

    median = sorted(numeric_values)[len(numeric_values) // 2]

    if median < 1:
        return "MWh"
    elif median < 100:
        return "kWh"
    elif median < 10000:
        return "kW"
    else:
        return "W"


def analyze_columns(df: pd.DataFrame) -> list[dict]:
    """Analyze each column and return analysis info."""
    results = []
    for col in df.columns:
        info = {"name": col, "sample": df[col].head(5).tolist()}
        series = df[col].astype(str).str.strip()
        non_empty = series[series != ""]

        # Try date detection
        date_fmt, date_desc = detect_date_format(non_empty)
        if date_fmt:
            info["type"] = "datetime"
            info["date_format"] = date_fmt
            info["date_desc"] = date_desc
        else:
            # Try numeric
            numeric_count = 0
            values = []
            for val in non_empty.head(100):
                parsed = parse_number_for_analysis(val)
                if parsed is not None:
                    numeric_count += 1
                    values.append(parsed)

            if numeric_count >= len(non_empty.head(100)) * 0.7:
                info["type"] = "numeric"
                if values:
                    info["min"] = min(values)
                    info["max"] = max(values)
                    info["median"] = sorted(values)[len(values) // 2]
                    info["unit_guess"] = detect_unit(non_empty)
            else:
                info["type"] = "text"

        results.append(info)
    return results


def display_analysis(analysis: list[dict]):
    """Display column analysis results."""
    table = Table(title="Column Analysis", show_lines=True)
    table.add_column("Col #", style="bold", width=6)
    table.add_column("Name", width=20)
    table.add_column("Type", width=12)
    table.add_column("Details", width=35)
    table.add_column("Sample Values", width=40)

    for i, info in enumerate(analysis):
        details = ""
        if info["type"] == "datetime":
            details = f"Format: {info.get('date_desc', 'unknown')}"
        elif info["type"] == "numeric":
            if "min" in info:
                details = (
                    f"Range: {info['min']:.2f} - {info['max']:.2f}\n"
                    f"Median: {info['median']:.2f}\n"
                    f"Likely unit: {info.get('unit_guess', '?')}"
                )

        samples = ", ".join(str(s)[:15] for s in info["sample"][:3])
        table.add_row(str(i), info["name"], info["type"], details, samples)

    console.print(table)
