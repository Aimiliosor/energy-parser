import re

import pandas as pd
from rich.console import Console

console = Console()


def parse_european_number(value) -> float:
    """Parse a number string handling European format.

    Handles:
      - European: 1.234,56 → 1234.56
      - European decimal only: 1234,56 → 1234.56
      - US: 1,234.56 → 1234.56
      - Plain: 246000 → 246000.0
      - Space thousands: 1 234,56 or 1 234.56
      - Non-breaking spaces (\xa0)
      - Leading/trailing units or whitespace
      - Empty/blank → NaN
    """
    # Already numeric — return directly
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return float("nan")
        return float(value)

    if pd.isna(value):
        return float("nan")

    value = str(value).strip()
    if not value:
        return float("nan")

    # Remove non-breaking spaces, thin spaces, and regular spaces used as
    # thousands separators (common in European locales)
    value = value.replace("\xa0", "").replace("\u202f", "").replace(" ", "")

    # Strip common trailing/leading unit suffixes (kW, W, kWh, MWh, etc.)
    value = re.sub(r'[a-zA-Z%°€$£]+$', '', value).strip()
    value = re.sub(r'^[€$£]+', '', value).strip()

    if not value:
        return float("nan")

    if "," in value and "." in value:
        last_comma = value.rfind(",")
        last_dot = value.rfind(".")
        if last_comma > last_dot:
            # European: 1.234,56
            value = value.replace(".", "").replace(",", ".")
        else:
            # US: 1,234.56
            value = value.replace(",", "")
    elif "," in value:
        # European decimal: 1234,56
        value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return float("nan")


def standardize_dates(series: pd.Series, date_format: str) -> pd.Series:
    """Parse dates and standardize to YYYY-MM-DD HH:MM format."""
    if date_format == "auto":
        dates = pd.to_datetime(series, dayfirst=True, errors="coerce", utc=True)
    else:
        dates = pd.to_datetime(series, format=date_format, errors="coerce", utc=True)

    # Strip timezone info — keep local time
    if hasattr(dates, "dt") and hasattr(dates.dt, "tz") and dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    return dates


def convert_to_kw(series: pd.Series, unit: str, hours_per_interval: float) -> pd.Series:
    """Convert values to kW based on the source unit.

    Conversions:
      W    → divide by 1000
      kW   → as-is
      Wh   → divide by (1000 * hours_per_interval)
      kWh  → divide by hours_per_interval
      MWh  → multiply by 1000 / hours_per_interval
      MW   → multiply by 1000
    """
    unit = unit.strip()
    if unit == "W":
        return series / 1000.0
    elif unit == "kW":
        return series
    elif unit == "Wh":
        return series / (1000.0 * hours_per_interval)
    elif unit == "kWh":
        return series / hours_per_interval
    elif unit == "MWh":
        return series * 1000.0 / hours_per_interval
    elif unit == "MW":
        return series * 1000.0
    else:
        console.print(f"[yellow]Unknown unit '{unit}', treating as kW[/yellow]")
        return series


def transform_data(
    df: pd.DataFrame,
    date_col: int,
    consumption_col: int,
    production_col: int | None,
    date_format: str,
    consumption_unit: str,
    production_unit: str | None,
    hours_per_interval: float,
) -> pd.DataFrame:
    """Transform raw DataFrame into standardized output.

    Returns a DataFrame with columns:
      - Date & Time (datetime)
      - Consumption (kW)
      - Production (kW) [optional]
    """
    console.print("\n[bold cyan]Phase 3: Transforming Data[/bold cyan]")

    # Parse dates
    console.print("  Parsing dates...")
    date_series = standardize_dates(df.iloc[:, date_col], date_format)
    invalid_dates = date_series.isna().sum()
    if invalid_dates > 0:
        console.print(f"  [yellow]Warning: {invalid_dates} dates could not be parsed[/yellow]")

    # Parse consumption values
    console.print("  Parsing consumption values...")
    consumption = df.iloc[:, consumption_col].apply(parse_european_number)
    nan_count = consumption.isna().sum()
    if nan_count > 0:
        console.print(f"  [yellow]{nan_count} consumption values are empty/unparseable[/yellow]")

    # Convert consumption to kW
    console.print(f"  Converting consumption from {consumption_unit} to kW...")
    consumption_kw = convert_to_kw(consumption, consumption_unit, hours_per_interval)

    # Build result
    result = pd.DataFrame({
        "Date & Time": date_series,
        "Consumption (kW)": consumption_kw,
    })

    # Handle production if present
    if production_col is not None and production_unit is not None:
        console.print("  Parsing production values...")
        production = df.iloc[:, production_col].apply(parse_european_number)
        nan_prod = production.isna().sum()
        if nan_prod > 0:
            console.print(f"  [yellow]{nan_prod} production values are empty/unparseable[/yellow]")

        console.print(f"  Converting production from {production_unit} to kW...")
        production_kw = convert_to_kw(production, production_unit, hours_per_interval)
        result["Production (kW)"] = production_kw

    # Sort by date
    result = result.sort_values("Date & Time").reset_index(drop=True)

    # Mark all rows as original data (corrections will update this later)
    result["data_source"] = "original"

    valid_rows = result["Date & Time"].notna().sum()
    console.print(f"  Transformation complete: [bold]{valid_rows}[/bold] valid rows")

    return result
