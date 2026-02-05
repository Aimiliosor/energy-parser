import pandas as pd
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def fill_from_previous_day(df: pd.DataFrame, col: str, row_idx: int) -> float | None:
    """Get value from the same hour on the previous day.
    Falls back to next day, then to interpolation."""
    target_time = df.at[row_idx, "Date & Time"]
    if pd.isna(target_time):
        return None

    # Try previous day
    prev_day = target_time - pd.Timedelta(days=1)
    match = df[df["Date & Time"] == prev_day]
    if not match.empty and pd.notna(match.iloc[0][col]):
        return match.iloc[0][col]

    # Try next day
    next_day = target_time + pd.Timedelta(days=1)
    match = df[df["Date & Time"] == next_day]
    if not match.empty and pd.notna(match.iloc[0][col]):
        return match.iloc[0][col]

    # Fallback: try 2 days back, 2 days forward
    for offset in [2, -2, 3, -3, 7, -7]:
        alt_day = target_time + pd.Timedelta(days=offset)
        match = df[df["Date & Time"] == alt_day]
        if not match.empty and pd.notna(match.iloc[0][col]):
            return match.iloc[0][col]

    return None


def correct_missing_values(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Handle missing values (NaN) in value columns."""
    missing = report.get("missing_values", {})
    if not missing:
        console.print("  [green]No missing values to correct.[/green]")
        return df

    total_missing = sum(v["count"] for v in missing.values())
    console.print(f"\n  [bold]Missing Values: {total_missing} total[/bold]")
    console.print("  How should missing values be filled?")
    console.print("    (a) Fill from same hour, previous day [bold](recommended)[/bold]")
    console.print("    (b) Linear interpolation from neighbors")
    console.print("    (c) Fill with 0")
    console.print("    (d) Leave as-is")

    choice = Prompt.ask("  Choice", choices=["a", "b", "c", "d"], default="a")

    value_cols = [c for c in df.columns if c != "Date & Time"]

    if choice == "a":
        filled = 0
        for col in value_cols:
            for idx in df[df[col].isna()].index:
                val = fill_from_previous_day(df, col, idx)
                if val is not None:
                    df.at[idx, col] = val
                    filled += 1
        # Remaining NaN: fallback to interpolation
        remaining = sum(df[col].isna().sum() for col in value_cols)
        if remaining > 0:
            for col in value_cols:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
            console.print(f"  Filled {filled} from previous day, {remaining} via interpolation fallback")
        else:
            console.print(f"  Filled {filled} values from previous day")

    elif choice == "b":
        for col in value_cols:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        console.print(f"  Interpolated {total_missing} missing values")

    elif choice == "c":
        for col in value_cols:
            df[col] = df[col].fillna(0)
        console.print(f"  Filled {total_missing} missing values with 0")

    elif choice == "d":
        console.print("  Missing values left as-is")

    return df


def correct_time_gaps(df: pd.DataFrame, report: dict, hours_per_interval: float) -> pd.DataFrame:
    """Insert rows for missing timestamps and fill values."""
    gaps = report.get("gaps", [])
    if not gaps:
        console.print("  [green]No time gaps to correct.[/green]")
        return df

    total_missing = sum(g["missing_count"] for g in gaps)
    console.print(f"\n  [bold]Time Gaps: {total_missing} missing timestamps[/bold]")
    console.print("  How should gaps be filled?")
    console.print("    (a) Insert rows, fill from same hour previous day [bold](recommended)[/bold]")
    console.print("    (b) Insert rows with interpolated values")
    console.print("    (c) Insert rows with 0")
    console.print("    (d) Skip (leave gaps)")

    choice = Prompt.ask("  Choice", choices=["a", "b", "c", "d"], default="a")

    if choice == "d":
        console.print("  Gaps left as-is")
        return df

    # Generate missing timestamps
    freq = pd.Timedelta(hours=hours_per_interval)
    value_cols = [c for c in df.columns if c != "Date & Time"]
    new_rows = []

    for gap in gaps:
        current = gap["from"] + freq
        while current < gap["to"]:
            row = {"Date & Time": current}
            for col in value_cols:
                row[col] = float("nan")
            new_rows.append(row)
            current += freq

    if not new_rows:
        return df

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.sort_values("Date & Time").reset_index(drop=True)

    console.print(f"  Inserted {len(new_rows)} timestamp rows")

    # Now fill the new rows
    if choice == "a":
        filled = 0
        for col in value_cols:
            for idx in df[df[col].isna()].index:
                val = fill_from_previous_day(df, col, idx)
                if val is not None:
                    df.at[idx, col] = val
                    filled += 1
        remaining = sum(df[col].isna().sum() for col in value_cols)
        if remaining > 0:
            for col in value_cols:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
            console.print(f"  Filled {filled} from previous day, {remaining} via interpolation fallback")
        else:
            console.print(f"  Filled {filled} values from previous day")

    elif choice == "b":
        for col in value_cols:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        console.print(f"  Interpolated values for {len(new_rows)} inserted rows")

    elif choice == "c":
        for col in value_cols:
            df[col] = df[col].fillna(0)
        console.print(f"  Filled {len(new_rows)} inserted rows with 0")

    return df


def correct_duplicates(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Handle duplicate timestamps."""
    duplicates = report.get("duplicates", [])
    if not duplicates:
        console.print("  [green]No duplicates to correct.[/green]")
        return df

    console.print(f"\n  [bold]Duplicates: {len(duplicates)} duplicate timestamp groups[/bold]")
    console.print("  How should duplicates be handled?")
    console.print("    (a) Keep first occurrence")
    console.print("    (b) Keep last occurrence")
    console.print("    (c) Average the values")

    choice = Prompt.ask("  Choice", choices=["a", "b", "c"], default="a")

    if choice == "a":
        df = df.drop_duplicates(subset="Date & Time", keep="first").reset_index(drop=True)
        console.print(f"  Kept first occurrence for {len(duplicates)} groups")
    elif choice == "b":
        df = df.drop_duplicates(subset="Date & Time", keep="last").reset_index(drop=True)
        console.print(f"  Kept last occurrence for {len(duplicates)} groups")
    elif choice == "c":
        value_cols = [c for c in df.columns if c != "Date & Time"]
        df = df.groupby("Date & Time", as_index=False)[value_cols].mean()
        df = df.sort_values("Date & Time").reset_index(drop=True)
        console.print(f"  Averaged values for {len(duplicates)} groups")

    return df


def correct_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Handle outlier values."""
    outliers = report.get("outliers", [])
    if not outliers:
        console.print("  [green]No outliers to correct.[/green]")
        return df

    console.print(f"\n  [bold]Outliers: {len(outliers)} detected[/bold]")
    console.print("  How should outliers be handled?")
    console.print("    (a) Replace with same hour previous day")
    console.print("    (b) Cap at threshold (3x median)")
    console.print("    (c) Leave as-is [bold](recommended - may be legitimate)[/bold]")

    choice = Prompt.ask("  Choice", choices=["a", "b", "c"], default="c")

    if choice == "c":
        console.print("  Outliers left as-is")
        return df

    if choice == "a":
        replaced = 0
        for o in outliers:
            val = fill_from_previous_day(df, o["column"], o["row"])
            if val is not None:
                df.at[o["row"], o["column"]] = val
                replaced += 1
        console.print(f"  Replaced {replaced}/{len(outliers)} outliers from previous day")

    elif choice == "b":
        for o in outliers:
            threshold = o["median"] * 3
            if o["type"] == "high":
                df.at[o["row"], o["column"]] = threshold
            else:
                df.at[o["row"], o["column"]] = o["median"] / 10
        console.print(f"  Capped {len(outliers)} outliers at threshold")

    return df


def run_corrections(df: pd.DataFrame, report: dict, hours_per_interval: float) -> pd.DataFrame:
    """Run all interactive corrections."""
    console.print("\n[bold cyan]Phase 5: Corrections[/bold cyan]")

    df = correct_duplicates(df, report)
    df = correct_time_gaps(df, report, hours_per_interval)
    df = correct_missing_values(df, report)
    df = correct_outliers(df, report)

    console.print("\n  [bold green]All corrections applied.[/bold green]")
    return df
