import os
import sys
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from energy_parser.file_reader import load_file, display_preview, clean_path
from energy_parser.analyzer import (
    analyze_columns,
    display_analysis,
    detect_granularity,
    detect_granularity_with_confidence,
    STANDARD_GRANULARITIES,
)
from energy_parser.transformer import transform_data
from energy_parser.quality_check import run_quality_check
from energy_parser.corrector import run_corrections
from energy_parser.exporter import save_xlsx

console = Console()

VALID_UNITS = ["W", "kW", "Wh", "kWh", "MWh", "MW"]

# Confidence threshold for automatic granularity acceptance
GRANULARITY_CONFIDENCE_THRESHOLD = 0.7


def prompt_granularity_selection(detected_info: dict | None = None) -> tuple[str, float]:
    """Prompt user to select data granularity when auto-detection fails.

    Args:
        detected_info: Optional dict from detect_granularity_with_confidence

    Returns:
        (label, hours_per_interval)
    """
    console.print("\n[bold yellow]Granularity Detection[/bold yellow]")

    if detected_info:
        console.print(f"  Auto-detection result: {detected_info['label']}")
        console.print(f"  Confidence: {detected_info['confidence']:.0%}")
        console.print(f"  Consistency: {detected_info['consistency']:.0%}")
        console.print("  [yellow]Confidence is too low for automatic selection.[/yellow]")
    else:
        console.print("  [yellow]Could not detect granularity automatically.[/yellow]")

    console.print("\n  Please select the data granularity:")
    for i, (label, minutes, _) in enumerate(STANDARD_GRANULARITIES):
        console.print(f"    ({i + 1}) {label}")

    while True:
        choice = Prompt.ask("  Select granularity", choices=["1", "2", "3", "4"], default="2")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(STANDARD_GRANULARITIES):
                label, minutes, hours = STANDARD_GRANULARITIES[idx]
                console.print(f"  [green]Selected: {label}[/green]")
                return label, hours
        except ValueError:
            pass
        console.print("  [red]Invalid selection, please try again.[/red]")


def prompt_file_path() -> str:
    """Prompt user for file path."""
    console.print(Panel(
        "[bold]Energy Parser[/bold]\n"
        "Ingests energy CSV/XLSX files, standardizes, quality-checks, and outputs clean XLSX.",
        title="Welcome",
        border_style="cyan",
    ))
    while True:
        path = Prompt.ask("\nEnter file path (drag & drop supported)")
        path = clean_path(path)
        if os.path.isfile(path):
            return path
        console.print(f"[red]File not found: {path}[/red]")


def prompt_column_selection(df: pd.DataFrame, analysis: list[dict]) -> dict:
    """Phase 2: Interactive column identification."""
    console.print("\n[bold cyan]Phase 2: Column Identification[/bold cyan]")

    n_cols = len(df.columns)

    # Date column
    date_candidates = [i for i, a in enumerate(analysis) if a["type"] == "datetime"]
    if date_candidates:
        default_date = str(date_candidates[0])
        console.print(f"  Auto-detected date column: [bold]#{date_candidates[0]}[/bold]")
    else:
        default_date = "0"

    while True:
        date_col = Prompt.ask(
            f"  Which column contains Date/Time? (0-{n_cols - 1})",
            default=default_date,
        )
        try:
            date_col = int(date_col)
            if 0 <= date_col < n_cols:
                break
        except ValueError:
            pass
        console.print(f"  [red]Please enter a number between 0 and {n_cols - 1}[/red]")

    # Consumption column
    numeric_candidates = [i for i, a in enumerate(analysis) if a["type"] == "numeric" and i != date_col]
    if numeric_candidates:
        default_cons = str(numeric_candidates[0])
        console.print(f"  Auto-detected consumption column: [bold]#{numeric_candidates[0]}[/bold]")
    else:
        default_cons = "1" if n_cols > 1 else "0"

    while True:
        cons_col = Prompt.ask(
            f"  Which column contains Consumption? (0-{n_cols - 1})",
            default=default_cons,
        )
        try:
            cons_col = int(cons_col)
            if 0 <= cons_col < n_cols:
                break
        except ValueError:
            pass
        console.print(f"  [red]Please enter a number between 0 and {n_cols - 1}[/red]")

    # Production column (optional)
    remaining_numeric = [i for i in numeric_candidates if i != cons_col]
    if remaining_numeric:
        default_prod = str(remaining_numeric[0])
    else:
        default_prod = "none"

    prod_col_str = Prompt.ask(
        f"  Which column contains Production/PV? (0-{n_cols - 1}, or 'none')",
        default=default_prod,
    )

    prod_col = None
    if prod_col_str.lower() != "none":
        try:
            prod_col = int(prod_col_str)
            if not (0 <= prod_col < n_cols):
                prod_col = None
                console.print("  [yellow]Invalid column number, skipping production.[/yellow]")
        except ValueError:
            prod_col = None

    # Unit confirmation for consumption
    cons_analysis = analysis[cons_col] if cons_col < len(analysis) else {}
    detected_cons_unit = cons_analysis.get("unit_guess", "kW")
    console.print(f"\n  Detected consumption unit: [bold]{detected_cons_unit}[/bold]")
    cons_unit = Prompt.ask(
        f"  Confirm consumption unit ({', '.join(VALID_UNITS)})",
        default=detected_cons_unit,
    )
    if cons_unit not in VALID_UNITS:
        console.print(f"  [yellow]Unknown unit '{cons_unit}', using kW[/yellow]")
        cons_unit = "kW"

    # Unit confirmation for production
    prod_unit = None
    if prod_col is not None:
        prod_analysis = analysis[prod_col] if prod_col < len(analysis) else {}
        detected_prod_unit = prod_analysis.get("unit_guess", "kW")
        console.print(f"  Detected production unit: [bold]{detected_prod_unit}[/bold]")
        prod_unit = Prompt.ask(
            f"  Confirm production unit ({', '.join(VALID_UNITS)})",
            default=detected_prod_unit,
        )
        if prod_unit not in VALID_UNITS:
            console.print(f"  [yellow]Unknown unit '{prod_unit}', using kW[/yellow]")
            prod_unit = "kW"

    # Date format
    date_analysis = analysis[date_col]
    date_format = date_analysis.get("date_format", "auto")

    return {
        "date_col": date_col,
        "consumption_col": cons_col,
        "production_col": prod_col,
        "consumption_unit": cons_unit,
        "production_unit": prod_unit,
        "date_format": date_format,
    }


def run():
    """Main orchestrator — sequences all 5 phases."""
    try:
        # ===== Phase 1: Input & Analysis =====
        file_path = prompt_file_path()
        df, metadata = load_file(file_path)
        display_preview(df)

        analysis = analyze_columns(df)
        display_analysis(analysis)

        # Detect granularity from the first date column found
        date_candidates = [i for i, a in enumerate(analysis) if a["type"] == "datetime"]
        hours_per_interval = 1.0
        granularity_label = "unknown"

        if date_candidates:
            date_col_idx = date_candidates[0]
            date_fmt = analysis[date_col_idx].get("date_format", "auto")
            if date_fmt == "auto":
                sample_dates = pd.to_datetime(df.iloc[:, date_col_idx], dayfirst=True, errors="coerce")
            else:
                sample_dates = pd.to_datetime(
                    df.iloc[:, date_col_idx], format=date_fmt, errors="coerce"
                )

            # Use enhanced detection with confidence
            detection_result = detect_granularity_with_confidence(sample_dates)

            if detection_result["confidence"] >= GRANULARITY_CONFIDENCE_THRESHOLD:
                # High confidence - auto-accept
                granularity_label = detection_result["label"]
                hours_per_interval = detection_result["hours_per_interval"]
                console.print(f"  [green]Detected granularity: [bold]{granularity_label}[/bold] "
                            f"(confidence: {detection_result['confidence']:.0%})[/green]")
            else:
                # Low confidence - prompt user
                granularity_label, hours_per_interval = prompt_granularity_selection(detection_result)
        else:
            # No date column found - prompt user
            granularity_label, hours_per_interval = prompt_granularity_selection(None)

        # ===== Phase 2: Column Identification =====
        selection = prompt_column_selection(df, analysis)

        # ===== Phase 3: Transform & Initial Output =====
        transformed = transform_data(
            df=df,
            date_col=selection["date_col"],
            consumption_col=selection["consumption_col"],
            production_col=selection["production_col"],
            date_format=selection["date_format"],
            consumption_unit=selection["consumption_unit"],
            production_unit=selection["production_unit"],
            hours_per_interval=hours_per_interval,
        )

        # Save initial XLSX
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.dirname(file_path)
        initial_path = os.path.join(output_dir, f"{base_name}_parsed.xlsx")
        save_xlsx(transformed, initial_path, is_final=False)

        # ===== Phase 4: Quality Check =====
        report = run_quality_check(transformed, granularity_label, hours_per_interval)

        if not report:
            console.print("\n[bold green]Done! Output saved to:[/bold green]")
            console.print(f"  {initial_path}")
            return

        # Check if there are any issues
        total_issues = (
            report.get("missing_timestamps", 0)
            + sum(v["count"] for v in report.get("missing_values", {}).values())
            + len(report.get("duplicates", []))
            + len(report.get("outliers", []))
        )

        if total_issues == 0:
            console.print("\n[bold green]No issues found! Output saved to:[/bold green]")
            console.print(f"  {initial_path}")
            return

        # ===== Phase 5: Corrections =====
        proceed = Confirm.ask("\nDo you want to proceed with quality check corrections?", default=True)

        if not proceed:
            console.print("\n[bold green]Done! Initial output saved to:[/bold green]")
            console.print(f"  {initial_path}")
            return

        corrected = run_corrections(transformed, report, hours_per_interval)

        # Ask for output filename
        default_clean = f"{base_name}_clean.xlsx"
        clean_name = Prompt.ask("  Output file name", default=default_clean)
        clean_path_out = os.path.join(output_dir, clean_name)

        save_xlsx(corrected, clean_path_out, is_final=True)

        console.print(f"\n[bold green]Done! Files saved:[/bold green]")
        console.print(f"  Initial: {initial_path}")
        console.print(f"  Clean:   {clean_path_out}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        console.print_exception(show_locals=False)
        sys.exit(1)
