import os
import sys
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

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
from energy_parser.data_validator import run_validation
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
        "[bold]Spartacus[/bold]\n"
        "ReVolta energy analysis tool. Ingests CSV/XLSX files, standardizes, quality-checks, and outputs clean XLSX.",
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


def _kpi_color(value, green_threshold, yellow_threshold, higher_is_better=True):
    """Return a Rich color tag based on threshold."""
    if higher_is_better:
        if value >= green_threshold:
            return "bold green"
        elif value >= yellow_threshold:
            return "bold yellow"
        return "bold red"
    else:
        if value <= green_threshold:
            return "bold green"
        elif value <= yellow_threshold:
            return "bold yellow"
        return "bold red"


def display_kpi_dashboard(kpi: dict):
    """Display a Rich table KPI dashboard."""
    table = Table(title="Data Quality KPI Dashboard", show_lines=True, expand=True)
    table.add_column("KPI", style="bold", width=25)
    table.add_column("Value", width=30)
    table.add_column("Status", width=10)

    # Quality Score
    score = kpi["quality_score"]
    color = _kpi_color(score, 80, 60)
    status = "PASS" if score >= 80 else ("WARN" if score >= 60 else "FAIL")
    table.add_row("Quality Score", f"[{color}]{score}/100[/{color}]", f"[{color}]{status}[/{color}]")

    # Completeness
    comp = kpi["completeness_pct"]
    color = _kpi_color(comp, 95, 80)
    table.add_row("Completeness", f"[{color}]{comp:.1f}%[/{color}]",
                  f"[{color}]{'PASS' if comp >= 95 else ('WARN' if comp >= 80 else 'FAIL')}[/{color}]")

    # Integrity
    integrity = kpi["integrity"]["status"]
    if integrity == "PASS":
        color = "bold green"
    elif integrity == "N/A":
        color = "dim"
    else:
        color = "bold red"
    table.add_row("Integrity", f"[{color}]{integrity}[/{color}]",
                  f"[{color}]{integrity}[/{color}]")

    # Missing Values
    mv = kpi["missing_values"]
    color = _kpi_color(mv, 0, 10, higher_is_better=False)
    table.add_row("Missing Values", f"[{color}]{mv}[/{color}]",
                  f"[{color}]{'PASS' if mv == 0 else ('WARN' if mv <= 10 else 'FAIL')}[/{color}]")

    # Timestamp Issues
    ti = kpi["timestamp_issues"]
    color = _kpi_color(ti, 0, 5, higher_is_better=False)
    table.add_row("Timestamp Issues", f"[{color}]{ti}[/{color}]",
                  f"[{color}]{'PASS' if ti == 0 else ('WARN' if ti <= 5 else 'FAIL')}[/{color}]")

    # Outliers
    outliers = kpi["outliers"]
    total = outliers["total"]
    color = _kpi_color(total, 0, 5, higher_is_better=False)
    breakdown = f"(low:{outliers['low']} med:{outliers['medium']} high:{outliers['high']})"
    table.add_row("Outliers", f"[{color}]{total} {breakdown}[/{color}]",
                  f"[{color}]{'PASS' if total == 0 else ('WARN' if total <= 5 else 'FAIL')}[/{color}]")

    # Value Range
    vr = kpi["value_range"]
    table.add_row("Value Range", f"min={vr['min']:.2f}  max={vr['max']:.2f}  avg={vr['avg']:.2f}", "")

    # Processing Accuracy
    acc = kpi["processing_accuracy_pct"]
    color = _kpi_color(acc, 95, 80)
    table.add_row("Processing Accuracy", f"[{color}]{acc:.1f}%[/{color}]",
                  f"[{color}]{'PASS' if acc >= 95 else ('WARN' if acc >= 80 else 'FAIL')}[/{color}]")

    # Untrustworthiness Score
    untrust = kpi.get("untrustworthiness", {})
    if untrust:
        u_pct = untrust["pct"]
        u_rating = untrust["rating"]
        tier = untrust["color_tier"]
        if tier == "green":
            u_color = "bold green"
        elif tier == "yellow":
            u_color = "bold yellow"
        elif tier == "orange":
            u_color = "bold dark_orange"
        else:
            u_color = "bold red"
        u_detail = f"{u_pct}% flagged ({untrust['flagged']} of {untrust['total']} records)"
        table.add_row("Untrustworthiness", f"[{u_color}]{u_detail}[/{u_color}]",
                      f"[{u_color}]{u_rating}[/{u_color}]")

    # Original Data %
    orig_pct = kpi.get("original_data_pct", 100.0)
    orig_color = _kpi_color(orig_pct, 90, 70)
    orig_status = "PASS" if orig_pct >= 90 else ("WARN" if orig_pct >= 70 else "FAIL")
    table.add_row("Original Data", f"[{orig_color}]{orig_pct:.1f}%[/{orig_color}]",
                  f"[{orig_color}]{orig_status}[/{orig_color}]")

    console.print(table)

    # Detailed results
    if kpi.get("detailed_results"):
        detail_table = Table(title="Detailed Validation Results", show_lines=True)
        detail_table.add_column("Check", width=25)
        detail_table.add_column("Status", width=8)
        detail_table.add_column("Details", width=60)

        for r in kpi["detailed_results"]:
            if r["status"] == "PASS":
                color = "green"
            elif r["status"] == "WARN":
                color = "yellow"
            else:
                color = "red"
            detail_table.add_row(r["name"], f"[{color}]{r['status']}[/{color}]", r["details"])

        console.print(detail_table)

    # Recommendations
    recs = kpi.get("recommendations", [])
    if recs:
        rec_panel_lines = []
        priority_labels = {1: "CRITICAL", 2: "IMPORTANT", 3: "ADVISORY", 4: "INFO"}
        priority_colors = {1: "bold red", 2: "bold yellow", 3: "cyan", 4: "dim"}
        for r in recs:
            p = r["priority"]
            label = priority_labels.get(p, "INFO")
            color = priority_colors.get(p, "dim")
            rec_panel_lines.append(
                f"[{color}][{label}][/{color}] [bold]{r['category']}[/bold]\n"
                f"  {r['message']}\n"
            )
        panel_content = "\n".join(rec_panel_lines)
        console.print(Panel(panel_content, title="Data Quality Recommendations",
                            border_style="cyan", expand=True))


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

        if report:
            kpi = run_validation(
                df=transformed, quality_report=report,
                original_df=df,
                consumption_col=selection["consumption_col"],
                production_col=selection["production_col"],
            )
            display_kpi_dashboard(kpi)

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

        # Post-correction validation
        post_report = run_quality_check(corrected, granularity_label, hours_per_interval, silent=True)
        if post_report:
            post_kpi = run_validation(
                df=corrected, quality_report=post_report,
                original_df=df, pre_correction_df=transformed,
                consumption_col=selection["consumption_col"],
                production_col=selection["production_col"],
            )
            console.print("\n[bold cyan]Post-Correction Validation[/bold cyan]")
            display_kpi_dashboard(post_kpi)

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
