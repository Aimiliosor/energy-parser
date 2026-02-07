"""Branded PDF report generation for energy data analysis.

Uses matplotlib (Agg backend) for chart generation and reportlab for PDF.
"""

import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether,
)

from energy_parser.statistics import DAY_ORDER


# ---------------------------------------------------------------------------
# Brand colors
# ---------------------------------------------------------------------------

BRAND_PRIMARY = "#2C495E"
BRAND_SECONDARY = "#EC465D"
BRAND_BG = "#F5F7FA"
BRAND_WHITE = "#FFFFFF"

_RL_PRIMARY = colors.HexColor(BRAND_PRIMARY)
_RL_SECONDARY = colors.HexColor(BRAND_SECONDARY)
_RL_BG = colors.HexColor(BRAND_BG)
_RL_WHITE = colors.HexColor(BRAND_WHITE)

# Line colors for day-of-week plots
_DAY_COLORS = [
    "#2C495E", "#EC465D", "#28A745", "#FFC107",
    "#6B46C1", "#FF8C00", "#17A2B8",
]

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ---------------------------------------------------------------------------
# Chart generation helpers
# ---------------------------------------------------------------------------

def generate_seasonal_chart(seasonal_data, column_name: str,
                             season: str) -> bytes:
    """Render one seasonal weekly profile as a PNG in memory.

    seasonal_data: dict from compute_seasonal_weekly_profile()[column_name][season]
                   — a DataFrame with index=hours (0-23), columns=day names.
    Returns PNG bytes.
    """
    profile_df = seasonal_data

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)

    for i, day in enumerate(DAY_ORDER):
        if day in profile_df.columns:
            ax.plot(profile_df.index, profile_df[day],
                    label=day, color=_DAY_COLORS[i % len(_DAY_COLORS)],
                    linewidth=1.2)

    ax.set_xlabel("Hour of Day", fontsize=9)
    ax.set_ylabel("Average Power (kW)", fontsize=9)
    ax.set_title(f"{column_name} — {season} Weekly Profile",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.set_xlim(0, 23)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_monthly_bar_chart(monthly_totals: dict,
                                column_name: str) -> bytes:
    """Render monthly kWh totals as a bar chart PNG in memory.

    monthly_totals: {1: float, 2: float, ..., 12: float}
    Returns PNG bytes.
    """
    months = list(range(1, 13))
    values = [monthly_totals.get(m, 0.0) for m in months]

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)

    bar_colors = [BRAND_PRIMARY if v >= 0 else BRAND_SECONDARY for v in values]
    ax.bar(MONTH_NAMES, values, color=bar_colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Month", fontsize=9)
    ax.set_ylabel("Energy (kWh)", fontsize=9)
    ax.set_title(f"{column_name} — Monthly Energy Totals",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _header_footer(canvas, doc, logo_path=None):
    """Draw header bar and footer on every page."""
    canvas.saveState()
    width, height = A4

    # Header bar
    canvas.setFillColor(_RL_PRIMARY)
    canvas.rect(0, height - 25 * mm, width, 25 * mm, fill=True, stroke=False)

    # Logo in header
    if logo_path and os.path.exists(logo_path):
        try:
            canvas.drawImage(logo_path, 10 * mm, height - 22 * mm,
                            width=30 * mm, height=18 * mm,
                            preserveAspectRatio=True, mask="auto")
        except Exception:
            pass

    # Header title
    canvas.setFillColor(_RL_WHITE)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(48 * mm, height - 17 * mm,
                      "Energy Data Analysis Report")

    # Header date
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 10 * mm, height - 17 * mm,
                           datetime.now().strftime("%Y-%m-%d"))

    # Footer bar
    canvas.setFillColor(_RL_PRIMARY)
    canvas.rect(0, 0, width, 12 * mm, fill=True, stroke=False)

    canvas.setFillColor(_RL_WHITE)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(10 * mm, 4.5 * mm,
                      "ReVolta srl — Energy Data Analysis Report")
    canvas.drawRightString(width - 10 * mm, 4.5 * mm,
                           f"Page {doc.page}")

    canvas.restoreState()


def _section_heading(text: str) -> Paragraph:
    """Create a styled section heading."""
    style = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=_RL_WHITE,
        backColor=_RL_PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
        leftIndent=6,
        rightIndent=6,
        leading=18,
    )
    return Paragraph(text, style)


def _body_text(text: str) -> Paragraph:
    """Create a body text paragraph."""
    style = ParagraphStyle(
        "BodyText",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#1A1A2E"),
        spaceBefore=2,
        spaceAfter=2,
    )
    return Paragraph(text, style)


def _build_stats_table(yearly_stats: dict, selected_metrics: list[str]) -> Table:
    """Build a formatted table of yearly statistics."""
    header_row = ["Metric"]
    columns = list(yearly_stats.keys())
    header_row.extend(columns)

    rows = [header_row]

    metric_labels = {
        "total_kwh": ("Total Energy (kWh)", "total_kwh"),
        "mean_kw": ("Mean Power (kW)", "mean_kw"),
        "median_kw": ("Median Power (kW)", "median_kw"),
        "std_kw": ("Std Deviation (kW)", "std_kw"),
        "min_max_kw": ("Min / Max Power (kW)", None),
        "peak_times": ("Peak Time", "peak_timestamp"),
        "daily_avg_kwh": ("Daily Average (kWh)", "daily_avg_kwh"),
    }

    for key, (label, stat_key) in metric_labels.items():
        if key not in selected_metrics:
            continue
        row = [label]
        for col in columns:
            stats = yearly_stats[col]
            if key == "min_max_kw":
                row.append(f"{stats['min_kw']:.2f} / {stats['max_kw']:.2f}")
            elif key == "peak_times":
                row.append(str(stats["peak_timestamp"])[:19])
            else:
                value = stats.get(stat_key, 0)
                row.append(f"{value:,.2f}")
        rows.append(row)

    col_widths = [120] + [140] * len(columns)
    table = Table(rows, colWidths=col_widths[:len(header_row)])

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]

    # Alternating row colors
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_commands.append(
                ("BACKGROUND", (0, i), (-1, i), _RL_BG))

    table.setStyle(TableStyle(style_commands))
    return table


def _build_kpi_table(kpi_data: dict) -> Table:
    """Build a summary KPI table."""
    rows = [
        ["KPI", "Value"],
        ["Quality Score", f"{kpi_data.get('quality_score', 'N/A')}/100"],
        ["Completeness", f"{kpi_data.get('completeness_pct', 'N/A')}%"],
        ["Missing Values", str(kpi_data.get("missing_values", "N/A"))],
        ["Timestamp Issues", str(kpi_data.get("timestamp_issues", "N/A"))],
        ["Processing Accuracy", f"{kpi_data.get('processing_accuracy_pct', 'N/A')}%"],
    ]

    untrust = kpi_data.get("untrustworthiness", {})
    if untrust:
        rows.append(["Untrustworthiness",
                      f"{untrust.get('pct', 0)}% ({untrust.get('rating', 'N/A')})"])

    table = Table(rows, colWidths=[160, 160])

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]

    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_commands.append(
                ("BACKGROUND", (0, i), (-1, i), _RL_BG))

    table.setStyle(TableStyle(style_commands))
    return table


def generate_pdf_report(output_path: str,
                         stats_result: dict,
                         kpi_data: dict | None = None,
                         logo_path: str | None = None,
                         quality_report: dict | None = None) -> str:
    """Generate branded PDF report using reportlab.

    Returns output_path on success.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=30 * mm,
        bottomMargin=18 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    story = []

    # --- KPI Summary ---
    if kpi_data:
        story.append(_section_heading("Data Quality Summary"))
        story.append(Spacer(1, 4 * mm))
        story.append(_build_kpi_table(kpi_data))
        story.append(Spacer(1, 6 * mm))

    # --- Yearly Statistics Table ---
    yearly = stats_result.get("yearly", {})
    selected = stats_result.get("selected_metrics", [])

    if yearly:
        story.append(_section_heading("Yearly Statistics"))
        story.append(Spacer(1, 4 * mm))

        # Filter out non-table metrics
        table_metrics = [m for m in selected if m not in ("monthly_totals", "seasonal_profile")]
        if table_metrics:
            story.append(_build_stats_table(yearly, table_metrics))
            story.append(Spacer(1, 6 * mm))

    # --- Monthly Totals Bar Charts ---
    if "monthly_totals" in selected and yearly:
        story.append(_section_heading("Monthly Energy Totals"))
        story.append(Spacer(1, 4 * mm))

        for col_name, col_stats in yearly.items():
            monthly = col_stats.get("monthly_totals", {})
            if monthly:
                chart_bytes = generate_monthly_bar_chart(monthly, col_name)
                img = RLImage(io.BytesIO(chart_bytes),
                              width=170 * mm, height=70 * mm)
                story.append(img)
                story.append(Spacer(1, 4 * mm))

    # --- Seasonal Weekly Profiles ---
    seasonal = stats_result.get("seasonal", {})
    if "seasonal_profile" in selected and seasonal:
        story.append(PageBreak())
        story.append(_section_heading("Seasonal Weekly Load Profiles"))
        story.append(Spacer(1, 4 * mm))

        for col_name, seasons_dict in seasonal.items():
            for season_name in ["Winter", "Spring", "Summer", "Autumn"]:
                profile_df = seasons_dict.get(season_name)
                if profile_df is None:
                    continue

                chart_bytes = generate_seasonal_chart(
                    profile_df, col_name, season_name)
                img = RLImage(io.BytesIO(chart_bytes),
                              width=170 * mm, height=70 * mm)
                story.append(img)
                story.append(Spacer(1, 3 * mm))

    # Build PDF with header/footer
    def on_page(canvas, doc_ref):
        _header_footer(canvas, doc_ref, logo_path=logo_path)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    return output_path
