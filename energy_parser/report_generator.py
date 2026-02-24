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

import numpy as np
from energy_parser.statistics import DAY_ORDER, MONTH_NAMES as STAT_MONTH_NAMES


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
# Peak analysis chart helpers
# ---------------------------------------------------------------------------

BRAND_LIGHT_GRAY = "#B4BCD6"


def generate_peak_timeline_chart(peak_timeline: list[dict],
                                  column_name: str) -> bytes:
    """Scatter plot showing when all major peaks occurred over time.

    peak_timeline: list of {"timestamp": Timestamp, "value": float}
    """
    if not peak_timeline:
        return _empty_chart("No peaks detected")

    timestamps = [p["timestamp"] for p in peak_timeline]
    values = [p["value"] for p in peak_timeline]

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    ax.scatter(timestamps, values, c=BRAND_SECONDARY, s=20, alpha=0.7,
               edgecolors=BRAND_PRIMARY, linewidths=0.5, zorder=3)

    # Trend line
    if len(timestamps) >= 2:
        x_num = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        coeffs = np.polyfit(x_num, values, 1)
        trend_y = np.polyval(coeffs, x_num)
        ax.plot(timestamps, trend_y, color=BRAND_PRIMARY, linewidth=1.5,
                linestyle="--", alpha=0.7, label="Trend")
        ax.legend(fontsize=7)

    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Peak Power (kW)", fontsize=9)
    ax.set_title(f"{column_name} — Peak Timeline",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_peak_heatmap(hourly_dist: dict, daily_dist: dict,
                           column_name: str) -> bytes:
    """Heatmap of peak occurrence by hour-of-day vs day-of-week.

    hourly_dist: {0: count, 1: count, ..., 23: count}
    daily_dist: {"Monday": count, ..., "Sunday": count}
    """
    # Build a 24 x 7 matrix — we proportionally distribute based on
    # hourly and daily counts since we don't have the full cross-tab.
    # Use outer product of normalised hourly and daily distributions.
    hours = np.array([hourly_dist.get(h, 0) for h in range(24)], dtype=float)
    days = np.array([daily_dist.get(d, 0) for d in DAY_ORDER], dtype=float)

    total = hours.sum()
    if total > 0:
        hours_norm = hours / total
        days_norm = days / total if days.sum() > 0 else days
        matrix = np.outer(hours_norm, days_norm) * total
    else:
        matrix = np.zeros((24, 7))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                    vmin=0)
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in DAY_ORDER], fontsize=8)
    ax.set_yticks(range(0, 24, 3))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8)
    ax.set_xlabel("Day of Week", fontsize=9)
    ax.set_ylabel("Hour of Day", fontsize=9)
    ax.set_title(f"{column_name} — Peak Occurrence Heatmap",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    fig.colorbar(im, ax=ax, label="Relative Peak Frequency", shrink=0.8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_peak_duration_chart(top_peaks: list[dict],
                                  column_name: str) -> bytes:
    """Bar chart of peak durations for the top peaks."""
    if not top_peaks:
        return _empty_chart("No peaks detected")

    labels = [f"#{p['rank']}\n{p['timestamp'][:10]}" for p in top_peaks]
    durations = [p["duration_hours"] for p in top_peaks]

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    bars = ax.bar(labels, durations, color=BRAND_PRIMARY, edgecolor="white",
                  linewidth=0.5)

    # Annotate bars with value
    for bar, dur in zip(bars, durations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{dur:.1f}h", ha="center", va="bottom", fontsize=7,
                color=BRAND_PRIMARY)

    ax.set_xlabel("Peak", fontsize=9)
    ax.set_ylabel("Duration (hours)", fontsize=9)
    ax.set_title(f"{column_name} — Peak Durations (above 90% of peak value)",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_peak_value_trend_chart(top_peaks: list[dict],
                                     column_name: str) -> bytes:
    """Horizontal bar chart showing top peak values ranked."""
    if not top_peaks:
        return _empty_chart("No peaks detected")

    labels = [f"#{p['rank']} — {p['timestamp']}" for p in reversed(top_peaks)]
    values = [p["value"] for p in reversed(top_peaks)]

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)

    bar_colors = [BRAND_SECONDARY if i == len(values) - 1 else BRAND_PRIMARY
                  for i in range(len(values))]
    bars = ax.barh(labels, values, color=bar_colors, edgecolor="white",
                   linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.1f} kW", ha="left", va="center", fontsize=7,
                color=BRAND_PRIMARY)

    ax.set_xlabel("Power (kW)", fontsize=9)
    ax.set_title(f"{column_name} — Top Peak Values",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_histogram_chart(hist_data: dict, column_name: str) -> bytes:
    """Render a frequency histogram for a value column.

    hist_data: dict with bin_edges, counts, mean, median from
               compute_frequency_histogram()[column_name].
    Returns PNG bytes.
    """
    bin_edges = hist_data.get("bin_edges", [])
    counts = hist_data.get("counts", [])
    mean_val = hist_data.get("mean", 0)
    median_val = hist_data.get("median", 0)

    if not counts or not bin_edges:
        return _empty_chart("No data for histogram")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

    # Bar chart using bin edges
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]
    bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0

    ax.bar(bin_centers, counts, width=bin_width * 0.9,
           color=BRAND_PRIMARY, edgecolor="white", linewidth=0.5, alpha=0.85)

    # Mean and median lines
    ax.axvline(mean_val, color=BRAND_SECONDARY, linewidth=1.8, linestyle="--",
               label=f"Mean: {mean_val:,.1f} kW")
    ax.axvline(median_val, color="#28A745", linewidth=1.8, linestyle="-.",
               label=f"Median: {median_val:,.1f} kW")

    ax.set_xlabel("Power (kW)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title(f"{column_name} — Frequency Distribution",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_cdf_chart(cdf_data: dict, column_name: str,
                        all_cdf_data: dict | None = None) -> bytes:
    """Render a cumulative distribution function (CDF) chart.

    If all_cdf_data is provided and has multiple columns, overlays them
    on a single chart. Otherwise plots only the given column.

    cdf_data: dict with values, cdf, percentiles from
              compute_cumulative_distribution()[column_name].
    all_cdf_data: full dict from compute_cumulative_distribution() for overlay.
    Returns PNG bytes.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

    line_colors = {
        0: BRAND_PRIMARY,       # consumption
        1: BRAND_LIGHT_GRAY,    # production
    }

    # Plot all columns if overlay data provided, otherwise just the one
    if all_cdf_data and len(all_cdf_data) > 1:
        columns_to_plot = all_cdf_data
    else:
        columns_to_plot = {column_name: cdf_data}

    for i, (col_name, col_cdf) in enumerate(columns_to_plot.items()):
        values = col_cdf.get("values", [])
        cdf = col_cdf.get("cdf", [])
        if not values:
            continue

        color = line_colors.get(i, BRAND_PRIMARY)
        ax.plot(values, cdf, color=color, linewidth=1.8, label=col_name)

    # Mark percentiles from the primary column
    percentiles = cdf_data.get("percentiles", {})
    for pct_name, pct_val in [("50th", percentiles.get("p50")),
                                ("90th", percentiles.get("p90")),
                                ("95th", percentiles.get("p95"))]:
        if pct_val is not None and pct_val > 0:
            pct_num = int(pct_name.replace("th", ""))
            ax.axvline(pct_val, color=BRAND_SECONDARY, linewidth=1.2,
                       linestyle=":", alpha=0.8)
            ax.axhline(pct_num, color=BRAND_SECONDARY, linewidth=0.5,
                       linestyle=":", alpha=0.4)
            ax.plot(pct_val, pct_num, marker="o", color=BRAND_SECONDARY,
                    markersize=6, zorder=5)
            ax.annotate(f"  {pct_name}: {pct_val:,.1f}",
                        xy=(pct_val, pct_num),
                        fontsize=7, color=BRAND_SECONDARY,
                        fontweight="bold",
                        verticalalignment="bottom")

    ax.set_xlabel("Power (kW)", fontsize=9)
    ax.set_ylabel("Cumulative Probability (%)", fontsize=9)
    ax.set_ylim(0, 105)
    title_col = column_name if len(columns_to_plot) == 1 else "All Columns"
    ax.set_title(f"{title_col} — Cumulative Distribution",
                 fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_peak_hour_frequency_chart(peak_hour_data: dict,
                                        column_name: str) -> bytes:
    """Bar chart of peak event frequency by hour of day.

    peak_hour_data: dict from compute_peak_hour_frequency()[column_name].
    Returns PNG bytes.
    """
    hourly_counts = peak_hour_data.get("hourly_counts", [0] * 24)
    avg_by_hour = peak_hour_data.get("avg_by_hour", [0.0] * 24)
    threshold = peak_hour_data.get("threshold_value", 0)

    if not any(c > 0 for c in hourly_counts):
        return _empty_chart("No peak events detected")

    hours = list(range(24))
    max_count = max(hourly_counts)

    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=120)

    # Bar colors: accent color for the highest-frequency hours
    bar_colors = []
    for c in hourly_counts:
        if max_count > 0 and c >= max_count * 0.8:
            bar_colors.append(BRAND_SECONDARY)
        else:
            bar_colors.append(BRAND_PRIMARY)

    ax1.bar(hours, hourly_counts, color=bar_colors, edgecolor="white",
            linewidth=0.5, alpha=0.85, zorder=3)

    ax1.set_xlabel("Hour of Day", fontsize=9)
    ax1.set_ylabel("Number of Peak Events", fontsize=9, color=BRAND_PRIMARY)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=7)
    ax1.set_xlim(-0.6, 23.6)
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)
    ax1.tick_params(axis="y", labelcolor=BRAND_PRIMARY)

    # Overlay: average consumption line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(hours, avg_by_hour, color=BRAND_LIGHT_GRAY, linewidth=1.5,
             linestyle="-", alpha=0.7, label=f"Avg Power (kW)", zorder=2)
    ax2.axhline(threshold, color=BRAND_SECONDARY, linewidth=1.0,
                linestyle="--", alpha=0.5,
                label=f"P{int(peak_hour_data.get('percentile_used', 90))} "
                      f"threshold: {threshold:,.0f} kW")
    ax2.set_ylabel("Average Power (kW)", fontsize=9, color=BRAND_LIGHT_GRAY)
    ax2.tick_params(axis="y", labelcolor=BRAND_LIGHT_GRAY)
    ax2.legend(fontsize=7, loc="upper left")

    ax1.set_title(f"{column_name} — Peak Event Frequency (24-Hour)",
                  fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _empty_chart(message: str) -> bytes:
    """Generate a small placeholder chart with a message."""
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=12, color=BRAND_LIGHT_GRAY, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_peak_summary_table(peak_data: dict,
                               grid_capacity_kw: float | None = None) -> Table:
    """Build a formatted table of top peaks."""
    top_peaks = peak_data.get("top_peaks", [])
    if not top_peaks:
        return Table([["No peaks detected"]], colWidths=[400])

    header = ["Rank", "Date & Time", "Day", "Month",
              "Value (kW)"]
    if grid_capacity_kw:
        header.append("% Grid")
    header.extend(["Duration (h)", "Rise (kW/h)", "Fall (kW/h)"])
    rows = [header]

    for p in top_peaks:
        row = [
            f"#{p['rank']}",
            p["timestamp"],
            p["day_of_week"],
            p["month"],
            f"{p['value']:,.1f}",
        ]
        if grid_capacity_kw:
            pct = p['value'] / grid_capacity_kw * 100
            row.append(f"{pct:.1f}%")
        row.extend([
            f"{p['duration_hours']:.1f}",
            f"{p['rise_rate']:,.1f}",
            f"{p['fall_rate']:,.1f}",
        ])
        rows.append(row)

    if grid_capacity_kw:
        col_widths = [30, 85, 50, 35, 55, 45, 50, 55, 50]
    else:
        col_widths = [35, 95, 60, 40, 60, 55, 60, 55]
    table = Table(rows, colWidths=col_widths)

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_commands.append(("BACKGROUND", (0, i), (-1, i), _RL_BG))

    table.setStyle(TableStyle(style_commands))
    return table


def _build_peak_characteristics_table(peak_data: dict,
                                       grid_capacity_kw: float | None = None) -> Table:
    """Build a table of peak characteristics and threshold analysis."""
    chars = peak_data.get("characteristics", {})
    thresholds = peak_data.get("thresholds", {})
    patterns = peak_data.get("patterns", {})
    clustering = chars.get("clustering", {})
    filtering = peak_data.get("data_filtering", {})

    rows = [
        ["Metric", "Value"],
        ["Total Peaks Detected", str(patterns.get("total_peaks_detected", 0))],
        ["Average Peak Duration", f"{chars.get('avg_duration_hours', 0):.1f} hours"],
        ["Average Rise Rate", f"{chars.get('avg_rise_rate', 0):,.1f} kW/hour"],
        ["Average Fall Rate", f"{chars.get('avg_fall_rate', 0):,.1f} kW/hour"],
        ["Clustered Peaks", str(clustering.get("clustered_count", 0))],
        ["Isolated Peaks", str(clustering.get("isolated_count", 0))],
        ["Avg Cluster Size", f"{clustering.get('avg_cluster_size', 0):.1f}"],
        ["90th Percentile", f"{thresholds.get('p90_value', 0):,.2f} kW"],
        ["95th Percentile", f"{thresholds.get('p95_value', 0):,.2f} kW"],
        ["Time Above 90th Pct", f"{thresholds.get('time_above_p90_hours', 0):,.1f} hours"],
        ["Time Above 95th Pct", f"{thresholds.get('time_above_p95_hours', 0):,.1f} hours"],
        ["Peak-to-Average Ratio", f"{thresholds.get('peak_to_avg_ratio', 0):.2f}x"],
    ]

    # Add grid capacity comparison
    if grid_capacity_kw and grid_capacity_kw > 0:
        top_peaks = peak_data.get("top_peaks", [])
        if top_peaks:
            max_peak = max(p["value"] for p in top_peaks)
            pct = max_peak / grid_capacity_kw * 100
            rows.append(["Grid Connection Capacity", f"{grid_capacity_kw:,.1f} kW"])
            rows.append(["Max Peak vs Grid Capacity", f"{pct:.1f}%"])

    # Add data filtering rows if applicable
    if filtering.get("filter_applied"):
        rows.append(["Data Points Used", f"{filtering.get('original_points', 0):,}"])
        rows.append(["Corrected Excluded", f"{filtering.get('excluded_points', 0):,} ({filtering.get('excluded_pct', 0):.1f}%)"])

    table = Table(rows, colWidths=[160, 160])
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_commands.append(("BACKGROUND", (0, i), (-1, i), _RL_BG))
    table.setStyle(TableStyle(style_commands))
    return table


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _header_footer(canvas, doc, logo_path=None, site_name=None,
                    report_title="Spartacus Energy Analysis Report"):
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
    header_title = f"Spartacus \u2014 {report_title}"
    canvas.setFillColor(_RL_WHITE)
    if site_name:
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(48 * mm, height - 14 * mm, header_title)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(48 * mm, height - 20 * mm, site_name)
    else:
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(48 * mm, height - 17 * mm, header_title)

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
                      f"ReVolta srl \u2014 {report_title}")
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


def generate_cost_breakdown_pie(cost_data: dict) -> bytes:
    """Render a cost breakdown pie chart PNG in memory.

    cost_data: dict with keys like 'energy_cost', 'grid_capacity_cost', etc.
    Returns PNG bytes.
    """
    labels = []
    values = []
    pie_colors = [
        BRAND_PRIMARY, BRAND_SECONDARY, "#28A745", "#FFC107",
        "#6B46C1", "#FF8C00", "#17A2B8",
    ]

    items = [
        ("Energy Cost", cost_data.get("energy_cost", 0)),
        ("Grid Capacity", cost_data.get("grid_capacity_cost", 0)),
        ("Grid Energy", cost_data.get("grid_energy_cost", 0)),
        ("Taxes & Levies", cost_data.get("taxes_and_levies", 0)),
        ("Overshoot Penalties", cost_data.get("overshoot_penalties", 0)),
        ("Prosumer Tariff", cost_data.get("prosumer_tariff", 0)),
    ]

    for label, val in items:
        if val > 0:
            labels.append(label)
            values.append(val)

    if not values:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.text(0.5, 0.5, "No cost data", ha="center", va="center",
                fontsize=14, color=BRAND_PRIMARY)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    wedges, texts, autotexts = ax.pie(
        values, autopct="%1.1f%%",
        colors=pie_colors[:len(values)],
        startangle=90, pctdistance=0.80,
        textprops={"fontsize": 9})

    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")

    ax.set_aspect("equal")
    ax.set_title("Cost Breakdown", fontsize=12, fontweight="bold",
                 color=BRAND_PRIMARY, pad=15)
    ax.legend(wedges, labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8,
              frameon=False)
    fig.subplots_adjust(bottom=0.15)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_monthly_cost_bar_chart(monthly_data: dict) -> bytes:
    """Render a monthly cost bar chart PNG in memory.

    monthly_data: dict of {month_key: CostBreakdown-like dict}
    Returns PNG bytes.
    """
    months = sorted(monthly_data.keys())
    if not months:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        ax.text(0.5, 0.5, "No monthly data", ha="center", va="center")
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    energy_costs = [monthly_data[m].get("energy_cost", 0) for m in months]
    grid_cap = [monthly_data[m].get("grid_capacity_cost", 0) for m in months]
    grid_en = [monthly_data[m].get("grid_energy_cost", 0) for m in months]
    taxes = [monthly_data[m].get("taxes_and_levies", 0) for m in months]
    penalties = [monthly_data[m].get("overshoot_penalties", 0) for m in months]

    x = np.arange(len(months))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

    bottom = np.zeros(len(months))
    bar_data = [
        ("Energy", energy_costs, BRAND_PRIMARY),
        ("Grid Capacity", grid_cap, BRAND_SECONDARY),
        ("Grid Energy", grid_en, "#28A745"),
        ("Taxes & Levies", taxes, "#FFC107"),
        ("Penalties", penalties, "#FF8C00"),
    ]

    for label, vals, color in bar_data:
        vals_arr = np.array(vals, dtype=float)
        if vals_arr.sum() > 0:
            ax.bar(x, vals_arr, width, bottom=bottom, label=label, color=color)
            bottom += vals_arr

    # Short month labels
    labels = []
    for m in months:
        try:
            parts = m.split("-")
            month_idx = int(parts[1]) - 1
            labels.append(MONTH_NAMES[month_idx])
        except (IndexError, ValueError):
            labels.append(m)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cost (\u20ac)", fontsize=9)
    ax.set_title("Monthly Cost Breakdown", fontsize=12,
                 fontweight="bold", color=BRAND_PRIMARY)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_scenario_comparison_chart(comparison_data: list[dict]) -> bytes:
    """Render a scenario comparison bar chart PNG in memory.

    comparison_data: list of dicts with 'Scenario' and cost component keys.
    Returns PNG bytes.
    """
    if not comparison_data:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center")
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    scenarios = [d["Scenario"] for d in comparison_data]
    totals = [d.get("Total Cost (excl. VAT)", 0) for d in comparison_data]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    colors_list = [BRAND_PRIMARY, BRAND_SECONDARY, "#28A745", "#FFC107"]
    bars = ax.bar(range(len(scenarios)), totals,
                  color=colors_list[:len(scenarios)])

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("Total Cost excl. VAT (\u20ac)", fontsize=9)
    ax.set_title("Scenario Comparison", fontsize=12,
                 fontweight="bold", color=BRAND_PRIMARY)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"\u20ac{val:,.0f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_pdf_report(output_path: str,
                         stats_result: dict | None = None,
                         kpi_data: dict | None = None,
                         logo_path: str | None = None,
                         quality_report: dict | None = None,
                         site_info: dict | None = None,
                         battery_data: dict | None = None,
                         cost_simulation_data: dict | None = None,
                         sections: list | None = None,
                         report_title: str = "Spartacus Energy Analysis Report",
                         data_overview: dict | None = None) -> str:
    """Generate branded PDF report using reportlab.

    Args:
        sections: List of section keys to include. If None, include all
            available sections (backward-compatible). Valid keys:
            "data_overview", "data_quality", "statistical", "peak_analysis",
            "battery", "cost_estimation".
        report_title: Custom title shown in the header/footer.
        data_overview: Optional dict with file info, granularity, date range.
        site_info: Optional dict with "site_name" and "grid_capacity_kw".
        battery_data: Optional dict from BatterySizer.generate_report_data().
        cost_simulation_data: Optional dict with cost simulation results.

    Returns output_path on success.
    """
    # Default: include everything (backward-compatible)
    if sections is None:
        sections = ["data_overview", "data_quality", "statistical",
                     "peak_analysis", "battery", "cost_estimation"]
    if stats_result is None:
        stats_result = {}

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=30 * mm,
        bottomMargin=18 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    story = []

    # --- Data Overview ---
    if "data_overview" in sections and data_overview:
        ov_rows = [["Field", "Value"]]
        if data_overview.get("file_name"):
            ov_rows.append(["Source File", data_overview["file_name"]])
        if data_overview.get("granularity"):
            ov_rows.append(["Data Granularity", data_overview["granularity"]])
        if data_overview.get("data_points"):
            ov_rows.append(["Data Points", f"{data_overview['data_points']:,}"])
        if data_overview.get("date_start"):
            ov_rows.append(["Date Range",
                            f"{data_overview['date_start']}  \u2014  "
                            f"{data_overview.get('date_end', '')}"])
        if data_overview.get("rows"):
            ov_rows.append(["Original Rows", f"{data_overview['rows']:,}"])
        if data_overview.get("columns"):
            ov_rows.append(["Original Columns", str(data_overview["columns"])])
        if data_overview.get("encoding"):
            ov_rows.append(["Encoding", data_overview["encoding"]])
        if data_overview.get("data_columns"):
            ov_rows.append(["Data Columns",
                            ", ".join(data_overview["data_columns"])])

        if len(ov_rows) > 1:
            ov_table = Table(ov_rows, colWidths=[160, 260])
            ov_style = [
                ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
            for i in range(1, len(ov_rows)):
                if i % 2 == 0:
                    ov_style.append(("BACKGROUND", (0, i), (-1, i), _RL_BG))
            ov_table.setStyle(TableStyle(ov_style))
            story.append(KeepTogether([
                _section_heading("Data Overview"),
                Spacer(1, 4 * mm),
                ov_table,
            ]))
            story.append(Spacer(1, 6 * mm))

    # --- Site Information ---
    if site_info:
        site_rows = [
            ["Field", "Value"],
            ["Site Name", site_info.get("site_name", "N/A")],
        ]
        grid_cap = site_info.get("grid_capacity_kw")
        if grid_cap is not None:
            site_rows.append(["Grid Connection Capacity", f"{grid_cap:,.1f} kW"])
        site_rows.append(["Report Date", datetime.now().strftime("%Y-%m-%d %H:%M")])

        site_table = Table(site_rows, colWidths=[160, 260])
        site_style = [
            ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]
        for i in range(1, len(site_rows)):
            if i % 2 == 0:
                site_style.append(("BACKGROUND", (0, i), (-1, i), _RL_BG))
        site_table.setStyle(TableStyle(site_style))
        story.append(KeepTogether([
            _section_heading("Site Information"),
            Spacer(1, 4 * mm),
            site_table,
        ]))
        story.append(Spacer(1, 6 * mm))

    # --- KPI Summary ---
    if "data_quality" in sections and kpi_data:
        story.append(KeepTogether([
            _section_heading("Data Quality Summary"),
            Spacer(1, 4 * mm),
            _build_kpi_table(kpi_data),
        ]))
        story.append(Spacer(1, 6 * mm))

    # --- Yearly Statistics Table ---
    yearly = stats_result.get("yearly", {})
    selected = stats_result.get("selected_metrics", [])

    if "statistical" in sections and yearly:
        # Filter out non-table metrics
        table_metrics = [m for m in selected
                         if m not in ("monthly_totals", "seasonal_profile",
                                      "peak_analysis", "frequency_histogram",
                                      "cumulative_distribution",
                                      "peak_hour_frequency")]
        if table_metrics:
            story.append(KeepTogether([
                _section_heading("Yearly Statistics"),
                Spacer(1, 4 * mm),
                _build_stats_table(yearly, table_metrics),
            ]))
            story.append(Spacer(1, 6 * mm))

    # --- Monthly Totals Bar Charts ---
    if "statistical" in sections and "monthly_totals" in selected and yearly:
        first_chart = True
        for col_name, col_stats in yearly.items():
            monthly = col_stats.get("monthly_totals", {})
            if monthly:
                chart_bytes = generate_monthly_bar_chart(monthly, col_name)
                img = RLImage(io.BytesIO(chart_bytes),
                              width=170 * mm, height=70 * mm)
                if first_chart:
                    story.append(KeepTogether([
                        _section_heading("Monthly Energy Totals"),
                        Spacer(1, 4 * mm),
                        img,
                    ]))
                    first_chart = False
                else:
                    story.append(img)
                story.append(Spacer(1, 4 * mm))

    # --- Seasonal Weekly Profiles ---
    seasonal = stats_result.get("seasonal", {})
    if "statistical" in sections and "seasonal_profile" in selected and seasonal:
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

    # --- Peak Consumption Analysis ---
    peaks = stats_result.get("peaks", {})
    if "peak_analysis" in sections and "peak_analysis" in selected and peaks:
        story.append(PageBreak())
        story.append(_section_heading("Peak Consumption Analysis"))
        story.append(Spacer(1, 4 * mm))

        # Extract grid capacity from site_info
        _grid_cap = None
        if site_info:
            _grid_cap = site_info.get("grid_capacity_kw")

        for col_name, peak_data in peaks.items():
            if not peak_data.get("top_peaks"):
                continue

            story.append(_body_text(f"<b>{col_name}</b>"))
            story.append(Spacer(1, 2 * mm))

            # Grid capacity note
            if _grid_cap and _grid_cap > 0:
                story.append(_body_text(
                    f"<i>Grid connection capacity: {_grid_cap:,.1f} kW.</i>"))
                story.append(Spacer(1, 1 * mm))

            # Data filtering transparency note
            filtering = peak_data.get("data_filtering", {})
            if filtering.get("filter_applied"):
                excluded = filtering.get("excluded_points", 0)
                excluded_pct = filtering.get("excluded_pct", 0)
                note = (f"<i>Analysis based on original data only &mdash; "
                        f"{excluded} corrected data point(s) ({excluded_pct}%) excluded.</i>")
                story.append(_body_text(note))

                excl_peaks = filtering.get("excluded_peaks_in_corrected", 0)
                if excl_peaks > 0:
                    story.append(_body_text(
                        f"<i>{excl_peaks} additional peak(s) found in corrected "
                        f"data periods (excluded from analysis).</i>"))
                story.append(Spacer(1, 2 * mm))

            # Top peaks table
            story.append(_body_text(
                "Top consumption peaks ranked by value, with duration "
                "(time above 90% of peak value), rise and fall rates:"))
            story.append(Spacer(1, 2 * mm))
            story.append(_build_peak_summary_table(peak_data,
                                                    grid_capacity_kw=_grid_cap))
            story.append(Spacer(1, 4 * mm))

            # Characteristics & thresholds table
            story.append(_build_peak_characteristics_table(peak_data,
                                                            grid_capacity_kw=_grid_cap))
            story.append(Spacer(1, 4 * mm))

            # Charts
            # 1. Peak value ranking
            chart_bytes = generate_peak_value_trend_chart(
                peak_data["top_peaks"], col_name)
            img = RLImage(io.BytesIO(chart_bytes),
                          width=170 * mm, height=70 * mm)
            story.append(img)
            story.append(Spacer(1, 3 * mm))

            # 2. Peak timeline
            chart_bytes = generate_peak_timeline_chart(
                peak_data.get("peak_timeline", []), col_name)
            img = RLImage(io.BytesIO(chart_bytes),
                          width=170 * mm, height=70 * mm)
            story.append(img)
            story.append(Spacer(1, 3 * mm))

            # 3. Peak durations
            chart_bytes = generate_peak_duration_chart(
                peak_data["top_peaks"], col_name)
            img = RLImage(io.BytesIO(chart_bytes),
                          width=170 * mm, height=70 * mm)
            story.append(img)
            story.append(Spacer(1, 3 * mm))

            # 4. Heatmap
            patterns = peak_data.get("patterns", {})
            chart_bytes = generate_peak_heatmap(
                patterns.get("hourly_distribution", {}),
                patterns.get("daily_distribution", {}),
                col_name)
            img = RLImage(io.BytesIO(chart_bytes),
                          width=170 * mm, height=85 * mm)
            story.append(img)
            story.append(Spacer(1, 6 * mm))

    # --- Frequency Histogram ---
    histogram = stats_result.get("histogram", {})
    if "statistical" in sections and "frequency_histogram" in selected and histogram:
        story.append(PageBreak())
        story.append(_section_heading("Frequency Distribution"))
        story.append(Spacer(1, 4 * mm))
        story.append(_body_text(
            "Distribution of power values showing frequency of occurrence "
            "across the measurement range. Mean and median lines indicate "
            "central tendency."))
        story.append(Spacer(1, 3 * mm))

        for col_name, hist_data in histogram.items():
            if not hist_data.get("counts"):
                continue
            chart_bytes = generate_histogram_chart(hist_data, col_name)
            img = RLImage(io.BytesIO(chart_bytes),
                          width=170 * mm, height=80 * mm)
            story.append(img)
            story.append(Spacer(1, 2 * mm))
            story.append(_body_text(
                f"<i>{col_name}: {hist_data.get('n_bins', 0)} bins, "
                f"bin width {hist_data.get('bin_width', 0):,.2f} kW. "
                f"Mean: {hist_data.get('mean', 0):,.2f} kW, "
                f"Median: {hist_data.get('median', 0):,.2f} kW, "
                f"Std Dev: {hist_data.get('std', 0):,.2f} kW.</i>"))
            story.append(Spacer(1, 4 * mm))

    # --- Cumulative Distribution ---
    cdf = stats_result.get("cdf", {})
    if "statistical" in sections and "cumulative_distribution" in selected and cdf:
        story.append(PageBreak())
        story.append(_section_heading("Cumulative Distribution"))
        story.append(Spacer(1, 4 * mm))
        story.append(_body_text(
            "Cumulative distribution function (CDF) showing the probability "
            "that power values fall below a given threshold. Key percentiles "
            "are marked for capacity planning and threshold analysis."))
        story.append(Spacer(1, 3 * mm))

        # Combined CDF chart (all columns overlaid)
        first_col = next(iter(cdf))
        chart_bytes = generate_cdf_chart(cdf[first_col], first_col, cdf)
        img = RLImage(io.BytesIO(chart_bytes),
                      width=170 * mm, height=80 * mm)
        story.append(img)
        story.append(Spacer(1, 2 * mm))

        # Percentile summary table
        pct_rows = [["Column", "50th Pct (kW)", "90th Pct (kW)", "95th Pct (kW)", "Count"]]
        for col_name, col_cdf in cdf.items():
            pcts = col_cdf.get("percentiles", {})
            pct_rows.append([
                col_name,
                f"{pcts.get('p50', 0):,.2f}",
                f"{pcts.get('p90', 0):,.2f}",
                f"{pcts.get('p95', 0):,.2f}",
                f"{col_cdf.get('count', 0):,}",
            ])

        pct_table = Table(pct_rows, colWidths=[120, 80, 80, 80, 60])
        pct_style = [
            ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
        for i in range(1, len(pct_rows)):
            if i % 2 == 0:
                pct_style.append(("BACKGROUND", (0, i), (-1, i), _RL_BG))
        pct_table.setStyle(TableStyle(pct_style))
        story.append(Spacer(1, 3 * mm))
        story.append(pct_table)
        story.append(Spacer(1, 4 * mm))

    # --- Peak Hour Frequency ---
    peak_hours = stats_result.get("peak_hours", {})
    if "peak_analysis" in sections and "peak_hour_frequency" in selected and peak_hours:
        story.append(PageBreak())
        story.append(_section_heading("Peak Event Frequency by Hour"))
        story.append(Spacer(1, 4 * mm))
        story.append(_body_text(
            "Distribution of peak events across the 24-hour day. "
            "Peak events are defined as readings at or above the "
            "90th percentile. High-frequency hours are highlighted "
            "to identify time-of-use demand patterns."))
        story.append(Spacer(1, 3 * mm))

        for col_name, ph_data in peak_hours.items():
            if ph_data.get("total_peaks", 0) == 0:
                continue

            chart_bytes = generate_peak_hour_frequency_chart(ph_data, col_name)
            img = RLImage(io.BytesIO(chart_bytes),
                          width=170 * mm, height=80 * mm)
            story.append(img)
            story.append(Spacer(1, 2 * mm))

            # Insight text
            peak_h = ph_data["peak_hour"]
            peak_hc = ph_data["peak_hour_count"]
            conc = ph_data.get("concentration", {})
            pf_hours = ph_data.get("peak_free_hours", [])

            insights = []
            insights.append(
                f"Peak events most common at <b>{peak_h:02d}:00</b> "
                f"({peak_hc} occurrences).")
            if conc.get("pct", 0) > 0:
                insights.append(
                    f"Peak concentration: <b>{conc['pct']:.0f}%</b> of peaks "
                    f"occur between {conc['start_hour']:02d}:00 &ndash; "
                    f"{conc['end_hour']:02d}:59.")
            if pf_hours:
                if len(pf_hours) <= 8:
                    pf_str = ", ".join(f"{h:02d}:00" for h in pf_hours)
                else:
                    pf_str = (f"{pf_hours[0]:02d}:00 &ndash; "
                              f"{pf_hours[-1]:02d}:00 "
                              f"({len(pf_hours)} hours)")
                insights.append(f"Peak-free hours: {pf_str}.")

            for line in insights:
                story.append(_body_text(f"<i>{line}</i>"))
            story.append(Spacer(1, 4 * mm))

    # --- Energy Cost Simulation ---
    if "cost_estimation" in sections and cost_simulation_data:
        story.append(PageBreak())
        story.append(_section_heading("Energy Cost Simulation"))
        story.append(Spacer(1, 4 * mm))

        # Contract summary
        contract_summary = cost_simulation_data.get("contract_summary", "")
        if contract_summary:
            for line in contract_summary.split("\n"):
                line = line.strip()
                if line:
                    story.append(_body_text(line))
            story.append(Spacer(1, 4 * mm))

        # Cost summary table
        cs_summary = cost_simulation_data.get("summary", {})
        if cs_summary:
            cs_rows = [
                ["Metric", "Value"],
                ["Total Consumption",
                 f"{cs_summary.get('total_consumption_kwh', 0) / 1000:,.1f} MWh"],
                ["Total Production",
                 f"{cs_summary.get('total_production_kwh', 0) / 1000:,.1f} MWh"],
                ["Self-Consumed",
                 f"{cs_summary.get('self_consumed_kwh', 0) / 1000:,.1f} MWh"],
                ["Self-Consumption Rate",
                 f"{cs_summary.get('self_consumption_rate', 0):.1%}"],
                ["Autarky Rate",
                 f"{cs_summary.get('autarky_rate', 0):.1%}"],
                ["Energy Cost",
                 f"\u20ac{cs_summary.get('energy_cost', 0):,.2f}"],
                ["Grid Capacity Cost",
                 f"\u20ac{cs_summary.get('grid_capacity_cost', 0):,.2f}"],
                ["Grid Energy Cost",
                 f"\u20ac{cs_summary.get('grid_energy_cost', 0):,.2f}"],
                ["Taxes & Levies",
                 f"\u20ac{cs_summary.get('taxes_and_levies', 0):,.2f}"],
                ["Overshoot Penalties",
                 f"\u20ac{cs_summary.get('overshoot_penalties', 0):,.2f}"],
                ["Injection Revenue",
                 f"-\u20ac{cs_summary.get('injection_revenue', 0):,.2f}"],
                ["Total Cost (excl. VAT)",
                 f"\u20ac{cs_summary.get('total_cost_excl_vat', 0):,.2f}"],
                ["Average Cost",
                 f"\u20ac{cs_summary.get('avg_eur_per_kwh', 0):.4f}/kWh"],
                ["Peak Demand",
                 f"{cs_summary.get('peak_demand_kw', 0):,.1f} kW"],
                ["Overshoots",
                 str(cs_summary.get("overshoots_count", 0))],
            ]

            cs_table = Table(cs_rows, colWidths=[180, 240])
            cs_style = [
                ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
            for i in range(1, len(cs_rows)):
                if i % 2 == 0:
                    cs_style.append(
                        ("BACKGROUND", (0, i), (-1, i), _RL_BG))
            cs_table.setStyle(TableStyle(cs_style))
            story.append(cs_table)
            story.append(Spacer(1, 6 * mm))

        # Charts
        cs_charts = cost_simulation_data.get("charts", {})
        for chart_key, chart_title in [
            ("breakdown_pie", "Cost Breakdown"),
            ("monthly_bar", "Monthly Cost Breakdown"),
            ("scenario_comparison", "Scenario Comparison"),
        ]:
            chart_bytes = cs_charts.get(chart_key)
            if chart_bytes:
                # Pie chart uses square dimensions to avoid distortion
                if chart_key == "breakdown_pie":
                    img = RLImage(io.BytesIO(chart_bytes),
                                  width=120 * mm, height=120 * mm)
                else:
                    img = RLImage(io.BytesIO(chart_bytes),
                                  width=170 * mm, height=75 * mm)
                story.append(KeepTogether([
                    _section_heading(chart_title),
                    Spacer(1, 4 * mm),
                    img,
                    Spacer(1, 4 * mm),
                ]))
                story.append(Spacer(1, 2 * mm))

        # Monthly breakdown table
        cs_monthly = cost_simulation_data.get("monthly", {})
        if cs_monthly:
            story.append(PageBreak())
            story.append(_section_heading("Monthly Cost Breakdown"))
            story.append(Spacer(1, 4 * mm))

            mt_header = ["Month", "Consumption\n(MWh)",
                         "Grid Cost\n(\u20ac)", "Energy\n(\u20ac)",
                         "Taxes\n(\u20ac)", "Total\n(\u20ac)",
                         "Avg\n(\u20ac/kWh)"]
            mt_rows = [mt_header]

            for month_key in sorted(cs_monthly.keys()):
                md = cs_monthly[month_key]
                cons_mwh = md.get("total_consumption_kwh", 0) / 1000
                total = md.get("total_cost_excl_vat", 0)
                cons_kwh = md.get("total_consumption_kwh", 1)
                avg = total / max(cons_kwh, 1)
                mt_rows.append([
                    month_key,
                    f"{cons_mwh:,.1f}",
                    f"{md.get('grid_capacity_cost', 0) + md.get('grid_energy_cost', 0):,.0f}",
                    f"{md.get('energy_cost', 0):,.0f}",
                    f"{md.get('taxes_and_levies', 0):,.0f}",
                    f"{total:,.0f}",
                    f"{avg:.4f}",
                ])

            mt_table = Table(mt_rows,
                             colWidths=[50, 60, 60, 60, 55, 60, 55])
            mt_style = [
                ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5,
                 colors.HexColor("#D0D5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
            for i in range(1, len(mt_rows)):
                if i % 2 == 0:
                    mt_style.append(
                        ("BACKGROUND", (0, i), (-1, i), _RL_BG))
            mt_table.setStyle(TableStyle(mt_style))
            story.append(mt_table)
            story.append(Spacer(1, 6 * mm))

        # Comparison table
        cs_comparison = cost_simulation_data.get("comparison", [])
        if cs_comparison:
            comp_header = ["Scenario", "Total Cost\n(\u20ac)",
                           "Avg\n(\u20ac/kWh)",
                           "Self-Cons\nRate", "Autarky\nRate",
                           "Peak\n(kW)"]
            comp_rows = [comp_header]
            for sc in cs_comparison:
                comp_rows.append([
                    sc.get("Scenario", ""),
                    f"{sc.get('Total Cost (excl. VAT)', 0):,.0f}",
                    str(sc.get("Avg \u20ac/kWh", "")),
                    str(sc.get("Self-Consumption Rate", "")),
                    str(sc.get("Autarky Rate", "")),
                    f"{sc.get('Peak Demand (kW)', 0):,.0f}",
                ])

            comp_table = Table(comp_rows,
                               colWidths=[90, 65, 55, 55, 55, 50])
            comp_style = [
                ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5,
                 colors.HexColor("#D0D5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
            for i in range(1, len(comp_rows)):
                if i % 2 == 0:
                    comp_style.append(
                        ("BACKGROUND", (0, i), (-1, i), _RL_BG))
            comp_table.setStyle(TableStyle(comp_style))
            story.append(KeepTogether([
                _section_heading("Scenario Comparison"),
                Spacer(1, 4 * mm),
                comp_table,
            ]))
            story.append(Spacer(1, 6 * mm))

    # --- Battery Dimensioning Analysis ---
    if "battery" in sections and battery_data:
        story.append(PageBreak())
        story.append(_section_heading("Battery Dimensioning Analysis"))
        story.append(Spacer(1, 4 * mm))

        # Input parameters table
        tariffs = battery_data.get("tariffs", {})
        param_rows = [
            ["Parameter", "Value"],
        ]
        if site_info:
            param_rows.append(["Site Name",
                                site_info.get("site_name", "N/A")])
        param_rows.extend([
            ["Offtake Tariff", f"{tariffs.get('offtake', 0):,.0f} \u20ac/MWh"],
            ["Injection Tariff",
             f"{tariffs.get('injection', 0):,.0f} \u20ac/MWh"],
            ["Peak Tariff", f"{tariffs.get('peak', 0):,.0f} \u20ac/kW"],
        ])
        param_table = Table(param_rows, colWidths=[160, 260])
        param_style = [
            ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]
        for i in range(1, len(param_rows)):
            if i % 2 == 0:
                param_style.append(
                    ("BACKGROUND", (0, i), (-1, i), _RL_BG))
        param_table.setStyle(TableStyle(param_style))
        story.append(param_table)
        story.append(Spacer(1, 6 * mm))

        # Battery sizing recommendations table
        rec = battery_data.get("recommendations", {})
        rec_rows = [
            ["Metric", "Value"],
            ["Maximum Capacity Needed",
             f"{rec.get('max_capacity', 0):,.1f} kWh"],
            ["Average Capacity Needed",
             f"{rec.get('avg_capacity', 0):,.1f} kWh"],
            ["Recommended Capacity",
             f"{rec.get('recommended_capacity', 0):,.1f} kWh"],
            ["Recommended Power Rating",
             f"{rec.get('recommended_power', 0):,.1f} kW"],
        ]
        rec_table = Table(rec_rows, colWidths=[200, 220])
        rec_style = [
            ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D5E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]
        for i in range(1, len(rec_rows)):
            if i % 2 == 0:
                rec_style.append(
                    ("BACKGROUND", (0, i), (-1, i), _RL_BG))
        rec_table.setStyle(TableStyle(rec_style))
        story.append(rec_table)
        story.append(Spacer(1, 6 * mm))

        # Economic analysis summary
        sav = battery_data.get("savings", {})

        econ_rows = [
            ["Metric", "Value"],
            ["Total Annual Savings",
             f"\u20ac{sav.get('annual_savings', 0):,.0f}"],
            ["Energy Arbitrage Savings",
             f"\u20ac{sav.get('energy_arbitrage', 0):,.0f}"],
            ["Peak Demand Reduction Savings",
             f"\u20ac{sav.get('peak_reduction', 0):,.0f}"],
            ["Peak Demand Reduction",
             f"{sav.get('peak_reduction_kw', 0):,.1f} kW"],
            ["Self-Consumption Rate",
             f"{sav.get('self_consumption_pct', 0):.1f}%"],
            ["Self-Consumption Increase",
             f"+{sav.get('self_consumption_increase', 0):.1f}%"],
        ]
        econ_table = Table(econ_rows, colWidths=[200, 220])
        econ_style = list(rec_style)  # same styling
        for i in range(1, len(econ_rows)):
            if i % 2 == 0:
                econ_style.append(
                    ("BACKGROUND", (0, i), (-1, i), _RL_BG))
        econ_table.setStyle(TableStyle(econ_style))
        story.append(KeepTogether([
            _section_heading("Economic Analysis"),
            Spacer(1, 4 * mm),
            econ_table,
        ]))
        story.append(Spacer(1, 6 * mm))

        # Charts
        charts = battery_data.get("charts", {})
        chart_order = [
            ("average_day_profile",
             "Typical Daily Production vs Consumption Profile"),
            ("monthly_storage", "Monthly Battery Storage Requirements"),
            ("annual_pattern", "Annual Storage Pattern"),
            ("duration_curve", "Storage Duration Curve"),
            ("monthly_savings", "Monthly Savings Potential"),
            ("self_consumption", "Energy Flow Analysis"),
        ]

        for chart_key, chart_title in chart_order:
            chart_bytes = charts.get(chart_key)
            if chart_bytes:
                story.append(PageBreak())
                story.append(_section_heading(chart_title))
                story.append(Spacer(1, 4 * mm))
                img = RLImage(io.BytesIO(chart_bytes),
                              width=170 * mm, height=80 * mm)
                story.append(img)
                story.append(Spacer(1, 4 * mm))

        # Monthly breakdown table
        monthly_table_data = battery_data.get("monthly_table", [])
        if monthly_table_data:
            story.append(PageBreak())
            story.append(_section_heading("Monthly Breakdown"))
            story.append(Spacer(1, 4 * mm))

            mt_header = ["Month", "Consumption\n(kWh)",
                         "Production\n(kWh)",
                         "Avg Storage\n(kWh)",
                         "Max Storage\n(kWh)",
                         "Est. Savings\n(\u20ac)"]
            mt_rows = [mt_header]
            for row in monthly_table_data:
                mt_rows.append([
                    row["month"],
                    f"{row['consumption']:,.0f}",
                    f"{row['production']:,.0f}",
                    f"{row['avg_storage']:,.1f}",
                    f"{row['max_storage']:,.1f}",
                    f"{row['est_savings']:,.0f}",
                ])

            mt_table = Table(mt_rows,
                             colWidths=[40, 75, 75, 70, 70, 70])
            mt_style = [
                ("BACKGROUND", (0, 0), (-1, 0), _RL_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), _RL_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5,
                 colors.HexColor("#D0D5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
            for i in range(1, len(mt_rows)):
                if i % 2 == 0:
                    mt_style.append(
                        ("BACKGROUND", (0, i), (-1, i), _RL_BG))
            mt_table.setStyle(TableStyle(mt_style))
            story.append(mt_table)
            story.append(Spacer(1, 6 * mm))

        # Professional recommendations text
        rec_text = battery_data.get("recommendation_text", "")
        if rec_text:
            # Build all paragraphs first
            rec_elements = [
                _section_heading("Professional Recommendation"),
                Spacer(1, 4 * mm),
            ]
            for paragraph in rec_text.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    para_html = paragraph.replace("\n", "<br/>")
                    rec_elements.append(_body_text(para_html))
                    rec_elements.append(Spacer(1, 3 * mm))
            # Keep heading with at least first paragraph
            if len(rec_elements) > 2:
                # KeepTogether for heading + first paragraph
                story.append(KeepTogether(rec_elements[:4]))
                for el in rec_elements[4:]:
                    story.append(el)
            else:
                story.extend(rec_elements)

    # Build PDF with header/footer
    _site_name = site_info.get("site_name") if site_info else None

    def on_page(canvas, doc_ref):
        _header_footer(canvas, doc_ref, logo_path=logo_path,
                        site_name=_site_name, report_title=report_title)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    return output_path
