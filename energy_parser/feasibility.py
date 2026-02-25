"""Spartacus - Feasibility Analysis Engine
==========================================
Evaluates a site's grid utilization and produces a scorecard with
recommendations for ReVolta's two battery offers:
  - Offer A: Grid Constraint / Leasing
  - Offer B: Joint Valorisation / BaaS

Works with consumption data + contract info; cost simulation results
and production data are optional enrichment.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional

from contract_model import EnergyContract


# === Scorecard Data Class ===

@dataclass
class FeasibilityScorecard:
    """Core metrics and results from feasibility analysis."""
    # Grid parameters
    pmax_kw: float = 0.0
    subscribed_power_kw: float = 0.0
    connection_power_kw: float = 0.0
    connection_size_category: str = ""
    total_consumption_kwh: float = 0.0
    has_production: bool = False

    # Core metrics
    grid_utilization_pct: float = 0.0
    headroom_kw: float = 0.0
    overshoot_count: int = 0
    max_overshoot_kw: float = 0.0
    available_capacity_kw: float = 0.0
    load_factor_pct: float = 0.0
    peak_concentration_pct: float = 0.0
    self_consumption_rate_pct: float = 0.0

    # Traffic lights: metric_name → "green" | "orange" | "red"
    traffic_lights: dict = field(default_factory=dict)

    # Offer relevance
    offer_a_relevance: str = "Low"       # "High" | "Medium" | "Low"
    offer_b_relevance: str = "Low"
    offer_a_rationale: str = ""
    offer_b_rationale: str = ""

    # Overall recommendation
    recommendation: str = ""

    # Chart data
    sorted_demand_kw: list = field(default_factory=list)
    demand_timeseries: list = field(default_factory=list)  # [(ts_str, kw), ...]
    overshoot_values: list = field(default_factory=list)


# === Threshold Constants ===

TRAFFIC_LIGHT_THRESHOLDS = {
    "grid_utilization_pct": {
        "green_max": 70.0,   # <70% → green
        "orange_max": 85.0,  # 70–85% → orange
        # >85% → red
        "higher_is_worse": True,
    },
    "headroom_kw": {
        "green_min": 100.0,  # >100kW → green
        "orange_min": 30.0,  # 30–100kW → orange
        # <30kW → red
        "higher_is_worse": False,
    },
    "overshoot_count": {
        "green_max": 0,
        "orange_max": 10,
        "higher_is_worse": True,
    },
    "max_overshoot_kw": {
        "green_max": 0.0,
        "orange_max": 20.0,
        "higher_is_worse": True,
    },
    "available_capacity_kw": {
        "green_min": 500.0,
        "orange_min": 200.0,
        "higher_is_worse": False,
    },
    "load_factor_pct": {
        "green_max": 30.0,
        "orange_max": 50.0,
        "higher_is_worse": True,
    },
    "peak_concentration_pct": {
        "green_min": 25.0,
        "orange_min": 15.0,
        "higher_is_worse": False,
    },
    "self_consumption_rate_pct": {
        "green_min": 70.0,
        "orange_min": 40.0,
        "higher_is_worse": False,
    },
}


# === Helper Functions ===

def _evaluate_traffic_light(value: float, thresholds: dict) -> str:
    """Determine traffic light color for a metric value."""
    if thresholds.get("higher_is_worse"):
        if value <= thresholds.get("green_max", 0):
            return "green"
        elif value <= thresholds.get("orange_max", 0):
            return "orange"
        else:
            return "red"
    else:
        if value >= thresholds.get("green_min", 0):
            return "green"
        elif value >= thresholds.get("orange_min", 0):
            return "orange"
        else:
            return "red"


def classify_connection_size(kw: float) -> str:
    """Classify the connection size category."""
    if kw <= 0:
        return "Unknown"
    elif kw < 30:
        return "Residential (<30 kW)"
    elif kw < 250:
        return "Small Commercial (30–250 kW)"
    elif kw < 800:
        return "Medium Industrial (250–800 kW)"
    else:
        return "Large Industrial (>800 kW)"


def _compute_peak_concentration(df: pd.DataFrame, contract: EnergyContract,
                                hours_per_interval: float) -> float:
    """Compute percentage of total consumption occurring during peak periods.

    Uses TimePeriod.matches(dt) to identify peak periods (names containing
    'pointe', 'peak', or 'hph'). Returns 0 if no time periods defined.
    """
    if not contract.energy.time_periods:
        return 0.0

    # Identify peak periods
    peak_periods = []
    for tp in contract.energy.time_periods:
        name_lower = tp.name.lower()
        if any(kw in name_lower for kw in ["pointe", "peak", "hph"]):
            peak_periods.append(tp)

    if not peak_periods:
        return 0.0

    total_kwh = df["Consumption (kW)"].sum() * hours_per_interval
    if total_kwh <= 0:
        return 0.0

    # Sum consumption during peak periods
    peak_kwh = 0.0
    timestamps = pd.to_datetime(df["Date & Time"])
    cons_values = df["Consumption (kW)"].values
    for i, ts in enumerate(timestamps):
        dt = ts.to_pydatetime()
        for tp in peak_periods:
            if tp.matches(dt):
                peak_kwh += cons_values[i] * hours_per_interval
                break

    return (peak_kwh / total_kwh) * 100.0


# === Offer Relevance Logic ===

def _determine_offer_a(sc: FeasibilityScorecard) -> tuple:
    """Determine Offer A (Grid Constraint / Leasing) relevance.

    Returns (relevance_level, rationale).
    Connection power gate is skipped when the value is not set (0).
    """
    grid_util = sc.grid_utilization_pct
    headroom = sc.headroom_kw
    conn_kw = sc.connection_power_kw
    # If connection power is not set, don't let it block the assessment
    conn_ok = (conn_kw >= 30) if conn_kw > 0 else True

    if (grid_util > 85 and headroom < 30 and conn_ok):
        return ("High", (
            f"Grid utilization is critically high at {grid_util:.1f}% "
            f"with only {headroom:.1f} kW of headroom. "
            f"A battery system can shave peaks and prevent overshoots, "
            f"avoiding costly grid upgrades or penalty charges."
        ))
    elif ((grid_util > 70 or headroom < 100) and conn_ok):
        return ("Medium", (
            f"Grid utilization ({grid_util:.1f}%) and headroom "
            f"({headroom:.1f} kW) suggest moderate grid stress. "
            f"A leasing battery could defer grid upgrade investments "
            f"and reduce overshoot risk."
        ))
    else:
        return ("Low", (
            f"Grid utilization is low at {grid_util:.1f}% "
            f"with {headroom:.1f} kW of headroom. "
            f"Peak shaving via Offer A is not a priority at this time."
        ))


def _determine_offer_b(sc: FeasibilityScorecard) -> tuple:
    """Determine Offer B (Joint Valorisation / BaaS) relevance.

    Returns (relevance_level, rationale).
    Connection power gate is skipped when the value is not set (0).
    """
    avail_cap = sc.available_capacity_kw
    load_factor = sc.load_factor_pct
    conn_kw = sc.connection_power_kw
    # If connection power is not set, don't let it block the assessment
    conn_high = (conn_kw >= 800) if conn_kw > 0 else True
    conn_med = (conn_kw >= 250) if conn_kw > 0 else True

    if (avail_cap > 500 and load_factor < 30 and conn_high):
        return ("High", (
            f"The site has {avail_cap:.0f} kW of available capacity "
            f"and a low load factor ({load_factor:.1f}%), indicating "
            f"significant unused grid capacity. A battery deployed for "
            f"joint valorisation (BaaS) can monetize this spare capacity "
            f"through ancillary services and arbitrage."
        ))
    elif ((200 <= avail_cap <= 500 or 30 <= load_factor <= 50)
          and conn_med):
        return ("Medium", (
            f"Available capacity ({avail_cap:.0f} kW) and load factor "
            f"({load_factor:.1f}%) show moderate potential for grid "
            f"services. Joint valorisation could generate additional "
            f"revenue, especially during off-peak periods."
        ))
    else:
        return ("Low", (
            f"With {avail_cap:.0f} kW available capacity and a "
            f"{load_factor:.1f}% load factor, the site's spare grid "
            f"capacity is limited. Joint valorisation (Offer B) would "
            f"have limited revenue potential."
        ))


# === Recommendation Generation ===

def _generate_recommendation(sc: FeasibilityScorecard) -> str:
    """Generate template-based recommendation text with actual values."""
    lines = []

    # Site overview
    lines.append(
        f"Site overview: The site has a {sc.connection_power_kw:.0f} kW "
        f"connection ({sc.connection_size_category}) with "
        f"{sc.subscribed_power_kw:.0f} kW subscribed power. "
        f"Peak demand reached {sc.pmax_kw:.1f} kW, resulting in "
        f"{sc.grid_utilization_pct:.1f}% grid utilization."
    )

    # Overshoot warning
    if sc.overshoot_count > 0:
        lines.append(
            f"\nOvershoot alert: {sc.overshoot_count} overshoot events "
            f"were detected, with a maximum exceedance of "
            f"{sc.max_overshoot_kw:.1f} kW above subscribed power. "
            f"This results in penalty charges and indicates the need "
            f"for peak management."
        )

    # Load factor
    if sc.load_factor_pct < 30:
        lines.append(
            f"\nLoad profile: The low load factor ({sc.load_factor_pct:.1f}%) "
            f"indicates a peaky consumption profile with significant "
            f"idle grid capacity that could be monetized."
        )
    elif sc.load_factor_pct > 50:
        lines.append(
            f"\nLoad profile: The high load factor ({sc.load_factor_pct:.1f}%) "
            f"indicates a relatively flat consumption profile with "
            f"consistent grid utilization."
        )
    else:
        lines.append(
            f"\nLoad profile: The moderate load factor "
            f"({sc.load_factor_pct:.1f}%) indicates a balanced consumption "
            f"profile with some peak-to-average variation."
        )

    # Self-consumption
    if sc.has_production and sc.self_consumption_rate_pct > 0:
        lines.append(
            f"\nSelf-consumption: On-site production achieves a "
            f"{sc.self_consumption_rate_pct:.1f}% self-consumption rate. "
            + ("Battery storage could further increase self-consumption "
               "and reduce grid dependency."
               if sc.self_consumption_rate_pct < 70
               else "Self-consumption is already well-optimized.")
        )

    # Offer A
    lines.append(
        f"\nOffer A (Grid Constraint / Leasing) — {sc.offer_a_relevance}: "
        f"{sc.offer_a_rationale}"
    )

    # Offer B
    lines.append(
        f"\nOffer B (Joint Valorisation / BaaS) — {sc.offer_b_relevance}: "
        f"{sc.offer_b_rationale}"
    )

    return "\n".join(lines)


# === Main Entry Point ===

def run_feasibility_analysis(
    df: pd.DataFrame,
    contract: EnergyContract,
    hours_per_interval: float,
    cost_summary=None,
) -> FeasibilityScorecard:
    """Run feasibility analysis on consumption data with contract parameters.

    Args:
        df: Transformed DataFrame with 'Date & Time' and 'Consumption (kW)'
            columns. May also contain 'Production (kW)'.
        contract: EnergyContract with grid parameters.
        hours_per_interval: Hours per data interval (e.g., 0.25 for 15-min).
        cost_summary: Optional CostBreakdown from cost simulation.

    Returns:
        FeasibilityScorecard with all computed metrics and recommendations.
    """
    sc = FeasibilityScorecard()

    # --- Grid parameters ---
    sc.subscribed_power_kw = contract.grid.subscribed_power_kw
    sc.connection_power_kw = contract.grid.connection_power_limit_kw
    sc.connection_size_category = classify_connection_size(sc.connection_power_kw)
    sc.has_production = (
        "Production (kW)" in df.columns
        and df["Production (kW)"].sum() > 0
    )

    # --- Consumption metrics ---
    consumption = df["Consumption (kW)"].values
    sc.total_consumption_kwh = float(consumption.sum() * hours_per_interval)

    # Net demand (consumption - production if available)
    if sc.has_production:
        production = df["Production (kW)"].values
        net_demand = np.maximum(consumption - production, 0)
    else:
        net_demand = consumption.copy()

    sc.pmax_kw = float(np.max(net_demand)) if len(net_demand) > 0 else 0.0

    # --- Core metrics ---
    # Grid utilization: Pmax / subscribed_power * 100
    if sc.subscribed_power_kw > 0:
        sc.grid_utilization_pct = (sc.pmax_kw / sc.subscribed_power_kw) * 100.0
    else:
        sc.grid_utilization_pct = 0.0

    # Headroom: subscribed_power - Pmax
    sc.headroom_kw = max(sc.subscribed_power_kw - sc.pmax_kw, 0.0)

    # Overshoots (exceedances above subscribed power)
    if sc.subscribed_power_kw > 0:
        overshoots = net_demand - sc.subscribed_power_kw
        overshoot_mask = overshoots > 0
        sc.overshoot_count = int(overshoot_mask.sum())
        sc.max_overshoot_kw = float(overshoots.max()) if sc.overshoot_count > 0 else 0.0
        sc.overshoot_values = overshoots[overshoot_mask].tolist()
    else:
        sc.overshoot_count = 0
        sc.max_overshoot_kw = 0.0
        sc.overshoot_values = []

    # Available capacity: connection_power - Pmax
    sc.available_capacity_kw = max(sc.connection_power_kw - sc.pmax_kw, 0.0)

    # Load factor: average demand / Pmax * 100
    avg_demand = float(np.mean(net_demand)) if len(net_demand) > 0 else 0.0
    if sc.pmax_kw > 0:
        sc.load_factor_pct = (avg_demand / sc.pmax_kw) * 100.0
    else:
        sc.load_factor_pct = 0.0

    # Peak concentration
    sc.peak_concentration_pct = _compute_peak_concentration(
        df, contract, hours_per_interval)

    # Self-consumption rate
    if cost_summary is not None and hasattr(cost_summary, 'self_consumption_rate'):
        sc.self_consumption_rate_pct = cost_summary.self_consumption_rate * 100.0
    elif sc.has_production:
        prod_total = float(df["Production (kW)"].sum() * hours_per_interval)
        self_consumed = 0.0
        if prod_total > 0:
            min_vals = np.minimum(consumption, df["Production (kW)"].values)
            self_consumed = float(min_vals.sum() * hours_per_interval)
            sc.self_consumption_rate_pct = (self_consumed / prod_total) * 100.0
    else:
        sc.self_consumption_rate_pct = 0.0

    # --- Traffic lights ---
    for metric_name, thresholds in TRAFFIC_LIGHT_THRESHOLDS.items():
        value = getattr(sc, metric_name, 0.0)
        # Skip self-consumption if no production
        if metric_name == "self_consumption_rate_pct" and not sc.has_production:
            sc.traffic_lights[metric_name] = "gray"
            continue
        # Skip peak concentration if no peak periods
        if metric_name == "peak_concentration_pct" and sc.peak_concentration_pct == 0:
            sc.traffic_lights[metric_name] = "gray"
            continue
        sc.traffic_lights[metric_name] = _evaluate_traffic_light(value, thresholds)

    # --- Offer relevance ---
    sc.offer_a_relevance, sc.offer_a_rationale = _determine_offer_a(sc)
    sc.offer_b_relevance, sc.offer_b_rationale = _determine_offer_b(sc)

    # --- Recommendation ---
    sc.recommendation = _generate_recommendation(sc)

    # --- Chart data ---
    # Sorted demand curve (descending) for load duration curve
    sorted_demand = np.sort(net_demand)[::-1]
    sc.sorted_demand_kw = sorted_demand.tolist()

    # Demand timeseries (downsampled to ≤2000 points)
    timestamps = pd.to_datetime(df["Date & Time"])
    n = len(net_demand)
    if n > 2000:
        step = max(n // 2000, 1)
        indices = np.arange(0, n, step)
        sc.demand_timeseries = [
            (str(timestamps.iloc[i]), float(net_demand[i]))
            for i in indices
        ]
    else:
        sc.demand_timeseries = [
            (str(timestamps.iloc[i]), float(net_demand[i]))
            for i in range(n)
        ]

    return sc


def scorecard_to_dict(sc: FeasibilityScorecard) -> dict:
    """Convert scorecard to a serializable dict."""
    return asdict(sc)
