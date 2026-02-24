"""
Spartacus - Energy Cost Simulation Engine
==========================================
Takes consumption (and optionally production) time series data + an EnergyContract
and computes the detailed energy cost breakdown per timestep.

Designed to integrate with Spartacus's existing data pipeline:
  - Uses the same DataFrame format (datetime index, kW columns)
  - Same granularity detection (15-min, 30-min, 60-min)
  - Compatible with corrected/quality-checked data
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from energy_parser.contract_model import (
    EnergyContract, MeteringMode, PriceType, Country
)


# === Simulation Result ===

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a simulation period."""
    # Volumes (kWh)
    total_consumption_kwh: float = 0.0
    total_production_kwh: float = 0.0
    self_consumed_kwh: float = 0.0
    grid_consumed_kwh: float = 0.0       # consumption - self_consumed
    injected_kwh: float = 0.0            # production - self_consumed
    self_consumption_rate: float = 0.0    # self_consumed / production (if > 0)
    autarky_rate: float = 0.0            # self_consumed / consumption

    # Costs (€) — positive = cost, negative = revenue
    energy_cost: float = 0.0             # Supplier energy charges
    grid_capacity_cost: float = 0.0      # Subscribed power fee
    grid_energy_cost: float = 0.0        # Grid energy component
    taxes_and_levies: float = 0.0        # Excise + renewable + other
    overshoot_penalties: float = 0.0     # Exceeding subscribed/connection power
    injection_revenue: float = 0.0       # Revenue from surplus injection
    green_certificate_revenue: float = 0.0
    prosumer_tariff: float = 0.0         # Belgium prosumer fee (if applicable)

    # Derived
    total_cost_excl_vat: float = 0.0
    vat_amount: float = 0.0
    total_cost_incl_vat: float = 0.0

    # Peak demand
    peak_demand_kw: float = 0.0
    overshoots_count: int = 0
    max_overshoot_kw: float = 0.0

    def compute_totals(self, contract: EnergyContract):
        """Compute derived totals from component costs."""
        self.total_cost_excl_vat = (
            self.energy_cost
            + self.grid_capacity_cost
            + self.grid_energy_cost
            + self.taxes_and_levies
            + self.overshoot_penalties
            + self.prosumer_tariff
            - self.injection_revenue
            - self.green_certificate_revenue
        )
        if contract.taxes.vat_applicable:
            self.vat_amount = self.total_cost_excl_vat * contract.taxes.vat_rate
        else:
            self.vat_amount = 0.0
        self.total_cost_incl_vat = self.total_cost_excl_vat + self.vat_amount


# === Main Simulation Engine ===

class CostSimulator:
    """
    Simulates energy costs based on consumption/production profiles and a contract.

    Usage:
        simulator = CostSimulator(contract)
        results_df, summary = simulator.simulate(consumption_df, production_df)
    """

    def __init__(self, contract: EnergyContract):
        self.contract = contract

    def simulate(
        self,
        consumption: pd.DataFrame,
        production: Optional[pd.DataFrame] = None,
        consumption_col: str = "consumption_kw",
        production_col: str = "production_kw",
    ) -> tuple[pd.DataFrame, CostBreakdown, dict[str, CostBreakdown]]:
        """
        Run the cost simulation.

        Args:
            consumption: DataFrame with datetime index, power in kW
            production: Optional DataFrame with datetime index, power in kW
            consumption_col: Column name for consumption data
            production_col: Column name for production data

        Returns:
            - Detailed DataFrame with cost per timestep
            - Overall CostBreakdown summary
            - Monthly CostBreakdown dict (key = "YYYY-MM")
        """
        c = self.contract
        df = self._prepare_data(consumption, production, consumption_col, production_col)

        # Detect granularity (minutes per timestep)
        granularity_min = self._detect_granularity(df)
        hours_per_step = granularity_min / 60.0

        # === Compute volumes per timestep (kWh) ===
        df["consumption_kwh"] = df["consumption_kw"] * hours_per_step
        df["production_kwh"] = df["production_kw"] * hours_per_step

        # Self-consumption and injection
        if c.production.has_production and "production_kw" in df.columns:
            df = self._compute_self_consumption(df, hours_per_step)
        else:
            df["self_consumed_kwh"] = 0.0
            df["grid_consumed_kwh"] = df["consumption_kwh"]
            df["injected_kwh"] = 0.0

        # === Energy cost ===
        df["energy_price"] = df.index.map(lambda dt: c.energy.get_energy_price(dt))
        df["energy_cost"] = df["grid_consumed_kwh"] * df["energy_price"]

        # === Grid energy cost ===
        if c.grid.time_differentiated:
            df["grid_energy_price"] = df.index.map(
                lambda dt: c.grid.get_grid_energy_price(dt, c.energy)
            )
        else:
            df["grid_energy_price"] = c.grid.flat_grid_energy_eur_per_kwh

        # For self-consumed kWh: apply grid fee reduction if applicable
        if c.production.has_production and c.production.avoids_grid_energy_fee:
            df["grid_energy_cost"] = df["grid_consumed_kwh"] * df["grid_energy_price"]
        elif c.production.has_production and c.production.grid_fee_reduction_pct > 0:
            reduced_price = df["grid_energy_price"] * (1 - c.production.grid_fee_reduction_pct)
            df["grid_energy_cost"] = (
                df["grid_consumed_kwh"] * df["grid_energy_price"]
                + df["self_consumed_kwh"] * reduced_price
            )
        else:
            # Grid fees apply to ALL consumption (even self-consumed), unless exempted
            df["grid_energy_cost"] = df["consumption_kwh"] * df["grid_energy_price"]

        # === Taxes & levies ===
        levies = c.taxes.total_levies_eur_per_kwh
        if c.production.has_production:
            # Determine which volumes are exempt
            exempt_kwh = 0.0
            taxed_kwh = df["consumption_kwh"].copy()

            if c.production.avoids_excise:
                # Self-consumed kWh exempt from excise
                df["taxes_levies"] = df["grid_consumed_kwh"] * levies
            else:
                df["taxes_levies"] = df["consumption_kwh"] * levies
        else:
            df["taxes_levies"] = df["consumption_kwh"] * levies

        # === Injection revenue ===
        df["injection_revenue"] = df["injected_kwh"] * c.production.injection_tariff_eur_per_kwh

        # === Overshoot detection ===
        # Net demand = consumption - production (what you actually draw from grid)
        net_demand = df["consumption_kw"] - df["production_kw"]
        net_demand = net_demand.clip(lower=0)
        df["net_demand_kw"] = net_demand

        if c.grid.overshoot_reference == "subscribed":
            reference_kw = c.grid.subscribed_power_kw
        else:
            reference_kw = c.grid.connection_power_limit_kw

        df["overshoot_kw"] = (net_demand - reference_kw).clip(lower=0)

        # Overshoot penalties are billed on MONTHLY PEAK overshoot, not per timestep
        # (this matches Belgian/French grid tariff rules)
        df["month"] = df.index.to_period("M")
        monthly_peak_overshoot = df.groupby("month")["overshoot_kw"].max()
        df["overshoot_penalty"] = 0.0
        for month_period, peak_os in monthly_peak_overshoot.items():
            if peak_os > 0:
                # Assign penalty once per month (to the timestep with max overshoot)
                month_mask = df["month"] == month_period
                month_data = df.loc[month_mask, "overshoot_kw"]
                max_idx = month_data.idxmax()
                df.loc[max_idx, "overshoot_penalty"] = peak_os * c.grid.overshoot_penalty_eur_per_kw

        # Monthly peak demand stats
        monthly_peaks = net_demand.groupby(df["month"]).max()

        # === Aggregate results ===
        summary = self._aggregate_summary(df, monthly_peaks, c, hours_per_step)
        monthly = self._aggregate_monthly(df, c)

        # Clean up temp columns
        df = df.drop(columns=["month"], errors="ignore")

        return df, summary, monthly

    def _prepare_data(
        self,
        consumption: pd.DataFrame,
        production: Optional[pd.DataFrame],
        consumption_col: str,
        production_col: str,
    ) -> pd.DataFrame:
        """Merge consumption and production into a single DataFrame."""
        df = pd.DataFrame(index=consumption.index)
        df["consumption_kw"] = consumption[consumption_col].values

        if production is not None and not production.empty:
            # Align production to consumption index
            prod_aligned = production[production_col].reindex(df.index, fill_value=0.0)
            df["production_kw"] = prod_aligned.values
        else:
            df["production_kw"] = 0.0

        df = df.fillna(0.0)
        return df

    def _detect_granularity(self, df: pd.DataFrame) -> float:
        """Detect timestep granularity in minutes."""
        if len(df) < 2:
            return 60.0
        diffs = pd.Series(df.index).diff().dropna()
        median_diff = diffs.median()
        minutes = median_diff.total_seconds() / 60
        # Round to nearest standard granularity
        for std in [1, 5, 10, 15, 30, 60]:
            if abs(minutes - std) < 2:
                return float(std)
        return minutes

    def _compute_self_consumption(self, df: pd.DataFrame, hours_per_step: float) -> pd.DataFrame:
        """Compute self-consumption, grid consumption, and injection."""
        c = self.contract

        if c.production.metering_mode == MeteringMode.GROSS:
            # Each timestep: self-consumed = min(consumption, production)
            df["self_consumed_kwh"] = np.minimum(
                df["consumption_kwh"], df["production_kwh"]
            )
            df["grid_consumed_kwh"] = df["consumption_kwh"] - df["self_consumed_kwh"]
            df["injected_kwh"] = df["production_kwh"] - df["self_consumed_kwh"]

        elif c.production.metering_mode == MeteringMode.NET:
            # Net metering: annual settlement, but we compute per timestep
            # Net = consumption - production (can be negative = injection)
            net = df["consumption_kwh"] - df["production_kwh"]
            df["grid_consumed_kwh"] = net.clip(lower=0)
            df["injected_kwh"] = (-net).clip(lower=0)
            df["self_consumed_kwh"] = np.minimum(
                df["consumption_kwh"], df["production_kwh"]
            )

        elif c.production.metering_mode == MeteringMode.SEMI_NET:
            # Same as net per timestep, but with periodic (e.g., monthly) settlement
            # For now, treat same as NET — can be refined later
            net = df["consumption_kwh"] - df["production_kwh"]
            df["grid_consumed_kwh"] = net.clip(lower=0)
            df["injected_kwh"] = (-net).clip(lower=0)
            df["self_consumed_kwh"] = np.minimum(
                df["consumption_kwh"], df["production_kwh"]
            )

        return df

    def _aggregate_summary(
        self,
        df: pd.DataFrame,
        monthly_peaks: pd.Series,
        contract: EnergyContract,
        hours_per_step: float,
    ) -> CostBreakdown:
        """Aggregate timestep results into an overall summary."""
        c = contract
        summary = CostBreakdown()

        # Volumes
        summary.total_consumption_kwh = df["consumption_kwh"].sum()
        summary.total_production_kwh = df["production_kwh"].sum()
        summary.self_consumed_kwh = df["self_consumed_kwh"].sum()
        summary.grid_consumed_kwh = df["grid_consumed_kwh"].sum()
        summary.injected_kwh = df["injected_kwh"].sum()

        if summary.total_production_kwh > 0:
            summary.self_consumption_rate = summary.self_consumed_kwh / summary.total_production_kwh
        if summary.total_consumption_kwh > 0:
            summary.autarky_rate = summary.self_consumed_kwh / summary.total_consumption_kwh

        # Costs
        summary.energy_cost = df["energy_cost"].sum()
        summary.grid_energy_cost = df["grid_energy_cost"].sum()
        summary.taxes_and_levies = df["taxes_levies"].sum()
        summary.injection_revenue = df["injection_revenue"].sum()
        summary.overshoot_penalties = df["overshoot_penalty"].sum()

        # Capacity cost (annual, pro-rated to data period)
        data_days = (df.index[-1] - df.index[0]).days + 1
        year_fraction = data_days / 365.25
        summary.grid_capacity_cost = (
            c.grid.capacity_charge_eur_per_kw_year
            * c.grid.subscribed_power_kw
            * year_fraction
        )

        # Green certificates
        if c.production.green_certificate_eligible:
            summary.green_certificate_revenue = (
                summary.total_production_kwh / 1000  # Convert to MWh
                * c.production.green_certificate_value_eur_per_mwh
            )

        # Prosumer tariff
        if c.production.prosumer_tariff_eur_per_year > 0:
            summary.prosumer_tariff = c.production.prosumer_tariff_eur_per_year * year_fraction

        # Peak demand stats
        summary.peak_demand_kw = (df["consumption_kw"] - df["production_kw"]).clip(lower=0).max()
        summary.overshoots_count = (df["overshoot_kw"] > 0).sum()
        if summary.overshoots_count > 0:
            summary.max_overshoot_kw = df["overshoot_kw"].max()

        summary.compute_totals(contract)
        return summary

    def _aggregate_monthly(
        self, df: pd.DataFrame, contract: EnergyContract
    ) -> dict[str, CostBreakdown]:
        """Aggregate results per month."""
        monthly_results = {}
        for period, group in df.groupby(df.index.to_period("M")):
            month_key = str(period)
            mb = CostBreakdown()
            mb.total_consumption_kwh = group["consumption_kwh"].sum()
            mb.total_production_kwh = group["production_kwh"].sum()
            mb.self_consumed_kwh = group["self_consumed_kwh"].sum()
            mb.grid_consumed_kwh = group["grid_consumed_kwh"].sum()
            mb.injected_kwh = group["injected_kwh"].sum()
            mb.energy_cost = group["energy_cost"].sum()
            mb.grid_energy_cost = group["grid_energy_cost"].sum()
            mb.taxes_and_levies = group["taxes_levies"].sum()
            mb.injection_revenue = group["injection_revenue"].sum()
            mb.overshoot_penalties = group["overshoot_penalty"].sum()

            # Monthly capacity cost
            mb.grid_capacity_cost = (
                contract.grid.capacity_charge_eur_per_kw_year
                * contract.grid.subscribed_power_kw
                / 12
            )

            mb.peak_demand_kw = (
                (group["consumption_kw"] - group["production_kw"]).clip(lower=0).max()
            )
            mb.overshoots_count = (group["overshoot_kw"] > 0).sum()
            if mb.overshoots_count > 0:
                mb.max_overshoot_kw = group["overshoot_kw"].max()

            if mb.total_production_kwh > 0:
                mb.self_consumption_rate = mb.self_consumed_kwh / mb.total_production_kwh
            if mb.total_consumption_kwh > 0:
                mb.autarky_rate = mb.self_consumed_kwh / mb.total_consumption_kwh

            mb.compute_totals(contract)
            monthly_results[month_key] = mb

        return monthly_results


# === Comparison Tool ===

def compare_scenarios(
    consumption: pd.DataFrame,
    production: Optional[pd.DataFrame],
    contracts: dict[str, EnergyContract],
    consumption_col: str = "consumption_kw",
    production_col: str = "production_kw",
) -> pd.DataFrame:
    """
    Compare energy costs across multiple contract scenarios.

    Args:
        contracts: Dict of {scenario_name: EnergyContract}

    Returns:
        DataFrame comparing key metrics across scenarios.
    """
    rows = []
    for name, contract in contracts.items():
        sim = CostSimulator(contract)
        _, summary, _ = sim.simulate(consumption, production, consumption_col, production_col)
        rows.append({
            "Scenario": name,
            "Total Cost (excl. VAT)": round(summary.total_cost_excl_vat, 2),
            "Energy Cost": round(summary.energy_cost, 2),
            "Grid Capacity": round(summary.grid_capacity_cost, 2),
            "Grid Energy": round(summary.grid_energy_cost, 2),
            "Taxes & Levies": round(summary.taxes_and_levies, 2),
            "Overshoot Penalties": round(summary.overshoot_penalties, 2),
            "Injection Revenue": round(summary.injection_revenue, 2),
            "GC Revenue": round(summary.green_certificate_revenue, 2),
            "Self-Consumption Rate": f"{summary.self_consumption_rate:.1%}",
            "Autarky Rate": f"{summary.autarky_rate:.1%}",
            "Peak Demand (kW)": round(summary.peak_demand_kw, 1),
            "Overshoots": summary.overshoots_count,
            "Avg €/kWh": round(
                summary.total_cost_excl_vat / max(summary.total_consumption_kwh, 1), 4
            ),
        })

    return pd.DataFrame(rows)
