"""Battery dimensioning analysis for energy data.

Calculates optimal battery storage size based on consumption and production
data, generates visualizations, and provides savings estimates.

Pure-function module with a BatterySizer class. Operates on the transformed
DataFrame (columns: "Date & Time", "Consumption (kW)", "Production (kW)").
"""

import io
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Brand colors
BRAND_PRIMARY = "#2C495E"
BRAND_SECONDARY = "#EC465D"
BRAND_LIGHT = "#B4BCD6"
BRAND_WHITE = "#FFFFFF"
BRAND_BG = "#F5F7FA"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


class BatterySizer:
    """Calculate battery sizing recommendations from energy data."""

    def __init__(self, df: pd.DataFrame, hours_per_interval: float,
                 tariffs: dict):
        """Initialize BatterySizer.

        Args:
            df: Transformed DataFrame with "Date & Time",
                "Consumption (kW)", "Production (kW)" columns.
            hours_per_interval: Hours between successive readings.
            tariffs: dict with keys 'offtake' (€/MWh), 'injection' (€/MWh),
                     'peak' (€/kW).
        """
        self.df = df.copy()
        self.hours_per_interval = hours_per_interval
        self.tariffs = tariffs

        # Ensure datetime
        self.df["Date & Time"] = pd.to_datetime(self.df["Date & Time"])
        self.df = self.df.sort_values("Date & Time").reset_index(drop=True)

        # Results (populated by run_analysis)
        self.daily_metrics = None
        self.hourly_profiles = None
        self.monthly_storage = None
        self.recommendations = None
        self.savings = None

    def _has_production(self) -> bool:
        """Check if production data exists and has non-zero values."""
        return ("Production (kW)" in self.df.columns
                and self.df["Production (kW)"].sum() > 0)

    def _data_coverage_months(self) -> int:
        """Return approximate number of months of data coverage."""
        dt = self.df["Date & Time"]
        span_days = (dt.max() - dt.min()).days
        return max(1, span_days // 30)

    def validate(self) -> dict:
        """Validate data suitability for battery analysis.

        Returns dict with:
            valid: bool
            message: str (error or warning)
            warning: str or None
        """
        if not self._has_production():
            return {
                "valid": False,
                "message": ("Battery dimensioning requires both consumption "
                            "and production data. This feature is only "
                            "available for sites with solar/renewable "
                            "generation."),
                "warning": None,
            }

        dt = self.df["Date & Time"]
        span_days = (dt.max() - dt.min()).days

        if span_days < 30:
            return {
                "valid": False,
                "message": ("Insufficient data for battery analysis. "
                            "At least 30 days of data is required. "
                            f"Current dataset spans {span_days} days."),
                "warning": None,
            }

        warning = None
        months = self._data_coverage_months()
        if months < 12:
            warning = (
                f"Analysis based on {months} months of data. Results may "
                "not capture full seasonal variation. For optimal battery "
                "sizing, 12 months of data is recommended."
            )

        return {"valid": True, "message": "OK", "warning": warning}

    def run_analysis(self) -> dict:
        """Run complete battery dimensioning analysis.

        Returns dict with all results, or raises ValueError if invalid.
        """
        validation = self.validate()
        if not validation["valid"]:
            raise ValueError(validation["message"])

        self.daily_metrics = self.calculate_daily_metrics()
        self.hourly_profiles = self.calculate_hourly_profiles()
        self.monthly_storage = self.calculate_monthly_storage()
        self.recommendations = self.recommend_battery_size()
        self.savings = self.calculate_savings()

        return {
            "daily_metrics": self.daily_metrics,
            "hourly_profiles": self.hourly_profiles,
            "monthly_storage": self.monthly_storage,
            "recommendations": self.recommendations,
            "savings": self.savings,
            "validation_warning": self.validate().get("warning"),
        }

    def calculate_daily_metrics(self) -> pd.DataFrame:
        """Calculate daily Grid_Out, Grid_In, Storable energy.

        For each interval:
        - net = Production - Consumption
        - If net > 0: excess production → grid_out (and storable)
        - If net < 0: deficit → grid_in

        Aggregate to daily sums (kWh by multiplying kW * hours_per_interval).
        """
        df = self.df.copy()
        cons = df["Consumption (kW)"].fillna(0)
        prod = df["Production (kW)"].fillna(0)

        # Per-interval values in kW
        net = prod - cons
        df["_excess"] = net.clip(lower=0)   # kW when prod > cons
        df["_deficit"] = (-net).clip(lower=0)  # kW when cons > prod

        df["_date"] = df["Date & Time"].dt.date

        # Convert to kWh by multiplying by interval duration
        h = self.hours_per_interval

        daily = df.groupby("_date").agg(
            grid_out=("_excess", lambda x: float(x.sum() * h)),
            grid_in=("_deficit", lambda x: float(x.sum() * h)),
            consumption=("Consumption (kW)", lambda x: float(x.sum() * h)),
            production=("Production (kW)", lambda x: float(x.sum() * h)),
        ).reset_index()
        daily.rename(columns={"_date": "date"}, inplace=True)

        # Storable = grid_out (excess production that could go to battery)
        daily["storable"] = daily["grid_out"]
        daily["month"] = pd.to_datetime(daily["date"]).dt.month

        # Daily savings potential
        tariff_diff = (self.tariffs["offtake"]
                       - self.tariffs["injection"]) / 1000  # €/kWh
        daily["savings_potential"] = daily["storable"] * tariff_diff

        return daily

    def calculate_hourly_profiles(self) -> dict:
        """Calculate average hourly profiles by month.

        Returns dict with keys 'overall' (DataFrame) and 'by_month'
        (dict of month -> DataFrame). Each DataFrame has columns:
        hour, avg_prod, avg_cons, rel_prod, rel_cons,
        avg_grid_out, avg_grid_in.
        """
        df = self.df.copy()
        df["_hour"] = df["Date & Time"].dt.hour
        df["_month"] = df["Date & Time"].dt.month
        cons = df["Consumption (kW)"].fillna(0)
        prod = df["Production (kW)"].fillna(0)
        net = prod - cons
        df["_excess"] = net.clip(lower=0)
        df["_deficit"] = (-net).clip(lower=0)

        by_month = {}
        for month in range(1, 13):
            month_data = df[df["_month"] == month]
            if month_data.empty:
                continue

            hourly = month_data.groupby("_hour").agg(
                avg_prod=("Production (kW)", "mean"),
                avg_cons=("Consumption (kW)", "mean"),
                avg_grid_out=("_excess", "mean"),
                avg_grid_in=("_deficit", "mean"),
            ).reindex(range(24), fill_value=0)

            # Normalized (relative) profiles
            total_prod = hourly["avg_prod"].sum()
            total_cons = hourly["avg_cons"].sum()
            hourly["rel_prod"] = (hourly["avg_prod"] / total_prod
                                  if total_prod > 0 else 0)
            hourly["rel_cons"] = (hourly["avg_cons"] / total_cons
                                  if total_cons > 0 else 0)
            hourly.index.name = "hour"
            by_month[month] = hourly.reset_index()

        # Overall average profile
        overall = df.groupby("_hour").agg(
            avg_prod=("Production (kW)", "mean"),
            avg_cons=("Consumption (kW)", "mean"),
            avg_grid_out=("_excess", "mean"),
            avg_grid_in=("_deficit", "mean"),
        ).reindex(range(24), fill_value=0)

        total_prod = overall["avg_prod"].sum()
        total_cons = overall["avg_cons"].sum()
        overall["rel_prod"] = (overall["avg_prod"] / total_prod
                               if total_prod > 0 else 0)
        overall["rel_cons"] = (overall["avg_cons"] / total_cons
                               if total_cons > 0 else 0)
        overall.index.name = "hour"

        return {
            "overall": overall.reset_index(),
            "by_month": by_month,
        }

    def calculate_monthly_storage(self) -> pd.DataFrame:
        """Calculate monthly storage requirements.

        Returns DataFrame with columns: month, month_name,
        consumption_total, production_total, avg_storage_need,
        max_storage_need.
        """
        if self.daily_metrics is None:
            self.daily_metrics = self.calculate_daily_metrics()

        monthly = self.daily_metrics.groupby("month").agg(
            consumption_total=("consumption", "sum"),
            production_total=("production", "sum"),
            avg_storage_need=("storable", "mean"),
            max_storage_need=("storable", "max"),
        ).reindex(range(1, 13), fill_value=0).reset_index()

        monthly["month_name"] = [MONTH_NAMES[m - 1] for m in monthly["month"]]
        return monthly

    def recommend_battery_size(self) -> dict:
        """Calculate battery sizing recommendations.

        Returns dict with:
            max_capacity: Largest single-day storage need (kWh)
            avg_capacity: Average daily storage need (kWh)
            recommended_capacity: 90th percentile of daily storage (kWh)
            recommended_power: Peak charge/discharge rate (kW)
        """
        if self.daily_metrics is None:
            self.daily_metrics = self.calculate_daily_metrics()

        storable = self.daily_metrics["storable"]
        storable_positive = storable[storable > 0]

        if storable_positive.empty:
            return {
                "max_capacity": 0.0,
                "avg_capacity": 0.0,
                "recommended_capacity": 0.0,
                "recommended_power": 0.0,
                "zero_storable": True,
            }

        max_capacity = float(storable.max())
        avg_capacity = float(storable_positive.mean())
        recommended_capacity = float(np.percentile(storable_positive, 90))

        # Power rating: max hourly excess production or deficit (kW)
        cons = self.df["Consumption (kW)"].fillna(0)
        prod = self.df["Production (kW)"].fillna(0)
        net = prod - cons
        max_charge = float(net.clip(lower=0).max())  # max excess prod (kW)
        max_discharge = float((-net).clip(lower=0).max())  # max deficit (kW)
        recommended_power = max(max_charge, max_discharge)

        return {
            "max_capacity": round(max_capacity, 1),
            "avg_capacity": round(avg_capacity, 1),
            "recommended_capacity": round(recommended_capacity, 1),
            "recommended_power": round(recommended_power, 1),
            "zero_storable": False,
        }

    def calculate_savings(self) -> dict:
        """Calculate annual savings with recommended battery capacity.

        Returns dict with:
            annual_savings: Total yearly savings (€)
            energy_arbitrage: Savings from energy arbitrage (€)
            peak_reduction: Savings from peak demand reduction (€)
            peak_reduction_kw: Peak demand reduction (kW)
            monthly_savings: DataFrame of monthly savings
            self_consumption_pct: Self-consumption percentage
            self_consumption_increase: Increase in self-consumption (%)
        """
        if self.daily_metrics is None:
            self.daily_metrics = self.calculate_daily_metrics()
        if self.recommendations is None:
            self.recommendations = self.recommend_battery_size()

        rec_cap = self.recommendations["recommended_capacity"]
        if rec_cap == 0:
            return self._empty_savings()

        daily = self.daily_metrics.copy()
        tariff_diff = (self.tariffs["offtake"]
                       - self.tariffs["injection"]) / 1000  # €/kWh

        # Usable stored energy per day (limited by battery capacity)
        daily["usable_stored"] = daily["storable"].clip(upper=rec_cap)
        daily["daily_savings"] = daily["usable_stored"] * tariff_diff

        # Monthly savings
        monthly_savings = daily.groupby("month").agg(
            total_savings=("daily_savings", "sum"),
            avg_daily_savings=("daily_savings", "mean"),
            total_stored=("usable_stored", "sum"),
        ).reindex(range(1, 13), fill_value=0).reset_index()
        monthly_savings["month_name"] = [MONTH_NAMES[m - 1]
                                         for m in monthly_savings["month"]]

        energy_arbitrage = float(daily["daily_savings"].sum())

        # Peak demand reduction
        # With battery, peak grid consumption can be reduced
        # by the battery discharge capacity
        cons = self.df["Consumption (kW)"].fillna(0)
        prod = self.df["Production (kW)"].fillna(0)
        peak_without_battery = float((cons - prod).clip(lower=0).max())
        rec_power = self.recommendations["recommended_power"]
        # Battery can reduce peak by its power rating (limited)
        peak_reduction_kw = min(rec_power, peak_without_battery * 0.15)
        peak_reduction = peak_reduction_kw * self.tariffs["peak"] * 12  # monthly charge

        annual_savings = energy_arbitrage + peak_reduction

        # Self-consumption analysis
        total_production = float(daily["production"].sum())
        total_consumption = float(daily["consumption"].sum())

        # Self-consumed = production used directly (not injected)
        total_grid_out = float(daily["grid_out"].sum())
        total_stored = float(daily["usable_stored"].sum())

        self_consumed_direct = total_production - total_grid_out
        self_consumed_with_battery = self_consumed_direct + total_stored

        sc_pct_before = ((self_consumed_direct / total_production * 100)
                         if total_production > 0 else 0)
        sc_pct_after = ((self_consumed_with_battery / total_production * 100)
                        if total_production > 0 else 0)
        sc_increase = sc_pct_after - sc_pct_before

        # Energy flow breakdown (annual kWh)
        excess_after_storage = max(0, total_grid_out - total_stored)
        grid_in_total = float(daily["grid_in"].sum())

        energy_flows = {
            "self_consumed": round(self_consumed_direct, 1),
            "stored_by_battery": round(total_stored, 1),
            "excess_to_grid": round(excess_after_storage, 1),
            "from_grid": round(grid_in_total, 1),
            "total_consumption": round(total_consumption, 1),
            "total_production": round(total_production, 1),
        }

        return {
            "annual_savings": round(annual_savings, 2),
            "energy_arbitrage": round(energy_arbitrage, 2),
            "peak_reduction": round(peak_reduction, 2),
            "peak_reduction_kw": round(peak_reduction_kw, 1),
            "monthly_savings": monthly_savings,
            "self_consumption_pct": round(sc_pct_after, 1),
            "self_consumption_increase": round(sc_increase, 1),
            "energy_flows": energy_flows,
        }

    def _empty_savings(self) -> dict:
        """Return empty savings result."""
        return {
            "annual_savings": 0.0,
            "energy_arbitrage": 0.0,
            "peak_reduction": 0.0,
            "peak_reduction_kw": 0.0,
            "monthly_savings": pd.DataFrame(
                columns=["month", "month_name", "total_savings",
                         "avg_daily_savings", "total_stored"]),
            "self_consumption_pct": 0.0,
            "self_consumption_increase": 0.0,
            "energy_flows": {
                "self_consumed": 0.0, "stored_by_battery": 0.0,
                "excess_to_grid": 0.0, "from_grid": 0.0,
                "total_consumption": 0.0, "total_production": 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Visualization methods
    # ------------------------------------------------------------------

    def generate_average_day_profile(self) -> bytes:
        """Line chart: typical daily production vs consumption profile."""
        if self.hourly_profiles is None:
            self.hourly_profiles = self.calculate_hourly_profiles()

        profile = self.hourly_profiles["overall"]
        hours = profile["hour"].values

        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

        ax.plot(hours, profile["rel_cons"], color=BRAND_PRIMARY,
                linewidth=2, label="Consumption", marker="o", markersize=3)
        ax.plot(hours, profile["rel_prod"], color=BRAND_LIGHT,
                linewidth=2, label="Production", marker="s", markersize=3)

        ax.fill_between(hours, profile["rel_cons"], profile["rel_prod"],
                        where=profile["rel_prod"] > profile["rel_cons"],
                        alpha=0.15, color=BRAND_LIGHT, label="Excess Production")
        ax.fill_between(hours, profile["rel_cons"], profile["rel_prod"],
                        where=profile["rel_prod"] <= profile["rel_cons"],
                        alpha=0.15, color=BRAND_PRIMARY, label="Grid Consumption")

        ax.set_xlabel("Hour of Day", fontsize=9)
        ax.set_ylabel("Relative Power (normalized)", fontsize=9)
        ax.set_title("Typical Daily Production vs Consumption Profile",
                     fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()

        return _fig_to_bytes(fig)

    def generate_monthly_storage_chart(self) -> bytes:
        """Bar chart: monthly storage requirements with recommendation line."""
        if self.monthly_storage is None:
            self.monthly_storage = self.calculate_monthly_storage()
        if self.recommendations is None:
            self.recommendations = self.recommend_battery_size()

        ms = self.monthly_storage
        x = np.arange(len(ms))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

        ax.bar(x - width / 2, ms["avg_storage_need"], width,
               label="Average Storage Need", color=BRAND_LIGHT,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, ms["max_storage_need"], width,
               label="Maximum Storage Need", color=BRAND_PRIMARY,
               edgecolor="white", linewidth=0.5)

        rec = self.recommendations["recommended_capacity"]
        if rec > 0:
            ax.axhline(rec, color=BRAND_SECONDARY, linewidth=2,
                       linestyle="--",
                       label=f"Recommended: {rec:,.0f} kWh")

        ax.set_xlabel("Month", fontsize=9)
        ax.set_ylabel("Energy (kWh)", fontsize=9)
        ax.set_title("Monthly Battery Storage Requirements",
                     fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
        ax.set_xticks(x)
        ax.set_xticklabels(ms["month_name"], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        return _fig_to_bytes(fig)

    def generate_annual_storage_pattern(self) -> bytes:
        """Line chart: daily storable energy across the year."""
        if self.daily_metrics is None:
            self.daily_metrics = self.calculate_daily_metrics()
        if self.recommendations is None:
            self.recommendations = self.recommend_battery_size()

        daily = self.daily_metrics
        dates = pd.to_datetime(daily["date"])

        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

        ax.fill_between(dates, daily["storable"], alpha=0.3,
                        color=BRAND_PRIMARY)
        ax.plot(dates, daily["storable"], color=BRAND_PRIMARY,
                linewidth=0.8, label="Daily Storable Energy")

        rec = self.recommendations["recommended_capacity"]
        if rec > 0:
            ax.axhline(rec, color=BRAND_SECONDARY, linewidth=2,
                       linestyle="--",
                       label=f"Recommended: {rec:,.0f} kWh")

        ax.set_xlabel("Date", fontsize=9)
        ax.set_ylabel("Storable Energy (kWh)", fontsize=9)
        ax.set_title("Annual Storage Pattern",
                     fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        return _fig_to_bytes(fig)

    def generate_duration_curve(self) -> bytes:
        """Sorted storable energy values (descending) — duration curve."""
        if self.daily_metrics is None:
            self.daily_metrics = self.calculate_daily_metrics()
        if self.recommendations is None:
            self.recommendations = self.recommend_battery_size()

        storable = self.daily_metrics["storable"].sort_values(
            ascending=False).values
        x = np.arange(1, len(storable) + 1)
        x_pct = x / len(storable) * 100

        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

        ax.fill_between(x_pct, storable, alpha=0.3, color=BRAND_PRIMARY)
        ax.plot(x_pct, storable, color=BRAND_PRIMARY, linewidth=1.5,
                label="Storable Energy")

        rec = self.recommendations["recommended_capacity"]
        if rec > 0:
            ax.axhline(rec, color=BRAND_SECONDARY, linewidth=2,
                       linestyle="--",
                       label=f"Recommended: {rec:,.0f} kWh")
            # Find intersection
            above = np.sum(storable >= rec)
            pct_above = above / len(storable) * 100
            ax.axvline(pct_above, color=BRAND_SECONDARY, linewidth=1,
                       linestyle=":", alpha=0.6)
            ax.annotate(f"  {pct_above:.0f}% of days",
                        xy=(pct_above, rec), fontsize=8,
                        color=BRAND_SECONDARY, fontweight="bold")

        ax.set_xlabel("Percentage of Days (%)", fontsize=9)
        ax.set_ylabel("Storable Energy (kWh)", fontsize=9)
        ax.set_title("Storage Duration Curve",
                     fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return _fig_to_bytes(fig)

    def generate_monthly_savings_chart(self) -> bytes:
        """Bar chart: potential monthly savings."""
        if self.savings is None:
            self.savings = self.calculate_savings()

        ms = self.savings["monthly_savings"]
        if ms.empty:
            return _empty_chart("No savings data available")

        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

        bars = ax.bar(ms["month_name"], ms["total_savings"],
                      color=BRAND_PRIMARY, edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, val in zip(bars, ms["total_savings"]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(ms["total_savings"]) * 0.02,
                        f"€{val:,.0f}", ha="center", va="bottom",
                        fontsize=7, color=BRAND_PRIMARY)

        ax.set_xlabel("Month", fontsize=9)
        ax.set_ylabel("Savings (€)", fontsize=9)
        ax.set_title("Monthly Savings Potential (Recommended Battery)",
                     fontsize=11, fontweight="bold", color=BRAND_PRIMARY)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        return _fig_to_bytes(fig)

    def generate_self_consumption_chart(self) -> bytes:
        """Pie chart: annual energy flow breakdown."""
        if self.savings is None:
            self.savings = self.calculate_savings()

        flows = self.savings["energy_flows"]

        labels = [
            "Self-Consumed\n(Direct)",
            "Stored by\nBattery",
            "Excess to\nGrid",
            "From Grid",
        ]
        values = [
            flows["self_consumed"],
            flows["stored_by_battery"],
            flows["excess_to_grid"],
            flows["from_grid"],
        ]
        colors_list = [BRAND_PRIMARY, BRAND_LIGHT, BRAND_SECONDARY, "#D0D5E0"]

        # Filter out zero values
        filtered = [(l, v, c) for l, v, c in zip(labels, values, colors_list)
                     if v > 0]
        if not filtered:
            return _empty_chart("No energy flow data")

        labels, values, colors_list = zip(*filtered)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors_list,
            autopct=lambda pct: f"{pct:.1f}%\n({pct / 100 * sum(values):,.0f} kWh)",
            startangle=90, textprops={"fontsize": 8},
            pctdistance=0.75)

        for t in autotexts:
            t.set_fontsize(7)

        ax.set_title("Annual Energy Flow Analysis",
                     fontsize=11, fontweight="bold", color=BRAND_PRIMARY,
                     pad=20)
        fig.tight_layout()

        return _fig_to_bytes(fig)

    def generate_all_charts(self) -> dict:
        """Generate all battery dimensioning charts.

        Returns dict of chart_name -> PNG bytes.
        """
        return {
            "average_day_profile": self.generate_average_day_profile(),
            "monthly_storage": self.generate_monthly_storage_chart(),
            "annual_pattern": self.generate_annual_storage_pattern(),
            "duration_curve": self.generate_duration_curve(),
            "monthly_savings": self.generate_monthly_savings_chart(),
            "self_consumption": self.generate_self_consumption_chart(),
        }

    def generate_report_data(self) -> dict:
        """Compile all data needed for PDF report.

        Returns dict with recommendations, savings, monthly data,
        charts, and recommendation text.
        """
        if self.recommendations is None:
            self.run_analysis()

        rec = self.recommendations
        sav = self.savings
        charts = self.generate_all_charts()

        # Monthly breakdown table data
        ms = self.monthly_storage
        ms_sav = sav["monthly_savings"]
        monthly_table = []
        for _, row in ms.iterrows():
            m = int(row["month"])
            sav_row = ms_sav[ms_sav["month"] == m]
            est_savings = (float(sav_row["total_savings"].iloc[0])
                          if not sav_row.empty else 0.0)
            monthly_table.append({
                "month": row["month_name"],
                "consumption": round(row["consumption_total"], 1),
                "production": round(row["production_total"], 1),
                "avg_storage": round(row["avg_storage_need"], 1),
                "max_storage": round(row["max_storage_need"], 1),
                "est_savings": round(est_savings, 2),
            })

        # Recommendation text
        rec_text = (
            f"Based on your annual consumption and production patterns, "
            f"we recommend a battery energy storage system with the "
            f"following specifications:\n\n"
            f"Recommended Capacity: {rec['recommended_capacity']:,.0f} kWh\n"
            f"- Captures 90% of available storage opportunities\n"
            f"- Accounts for seasonal variation in production\n\n"
            f"Recommended Power Rating: {rec['recommended_power']:,.0f} kW\n"
            f"- Handles peak charge rates during high production periods\n"
            f"- Supports discharge rates during evening consumption peaks\n\n"
            f"Expected Performance:\n"
            f"- Annual energy savings: \u20ac{sav['annual_savings']:,.0f}\n"
            f"- Self-consumption increase: "
            f"{sav['self_consumption_increase']:.1f}%\n"
            f"- Peak demand reduction: "
            f"{sav['peak_reduction_kw']:.1f} kW\n\n"
            f"This system would optimally balance upfront investment with "
            f"energy savings while avoiding oversizing for rare peak "
            f"production days."
        )

        return {
            "recommendations": rec,
            "savings": sav,
            "monthly_table": monthly_table,
            "charts": charts,
            "recommendation_text": rec_text,
            "tariffs": self.tariffs,
        }


def _fig_to_bytes(fig) -> bytes:
    """Save matplotlib figure to PNG bytes and close."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _empty_chart(message: str) -> bytes:
    """Generate a placeholder chart with a message."""
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=12, color=BRAND_LIGHT, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return _fig_to_bytes(fig)
