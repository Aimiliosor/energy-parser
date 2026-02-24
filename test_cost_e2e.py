"""
End-to-end cost simulation test using real CSV data and all DB contracts.
"""
import os
import sys
import io
import tempfile
import traceback
import pandas as pd
import numpy as np

# Fix Windows console encoding for unicode characters like ≤
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace")

# Ensure project root on path
sys.path.insert(0, os.path.dirname(__file__))

from energy_parser.contract_model import (
    load_contracts_db, get_default_db_path, contract_from_dict, contract_to_dict,
)
from energy_parser.cost_simulator import CostSimulator, CostBreakdown, compare_scenarios
from energy_parser.report_generator import (
    generate_cost_breakdown_pie, generate_monthly_cost_bar_chart,
    generate_scenario_comparison_chart, generate_pdf_report,
)
from energy_parser.exporter import save_cost_simulation_xlsx

CSV_PATH = os.path.join(os.path.dirname(__file__),
                         "Raw_input_files", "test_cons_prod_2024.csv")

passed = 0
errors = []


def test(name):
    """Decorator-style header."""
    print(f"\n[{name}]")


# === 1. Load real CSV data ===
test("1. Load CSV data")
try:
    df_raw = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    df_raw["DateTime"] = pd.to_datetime(df_raw["DateTime"], dayfirst=True)
    df_raw = df_raw.set_index("DateTime").sort_index()
    print(f"  Loaded: {len(df_raw)} rows, "
          f"{df_raw.index[0]} to {df_raw.index[-1]}")
    print(f"  Columns: {list(df_raw.columns)}")
    print(f"  Consumption: {df_raw['Consumption_kW'].sum() * 0.25 / 1000:.1f} MWh")
    print(f"  Production:  {df_raw['Production_kW'].sum() * 0.25 / 1000:.1f} MWh")
    print(f"  Peak demand: {df_raw['Consumption_kW'].max():.1f} kW")

    # Prepare DataFrames for simulator
    cons_df = pd.DataFrame({"consumption_kw": df_raw["Consumption_kW"]},
                            index=df_raw.index)
    prod_df = pd.DataFrame({"production_kw": df_raw["Production_kW"]},
                            index=df_raw.index)
    passed += 1
    print("  PASS")
except Exception as e:
    errors.append(("Load CSV", e))
    print(f"  FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# === 2. Load all contracts from DB ===
test("2. Load contracts database")
try:
    hierarchy, metadata = load_contracts_db(get_default_db_path())
    all_contracts = {}
    for country, regions in hierarchy.items():
        for region, contracts in regions.items():
            for name, contract in contracts.items():
                cid = metadata[name]["id"]
                all_contracts[cid] = contract
                print(f"  [{cid}] {name}")
    assert len(all_contracts) == 3, f"Expected 3, got {len(all_contracts)}"
    passed += 1
    print(f"  PASS: {len(all_contracts)} contracts loaded")
except Exception as e:
    errors.append(("Load DB", e))
    print(f"  FAIL: {e}")
    traceback.print_exc()

# === 3. Simulate each contract ===
test("3. Simulate all contracts")
results = {}
try:
    for cid, contract in all_contracts.items():
        sim = CostSimulator(contract)
        detail_df, summary, monthly = sim.simulate(cons_df, prod_df)
        results[cid] = (detail_df, summary, monthly)
        avg_kwh = summary.total_cost_excl_vat / max(summary.total_consumption_kwh, 1)
        prod_info = ""
        if summary.total_production_kwh > 0:
            prod_info = (f", self-cons={summary.self_consumption_rate:.0%}, "
                         f"autarky={summary.autarky_rate:.0%}")
        print(f"  {cid}: EUR {summary.total_cost_excl_vat:,.2f} "
              f"(avg {avg_kwh:.4f}/kWh, peak {summary.peak_demand_kw:.0f} kW"
              f"{prod_info})")
    assert len(results) == 3
    passed += 1
    print(f"  PASS: All {len(results)} simulations completed")
except Exception as e:
    errors.append(("Simulate all", e))
    print(f"  FAIL: {e}")
    traceback.print_exc()

# === 4. Scenario comparison ===
test("4. Scenario comparison")
try:
    sorted_results = sorted(results.items(),
                             key=lambda x: x[1][1].total_cost_excl_vat)
    top_contracts = {cid: all_contracts[cid] for cid, _ in sorted_results}
    comp_df = compare_scenarios(cons_df, prod_df, top_contracts)
    print(comp_df[["Scenario", "Total Cost (excl. VAT)", "Avg \u20ac/kWh",
                    "Self-Consumption Rate"]].to_string(index=False))
    assert len(comp_df) == 3
    passed += 1
    print("  PASS")
except Exception as e:
    errors.append(("Comparison", e))
    print(f"  FAIL: {e}")
    traceback.print_exc()

# === 5. Chart generation ===
test("5. Chart generation")
try:
    # Use cheapest contract for charts
    cheapest_id = sorted_results[0][0]
    cheapest_summary = results[cheapest_id][1]
    cheapest_monthly = results[cheapest_id][2]
    summary_dict = {
        "energy_cost": cheapest_summary.energy_cost,
        "grid_capacity_cost": cheapest_summary.grid_capacity_cost,
        "grid_energy_cost": cheapest_summary.grid_energy_cost,
        "taxes_and_levies": cheapest_summary.taxes_and_levies,
        "overshoot_penalties": cheapest_summary.overshoot_penalties,
    }
    monthly_dict = {}
    for mk, mb in cheapest_monthly.items():
        monthly_dict[mk] = {
            "total_cost_excl_vat": mb.total_cost_excl_vat,
            "energy_cost": mb.energy_cost,
            "grid_capacity_cost": mb.grid_capacity_cost,
            "grid_energy_cost": mb.grid_energy_cost,
            "taxes_and_levies": mb.taxes_and_levies,
        }

    pie_img = generate_cost_breakdown_pie(summary_dict)
    bar_img = generate_monthly_cost_bar_chart(monthly_dict)
    comp_records = comp_df.to_dict("records")
    comp_img = generate_scenario_comparison_chart(comp_records)
    print(f"  Pie: {len(pie_img)}B, Bar: {len(bar_img)}B, Comp: {len(comp_img)}B")
    assert all(len(img) > 1000 for img in [pie_img, bar_img, comp_img])
    passed += 1
    print("  PASS")
except Exception as e:
    errors.append(("Charts", e))
    print(f"  FAIL: {e}")
    traceback.print_exc()

# === 6. Excel export ===
test("6. Excel export")
try:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        xlsx_path = f.name
    summary_export = {
        "total_consumption_kwh": cheapest_summary.total_consumption_kwh,
        "total_production_kwh": cheapest_summary.total_production_kwh,
        "self_consumed_kwh": cheapest_summary.self_consumed_kwh,
        "grid_consumed_kwh": cheapest_summary.grid_consumed_kwh,
        "injected_kwh": cheapest_summary.injected_kwh,
        "self_consumption_rate": cheapest_summary.self_consumption_rate,
        "autarky_rate": cheapest_summary.autarky_rate,
        "energy_cost": cheapest_summary.energy_cost,
        "grid_capacity_cost": cheapest_summary.grid_capacity_cost,
        "grid_energy_cost": cheapest_summary.grid_energy_cost,
        "taxes_and_levies": cheapest_summary.taxes_and_levies,
        "overshoot_penalties": cheapest_summary.overshoot_penalties,
        "injection_revenue": cheapest_summary.injection_revenue,
        "total_cost_excl_vat": cheapest_summary.total_cost_excl_vat,
        "peak_demand_kw": cheapest_summary.peak_demand_kw,
        "overshoots_count": cheapest_summary.overshoots_count,
    }
    monthly_export = {}
    for mk, mb in cheapest_monthly.items():
        monthly_export[mk] = {
            "total_consumption_kwh": mb.total_consumption_kwh,
            "total_production_kwh": mb.total_production_kwh,
            "self_consumption_rate": mb.self_consumption_rate,
            "energy_cost": mb.energy_cost,
            "grid_capacity_cost": mb.grid_capacity_cost,
            "grid_energy_cost": mb.grid_energy_cost,
            "taxes_and_levies": mb.taxes_and_levies,
            "overshoot_penalties": mb.overshoot_penalties,
            "total_cost_excl_vat": mb.total_cost_excl_vat,
            "peak_demand_kw": mb.peak_demand_kw,
        }
    save_cost_simulation_xlsx(xlsx_path, summary_export, monthly_export,
                               comp_records)
    sz = os.path.getsize(xlsx_path)
    print(f"  Saved: {xlsx_path} ({sz / 1024:.0f} KB)")
    os.unlink(xlsx_path)
    assert sz > 2000
    passed += 1
    print("  PASS")
except Exception as e:
    errors.append(("Excel", e))
    print(f"  FAIL: {e}")
    traceback.print_exc()

# === 7. Monthly breakdown sanity ===
test("7. Monthly breakdown sanity")
try:
    for cid in list(all_contracts.keys()):
        summary = results[cid][1]
        monthly = results[cid][2]
        monthly_cons = sum(m.total_consumption_kwh for m in monthly.values())
        diff_pct = abs(monthly_cons - summary.total_consumption_kwh) / max(
            summary.total_consumption_kwh, 1) * 100
        assert diff_pct < 0.1, f"Consumption mismatch {cid}: {diff_pct:.2f}%"
        print(f"  {cid}: {len(monthly)} months, cons diff={diff_pct:.4f}%")
    passed += 1
    print("  PASS")
except Exception as e:
    errors.append(("Monthly sanity", e))
    print(f"  FAIL: {e}")
    traceback.print_exc()

# === Summary ===
total = passed + len(errors)
print(f"\n{'='*70}")
if errors:
    print(f"RESULT: {passed}/{total} PASSED, {len(errors)} FAILED")
    for name, err in errors:
        print(f"  FAILED: {name}: {err}")
else:
    print(f"RESULT: ALL {passed} TESTS PASSED")
print(f"{'='*70}")
