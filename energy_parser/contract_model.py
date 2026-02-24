"""
Spartacus - Energy Contract Data Model
=======================================
Defines the structure for energy contracts, tariffs, and on-site production parameters.
Used by the cost simulation engine to compute energy costs over a given period.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from datetime import datetime, time


# === Enums ===

class Country(Enum):
    BELGIUM = "BE"
    FRANCE = "FR"
    NETHERLANDS = "NL"
    GERMANY = "DE"


class VoltageLevel(Enum):
    LOW = "LV"        # < 1 kV
    MEDIUM = "MV"     # 1 - 36 kV
    HIGH = "HV"       # > 36 kV


class MeteringMode(Enum):
    GROSS = "gross"           # Injection and consumption measured separately
    NET = "net"               # Net metering (compensation)
    SEMI_NET = "semi_net"     # Partial net metering (e.g., annual settlement)


class PriceType(Enum):
    FIXED = "fixed"           # Fixed €/kWh for the contract duration
    INDEXED = "indexed"       # Base price + index (e.g., ENDEX, EPEX spot)
    SPOT = "spot"             # Pure spot-based (e.g., dynamic contract)
    BLOC_SPOT = "bloc_spot"   # Bloc (baseload) + spot supplement


# === Time-of-Use Period ===

@dataclass
class TimePeriod:
    """Defines a time-of-use period (e.g., peak, off-peak, shoulder)."""
    name: str                          # e.g., "Heures Pleines Hiver", "Peak", "Nacht"
    price_eur_per_kwh: float           # Energy price for this period
    grid_energy_eur_per_kwh: float = 0.0  # Grid energy component for this period (if time-differentiated)
    months: list[int] = field(default_factory=lambda: list(range(1, 13)))  # 1-12, default all year
    days_of_week: list[int] = field(default_factory=lambda: list(range(7)))  # 0=Mon, 6=Sun
    start_time: time = time(0, 0)
    end_time: time = time(23, 59)

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime falls within this period."""
        if dt.month not in self.months:
            return False
        if dt.weekday() not in self.days_of_week:
            return False
        t = dt.time()
        if self.start_time <= self.end_time:
            return self.start_time <= t <= self.end_time
        else:  # Overnight period (e.g., 22:00 - 06:00)
            return t >= self.start_time or t <= self.end_time


# === Energy Charges ===

@dataclass
class EnergyCharges:
    """Energy supply charges (what you pay the supplier for kWh)."""
    price_type: PriceType = PriceType.FIXED
    time_periods: list[TimePeriod] = field(default_factory=list)

    # For INDEXED contracts
    index_name: Optional[str] = None       # e.g., "ENDEX BE Y+1", "EPEX Spot FR"
    index_base_eur_per_kwh: float = 0.0    # Fixed base added to index

    # For BLOC_SPOT contracts
    bloc_price_eur_per_kwh: float = 0.0    # Baseload bloc price
    bloc_volume_kwh: float = 0.0           # Contracted bloc volume (flat per hour)

    # Fallback flat price (if no time_periods defined)
    flat_price_eur_per_kwh: float = 0.0

    def get_energy_price(self, dt: datetime) -> float:
        """Return the applicable energy price (€/kWh) for a given datetime."""
        for period in self.time_periods:
            if period.matches(dt):
                return period.price_eur_per_kwh
        return self.flat_price_eur_per_kwh


# === Grid / Network Fees ===

@dataclass
class GridFees:
    """
    Grid/network charges (TURPE in France, Nettarieven in Belgium).
    These are typically regulated and depend on voltage level and region.
    """
    # Capacity / demand charge
    capacity_charge_eur_per_kw_year: float = 0.0   # Annual capacity fee
    subscribed_power_kw: float = 0.0                # Contracted/subscribed power
    connection_power_limit_kw: float = 0.0          # Physical connection limit (kVA or kW)

    # Energy component of grid fees (can be time-differentiated via TimePeriods)
    flat_grid_energy_eur_per_kwh: float = 0.0       # If not time-differentiated
    time_differentiated: bool = False                # If True, use TimePeriod.grid_energy_eur_per_kwh

    # Reactive power
    reactive_power_penalty_eur_per_kvarh: float = 0.0
    cos_phi_threshold: float = 0.9                   # Penalty applies below this

    # Overshoot penalties
    overshoot_penalty_eur_per_kw: float = 0.0        # Per kW exceeding subscribed power
    overshoot_reference: str = "subscribed"           # "subscribed" or "connection"

    def get_grid_energy_price(self, dt: datetime, energy_charges: EnergyCharges) -> float:
        """Return applicable grid energy component (€/kWh) for a datetime."""
        if self.time_differentiated:
            for period in energy_charges.time_periods:
                if period.matches(dt):
                    return period.grid_energy_eur_per_kwh
        return self.flat_grid_energy_eur_per_kwh


# === Taxes & Levies ===

@dataclass
class TaxesAndLevies:
    """
    Country-specific taxes and levies on electricity.
    Examples:
      - Belgium: accijnzen, bijdrage energiefonds, federale bijdrage
      - France: TICFE/accise, CTA
      - All: VAT
    """
    excise_eur_per_kwh: float = 0.0          # Accijnzen (BE) / TICFE-Accise (FR)
    renewable_levy_eur_per_kwh: float = 0.0  # Green certificate obligation / surcharge
    other_levies_eur_per_kwh: float = 0.0    # Federal contribution, CTA, etc.
    vat_rate: float = 0.21                   # VAT rate (0.21 = 21%)
    vat_applicable: bool = True              # Some B2B customers recover VAT

    @property
    def total_levies_eur_per_kwh(self) -> float:
        return self.excise_eur_per_kwh + self.renewable_levy_eur_per_kwh + self.other_levies_eur_per_kwh


# === On-Site Production ===

@dataclass
class OnSiteProduction:
    """
    Parameters for on-site generation (PV, CHP, wind, etc.).
    Determines how self-consumed vs. injected energy is valued.
    """
    has_production: bool = False
    technology: str = ""                          # "PV", "CHP", "Wind", etc.
    installed_capacity_kwp: float = 0.0

    # Metering
    metering_mode: MeteringMode = MeteringMode.GROSS

    # Injection tariff
    injection_tariff_eur_per_kwh: float = 0.0     # Revenue for surplus injection
    injection_tariff_indexed: bool = False          # If True, follows market price

    # Self-consumption benefits (what costs are AVOIDED per self-consumed kWh)
    avoids_energy_charge: bool = True               # Avoids supplier energy price
    avoids_grid_energy_fee: bool = False             # Country-dependent
    avoids_excise: bool = False                      # Country-dependent
    avoids_renewable_levy: bool = False              # Country-dependent
    grid_fee_reduction_pct: float = 0.0             # % reduction on grid fees for self-consumed kWh

    # Green certificates (Belgium)
    green_certificate_value_eur_per_mwh: float = 0.0
    green_certificate_eligible: bool = False

    # Prosumer tariff (Belgium, being phased out)
    prosumer_tariff_eur_per_year: float = 0.0


# === Penalties ===

@dataclass
class Penalties:
    """Contract penalties and minimum offtake clauses."""
    minimum_offtake_kwh_year: float = 0.0           # Minimum annual consumption
    minimum_offtake_penalty_eur_per_kwh: float = 0.0  # Penalty per kWh below minimum
    demand_overshoot_penalty_eur_per_kw: float = 0.0   # Duplicate of grid, but some suppliers add their own


# === Main Contract ===

@dataclass
class EnergyContract:
    """
    Complete energy contract definition.
    Aggregates all components needed for cost simulation.
    """
    # Identification
    contract_name: str = ""
    supplier: str = ""
    country: Country = Country.BELGIUM
    voltage_level: VoltageLevel = VoltageLevel.LOW
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Components
    energy: EnergyCharges = field(default_factory=EnergyCharges)
    grid: GridFees = field(default_factory=GridFees)
    taxes: TaxesAndLevies = field(default_factory=TaxesAndLevies)
    production: OnSiteProduction = field(default_factory=OnSiteProduction)
    penalties: Penalties = field(default_factory=Penalties)

    # Metadata
    source: str = "manual"       # "manual", "database", "ocr", "csv_import", "clone"
    validated: bool = False       # User confirmed values (especially for OCR)
    notes: str = ""

    def summary(self) -> str:
        """Return a human-readable summary of the contract."""
        lines = [
            f"Contract: {self.contract_name}",
            f"Supplier: {self.supplier} ({self.country.value})",
            f"Voltage: {self.voltage_level.value}",
            f"Source: {self.source} | Validated: {self.validated}",
            f"Energy periods: {len(self.energy.time_periods)}",
            f"Subscribed power: {self.grid.subscribed_power_kw} kW",
            f"Connection limit: {self.grid.connection_power_limit_kw} kW",
            f"On-site production: {'Yes' if self.production.has_production else 'No'}",
        ]
        if self.production.has_production:
            lines.append(f"  → {self.production.technology} {self.production.installed_capacity_kwp} kWp")
            lines.append(f"  → Injection: {self.production.injection_tariff_eur_per_kwh} €/kWh")
            lines.append(f"  → Metering: {self.production.metering_mode.value}")
        return "\n".join(lines)


# === Contract Templates / Presets ===

def create_belgium_mv_contract() -> EnergyContract:
    """Example: Belgian MV customer with standard tariff structure."""
    return EnergyContract(
        contract_name="Belgium MV - Standard",
        supplier="Generic BE Supplier",
        country=Country.BELGIUM,
        voltage_level=VoltageLevel.MEDIUM,
        energy=EnergyCharges(
            price_type=PriceType.FIXED,
            time_periods=[
                TimePeriod(
                    name="Day",
                    price_eur_per_kwh=0.085,
                    grid_energy_eur_per_kwh=0.025,
                    start_time=time(7, 0),
                    end_time=time(21, 59),
                    days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri
                ),
                TimePeriod(
                    name="Night",
                    price_eur_per_kwh=0.065,
                    grid_energy_eur_per_kwh=0.015,
                    start_time=time(22, 0),
                    end_time=time(6, 59),
                ),
                TimePeriod(
                    name="Weekend",
                    price_eur_per_kwh=0.065,
                    grid_energy_eur_per_kwh=0.015,
                    days_of_week=[5, 6],  # Sat-Sun
                ),
            ],
        ),
        grid=GridFees(
            capacity_charge_eur_per_kw_year=45.0,
            subscribed_power_kw=500,
            connection_power_limit_kw=630,
            time_differentiated=True,
            overshoot_penalty_eur_per_kw=120.0,
        ),
        taxes=TaxesAndLevies(
            excise_eur_per_kwh=0.02586,
            renewable_levy_eur_per_kwh=0.005,
            other_levies_eur_per_kwh=0.002,
            vat_rate=0.21,
            vat_applicable=False,  # B2B: VAT recoverable
        ),
    )


def create_france_tarif_vert_contract() -> EnergyContract:
    """Example: French Tarif Vert structure (MV customer)."""
    return EnergyContract(
        contract_name="France MV - Tarif Vert (5 periods)",
        supplier="Generic FR Supplier",
        country=Country.FRANCE,
        voltage_level=VoltageLevel.MEDIUM,
        energy=EnergyCharges(
            price_type=PriceType.FIXED,
            time_periods=[
                TimePeriod(
                    name="Heures de Pointe (HPH)",
                    price_eur_per_kwh=0.2641,
                    grid_energy_eur_per_kwh=0.035,
                    months=[12, 1, 2],  # Winter
                    start_time=time(9, 0),
                    end_time=time(10, 59),
                    days_of_week=[0, 1, 2, 3, 4],
                ),
                TimePeriod(
                    name="Heures Pleines Hiver (HH)",
                    price_eur_per_kwh=0.1850,
                    grid_energy_eur_per_kwh=0.028,
                    months=[12, 1, 2, 11, 3],
                    start_time=time(6, 0),
                    end_time=time(21, 59),
                    days_of_week=[0, 1, 2, 3, 4],
                ),
                TimePeriod(
                    name="Heures Creuses Hiver (HCH)",
                    price_eur_per_kwh=0.1350,
                    grid_energy_eur_per_kwh=0.018,
                    months=[12, 1, 2, 11, 3],
                    start_time=time(22, 0),
                    end_time=time(5, 59),
                ),
                TimePeriod(
                    name="Heures Pleines Été (HPE)",
                    price_eur_per_kwh=0.1280,
                    grid_energy_eur_per_kwh=0.022,
                    months=[4, 5, 6, 7, 8, 9, 10],
                    start_time=time(6, 0),
                    end_time=time(21, 59),
                    days_of_week=[0, 1, 2, 3, 4],
                ),
                TimePeriod(
                    name="Heures Creuses Été (HCE)",
                    price_eur_per_kwh=0.0965,
                    grid_energy_eur_per_kwh=0.012,
                    months=[4, 5, 6, 7, 8, 9, 10],
                    start_time=time(22, 0),
                    end_time=time(5, 59),
                ),
            ],
        ),
        grid=GridFees(
            capacity_charge_eur_per_kw_year=55.0,
            subscribed_power_kw=400,
            connection_power_limit_kw=500,
            time_differentiated=True,
            overshoot_penalty_eur_per_kw=150.0,
        ),
        taxes=TaxesAndLevies(
            excise_eur_per_kwh=0.02250,  # TICFE / accise
            renewable_levy_eur_per_kwh=0.0,  # Included in TURPE since reform
            other_levies_eur_per_kwh=0.003,  # CTA etc.
            vat_rate=0.20,
            vat_applicable=False,  # B2B
        ),
    )


# === Serialization / Deserialization ===

def contract_to_dict(contract: EnergyContract) -> dict:
    """Serialize an EnergyContract to a JSON-compatible dict."""
    d = asdict(contract)
    # Convert enums to their values
    d["country"] = contract.country.value
    d["voltage_level"] = contract.voltage_level.value
    d["energy"]["price_type"] = contract.energy.price_type.value
    d["production"]["metering_mode"] = contract.production.metering_mode.value
    # Convert time objects in time_periods
    for tp in d["energy"]["time_periods"]:
        tp["start_time"] = tp["start_time"].strftime("%H:%M")
        tp["end_time"] = tp["end_time"].strftime("%H:%M")
    # Convert datetime objects
    if d["start_date"]:
        d["start_date"] = d["start_date"].isoformat()
    if d["end_date"]:
        d["end_date"] = d["end_date"].isoformat()
    return d


def contract_from_dict(d: dict) -> EnergyContract:
    """Deserialize a dict (from JSON) into an EnergyContract."""
    # Parse enums
    country = Country(d.get("country", "BE"))
    voltage_level = VoltageLevel(d.get("voltage_level", "LV"))
    price_type = PriceType(d.get("energy", {}).get("price_type", "fixed"))
    metering_mode = MeteringMode(
        d.get("production", {}).get("metering_mode", "gross"))

    # Parse time periods
    time_periods = []
    for tp_data in d.get("energy", {}).get("time_periods", []):
        st = tp_data.get("start_time", "00:00")
        et = tp_data.get("end_time", "23:59")
        if isinstance(st, str):
            parts = st.split(":")
            st = time(int(parts[0]), int(parts[1]))
        if isinstance(et, str):
            parts = et.split(":")
            et = time(int(parts[0]), int(parts[1]))
        time_periods.append(TimePeriod(
            name=tp_data.get("name", ""),
            price_eur_per_kwh=tp_data.get("price_eur_per_kwh", 0.0),
            grid_energy_eur_per_kwh=tp_data.get("grid_energy_eur_per_kwh", 0.0),
            months=tp_data.get("months", list(range(1, 13))),
            days_of_week=tp_data.get("days_of_week", list(range(7))),
            start_time=st,
            end_time=et,
        ))

    energy_d = d.get("energy", {})
    grid_d = d.get("grid", {})
    taxes_d = d.get("taxes", {})
    prod_d = d.get("production", {})
    pen_d = d.get("penalties", {})

    # Parse dates
    start_date = None
    end_date = None
    if d.get("start_date"):
        try:
            start_date = datetime.fromisoformat(d["start_date"])
        except (ValueError, TypeError):
            pass
    if d.get("end_date"):
        try:
            end_date = datetime.fromisoformat(d["end_date"])
        except (ValueError, TypeError):
            pass

    return EnergyContract(
        contract_name=d.get("contract_name", ""),
        supplier=d.get("supplier", ""),
        country=country,
        voltage_level=voltage_level,
        start_date=start_date,
        end_date=end_date,
        energy=EnergyCharges(
            price_type=price_type,
            time_periods=time_periods,
            index_name=energy_d.get("index_name"),
            index_base_eur_per_kwh=energy_d.get("index_base_eur_per_kwh", 0.0),
            bloc_price_eur_per_kwh=energy_d.get("bloc_price_eur_per_kwh", 0.0),
            bloc_volume_kwh=energy_d.get("bloc_volume_kwh", 0.0),
            flat_price_eur_per_kwh=energy_d.get("flat_price_eur_per_kwh", 0.0),
        ),
        grid=GridFees(
            capacity_charge_eur_per_kw_year=grid_d.get(
                "capacity_charge_eur_per_kw_year", 0.0),
            subscribed_power_kw=grid_d.get("subscribed_power_kw", 0.0),
            connection_power_limit_kw=grid_d.get(
                "connection_power_limit_kw", 0.0),
            flat_grid_energy_eur_per_kwh=grid_d.get(
                "flat_grid_energy_eur_per_kwh", 0.0),
            time_differentiated=grid_d.get("time_differentiated", False),
            reactive_power_penalty_eur_per_kvarh=grid_d.get(
                "reactive_power_penalty_eur_per_kvarh", 0.0),
            cos_phi_threshold=grid_d.get("cos_phi_threshold", 0.9),
            overshoot_penalty_eur_per_kw=grid_d.get(
                "overshoot_penalty_eur_per_kw", 0.0),
            overshoot_reference=grid_d.get("overshoot_reference", "subscribed"),
        ),
        taxes=TaxesAndLevies(
            excise_eur_per_kwh=taxes_d.get("excise_eur_per_kwh", 0.0),
            renewable_levy_eur_per_kwh=taxes_d.get(
                "renewable_levy_eur_per_kwh", 0.0),
            other_levies_eur_per_kwh=taxes_d.get(
                "other_levies_eur_per_kwh", 0.0),
            vat_rate=taxes_d.get("vat_rate", 0.21),
            vat_applicable=taxes_d.get("vat_applicable", True),
        ),
        production=OnSiteProduction(
            has_production=prod_d.get("has_production", False),
            technology=prod_d.get("technology", ""),
            installed_capacity_kwp=prod_d.get("installed_capacity_kwp", 0.0),
            metering_mode=metering_mode,
            injection_tariff_eur_per_kwh=prod_d.get(
                "injection_tariff_eur_per_kwh", 0.0),
            injection_tariff_indexed=prod_d.get(
                "injection_tariff_indexed", False),
            avoids_energy_charge=prod_d.get("avoids_energy_charge", True),
            avoids_grid_energy_fee=prod_d.get("avoids_grid_energy_fee", False),
            avoids_excise=prod_d.get("avoids_excise", False),
            avoids_renewable_levy=prod_d.get("avoids_renewable_levy", False),
            grid_fee_reduction_pct=prod_d.get("grid_fee_reduction_pct", 0.0),
            green_certificate_value_eur_per_mwh=prod_d.get(
                "green_certificate_value_eur_per_mwh", 0.0),
            green_certificate_eligible=prod_d.get(
                "green_certificate_eligible", False),
            prosumer_tariff_eur_per_year=prod_d.get(
                "prosumer_tariff_eur_per_year", 0.0),
        ),
        penalties=Penalties(
            minimum_offtake_kwh_year=pen_d.get(
                "minimum_offtake_kwh_year", 0.0),
            minimum_offtake_penalty_eur_per_kwh=pen_d.get(
                "minimum_offtake_penalty_eur_per_kwh", 0.0),
            demand_overshoot_penalty_eur_per_kw=pen_d.get(
                "demand_overshoot_penalty_eur_per_kw", 0.0),
        ),
        source=d.get("source", "database"),
        validated=d.get("validated", False),
        notes=d.get("notes", ""),
    )


def save_contract_json(contract: EnergyContract, path: str):
    """Save an EnergyContract to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contract_to_dict(contract), f, indent=2, ensure_ascii=False)


def load_contract_json(path: str) -> EnergyContract:
    """Load an EnergyContract from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return contract_from_dict(json.load(f))


def _get_region_dso(contract_id: str, country_code: str) -> str:
    """Extract region/DSO label from contract ID prefix."""
    _prefix_map = {
        "BE-FL": "Flanders — Fluvius",
        "BE-WAL": "Wallonia — ORES/RESA",
        "BE-BXL": "Brussels — Sibelga",
    }
    # Check Belgian regional prefixes (longest first)
    for prefix, label in _prefix_map.items():
        if contract_id.startswith(prefix):
            return label
    # Default by country
    _country_defaults = {
        "FR": "Enedis (national)",
        "BE": "Belgium (general)",
        "NL": "Netherlands",
        "DE": "Germany",
    }
    return _country_defaults.get(country_code, country_code)


def _country_display_name(country_code: str) -> str:
    """Convert country code to display name."""
    _names = {"FR": "France", "BE": "Belgium", "NL": "Netherlands", "DE": "Germany"}
    return _names.get(country_code, country_code)


def load_contracts_db(path: str) -> tuple[dict, dict]:
    """Load the contracts database JSON.

    Returns:
        hierarchy: {country_display: {region_dso: {contract_name: EnergyContract}}}
        metadata: {contract_name: {"id": str, "description": str, "notes": str}}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hierarchy = {}
    metadata = {}
    for entry in data.get("contracts", []):
        contract_id = entry.get("id", "")
        country_code = entry.get("country", "Unknown")
        name = entry.get("contract_name", "Unnamed")
        description = entry.get("description", "")

        # Collect all notes from sub-sections
        notes_parts = []
        for section in ("energy", "grid", "taxes", "production"):
            section_notes = entry.get(section, {}).get("notes", "")
            if section_notes:
                notes_parts.append(f"[{section.title()}] {section_notes}")

        contract = contract_from_dict(entry)

        country_display = _country_display_name(country_code)
        region_dso = _get_region_dso(contract_id, country_code)

        hierarchy.setdefault(country_display, {}).setdefault(
            region_dso, {})[name] = contract
        metadata[name] = {
            "id": contract_id,
            "description": description,
            "notes": "\n".join(notes_parts),
        }

    return hierarchy, metadata


def save_contracts_db(db: dict, path: str):
    """Save the contracts database to JSON.

    db: {country_display: {region_dso: {contract_name: EnergyContract}}}
    """
    contracts_list = []
    for region_contracts in db.values():
        for contracts in region_contracts.values():
            for contract in contracts.values():
                contracts_list.append(contract_to_dict(contract))

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"contracts": contracts_list}, f,
                  indent=2, ensure_ascii=False)


def get_default_db_path() -> str:
    """Return the default path for contracts_db.json."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "contracts_db.json")
