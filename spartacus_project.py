"""Spartacus Project File — Save/Load serialization module.

Handles .spartacus project files (JSON with custom extension) for
saving and restoring the complete session state.

Property of ReVolta srl. All rights reserved.
"""

import json
import os
import io
from datetime import datetime, time

import numpy as np
import pandas as pd

from energy_parser.contract_model import (
    contract_to_dict, contract_from_dict,
)


# ============================================================
# Custom JSON Encoder / Decoder
# ============================================================

class SpartacusEncoder(json.JSONEncoder):
    """Encode pandas / numpy / datetime types to JSON-safe dicts."""

    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return {"__type__": "Timestamp", "value": obj.isoformat()}
        if isinstance(obj, pd.DataFrame):
            return {"__type__": "DataFrame", "csv": obj.to_csv(index=False)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return {"__type__": "time", "value": obj.strftime("%H:%M:%S")}
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


def spartacus_object_hook(d: dict):
    """Reverse of SpartacusEncoder — reconstruct special types."""
    t = d.get("__type__")
    if t == "Timestamp":
        return pd.Timestamp(d["value"])
    if t == "DataFrame":
        return pd.read_csv(io.StringIO(d["csv"]))
    if t == "time":
        parts = d["value"].split(":")
        return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    return d


# ============================================================
# Core save / load
# ============================================================

def save_project(path: str, state_dict: dict) -> None:
    """Write *state_dict* to *path* as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, cls=SpartacusEncoder, ensure_ascii=False, indent=2)


def load_project(path: str) -> dict:
    """Read a .spartacus file and return the state dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_hook=spartacus_object_hook)


# ============================================================
# Specialized serializers for tricky structures
# ============================================================

def serialize_quality_report(report: dict) -> dict:
    """Convert quality_report timestamps to ISO strings for JSON."""
    if report is None:
        return None
    out = {}
    for k, v in report.items():
        if k == "date_range" and isinstance(v, (list, tuple)):
            out[k] = [
                v[0].isoformat() if hasattr(v[0], "isoformat") else str(v[0]),
                v[1].isoformat() if hasattr(v[1], "isoformat") else str(v[1]),
            ]
        elif k == "gaps" and isinstance(v, list):
            gaps = []
            for g in v:
                gc = dict(g)
                if "from" in gc and hasattr(gc["from"], "isoformat"):
                    gc["from"] = gc["from"].isoformat()
                if "to" in gc and hasattr(gc["to"], "isoformat"):
                    gc["to"] = gc["to"].isoformat()
                gaps.append(gc)
            out[k] = gaps
        elif k == "duplicates" and isinstance(v, list):
            dups = []
            for d in v:
                dc = dict(d)
                if "timestamp" in dc and hasattr(dc["timestamp"], "isoformat"):
                    dc["timestamp"] = dc["timestamp"].isoformat()
                dups.append(dc)
            out[k] = dups
        else:
            out[k] = v
    return out


def deserialize_quality_report(data: dict) -> dict:
    """Reverse of serialize_quality_report."""
    if data is None:
        return None
    out = {}
    for k, v in data.items():
        if k == "date_range" and isinstance(v, list) and len(v) == 2:
            out[k] = (pd.Timestamp(v[0]), pd.Timestamp(v[1]))
        elif k == "gaps" and isinstance(v, list):
            gaps = []
            for g in v:
                gc = dict(g)
                if "from" in gc:
                    gc["from"] = pd.Timestamp(gc["from"])
                if "to" in gc:
                    gc["to"] = pd.Timestamp(gc["to"])
                gaps.append(gc)
            out[k] = gaps
        elif k == "duplicates" and isinstance(v, list):
            dups = []
            for d in v:
                dc = dict(d)
                if "timestamp" in dc:
                    dc["timestamp"] = pd.Timestamp(dc["timestamp"])
                dups.append(dc)
            out[k] = dups
        else:
            out[k] = v
    return out


def _cost_breakdown_to_dict(cb) -> dict:
    """Convert a CostBreakdown dataclass to a plain dict."""
    from dataclasses import asdict
    return asdict(cb)


def serialize_cost_simulation(result_tuple, contract, db_base_contract) -> dict:
    """Serialize cost simulation state (skips detail_df — too large, regenerated)."""
    if result_tuple is None:
        return None
    _, summary, monthly = result_tuple

    monthly_dict = {}
    for mk, mb in monthly.items():
        monthly_dict[str(mk)] = _cost_breakdown_to_dict(mb)

    out = {
        "summary": _cost_breakdown_to_dict(summary),
        "monthly": monthly_dict,
    }
    if contract is not None:
        out["contract"] = contract_to_dict(contract)
    if db_base_contract is not None:
        out["db_base_contract"] = contract_to_dict(db_base_contract)
    return out


def deserialize_cost_simulation(data: dict) -> dict:
    """Reverse of serialize_cost_simulation."""
    if data is None:
        return None
    result = {
        "summary_dict": data.get("summary", {}),
        "monthly_dict": data.get("monthly", {}),
        "contract": None,
        "db_base_contract": None,
    }
    if data.get("contract"):
        result["contract"] = contract_from_dict(data["contract"])
    if data.get("db_base_contract"):
        result["db_base_contract"] = contract_from_dict(data["db_base_contract"])
    return result


# ============================================================
# Recent projects config
# ============================================================

def get_config_path() -> str:
    """Return the path to the Spartacus config file."""
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        folder = os.path.join(base, "Spartacus")
    else:
        folder = os.path.join(os.path.expanduser("~"), ".spartacus")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "spartacus_config.json")


def load_recent_projects() -> list:
    """Load recent projects list. Each entry: {path, name, last_opened}."""
    cfg_path = get_config_path()
    if not os.path.exists(cfg_path):
        return []
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("recent_projects", [])
    except Exception:
        return []


def save_recent_projects(projects: list) -> None:
    """Persist the recent projects list (max 5)."""
    cfg_path = get_config_path()
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
    except Exception:
        data = {}
    data["recent_projects"] = projects[:5]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_to_recent_projects(path: str, name: str) -> list:
    """Add or move *path* to the top of recent projects. Returns updated list."""
    projects = load_recent_projects()
    # Remove existing entry for same path
    projects = [p for p in projects if p.get("path") != path]
    # Prepend
    projects.insert(0, {
        "path": path,
        "name": name,
        "last_opened": datetime.now().isoformat(),
    })
    projects = projects[:5]
    save_recent_projects(projects)
    return projects
