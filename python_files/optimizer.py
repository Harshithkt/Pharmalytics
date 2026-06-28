"""
optimizer.py
------------
Linear-programming allocation optimizer using PuLP.

Problem:
    Allocate demand at each pharmacy cluster to available warehouses
    to minimise total transport cost, subject to:
      1. Demand satisfaction  — all cluster demand is served
      2. Capacity constraints — each warehouse's total, cold, and
                                critical-cold capacity is respected
      3. Cold-chain routing   — cold/critical demand can only be served
                                by cold-capable warehouses

Decision variables:
    x[w, c, t] = fraction of drug-type t demand at cluster c served by warehouse w

Objective:
    Minimise Σ cost[w,c] · demand[c,t] · x[w,c,t]

Returns:
    - Allocation DataFrame (warehouse, cluster, drug_type, allocated_units)
    - Objective value (total cost)
    - Constraint satisfaction report
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pulp

from data_loader import load_all_workbooks, WorkbookBundle

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
REGIONS = ["Rize", "Trabzon", "Ordu", "Giresun"]
DRUG_TYPES = ["normal", "cold", "critical_cold"]

# Map drug_type -> demand column and capacity metric
DRUG_DEMAND_MAP = {
    "normal":       "demandNorm",
    "cold":         "demandCold",
    "critical_cold":"demandColdCrit",
}
DRUG_CAP_MAP = {
    "normal":        "capacityNorm",
    "cold":          "capacityCold",
    "critical_cold": "capCritCold",
}

# Warehouses (source nodes) — labelled from CostsMWC-Clustered
WAREHOUSES = {
    "Rize":    ["RC0", "RC1"],
    "Trabzon": ["TC0", "TC1", "TC2", "TC3"],
    "Ordu":    ["OC0", "OC1"],
    "Giresun": ["GC0", "GC1"],
}

# Cold-capable warehouses (can handle cold & critical-cold shipments)
COLD_CAPABLE = {"RC0", "RC1", "TC0", "TC1", "GC0", "GC1", "OC0", "OC1"}


# --------------------------------------------------------------------------- #
# Data extraction helpers
# --------------------------------------------------------------------------- #

def _get_cluster_demand(demand_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return demand per (region, cluster, drug_type)."""
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
    rows = []
    dtype_map = {
        "demandNorm":     "normal",
        "demandCold":     "cold",
        "demandColdCrit": "critical_cold",
    }
    for frame in demand_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "cluster", "demand_type", "demand"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
        frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["demand"])
        if frame.empty:
            continue
        for dtype_raw, label in dtype_map.items():
            sub = frame[frame["demand_type"] == dtype_raw].copy()
            if sub.empty:
                continue
            grp = sub.groupby(["region", "cluster"])["demand"].sum().reset_index()
            grp["drug_type"] = label
            rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["region", "cluster", "drug_type", "demand"])
    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["region", "cluster", "drug_type"])["demand"]
        .sum()
        .reset_index()
    )


def _get_warehouse_capacity(capacity_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return capacity per (region, warehouse_key, cap_type).
    Use P=1 configuration as the baseline.
    """
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
    rows = []
    metric_map = {
        "capacityNorm":  "normal",
        "capacityCold":  "cold",
        "capCritCold":   "critical_cold",
        "capacity":      "total",
    }
    for frame in capacity_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "configuration", "metric", "value"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["value"])
        p1 = frame[frame["configuration"] == "P=1"].copy()
        if p1.empty:
            continue
        for mname, label in metric_map.items():
            sub = p1[p1["metric"] == mname]
            if sub.empty:
                continue
            grp = sub.groupby("region")["value"].max().reset_index()
            grp["cap_type"] = label
            rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["region", "cap_type", "capacity"])
    result = pd.concat(rows, ignore_index=True)
    result.columns = [c if c != "value" else "capacity" for c in result.columns]
    if "value" in result.columns and "capacity" not in result.columns:
        result = result.rename(columns={"value": "capacity"})
    # Rename value -> capacity if needed
    for frame_tmp in [result]:
        if "value" in frame_tmp.columns:
            frame_tmp.rename(columns={"value": "capacity"}, inplace=True)
    return (
        result.groupby(["region", "cap_type"])["capacity"]
        .max()
        .reset_index()
    )


def _get_transport_costs(fixed_cost_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return fixed warehouse transport cost per (region, warehouse_key)."""
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
    rows = []
    for frame in fixed_cost_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "cost_type", "fixed_cost"}.issubset(frame.columns):
            continue
        rows.append(frame[["region", "cost_type", "fixed_cost"]].copy())
    if not rows:
        return pd.DataFrame(columns=["region", "cost_type", "fixed_cost"])
    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["region", "cost_type"])["fixed_cost"]
        .mean()
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# Build and solve LP
# --------------------------------------------------------------------------- #

def build_and_solve(
    bundles: dict[str, WorkbookBundle],
    solver_name: str = "HiGHS_CMD",
    time_limit_secs: int = 120,
    msg: bool = False,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """
    Formulate and solve the warehouse-allocation LP.

    Returns:
        allocation_df  – DataFrame of decision variable values
        total_cost     – LP objective value
        report         – constraint satisfaction dictionary
    """
    demand_df   = _get_cluster_demand(bundles["demand"].cleaned)
    capacity_df = _get_warehouse_capacity(bundles["capacity"].cleaned)
    cost_df     = _get_transport_costs(bundles["fixed_cost"].cleaned)

    # Flatten cost lookup: cost_type -> cost value (per unit shipped)
    cost_lookup: dict[str, float] = dict(
        zip(cost_df["cost_type"].astype(str), cost_df["fixed_cost"].astype(float))
    )

    # Capacity lookup: (region, cap_type) -> max capacity
    cap_lookup: dict[tuple, float] = {
        (str(row["region"]), str(row["cap_type"])): float(row["capacity"])
        for _, row in capacity_df.iterrows()
    }

    # --- Build problem ---
    prob = pulp.LpProblem("Pharma_Warehouse_Allocation", pulp.LpMinimize)

    # Enumerate: (warehouse_key, cluster_id, drug_type) combinations
    vars_: dict[tuple, pulp.LpVariable] = {}

    demand_records = demand_df.to_dict("records")

    for region, wh_list in WAREHOUSES.items():
        region_demand = [r for r in demand_records if r["region"] == region]
        for rec in region_demand:
            cluster  = rec["cluster"]
            drug_type = rec["drug_type"]
            for wh_key in wh_list:
                # Cold-chain constraint: cold/critical only from cold-capable WH
                if drug_type in ("cold", "critical_cold") and wh_key not in COLD_CAPABLE:
                    continue
                var_name = f"x_{wh_key}_{cluster}_{drug_type}".replace(" ", "_")
                v = pulp.LpVariable(var_name, lowBound=0, upBound=1)
                vars_[(wh_key, cluster, drug_type, region)] = v

    # --- Objective ---
    obj_terms = []
    for (wh_key, cluster, drug_type, region), var in vars_.items():
        d_rows = demand_df[
            (demand_df["region"] == region) &
            (demand_df["cluster"] == cluster) &
            (demand_df["drug_type"] == drug_type)
        ]
        if d_rows.empty:
            continue
        demand_val = float(d_rows["demand"].iloc[0])
        unit_cost  = cost_lookup.get(wh_key, 1.0)  # fallback = 1
        obj_terms.append(unit_cost * demand_val * var)

    prob += pulp.lpSum(obj_terms), "total_transport_cost"

    # --- Demand satisfaction: for each (region, cluster, drug_type), sum x = 1 ---
    for rec in demand_records:
        region    = rec["region"]
        cluster   = rec["cluster"]
        drug_type = rec["drug_type"]
        relevant  = [
            var for (wh_key, cl, dt, rg), var in vars_.items()
            if cl == cluster and dt == drug_type and rg == region
        ]
        if not relevant:
            continue
        cname = f"demand_{region}_{cluster}_{drug_type}".replace(" ", "_")
        prob += pulp.lpSum(relevant) == 1, cname

    # --- Capacity constraints per (region, warehouse_key, cap_type) ---
    for region, wh_list in WAREHOUSES.items():
        for wh_key in wh_list:
            for drug_type, cap_type in DRUG_CAP_MAP.items():
                cap_val = cap_lookup.get((region, cap_type), None)
                if cap_val is None or cap_val == 0:
                    continue
                region_demand = [r for r in demand_records if r["region"] == region]
                load_terms = []
                for rec in region_demand:
                    if rec["drug_type"] != drug_type:
                        continue
                    key = (wh_key, rec["cluster"], drug_type, region)
                    if key not in vars_:
                        continue
                    load_terms.append(float(rec["demand"]) * vars_[key])
                if not load_terms:
                    continue
                cname = f"cap_{region}_{wh_key}_{drug_type}".replace(" ", "_")
                prob += pulp.lpSum(load_terms) <= cap_val, cname

    # --- Solve (try HiGHS Python API → CBC → any available) ---
    _solved = False
    status_str = "Not Solved"
    total_cost = 0.0
    try:
        # HiGHS is available as a Python-API solver (not CMD) on Apple Silicon
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit_secs)
        status = prob.solve(solver)
        _solved = True
    except Exception:
        pass

    if not _solved:
        # Try all available solvers
        for _sname in pulp.listSolvers(onlyAvailable=True):
            try:
                _s = pulp.getSolver(_sname, msg=msg, timeLimit=time_limit_secs)
                status = prob.solve(_s)
                _solved = True
                break
            except Exception:
                continue

    if not _solved:
        print("  ⚠ No compatible LP solver found. Returning empty allocation.")
        return pd.DataFrame(), 0.0, {"status": "No solver", "total_cost": 0}

    status_str  = pulp.LpStatus[prob.status]
    total_cost  = float(pulp.value(prob.objective) or 0.0)

    # --- Extract solution ---
    alloc_rows = []
    for (wh_key, cluster, drug_type, region), var in vars_.items():
        val = pulp.value(var)
        if val is not None and val > 1e-6:
            d_rows = demand_df[
                (demand_df["region"] == region) &
                (demand_df["cluster"] == cluster) &
                (demand_df["drug_type"] == drug_type)
            ]
            demand_val = float(d_rows["demand"].iloc[0]) if not d_rows.empty else 0.0
            alloc_rows.append({
                "region":           region,
                "warehouse":        wh_key,
                "cluster":          cluster,
                "drug_type":        drug_type,
                "allocation_frac":  round(val, 4),
                "allocated_units":  round(val * demand_val, 2),
                "unit_cost":        cost_lookup.get(wh_key, 0.0),
                "route_cost":       round(val * demand_val * cost_lookup.get(wh_key, 0.0), 2),
            })

    allocation_df = pd.DataFrame(alloc_rows).sort_values(
        ["region", "warehouse", "drug_type"], ignore_index=True
    )

    report = {
        "status":             status_str,
        "total_cost":         round(total_cost, 2),
        "n_decision_vars":    len(vars_),
        "n_constraints":      len(prob.constraints),
        "n_allocated_routes": len(allocation_df),
        "cost_by_region":     (
            allocation_df.groupby("region")["route_cost"].sum().round(2).to_dict()
            if not allocation_df.empty else {}
        ),
        "cost_by_drug_type":  (
            allocation_df.groupby("drug_type")["route_cost"].sum().round(2).to_dict()
            if not allocation_df.empty else {}
        ),
    }

    return allocation_df, total_cost, report


def run(base_path: str | Path | None = None) -> tuple[pd.DataFrame, float, dict]:
    """
    Full optimizer run.

    Returns:
        allocation_df – allocation results
        total_cost    – minimised total transport cost
        report        – summary report dict
    """
    bundles = load_all_workbooks(base_path)
    return build_and_solve(bundles)


if __name__ == "__main__":
    alloc_df, total_cost, report = run()
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"  Status:      {report['status']}")
    print(f"  Total Cost:  {report['total_cost']}")
    print(f"  Variables:   {report['n_decision_vars']}")
    print(f"  Constraints: {report['n_constraints']}")
    print(f"\n  Cost by region:")
    for r, c in report["cost_by_region"].items():
        print(f"    {r}: {c}")
    print(f"\n  Cost by drug type:")
    for d, c in report["cost_by_drug_type"].items():
        print(f"    {d}: {c}")
    print("\n  Allocation (top 15 routes):")
    print(alloc_df.head(15).to_string(index=False))
