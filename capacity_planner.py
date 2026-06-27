"""
capacity_planner.py
-------------------
Module 6: Capacity Planning Optimizer
Determines the minimum warehouse configuration P* (P=1 to P=5) for each region
such that capacity >= demand across total, cold, and critical-cold drug types.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from data_loader import load_all_workbooks

OUTPUT_FILE = Path(__file__).resolve().parent / "pipeline_outputs" / "capacity_plan.csv"

def run(base_path: str | Path | None = None) -> pd.DataFrame:
    bundles = load_all_workbooks(base_path)
    
    # 1. Demand per region, drug type
    demand_raw = [v for v in bundles["demand"].cleaned.values() if isinstance(v, pd.DataFrame)]
    demand_df = pd.concat(demand_raw, ignore_index=True)
    
    def map_dt(dt):
        dt_lower = str(dt).lower()
        if "critical" in dt_lower and "cold" in dt_lower: return "critical_cold"
        if "cold" in dt_lower: return "cold"
        return "normal"
        
    demand_df["dt"] = demand_df["demand_type"].apply(map_dt)
    dem_grp = demand_df.groupby(["region", "dt"])["demand"].sum().reset_index()
    
    reg_demand = {}
    for r in dem_grp["region"].unique():
        rdf = dem_grp[dem_grp["region"] == r]
        total = rdf["demand"].sum()
        cold = rdf[rdf["dt"] == "cold"]["demand"].sum()
        crit = rdf[rdf["dt"] == "critical_cold"]["demand"].sum()
        reg_demand[r] = {"total": total, "cold": cold, "critical_cold": crit}

    # 2. Capacity per region, P, drug type
    cap_raw = [v for v in bundles["capacity"].cleaned.values() if isinstance(v, pd.DataFrame)]
    cap_df = pd.concat(cap_raw, ignore_index=True)
    
    def map_cap(m):
        m_lower = str(m).lower()
        if "crit" in m_lower and "cold" in m_lower: return "critical_cold"
        if "cold" in m_lower: return "cold"
        if "norm" in m_lower: return "normal"
        return "total"
        
    cap_df["dt"] = cap_df["metric"].apply(map_cap)
    cap_grp = cap_df.groupby(["region", "configuration", "dt"])["value"].sum().reset_index()
    
    reg_cap = {}
    for r in cap_grp["region"].unique():
        reg_cap[r] = {}
        rdf = cap_grp[cap_grp["region"] == r]
        for config in rdf["configuration"].unique():
            cdf = rdf[rdf["configuration"] == config]
            tot = cdf[cdf["dt"] == "total"]["value"].sum()
            if pd.isna(tot) or tot == 0:
                tot = cdf[cdf["dt"].isin(["normal", "cold", "critical_cold"])]["value"].sum()
            cold = cdf[cdf["dt"] == "cold"]["value"].sum()
            crit = cdf[cdf["dt"] == "critical_cold"]["value"].sum()
            reg_cap[r][config] = {
                "total": tot if not pd.isna(tot) else 0.0,
                "cold": cold if not pd.isna(cold) else 0.0,
                "critical_cold": crit if not pd.isna(crit) else 0.0
            }

    # 3. Minimum unit cost per region from CostCluster
    cost_raw = [v for v in bundles["cost"].cleaned.values() if isinstance(v, pd.DataFrame)]
    cost_df = pd.concat(cost_raw, ignore_index=True)
    cost_df["value"] = pd.to_numeric(cost_df["value"], errors="coerce")
    min_costs = cost_df.dropna(subset=["value"]).groupby("region")["value"].min().to_dict()

    # 4. Find recommended P
    results = []
    all_configs = sorted(list(set(cap_grp["configuration"].unique())), reverse=True)
    
    for r, d in reg_demand.items():
        if r not in reg_cap:
            continue
        c_p1 = reg_cap[r].get("P=1", {"total": 0, "cold": 0, "critical_cold": 0})
        
        recommended_P = "P=1" # fallback
        for config in all_configs:
            if config not in reg_cap[r]: continue
            c = reg_cap[r][config]
            if (c["total"] >= d["total"]) and (c["cold"] >= d["cold"]) and (c["critical_cold"] >= d["critical_cold"]):
                recommended_P = config
                break
                
        c_rec = reg_cap[r].get(recommended_P, c_p1)
        gap_total = max(0, c_rec["total"] - c_p1["total"])
        gap_cold = max(0, c_rec["cold"] - c_p1["cold"])
        
        min_cost = min_costs.get(r, 0.0)
        upgrade_cost_delta = gap_total * min_cost
        
        results.append({
            "region": r,
            "current_P": "P=1",
            "recommended_P": recommended_P,
            "capacity_gap_closed_total": gap_total,
            "capacity_gap_closed_cold": gap_cold,
            "upgrade_cost_delta": upgrade_cost_delta
        })
        
    res_df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUTPUT_FILE, index=False)
    return res_df

if __name__ == "__main__":
    df = run()
    print("=== CAPACITY PLANNER ===")
    print(df.to_string(index=False))
