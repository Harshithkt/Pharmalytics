"""
rerouting.py
------------
Module 7: Rerouting Recommender
Identifies region-drug_type pairs where the assigned warehouse unit cost 
exceeds the minimum available unit cost across cold-capable alternatives.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from data_loader import load_all_workbooks

OUTPUT_FILE = Path(__file__).resolve().parent / "pipeline_outputs" / "rerouting_recommendations.csv"

# Same definition as optimizer.py for cold-capable warehouses
COLD_CAPABLE = {"RC0", "RC1", "TC0", "TC1", "GC0", "GC1", "OC0", "OC1"}

def run(base_path: str | Path | None = None) -> pd.DataFrame:
    bundles = load_all_workbooks(base_path)
    
    # 1. Load allocations
    alloc_path = Path(__file__).resolve().parent / "pipeline_outputs" / "allocation_results.csv"
    if not alloc_path.exists():
        print("No allocation_results.csv found. Run LP first.")
        return pd.DataFrame()
    alloc_df = pd.read_csv(alloc_path)
    
    # 2. Get fixed warehouse costs (unit cost per warehouse)
    cost_raw = [v for v in bundles["fixed_cost"].cleaned.values() if isinstance(v, pd.DataFrame)]
    if not cost_raw:
        return pd.DataFrame()
    cost_df = pd.concat(cost_raw, ignore_index=True)
    cost_df["fixed_cost"] = pd.to_numeric(cost_df["fixed_cost"], errors="coerce")
    cost_map = {}
    for _, row in cost_df.dropna(subset=["fixed_cost"]).iterrows():
        reg = row["region"]
        wh = row["cost_type"]
        val = row["fixed_cost"]
        if reg not in cost_map:
            cost_map[reg] = {}
        cost_map[reg][wh] = val

    # 3. Aggregate allocations by region, drug_type, warehouse
    agg_alloc = alloc_df.groupby(["region", "drug_type", "warehouse", "unit_cost"])["allocated_units"].sum().reset_index()

    results = []
    
    for _, row in agg_alloc.iterrows():
        r = row["region"]
        dt = row["drug_type"]
        wh = row["warehouse"]
        current_cost = row["unit_cost"]
        demand = row["allocated_units"]
        
        if r not in cost_map:
            continue
            
        valid_alts = {}
        for alt_wh, alt_cost in cost_map[r].items():
            if dt in ["cold", "critical_cold"]:
                if alt_wh in COLD_CAPABLE:
                    valid_alts[alt_wh] = alt_cost
            else:
                valid_alts[alt_wh] = alt_cost
                
        if not valid_alts:
            continue
            
        min_alt_wh = min(valid_alts, key=valid_alts.get)
        min_alt_cost = valid_alts[min_alt_wh]
        
        if current_cost > min_alt_cost:
            savings = (current_cost - min_alt_cost) * demand
            results.append({
                "region": r,
                "drug_type": dt,
                "current_warehouse": wh,
                "current_unit_cost": current_cost,
                "recommended_warehouse": min_alt_wh,
                "recommended_unit_cost": min_alt_cost,
                "annual_cost_saving": savings
            })
            
    columns = ["region", "drug_type", "current_warehouse", "current_unit_cost", "recommended_warehouse", "recommended_unit_cost", "annual_cost_saving"]
    res_df = pd.DataFrame(results, columns=columns)
    if not res_df.empty:
        # Aggregate any split clusters
        res_df = res_df.groupby(["region", "drug_type", "current_warehouse", "recommended_warehouse"]).agg({
            "current_unit_cost": "mean",
            "recommended_unit_cost": "mean",
            "annual_cost_saving": "sum"
        }).reset_index()
        # Sort by savings
        res_df = res_df.sort_values(by="annual_cost_saving", ascending=False)
        
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUTPUT_FILE, index=False)
    return res_df

if __name__ == "__main__":
    df = run()
    print("=== REROUTING RECOMMENDATIONS ===")
    if df.empty:
        print("No suboptimal routings found.")
    else:
        print(df.to_string(index=False))
