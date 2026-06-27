"""
risk_engine.py
--------------
Composite risk scoring engine for warehouse-to-pharmacy supply chain routes.

Risk score (0-100) = weighted combination of:
  - Normalised transport cost          (weight: 0.25)
  - Normalised delivery time           (weight: 0.30)
  - Demand-capacity gap               (weight: 0.30)
  - Cold-chain constraint violation    (weight: 0.15)

Routes with risk_score >= 75 are flagged as critical bottlenecks.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import load_all_workbooks, WorkbookBundle

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WEIGHTS = {
    "cost":       0.25,
    "time":       0.30,
    "gap":        0.30,
    "cold":       0.15,
}
RISK_CRITICAL_THRESHOLD = 75.0   # scores >= this are "critical"
RISK_HIGH_THRESHOLD     = 50.0   # scores >= this are "high"

REGIONS = ["Rize", "Trabzon", "Ordu", "Giresun"]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _extract_time_stats(time_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return mean/max delivery time per (region, source_zone)."""
    rows = []
    for frame in time_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"source_zone", "delivery_time", "region"}.issubset(frame.columns):
            continue
        grp = frame.groupby(["region", "source_zone"])["delivery_time"].agg(
            mean_time="mean", max_time="max"
        ).reset_index()
        rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["region", "source_zone", "mean_time", "max_time"])
    return pd.concat(rows, ignore_index=True)


def _extract_cost_stats(cost_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return mean/max cost per (region, mode) — skipping junk/index sheets."""
    rows = []
    # Only accept sheets that clearly belong to a named region
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
    for frame in cost_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "mode", "value"}.issubset(frame.columns):
            continue
        # Skip sheets whose region is not one of the four known regions
        region_vals = frame["region"].dropna().unique()
        if not any(str(r) in VALID_REGIONS for r in region_vals):
            continue
        # Coerce value to numeric, drop non-numeric rows
        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["value"])
        if frame.empty:
            continue
        grp = frame.groupby(["region", "mode"])["value"].agg(
            mean_cost="mean", max_cost="max"
        ).reset_index()
        rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["region", "mode", "mean_cost", "max_cost"])
    return pd.concat(rows, ignore_index=True)


def _extract_demand_capacity(
    demand_cleaned: dict[str, pd.DataFrame],
    capacity_cleaned: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return per-region demand/capacity ratios."""
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}

    # ---- demand ----
    d_rows = []
    for frame in demand_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "demand_type", "demand"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
        frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["demand"])
        if frame.empty:
            continue
        total = frame.groupby("region")["demand"].sum().reset_index(name="total_demand")
        cold  = (
            frame[frame["demand_type"].str.lower().str.contains("cold", na=False)]
            .groupby("region")["demand"].sum().reset_index(name="cold_demand")
        )
        d_rows.append(total.merge(cold, on="region", how="left"))
    if not d_rows:
        demand_df = pd.DataFrame(columns=["region", "total_demand", "cold_demand"])
    else:
        demand_df = pd.concat(d_rows, ignore_index=True).groupby("region").sum().reset_index()

    # ---- capacity ----
    c_rows = []
    for frame in capacity_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "metric", "value"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["value"])
        if frame.empty:
            continue
        cap      = frame[frame["metric"] == "capacity"].groupby("region")["value"].max().reset_index(name="total_cap")
        cold_cap = frame[frame["metric"] == "capacityCold"].groupby("region")["value"].max().reset_index(name="cold_cap")
        c_rows.append(cap.merge(cold_cap, on="region", how="left"))
    if not c_rows:
        cap_df = pd.DataFrame(columns=["region", "total_cap", "cold_cap"])
    else:
        cap_df = pd.concat(c_rows, ignore_index=True).groupby("region").max().reset_index()

    merged = demand_df.merge(cap_df, on="region", how="outer")
    merged["gap_ratio"] = (
        (merged["total_demand"] - merged["total_cap"]).clip(lower=0) / merged["total_cap"].replace(0, np.nan)
    ).fillna(0)
    merged["cold_violation_ratio"] = (
        merged["cold_demand"] / merged["cold_cap"].replace(0, np.nan)
    ).fillna(0).clip(upper=2)
    return merged


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def compute_risk_scores(bundles: dict[str, WorkbookBundle]) -> pd.DataFrame:
    """
    Compute composite risk scores for each warehouse-region route.

    Returns a DataFrame with columns:
        region, source_zone, mean_time, max_time,
        mean_cost, total_demand, cold_demand, total_cap, cold_cap,
        gap_ratio, cold_violation_ratio,
        norm_time, norm_cost, norm_gap, norm_cold,
        risk_score, risk_label
    """
    time_stats = _extract_time_stats(bundles["time"].cleaned)
    cost_stats = _extract_cost_stats(bundles["cost"].cleaned)
    dc_stats   = _extract_demand_capacity(
        bundles["demand"].cleaned, bundles["capacity"].cleaned
    )

    # Average cost per region (across all modes)
    avg_cost = cost_stats.groupby("region")["mean_cost"].mean().reset_index(name="avg_cost")
    # Average time per region (across all zones)
    avg_time = time_stats.groupby("region")["mean_time"].mean().reset_index(name="avg_time")

    base = dc_stats.merge(avg_cost, on="region", how="left")
    base = base.merge(avg_time, on="region", how="left")

    # Also build per-(region, source_zone) granular scores for the ML model
    granular = time_stats.merge(dc_stats, on="region", how="left")
    granular = granular.merge(avg_cost, on="region", how="left")

    # Normalise
    for col, norm_col in [
        ("avg_cost",             "norm_cost"),
        ("avg_time",             "norm_time"),
        ("gap_ratio",            "norm_gap"),
        ("cold_violation_ratio", "norm_cold"),
    ]:
        if col in base.columns:
            base[norm_col] = _minmax(base[col].fillna(0))
        else:
            base[norm_col] = 0.0

    base["risk_score"] = (
        WEIGHTS["cost"]  * base["norm_cost"]  * 100 +
        WEIGHTS["time"]  * base["norm_time"]  * 100 +
        WEIGHTS["gap"]   * base["norm_gap"]   * 100 +
        WEIGHTS["cold"]  * base["norm_cold"]  * 100
    )

    def _label(s: float) -> str:
        if s >= RISK_CRITICAL_THRESHOLD:
            return "CRITICAL"
        if s >= RISK_HIGH_THRESHOLD:
            return "HIGH"
        return "LOW"

    base["risk_label"] = base["risk_score"].apply(_label)
    base["is_bottleneck"] = base["risk_score"] >= RISK_CRITICAL_THRESHOLD

    # ---- Granular (per route) risk table ----
    for col, norm_col in [
        ("mean_time",            "norm_time"),
        ("avg_cost",             "norm_cost"),
        ("gap_ratio",            "norm_gap"),
        ("cold_violation_ratio", "norm_cold"),
    ]:
        if col in granular.columns:
            granular[norm_col] = _minmax(granular[col].fillna(0))
        else:
            granular[norm_col] = 0.0

    granular["risk_score"] = (
        WEIGHTS["cost"]  * granular["norm_cost"]  * 100 +
        WEIGHTS["time"]  * granular["norm_time"]  * 100 +
        WEIGHTS["gap"]   * granular["norm_gap"]   * 100 +
        WEIGHTS["cold"]  * granular["norm_cold"]  * 100
    )
    granular["risk_label"]    = granular["risk_score"].apply(_label)
    granular["is_bottleneck"] = granular["risk_score"] >= RISK_CRITICAL_THRESHOLD

    return base, granular


def identify_bottlenecks(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only critical/bottleneck routes."""
    return risk_df[risk_df["is_bottleneck"]].sort_values("risk_score", ascending=False).reset_index(drop=True)


def risk_summary(risk_df: pd.DataFrame) -> dict[str, Any]:
    """Return summary statistics of the risk distribution."""
    return {
        "total_routes":    len(risk_df),
        "critical_routes": int(risk_df["is_bottleneck"].sum()),
        "high_risk":       int((risk_df["risk_label"] == "HIGH").sum()),
        "low_risk":        int((risk_df["risk_label"] == "LOW").sum()),
        "mean_risk_score": round(float(risk_df["risk_score"].mean()), 2),
        "max_risk_score":  round(float(risk_df["risk_score"].max()), 2),
        "min_risk_score":  round(float(risk_df["risk_score"].min()), 2),
    }


def run(base_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full risk engine run.

    Returns:
        region_risk_df  – risk scores aggregated by region
        granular_df     – risk scores per (region, source_zone)
        summary         – dict with headline stats
    """
    bundles = load_all_workbooks(base_path)
    region_df, granular_df = compute_risk_scores(bundles)
    summary = risk_summary(granular_df)
    return region_df, granular_df, summary


if __name__ == "__main__":
    region_df, granular_df, summary = run()
    print("\n=== RISK ENGINE SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n=== REGION-LEVEL RISK SCORES ===")
    print(region_df[["region", "risk_score", "risk_label", "is_bottleneck"]].to_string(index=False))
    print("\n=== GRANULAR RISK TABLE (TOP 10) ===")
    print(granular_df.nlargest(10, "risk_score")[
        ["region", "source_zone", "risk_score", "risk_label"]
    ].to_string(index=False))
