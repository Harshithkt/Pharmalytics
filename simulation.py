"""
simulation.py
-------------
Monte Carlo disruption simulation for pharmaceutical supply chain risk.

Models two disruption scenarios per cluster:
  1. Demand Surge   – demand increases by a random % (up to surge_pct)
  2. Capacity Shock – warehouse capacity drops by a random % (up to failure_pct)

Outputs:
  - shortage_probability per (region, drug_type) — P(demand > available_capacity)
  - expected_shortage_volume — mean units short across simulations
  - worst_case_shortage      — 95th-percentile shortage
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
N_SIMS          = 2_000          # Monte Carlo iterations
SURGE_PCT       = 0.40           # max demand surge (40%)
FAILURE_PCT     = 0.35           # max capacity failure (35%)
RANDOM_SEED     = 42
SHORTAGE_THRESH = 0.0            # shortage if (demand - capacity) > 0


# --------------------------------------------------------------------------- #
# Data extraction helpers
# --------------------------------------------------------------------------- #

def _get_demand_by_region(demand_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return per-(region, drug_type) total demand from cleaned demand frames."""
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
    rows = []
    type_map = {
        "demand":         "total",
        "demandCold":     "cold",
        "demandNorm":     "normal",
        "demandColdCrit": "critical_cold",
        "demandNormCrit": "critical_normal",
    }
    for frame in demand_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "demand_type", "demand"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
        frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["demand"])
        for dtype, label in type_map.items():
            sub = frame[frame["demand_type"] == dtype]
            if sub.empty:
                continue
            grp = sub.groupby("region")["demand"].sum().reset_index()
            grp.columns = ["region", "demand"]
            grp["drug_type"] = label
            rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["region", "drug_type", "demand"])
    result = pd.concat(rows, ignore_index=True)
    # aggregate in case duplicates across sheets
    return result.groupby(["region", "drug_type"])["demand"].sum().reset_index()


def _get_capacity_by_region(capacity_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return per-(region, metric) max capacity. Use P=1 configuration as baseline."""
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
    rows = []
    metric_map = {
        "capacity":     "total",
        "capacityCold": "cold",
        "capacityNorm": "normal",
        "capCritCold":  "critical_cold",
        "capCritNorm":  "critical_normal",
    }
    for frame in capacity_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "metric", "value"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        # Ensure we only calculate base capacity from P=1 configuration
        if "configuration" in frame.columns:
            frame = frame[frame["configuration"] == "P=1"]
        frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["value"])
        for mname, label in metric_map.items():
            sub = frame[frame["metric"] == mname]
            if sub.empty:
                continue
            # Sum across all cluster points in the region
            grp = sub.groupby("region")["value"].sum().reset_index()
            grp.columns = ["region", "capacity"]
            grp["cap_type"] = label
            rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["region", "cap_type", "capacity"])
    result = pd.concat(rows, ignore_index=True)
    return result.groupby(["region", "cap_type"])["capacity"].sum().reset_index()


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #

def _run_scenario(
    demand: float,
    capacity: float,
    n_sims: int,
    surge_pct: float,
    failure_pct: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Single Monte Carlo scenario for one (region, drug_type) pair.

    Returns shortage statistics across n_sims iterations.
    """
    # Demand surge: D_sim = D * (1 + U[0, surge_pct])
    d_sim = demand * (1.0 + rng.uniform(0, surge_pct, size=n_sims))
    # Capacity failure: C_sim = C * (1 - U[0, failure_pct])
    c_sim = capacity * (1.0 - rng.uniform(0, failure_pct, size=n_sims))
    c_sim = np.clip(c_sim, a_min=0, a_max=None)

    shortage = np.maximum(0.0, d_sim - c_sim)
    shortage_flag = shortage > SHORTAGE_THRESH

    return {
        "shortage_probability":    float(shortage_flag.mean()),
        "expected_shortage":       float(shortage.mean()),
        "worst_case_shortage_p95": float(np.percentile(shortage, 95)),
        "worst_case_shortage_p99": float(np.percentile(shortage, 99)),
        "mean_demand_sim":         float(d_sim.mean()),
        "mean_capacity_sim":       float(c_sim.mean()),
    }


def run_monte_carlo(
    bundles: dict[str, WorkbookBundle],
    n_sims: int = N_SIMS,
    surge_pct: float = SURGE_PCT,
    failure_pct: float = FAILURE_PCT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Run Monte Carlo disruption simulation across all regions and drug types.

    Parameters
    ----------
    bundles     : output of load_all_workbooks()
    n_sims      : number of Monte Carlo iterations
    surge_pct   : maximum demand surge fraction
    failure_pct : maximum warehouse capacity loss fraction
    seed        : random seed for reproducibility

    Returns
    -------
    DataFrame with columns:
        region, drug_type, base_demand, base_capacity,
        shortage_probability, expected_shortage,
        worst_case_shortage_p95, worst_case_shortage_p99,
        mean_demand_sim, mean_capacity_sim
    """
    rng = np.random.default_rng(seed)

    demand_df   = _get_demand_by_region(bundles["demand"].cleaned)
    capacity_df = _get_capacity_by_region(bundles["capacity"].cleaned)

    # align drug_type <-> cap_type naming
    type_pairs = [
        ("total",          "total"),
        ("cold",           "cold"),
        ("normal",         "normal"),
        ("critical_cold",  "critical_cold"),
        ("critical_normal","critical_normal"),
    ]

    results = []
    for drug_type, cap_type in type_pairs:
        d_sub = demand_df[demand_df["drug_type"] == drug_type].copy()
        c_sub = capacity_df[capacity_df["cap_type"] == cap_type].copy()
        merged = d_sub.merge(c_sub, on="region", how="inner")
        merged = merged.rename(columns={"demand": "base_demand", "capacity": "base_capacity"})

        for _, row in merged.iterrows():
            stats = _run_scenario(
                demand=float(row["base_demand"]),
                capacity=float(row["base_capacity"]),
                n_sims=n_sims,
                surge_pct=surge_pct,
                failure_pct=failure_pct,
                rng=rng,
            )
            results.append({
                "region":     row["region"],
                "drug_type":  drug_type,
                "base_demand":   round(row["base_demand"], 2),
                "base_capacity": round(row["base_capacity"], 2),
                **{k: round(v, 4) for k, v in stats.items()},
            })

    return pd.DataFrame(results).sort_values(
        ["shortage_probability", "expected_shortage"], ascending=False
    ).reset_index(drop=True)


def shortage_summary(sim_df: pd.DataFrame) -> dict[str, Any]:
    """High-level summary of simulation results."""
    return {
        "n_scenarios":             len(sim_df),
        "high_shortage_risk":      int((sim_df["shortage_probability"] > 0.5).sum()),
        "critical_shortage_risk":  int((sim_df["shortage_probability"] > 0.8).sum()),
        "max_shortage_probability": round(float(sim_df["shortage_probability"].max()), 4),
        "mean_expected_shortage":  round(float(sim_df["expected_shortage"].mean()), 2),
        "worst_p99_shortage":      round(float(sim_df["worst_case_shortage_p99"].max()), 2),
        "most_vulnerable_region":  sim_df.loc[sim_df["shortage_probability"].idxmax(), "region"],
        "most_vulnerable_type":    sim_df.loc[sim_df["shortage_probability"].idxmax(), "drug_type"],
    }


def sensitivity_sweep(
    bundles: dict[str, WorkbookBundle],
    surge_levels: list[float] | None = None,
    failure_levels: list[float] | None = None,
    n_sims: int = 500,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sweep over combinations of surge_pct and failure_pct.

    Returns a DataFrame with mean shortage_probability at each combination.
    Useful for the dashboard slider visualisation.
    """
    if surge_levels is None:
        surge_levels = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    if failure_levels is None:
        failure_levels = [0.0, 0.10, 0.20, 0.30, 0.35]

    rows = []
    for sp in surge_levels:
        for fp in failure_levels:
            df = run_monte_carlo(bundles, n_sims=n_sims, surge_pct=sp, failure_pct=fp, seed=seed)
            rows.append({
                "surge_pct":               sp,
                "failure_pct":             fp,
                "mean_shortage_prob":      round(df["shortage_probability"].mean(), 4),
                "max_shortage_prob":       round(df["shortage_probability"].max(), 4),
                "critical_scenarios":      int((df["shortage_probability"] > 0.5).sum()),
                "mean_expected_shortage":  round(df["expected_shortage"].mean(), 2),
            })
    return pd.DataFrame(rows)


def run(base_path: str | Path | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Full simulation run.

    Returns:
        sim_df      – per-(region, drug_type) Monte Carlo results
        summary     – headline stats dict
        sweep_df    – sensitivity sweep results
    """
    bundles  = load_all_workbooks(base_path)
    sim_df   = run_monte_carlo(bundles)
    summary  = shortage_summary(sim_df)
    sweep_df = sensitivity_sweep(bundles)
    return sim_df, summary, sweep_df


if __name__ == "__main__":
    sim_df, summary, sweep_df = run()
    print("\n=== DISRUPTION SIMULATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n=== SHORTAGE PROBABILITIES BY REGION & DRUG TYPE ===")
    print(sim_df[["region", "drug_type", "shortage_probability", "expected_shortage",
                  "worst_case_shortage_p95"]].to_string(index=False))
    print("\n=== SENSITIVITY SWEEP (mean shortage prob) ===")
    print(sweep_df.to_string(index=False))
