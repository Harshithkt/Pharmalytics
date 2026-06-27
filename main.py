"""
main.py
-------
Orchestrates the full pharmaceutical supply chain risk analytics pipeline:

  1.  Data loading & validation        (data_loader)
  2.  Risk scoring engine              (risk_engine)
  3.  Monte Carlo disruption sim       (simulation)
  4.  XGBoost bottleneck predictor     (ml_model)
  5.  LP allocation optimizer          (optimizer)
  6.  Interactive Dash dashboard       (dashboard)

Usage:
    python main.py                    # run full pipeline + launch dashboard
    python main.py --no-dash          # run pipeline only, print results
    python main.py --port 8080        # custom port
    python main.py --debug            # Dash debug mode
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
OUTPUT_DIR   = PROJECT_ROOT / "pipeline_outputs"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _hr(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _header(title: str) -> None:
    _hr("═")
    print(f"  {title}")
    _hr("═")


def _step(n: int, title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  Step {n}: {title}")
    print(f"{'─'*70}")


def _save(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"  ✓ Saved → {path.relative_to(PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #

def step1_load() -> dict:
    from data_loader import load_and_validate, print_quality_summary
    _step(1, "Data Loading & Validation")
    bundles, report = load_and_validate(DATASET_DIR)
    print_quality_summary(report)
    return bundles


def step2_risk(bundles: dict) -> tuple:
    from risk_engine import compute_risk_scores, risk_summary
    _step(2, "Risk Scoring Engine")
    t0 = time.time()
    region_df, granular_df = compute_risk_scores(bundles)
    summary = risk_summary(granular_df)
    print(f"  Computed risk scores in {time.time()-t0:.2f}s")
    print(f"  Total routes : {summary['total_routes']}")
    print(f"  Critical     : {summary['critical_routes']}")
    print(f"  High         : {summary['high_risk']}")
    print(f"  Low          : {summary['low_risk']}")
    print(f"  Mean score   : {summary['mean_risk_score']}")
    print(f"  Max score    : {summary['max_risk_score']}")
    print("\n  Regional Risk Scores:")
    print(region_df[["region", "risk_score", "risk_label"]].to_string(index=False))
    _save(region_df, "risk_region.csv")
    _save(granular_df, "risk_granular.csv")
    return region_df, granular_df, summary


def step3_simulate(bundles: dict) -> tuple:
    from simulation import run_monte_carlo, shortage_summary, sensitivity_sweep
    _step(3, "Monte Carlo Disruption Simulation")
    t0 = time.time()
    sim_df = run_monte_carlo(bundles, n_sims=2000)
    summary = shortage_summary(sim_df)
    print(f"  Simulated {2000:,} scenarios in {time.time()-t0:.2f}s")
    print(f"  High shortage risk scenarios : {summary['high_shortage_risk']}")
    print(f"  Critical shortage risk       : {summary['critical_shortage_risk']}")
    print(f"  Max shortage probability     : {summary['max_shortage_probability']:.1%}")
    print(f"  Most vulnerable region       : {summary['most_vulnerable_region']}")
    print(f"  Most vulnerable drug type    : {summary['most_vulnerable_type']}")
    print("\n  Shortage probabilities (top 10):")
    top = sim_df[["region", "drug_type", "shortage_probability", "expected_shortage"]].head(10)
    print(top.to_string(index=False))
    print("\n  Running sensitivity sweep …")
    sweep_df = sensitivity_sweep(bundles, n_sims=200)
    _save(sim_df, "simulation_results.csv")
    _save(sweep_df, "sensitivity_sweep.csv")
    return sim_df, summary, sweep_df


def step4_ml(bundles: dict) -> tuple:
    from ml_model import build_feature_matrix, train_model, predict
    _step(4, "XGBoost Bottleneck Predictor")
    t0 = time.time()
    features    = build_feature_matrix(bundles)
    model, metrics = train_model(features)
    predictions = predict(model, features)
    print(f"  Training complete in {time.time()-t0:.2f}s")
    print(f"  Samples  : {metrics['n_samples']}")
    print(f"  Features : {metrics['n_features']} → {metrics['feature_cols']}")
    print(f"  CV ROC-AUC  : {metrics['cv_roc_auc_mean']} ± {metrics['cv_roc_auc_std']}")
    print(f"  In-sample AUC: {metrics['insample_roc_auc']}")
    print("\n  Feature importances:")
    for feat, imp in metrics["feature_importances"].items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:35s} {bar} {imp:.4f}")
    cr = metrics.get("classification_report", {})
    if "1" in cr:
        print(f"\n  High-risk class precision : {cr['1'].get('precision', 0):.3f}")
        print(f"  High-risk class recall    : {cr['1'].get('recall', 0):.3f}")
        print(f"  High-risk class F1        : {cr['1'].get('f1-score', 0):.3f}")
    _save(predictions, "ml_predictions.csv")
    return model, predictions, metrics


def step5_optimize(bundles: dict) -> tuple:
    from optimizer import build_and_solve
    _step(5, "LP Optimization Engine")
    t0 = time.time()
    alloc_df, total_cost, report = build_and_solve(bundles, msg=False)
    print(f"  Solved in {time.time()-t0:.2f}s")
    print(f"  Status       : {report['status']}")
    print(f"  Total cost   : {report['total_cost']:,.2f}")
    print(f"  Variables    : {report['n_decision_vars']}")
    print(f"  Constraints  : {report['n_constraints']}")
    print(f"  Routes found : {report['n_allocated_routes']}")
    print("\n  Cost by region:")
    for r, c in report.get("cost_by_region", {}).items():
        print(f"    {r:12s}: {c:,.2f}")
    print("\n  Cost by drug type:")
    for d, c in report.get("cost_by_drug_type", {}).items():
        print(f"    {d:20s}: {c:,.2f}")
    if not alloc_df.empty:
        _save(alloc_df, "allocation_results.csv")
    return alloc_df, total_cost, report


def step6_capacity_planner(base_path: Path) -> pd.DataFrame:
    import capacity_planner
    _step(6, "Capacity Planning Optimizer")
    t0 = time.time()
    df = capacity_planner.run(base_path)
    print(f"  Processed {len(df)} regions in {time.time() - t0:.2f}s")
    print(f"  Total upgrade cost: {df['upgrade_cost_delta'].sum():,.2f}")
    return df

def step7_rerouting(base_path: Path) -> pd.DataFrame:
    import rerouting
    _step(7, "Rerouting Recommender")
    t0 = time.time()
    df = rerouting.run(base_path)
    print(f"  Processed {len(df)} alternative routes in {time.time() - t0:.2f}s")
    if not df.empty:
        print(f"  Total potential savings: {df['annual_cost_saving'].sum():,.2f}")
    else:
        print("  No suboptimal routings found.")
    return df

def step8_dashboard(base_path: Path, port: int, debug: bool) -> None:
    from dashboard import create_app
    _step(8, f"Launching Plotly Dash Dashboard on http://0.0.0.0:{port}")
    app, _ = create_app(base_path)
    print(f"\n  Open in browser → http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=debug, host="0.0.0.0", port=port)


# --------------------------------------------------------------------------- #
# Insight report
# --------------------------------------------------------------------------- #

def print_insight_report(
    risk_summary: dict,
    sim_summary: dict,
    ml_metrics: dict,
    opt_report: dict,
    cap_df: pd.DataFrame,
    reroute_df: pd.DataFrame,
) -> None:
    _hr("═")
    print("  PIPELINE INSIGHT REPORT")
    _hr("═")
    print("\n  ── Risk ──────────────────────────────────────────────")
    print(f"  Critical bottleneck routes : {risk_summary['critical_routes']}")
    print(f"  Max risk score             : {risk_summary['max_risk_score']}")
    print(f"  Mean risk score            : {risk_summary['mean_risk_score']}")

    print("\n  ── Disruption Simulation ─────────────────────────────")
    print(f"  High-shortage-risk scenarios  : {sim_summary['high_shortage_risk']}")
    print(f"  Max shortage probability      : {sim_summary['max_shortage_probability']:.1%}")
    print(f"  Most vulnerable region        : {sim_summary['most_vulnerable_region']}")
    print(f"  Worst-case P99 shortage       : {sim_summary['worst_p99_shortage']:,.0f} units")

    print("\n  ── ML Model ──────────────────────────────────────────")
    print(f"  CV ROC-AUC     : {ml_metrics.get('cv_roc_auc_mean','N/A')} ± "
          f"{ml_metrics.get('cv_roc_auc_std','N/A')}")
    print(f"  In-sample AUC  : {ml_metrics.get('insample_roc_auc','N/A')}")
    top_feat = next(iter(ml_metrics.get("feature_importances", {})), "N/A")
    print(f"  Top feature    : {top_feat}")

    print("\n  ── LP Optimizer ──────────────────────────────────────")
    print(f"  Status         : {opt_report.get('status','N/A')}")
    print(f"  Min total cost : {opt_report.get('total_cost', 0):,.2f}")
    print(f"  Routes solved  : {opt_report.get('n_allocated_routes', 0)}")
    
    print("\n  ── Upgrades & Rerouting ──────────────────────────────")
    print(f"  Total upgrade delta cost : {cap_df['upgrade_cost_delta'].sum() if not cap_df.empty else 0:,.2f}")
    print(f"  Rerouting opportunities  : {len(reroute_df)}")
    print(f"  Potential annual savings : {reroute_df['annual_cost_saving'].sum() if not reroute_df.empty else 0:,.2f}")
    _hr("═")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PharmaChain Risk Analytics Pipeline"
    )
    p.add_argument("--no-dash",  action="store_true",
                   help="Run pipeline only, skip dashboard launch")
    p.add_argument("--port",     type=int, default=8050,
                   help="Dash server port (default: 8050)")
    p.add_argument("--debug",    action="store_true",
                   help="Enable Dash debug mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _header("PharmaChain Supply Chain Risk Analytics  |  IEEE Dataport Dataset")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Dataset      : {DATASET_DIR}")
    print()

    t_start = time.time()

    # 1. Load
    bundles = step1_load()

    # 2. Risk
    region_df, granular_df, r_summary = step2_risk(bundles)

    # 3. Simulate
    sim_df, s_summary, sweep_df = step3_simulate(bundles)

    # 4. ML
    model, predictions, ml_metrics = step4_ml(bundles)

    # 5. Optimize
    alloc_df, total_cost, opt_report = step5_optimize(bundles)
    
    # 6. Capacity Planner
    cap_df = step6_capacity_planner(DATASET_DIR)
    
    # 7. Rerouting
    reroute_df = step7_rerouting(DATASET_DIR)

    # Insight report
    print_insight_report(r_summary, s_summary, ml_metrics, opt_report, cap_df, reroute_df)

    elapsed = time.time() - t_start
    print(f"\n  ✓ Full pipeline completed in {elapsed:.1f}s")
    print(f"  ✓ Outputs saved to: {OUTPUT_DIR}")

    if not args.no_dash:
        step8_dashboard(DATASET_DIR, args.port, args.debug)


if __name__ == "__main__":
    main()

