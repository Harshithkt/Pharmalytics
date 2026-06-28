"""
ml_model.py
-----------
XGBoost-based bottleneck predictor for warehouse-to-pharmacy routes.

Features used:
  - avg_cost              : mean transport cost across modes for the region
  - avg_time              : mean delivery time across source zones
  - gap_ratio             : (demand - capacity) / capacity, clipped >= 0
  - cold_violation_ratio  : demandCold / capacityCold
  - distance_mean         : mean inter-cluster distance for the region
  - capacity_utilization  : total_demand / total_capacity

Target: is_high_risk (binary) — 1 if risk_score >= 50th percentile

Outputs:
  - Trained XGBoost classifier (saved as model.json)
  - Feature importance table
  - Cross-validation AUC
  - Prediction DataFrame
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    _USE_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as _GBC
    _USE_XGB = False
    print("[ml_model] XGBoost unavailable (missing libomp?). Falling back to "
          "sklearn GradientBoostingClassifier.  Run `brew install libomp` to fix.")

from data_loader import load_all_workbooks, WorkbookBundle
from risk_engine import compute_risk_scores

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODEL_FILE    = Path(__file__).resolve().parent / "xgb_model.json"
SCALER_PARAMS = Path(__file__).resolve().parent / "scaler_params.json"
RANDOM_SEED   = 42
CV_FOLDS      = 5

XGB_PARAMS = {
    "n_estimators":      200,
    "max_depth":         4,
    "learning_rate":     0.08,
    "subsample":         0.85,
    "colsample_bytree":  0.85,
    "min_child_weight":  2,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.5,
    "eval_metric":       "logloss",
    "random_state":      RANDOM_SEED,
    "n_jobs":            -1,
}


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def _extract_distance_feature(distance_cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return mean distance per region, skipping junk/coordinate sheets."""
    VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
    rows = []
    for frame in distance_cleaned.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if not {"region", "value"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["value"])
        if frame.empty:
            continue
        grp = frame.groupby("region")["value"].mean().reset_index(name="distance_mean")
        rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=["region", "distance_mean"])
    return pd.concat(rows, ignore_index=True).groupby("region")["distance_mean"].mean().reset_index()


def build_feature_matrix(
    bundles: dict[str, WorkbookBundle],
) -> pd.DataFrame:
    """
    Assemble the ML feature matrix.
    Training data expanded via cross-combination of dataset dimensions — all feature values 
    sourced exclusively from the 7 provided IEEE DataPort Excel files. No external or 
    synthetic data introduced.
    """
    import itertools
    from risk_engine import WEIGHTS
    
    try:
        time_df = pd.read_csv("pipeline_outputs/risk_granular.csv")[["region", "source_zone", "mean_time", "norm_time"]]
        cost_df = pd.read_csv("pipeline_outputs/risk_region.csv")[["region", "avg_cost", "norm_cost"]]
        sim_df  = pd.read_csv("pipeline_outputs/simulation_results.csv")[["region", "drug_type", "shortage_probability", "expected_shortage"]]
        alloc_df = pd.read_csv("pipeline_outputs/allocation_results.csv")
    except FileNotFoundError:
        # Fallback if pipeline outputs not present yet
        from risk_engine import compute_risk_scores
        _, granular_df = compute_risk_scores(bundles)
        dist_df = _extract_distance_feature(bundles["distance"].cleaned)
        features = granular_df.merge(dist_df, on="region", how="left")
        features["distance_mean"] = features["distance_mean"].fillna(features["distance_mean"].median())
        features["capacity_utilization"] = (features["total_demand"] / features["total_cap"].replace(0, np.nan)).fillna(0).clip(upper=5)
        features["is_high_risk"] = (features["risk_score"] >= features["risk_score"].median()).astype(int)
        return features.reset_index(drop=True)

    dist_df = _extract_distance_feature(bundles["distance"].cleaned)
    dist_df["distance_mean"] = dist_df["distance_mean"].fillna(dist_df["distance_mean"].median())

    demand_raw = [v for v in bundles["demand"].cleaned.values() if isinstance(v, pd.DataFrame)]
    demand_df = pd.concat(demand_raw, ignore_index=True) if demand_raw else pd.DataFrame()
    
    def map_dt(dt):
        dt_lower = str(dt).lower()
        if "critical" in dt_lower and "cold" in dt_lower: return "critical_cold"
        if "cold" in dt_lower: return "cold"
        return "normal"
        
    demand_df["dt"] = demand_df["demand_type"].apply(map_dt)
    demand_final = demand_df.groupby(["region", "dt"])["demand"].sum().reset_index()
    total_demand = demand_final.groupby("region")["demand"].sum().reset_index()
    total_demand["dt"] = "total"
    demand_final = pd.concat([demand_final, total_demand], ignore_index=True)

    cap_raw = [v for v in bundles["capacity"].cleaned.values() if isinstance(v, pd.DataFrame)]
    cap_df = pd.concat(cap_raw, ignore_index=True) if cap_raw else pd.DataFrame()
    
    def map_cap(m):
        m_lower = str(m).lower()
        if "crit" in m_lower and "cold" in m_lower: return "critical_cold"
        if "cold" in m_lower: return "cold"
        if "norm" in m_lower: return "normal"
        return "total"
        
    cap_df["dt"] = cap_df["metric"].apply(map_cap)
    cap_final = cap_df.groupby(["region", "configuration", "dt"])["value"].max().reset_index()

    regions = time_df["region"].unique()
    zones = time_df["source_zone"].unique()
    dts = ["total", "normal", "cold", "critical_cold"]
    configs = cap_final["configuration"].unique()
    
    rows = [{"region": r, "source_zone": z, "drug_type": dt, "configuration": c}
            for r, z, dt, c in itertools.product(regions, zones, dts, configs)]
    df = pd.DataFrame(rows)

    df = df.merge(time_df, on=["region", "source_zone"], how="left")
    df = df.merge(cost_df, on="region", how="left")
    df = df.merge(dist_df, on="region", how="left")
    df = df.merge(demand_final, left_on=["region", "drug_type"], right_on=["region", "dt"], how="left").drop(columns=["dt"])
    df = df.merge(cap_final, left_on=["region", "configuration", "drug_type"], right_on=["region", "configuration", "dt"], how="left").drop(columns=["dt"])
    df = df.merge(sim_df, on=["region", "drug_type"], how="left")
    
    alloc_grp = alloc_df.groupby(["region", "drug_type"])["route_cost"].mean().reset_index()
    df = df.merge(alloc_grp, on=["region", "drug_type"], how="left")

    df["demand"] = df["demand"].fillna(0)
    df["value"] = df["value"].fillna(0)
    
    df["gap_ratio"] = ((df["demand"] - df["value"]).clip(lower=0) / df["value"].replace(0, np.nan)).fillna(0)
    
    df["cold_violation_ratio"] = 0.0
    cold_mask = df["drug_type"].isin(["cold", "critical_cold"])
    df.loc[cold_mask, "cold_violation_ratio"] = (df.loc[cold_mask, "demand"] / df.loc[cold_mask, "value"].replace(0, np.nan)).fillna(0).clip(upper=2)
    
    def _minmax(s):
        lo, hi = s.min(), s.max()
        if hi == lo: return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)
        
    df["norm_gap"] = _minmax(df["gap_ratio"])
    df["norm_cold"] = _minmax(df["cold_violation_ratio"])
    df["capacity_utilization"] = (df["demand"] / df["value"].replace(0, np.nan)).fillna(0).clip(upper=5)
    
    df["risk_score"] = (
        WEIGHTS["cost"] * df["norm_cost"] * 100 +
        WEIGHTS["time"] * df["norm_time"] * 100 +
        WEIGHTS["gap"]  * df["norm_gap"]  * 100 +
        WEIGHTS["cold"] * df["norm_cold"] * 100
    )
    
    median_risk = df["risk_score"].median()
    df["is_high_risk"] = (df["risk_score"] >= median_risk).astype(int)

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

FEATURE_COLS = [
    "norm_cost",
    "norm_time",
    "norm_gap",
    "norm_cold",
    "distance_mean",
    "capacity_utilization",
]

def train_model(
    features: pd.DataFrame,
) -> tuple:
    """
    Train XGBoost and Random Forest classifiers, compute cross-validation metrics,
    compare them, and return the best model.

    Returns:
        model   – trained best classifier
        metrics – dict with metrics and feature importances for both models
    """
    available = [c for c in FEATURE_COLS if c in features.columns]
    X = features[available].fillna(0).values
    y = features["is_high_risk"].values

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    models = {}
    if _USE_XGB:
        models["XGBoost"] = xgb.XGBClassifier(**XGB_PARAMS)
    else:
        models["GradientBoosting"] = _GBC(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.85, random_state=RANDOM_SEED)
    
    models["Random Forest"] = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1)

    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]
    
    metrics_list = []
    trained_models = {}
    importances_dict = {}
    
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
        model.fit(X, y)
        trained_models[name] = model
        
        imp = model.feature_importances_
        sorted_imp = dict(sorted(zip(available, imp.tolist()), key=lambda x: x[1], reverse=True))
        importances_dict[name] = sorted_imp
        
        metrics_list.append({
            "model": name,
            "accuracy": round(float(cv_res["test_accuracy"].mean()), 4),
            "precision": round(float(cv_res["test_precision_weighted"].mean()), 4),
            "recall": round(float(cv_res["test_recall_weighted"].mean()), 4),
            "f1_weighted": round(float(cv_res["test_f1_weighted"].mean()), 4),
            "roc_auc_mean": round(float(cv_res["test_roc_auc"].mean()), 4),
            "roc_auc_std": round(float(cv_res["test_roc_auc"].std()), 4),
        })

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv("pipeline_outputs/ml_metrics_comparison.csv", index=False)
    
    best_model_name = df_metrics.loc[df_metrics["roc_auc_mean"].idxmax(), "model"]
    best_model = trained_models[best_model_name]
    
    if best_model_name == "XGBoost" and _USE_XGB:
        best_model.save_model(str(MODEL_FILE))
    else:
        # If Random Forest or GradientBoosting is best, we would serialize it using pickle or joblib
        # For simplicity, we can just save a dummy if we expect XGBoost, but we will use the in-memory returned model.
        import joblib
        joblib.dump(best_model, str(MODEL_FILE.with_suffix('.joblib')))

    y_pred = best_model.predict(X)

    metrics = {
        "best_model": best_model_name,
        "importances_dict": importances_dict,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "n_samples": len(y),
        "n_features": len(available),
        "feature_cols": available,
        "cv_roc_auc_mean": df_metrics.loc[df_metrics["roc_auc_mean"].idxmax(), "roc_auc_mean"],
        "accuracy": df_metrics.loc[df_metrics["roc_auc_mean"].idxmax(), "accuracy"],
        "xgb_auc": df_metrics.loc[df_metrics["model"] == "XGBoost", "roc_auc_mean"].values[0] if "XGBoost" in df_metrics["model"].values else 0,
        "rf_auc": df_metrics.loc[df_metrics["model"] == "Random Forest", "roc_auc_mean"].values[0] if "Random Forest" in df_metrics["model"].values else 0,
    }

    return best_model, metrics


def predict(
    model,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions on the feature DataFrame.

    Returns the feature DataFrame with added columns:
        predicted_high_risk, risk_probability
    """
    available  = [c for c in FEATURE_COLS if c in features.columns]
    X          = features[available].fillna(0).values
    preds      = model.predict(X)
    proba      = model.predict_proba(X)[:, 1]

    result = features.copy()
    result["predicted_high_risk"] = preds
    result["risk_probability"]    = proba.round(4)
    return result


def load_model() -> xgb.XGBClassifier | None:
    """Load a previously saved model from disk, or None if not found."""
    if MODEL_FILE.exists():
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_FILE))
        return m
    return None


# --------------------------------------------------------------------------- #
# Public run entry point
# --------------------------------------------------------------------------- #

def run(
    base_path: str | Path | None = None,
    retrain: bool = True,
) -> tuple[xgb.XGBClassifier, pd.DataFrame, dict[str, Any]]:
    """
    Full ML pipeline run.

    Returns:
        model       – trained XGBClassifier
        predictions – DataFrame with features + predictions
        metrics     – training / evaluation metrics
    """
    bundles  = load_all_workbooks(base_path)
    features = build_feature_matrix(bundles)

    if retrain or not MODEL_FILE.exists():
        model, metrics = train_model(features)
    else:
        model   = load_model()
        metrics = {"note": "loaded existing model from disk"}

    predictions = predict(model, features)
    return model, predictions, metrics


if __name__ == "__main__":
    model, predictions, metrics = run()
    print("\n=== ML BOTTLENECK PREDICTOR ===")
    print(f"  Best Model: {metrics.get('best_model', 'N/A')}")
    print("\n  Predictions (top 10 high-risk routes):")
    top = predictions.nlargest(10, "risk_probability")[
        ["region", "source_zone", "risk_score", "risk_probability", "predicted_high_risk"]
    ]
    print(top.to_string(index=False))
