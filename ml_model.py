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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
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


def _make_model():
    """Return an XGBClassifier or fallback GradientBoostingClassifier."""
    if _USE_XGB:
        return xgb.XGBClassifier(**XGB_PARAMS)
    return _GBC(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.85, random_state=RANDOM_SEED,
    )


def train_model(
    features: pd.DataFrame,
) -> tuple:
    """
    Train classifier and compute cross-validation metrics.

    Returns:
        model   – trained classifier
        metrics – dict with cv_auc, feature_importances, etc.
    """
    available = [c for c in FEATURE_COLS if c in features.columns]
    X = features[available].fillna(0).values
    y = features["is_high_risk"].values

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    model_cv = _make_model()
    cv_scores = cross_val_score(model_cv, X, y, cv=cv, scoring="roc_auc")

    # Final model on full data
    model = _make_model()
    if _USE_XGB:
        model.fit(X, y, eval_set=[(X, y)], verbose=False)
        model.save_model(str(MODEL_FILE))
        importances = dict(zip(available, model.feature_importances_.tolist()))
    else:
        model.fit(X, y)
        importances = dict(zip(available, model.feature_importances_.tolist()))

    sorted_imp  = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # In-sample classification report
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "cv_roc_auc_mean":    round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std":     round(float(cv_scores.std()), 4),
        "cv_roc_auc_scores":  [round(s, 4) for s in cv_scores.tolist()],
        "insample_roc_auc":   round(float(roc_auc_score(y, y_proba)), 4),
        "feature_importances": sorted_imp,
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "confusion_matrix":   confusion_matrix(y, y_pred).tolist(),
        "n_samples":          len(y),
        "n_features":         len(available),
        "feature_cols":       available,
        "backend":            "xgboost" if _USE_XGB else "sklearn-gbm",
    }

    return model, metrics


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
    print("\n=== XGBoost BOTTLENECK PREDICTOR ===")
    print(f"  CV ROC-AUC:  {metrics.get('cv_roc_auc_mean', 'N/A')} "
          f"± {metrics.get('cv_roc_auc_std', 'N/A')}")
    print(f"  In-sample AUC: {metrics.get('insample_roc_auc', 'N/A')}")
    print("\n  Feature importances:")
    for feat, imp in metrics.get("feature_importances", {}).items():
        print(f"    {feat:30s}: {imp:.4f}")
    print("\n  Predictions (top 10 high-risk routes):")
    top = predictions.nlargest(10, "risk_probability")[
        ["region", "source_zone", "risk_score", "risk_probability", "predicted_high_risk"]
    ]
    print(top.to_string(index=False))
