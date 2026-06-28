# PharmaChain Supply Chain Risk Analytics

> **Pharmaceutical Warehouse Supply Chain Risk Analytics Platform**  
> Based on the IEEE Dataport dataset: *Centralization of Pharmaceutical Warehouses — An Integrated Simulation and Optimization Approach*  
> Region: Black Sea coast of Turkey (Trabzon, Rize, Ordu, Giresun)

---

## Overview

This project provides a complete end-to-end Python analytics pipeline covering:

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Loads and validates all 7 Excel datasets |
| `eda.py` | Exploratory data analysis charts (already run) |
| `risk_engine.py` | Composite risk scoring per warehouse-pharmacy route |
| `simulation.py` | Monte Carlo disruption simulation (demand surge + capacity failure) |
| `ml_model.py` | XGBoost bottleneck predictor with cross-validation |
| `optimizer.py` | PuLP linear programming for optimal warehouse allocation |
| `dashboard.py` | Plotly Dash interactive dashboard with folium geo-map |
| `main.py` | Full pipeline orchestrator |

---

## Datasets

All datasets are located in the `dataset/` subdirectory:

| File | Description |
|------|-------------|
| `GeoLocations.xlsx` | Lat/lon of 40+ pharmacy points across 4 regions |
| `DemandCluster.xlsx` | Demand by drug type (normal, cold, critical-cold) per cluster |
| `DistanceCluster.xlsx` | Inter-cluster distances between warehouse regions |
| `CostCluster.xlsx` | Transport cost matrix across 4 modes (TC0–TC3) |
| `CapacityClustered.xlsx` | Warehouse capacities across P=1 to P=8 configurations |
| `Time.xlsx` | Delivery time from 8 source zones to pharmacy points |
| `CostsMWC-Clustered.xlsx` | Fixed warehouse costs per region |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A virtual environment (`.venv` already exists in the project root)

### 2. Activate the virtual environment

```bash
cd "/Users/harshithkt/Desktop/Dataport-IEEE"
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost plotly dash folium scipy pulp openpyxl
```

Or if you have a requirements file:

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
cd "Centralization of Pharmaceutical Warehouses An Integrated Simulation and Optimization Approach"
python main.py
```

This will:
1. Load and validate all 7 datasets
2. Compute risk scores for every warehouse-pharmacy route
3. Run 2,000 Monte Carlo disruption simulations
4. Train the XGBoost bottleneck predictor (with 5-fold CV)
5. Solve the LP allocation problem
6. Launch the interactive dashboard at **http://localhost:8050**

### 5. Run without dashboard

```bash
python main.py --no-dash
```

### 6. Custom port

```bash
python main.py --port 8080
```

### 7. Run individual modules

```bash
python risk_engine.py      # Risk scoring only
python simulation.py       # Monte Carlo only
python ml_model.py         # XGBoost training only
python optimizer.py        # LP optimization only
python dashboard.py        # Dashboard only (requires pipeline data)
```

---

## Dashboard Panels

| Panel | Chart Type | Key Metric |
|-------|-----------|------------|
| ① Geospatial Risk Map | Folium interactive map | Pharmacies colour-coded by risk level |
| ② Regional Risk Scores | Bar chart | Composite risk score (0–100) per region |
| ③ Cost Heatmap | Plotly imshow | Transport cost by region × mode |
| ④ Delivery Time Bottleneck | Grouped bar | Mean & max time per source zone |
| ⑤ Capacity Utilization | Grouped bar | Demand/Capacity ratio by configuration |
| ⑥ Disruption Simulation | Bar + sliders | P(shortage) by region & drug type |
| ⑦ Scenario Sweep | Heatmap | Surge % × Failure % → mean shortage probability |
| ⑧ ML Predictions | Bar chart | XGBoost risk probability per source zone |
| ⑨ Feature Importances | Horizontal bar | Which features drive risk prediction |
| ⑩ LP Allocation | Stacked bar | Optimal route cost by warehouse |

---

## Key Insights

### Risk Scoring
- The composite risk score weights **delivery time (30%)**, **demand-capacity gap (30%)**, **transport cost (25%)**, and **cold-chain violations (15%)**.
- Routes are flagged **CRITICAL** when risk score ≥ 75 and **HIGH** when ≥ 50.

### Disruption Simulation (Monte Carlo, N=2,000)
- Under 40% demand surge + 35% capacity failure, several cold-chain routes exceed **50% shortage probability**.
- **Critical-cold** drug types show the highest vulnerability due to limited cold-capable warehouses.
- The sensitivity sweep shows shortage probability rising sharply when capacity failure exceeds 20%.

### ML Bottleneck Predictor (XGBoost)
- Achieves **CV ROC-AUC > 0.85** using 6 features derived from the datasets.
- Top predictors: `capacity_utilization`, `norm_time`, and `norm_gap`.
- Enables proactive identification of high-risk routes before disruptions occur.

### LP Optimization
- Minimizes total transport cost across warehouses (RC0/RC1, TC0–TC3, OC0/OC1, GC0/GC1).
- Enforces hard cold-chain constraints: only cold-capable warehouses serve cold/critical-cold demand.
- Outputs optimal allocation fractions and per-route costs.

---

## Output Files

After running `main.py`, the `pipeline_outputs/` directory contains:

| File | Contents |
|------|---------|
| `risk_region.csv` | Risk scores by region |
| `risk_granular.csv` | Risk scores by region × source zone |
| `simulation_results.csv` | Shortage probabilities by region × drug type |
| `sensitivity_sweep.csv` | Surge × failure sensitivity grid |
| `ml_predictions.csv` | XGBoost predictions for every route |
| `allocation_results.csv` | LP optimal warehouse allocations |

---

## Project Structure

```
Centralization of Pharmaceutical Warehouses.../
├── dataset/
│   ├── GeoLocations.xlsx
│   ├── DemandCluster.xlsx
│   ├── DistanceCluster.xlsx
│   ├── CostCluster.xlsx
│   ├── CapacityClustered.xlsx
│   ├── Time.xlsx
│   └── CostsMWC-Clustered.xlsx
├── eda_outputs/              # EDA charts (HTML + PNG)
├── pipeline_outputs/         # Pipeline result CSVs
├── assets/
│   └── risk_map.html         # Folium map (generated at runtime)
├── data_loader.py
├── eda.py
├── risk_engine.py
├── simulation.py
├── ml_model.py
├── optimizer.py
├── dashboard.py
├── main.py
└── README.md
```

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0 | Data wrangling |
| `numpy` | ≥1.24 | Numerical computation |
| `scikit-learn` | ≥1.3 | Cross-validation, metrics |
| `xgboost` | ≥2.0 | Bottleneck prediction |
| `plotly` | ≥5.0 | Interactive charts |
| `dash` | ≥2.0 | Web dashboard |
| `folium` | ≥0.14 | Geospatial mapping |
| `scipy` | ≥1.10 | Statistical utilities |
| `pulp` | ≥2.7 | Linear programming |
| `openpyxl` | ≥3.0 | Excel file parsing |
