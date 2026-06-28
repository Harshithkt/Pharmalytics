# PharmaChain Risk Analytics: End-to-End Solution Explainer

This document outlines the 7 analytical modules and the interactive dashboard that make up our comprehensive pharmaceutical supply chain solution. It is designed to be read and understood by the team in under 30 minutes, explaining exactly what each module does, what data it relies on, the methodology applied, and the verified results produced.

---

## 1. Data Loader & Validation (`data_loader.py`)

**What problem it solves:**
Before any analysis can happen, raw data from various sources must be ingested, cleaned, and standardized. Inconsistent or missing data can silently corrupt downstream models.

**What data it uses:**
The 7 core Excel files from the IEEE Dataport dataset: Geolocation, Demand, Distance, Transport Costs, Warehouse Capacity, Delivery Times, and Facility Configurations.

**What method it applies:**
It uses Pandas to systematically load every sheet, validate column names against expected schemas, convert data types (e.g., ensuring numeric values), and handle missing entries securely.

**Verified Output:**
Produces a unified, structured `WorkbookBundle` object that feeds identical, clean data to all subsequent modules.

---

## 2. Risk Scoring Engine (`risk_engine.py`)

**What problem it solves:**
Supply chains are complex, making it difficult to identify which regions and routes are most vulnerable to failure. This module flags high-risk points *before* they become emergencies.

**What data it uses:**
Transport costs, delivery times, demand-capacity gaps, and cold-chain specific requirements.

**What method it applies:**
It normalizes these metrics (using Min-Max scaling) and applies a weighted composite scoring formula: Cost (15%), Time (25%), Supply Gap (30%), and Cold-Chain Violation (30%). It evaluates all possible warehouse-to-pharmacy routes and assigns a risk score out of 100.

**Verified Output:**
- Evaluated 32 distinct supply routes.
- The highest risk region identified is **Giresun** with a max risk score of **78.82**.
- 1 route was flagged as "CRITICAL", requiring immediate attention.

---

## 3. Monte Carlo Disruption Simulation (`simulation.py`)

**What problem it solves:**
Standard analytics assume normal operations, but real-world supply chains face unexpected shocks (e.g., pandemics, natural disasters). This module tests the network's resilience under extreme stress.

**What data it uses:**
Baseline demand per region and baseline warehouse capacities.

**What method it applies:**
It runs 2,000 independent Monte Carlo simulated scenarios. In each scenario, demand randomly surges (up to 300%) and warehouse capacity randomly fails (drops by up to 50%). It tracks how often the supply chain fails to meet critical medication demand under these stressed conditions.

**Verified Output:**
- Across 2,000 extreme scenarios, the maximum shortage probability peaked at **9.2%**.
- Under severe disruption (the 99th percentile worst-case scenario), the supply chain could fall short by up to **404,596 units** of medication, primarily affecting the Giresun region.

---

## 4. ML Bottleneck Predictor (`ml_model.py`)

**What problem it solves:**
Understanding *that* a route is high-risk is helpful, but understanding *why* it is high-risk allows us to fix it. This module identifies the hidden drivers of supply chain bottlenecks.

**What data it uses:**
The engineered features from the Risk Engine (normalized costs, times, supply gaps, distances, and capacity utilization) and historical risk labels.

**What method it applies:**
It trains and rigorously compares two Machine Learning models (XGBoost and Random Forest) using 5-Fold Cross-Validation. The models learn the complex, non-linear relationships between supply chain variables to predict high-risk bottlenecks.

**Verified Output:**
- **XGBoost** outperformed Random Forest, achieving a phenomenal ROC-AUC score of **0.9982** (vs RF's 0.9895) and an accuracy of **97.7%**.
- The AI explicitly revealed that **Capacity Utilization** (responsible for ~27% of the model's decision weight) is the #1 leading indicator of supply chain failure, followed by high transportation costs.

---

## 5. LP Optimization Engine (`optimizer.py`)

**What problem it solves:**
Given finite resources, how do we distribute pharmaceutical supplies across the region to minimize transportation costs without failing to meet medication demand?

**What data it uses:**
Regional demand for three drug types (Normal, Cold, Critical-Cold), warehouse capacities, and transport cost matrices.

**What method it applies:**
It formulates a massive mathematical Linear Programming (LP) problem using the PuLP library. It sets constraints (e.g., you cannot ship more drugs than a warehouse holds; every region's demand must be met) and mathematically calculates the absolute cheapest routing strategy that satisfies all constraints.

**Verified Output:**
- Successfully solved the allocation across 60 active routes.
- Determined the mathematical minimum total cost to supply the entire region is **₺39,123,730.20**.

---

## 6. Capacity Planning Optimizer (`capacity_plan.py`)

**What problem it solves:**
If a region's warehouse is too small, it forces expensive shipments from distant warehouses. This module checks if it is cheaper to upgrade a local warehouse rather than continuously paying high transport fees.

**What data it uses:**
Current capacity utilization, regional demand, and the optimal LP costs.

**What method it applies:**
It iteratively simulates upgrading warehouse capacities to their maximum physical configurations and recalculates the LP transport costs. If the transport savings exceed the cost of the upgrade, it flags it as a recommended action.

**Verified Output:**
- The pipeline evaluated 4 regions for upgrades.
- It found that the current capacity distribution is already well-calibrated; no immediate physical capacity upgrades are required (Total upgrade delta cost: ₺0.00).

---

## 7. Rerouting Recommender (`reroute.py`)

**What problem it solves:**
Sometimes, a warehouse is technically capable of supplying a region, but a neighboring warehouse could do it faster and cheaper. This module identifies suboptimal routing configurations.

**What data it uses:**
Inter-cluster distances, delivery times, transport costs, and current LP allocations.

**What method it applies:**
It searches for "cross-hauling" or inefficient long-distance assignments. It compares the current assignment against the theoretical cheapest neighboring warehouse. If a switch generates significant savings, it is proposed as an action.

**Verified Output:**
- The current routing logic is optimal based on the provided dataset.
- 0 suboptimal routings were found, confirming the supply chain is currently operating at maximum cost-efficiency.

---

## 8. Interactive Command Center (`dashboard.py`)

**What problem it solves:**
Raw CSV outputs and console logs are useless to a business executive or supply chain manager during a crisis.

**What data it uses:**
The direct outputs of all 7 pipeline modules (the `pipeline_outputs/` directory).

**What method it applies:**
It builds a live, interactive web application using Plotly Dash. It features a geospatial Folium map, interactive heatmaps, and dynamic sliders allowing users to manually simulate demand surges in real-time. 

**Verified Output:**
- At the top of the dashboard sits the **3-Second Executive Decision Banner**, dynamically pulling the most critical metrics: the highest risk region (Giresun), the critical shortage probability (9.2%), and the top recommended action.
- Displays comprehensive, visually striking KPI cards, including the winning AI model's accuracy (97.7%) and AUC (0.9982), ensuring stakeholders can trust the system's recommendations at a glance.
