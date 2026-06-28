# Technical Architecture & Solution Explainer

This document serves as the deeply technical documentation for the PharmaChain Risk Analytics pipeline. It details the underlying mathematics, algorithmic methodologies, architectural decisions, and precise data flows of the end-to-end Python solution.

## Architecture Overview

The system operates as a sequential Directed Acyclic Graph (DAG) orchestrated by `main.py`. The pipeline processes 7 raw datasets through 7 independent, decoupled analytical modules. Each module ingests data, performs heavy computation, and outputs deterministic CSV artifacts into the `pipeline_outputs/` directory.

This architecture ensures zero state leakage between steps and allows the final Plotly Dash application to render strictly from cached pipeline outputs, achieving sub-second load times.

---

## 1. Data Ingestion & Sanitization (`data_loader.py`)

**Algorithmic Approach:**
- **Vectorized Ingestion:** Leverages `pandas.read_excel()` to load the 7 IEEE datasets simultaneously.
- **Data Normalization:** Translates inconsistent schema definitions (e.g., Turkish sheet names, variable column lengths) into a strict `WorkbookBundle` dataclass.
- **Zero-Imputation Strategy:** Any missing values or zero-demand nodes (e.g., Trabzon Demand = 0) are deliberately handled rather than dropped, ensuring dimension parity across matrix multiplications downstream.

---

## 2. Risk Scoring Engine (`risk_engine.py`)

**Mathematical Formulation:**
The Risk Engine assigns a continuous risk score $R \in [0, 100]$ to every possible warehouse-pharmacy transport route. The score is a weighted linear combination of Min-Max normalized metrics:

$$ R = (w_1 \cdot \bar{C}) + (w_2 \cdot \bar{T}) + (w_3 \cdot \bar{G}) + (w_4 \cdot \bar{V}) $$

Where the engineered features are:
- **Normalized Cost ($\bar{C}$):** $w_1 = 15\%$. Min-Max scaled transport cost per unit.
- **Normalized Time ($\bar{T}$):** $w_2 = 25\%$. Min-Max scaled delivery time from source zone to region.
- **Supply Gap Ratio ($\bar{G}$):** $w_3 = 30\%$. Calculated as $\max(0, Demand - Capacity) / Capacity$.
- **Cold-Chain Violation ($\bar{V}$):** $w_4 = 30\%$. A boolean constraint penalty triggered if cold or critical-cold demand is forced into a non-cold-capable warehouse (TC0/TC1/TC2 limits).

**Current Output:**
Evaluates 32 regional routes. Giresun holds the maximum global risk score (78.82) classifying it as the sole `CRITICAL` chokepoint.

---

## 3. Stochastic Monte Carlo Simulation (`simulation.py`)

**Stochastic Framework:**
Traditional static analysis fails to capture catastrophic network shocks. This module applies a dual-variable Monte Carlo simulation simulating 2,000 independent network states.

- **Demand Surge Function:** $D_{sim} = D_{base} \times (1 + \text{Uniform}(0.0, 3.0))$
- **Capacity Failure Function:** $C_{sim} = C_{base} \times (1 - \text{Uniform}(0.0, 0.5))$

**Simulation Loop:**
For every iteration $i$ across 2,000 simulations, the engine calculates the net shortage:
$S_i = \max(0, D_{sim} - C_{sim})$

It aggregates the $S_i$ distribution to compute:
1. **Shortage Probability:** The frequency where $S_i > 0$. Max observed is 9.2%.
2. **$P_{99}$ Worst-Case Shortage:** The 99th percentile of the $S_i$ distribution, culminating in 404,596 missing units for Giresun.

---

## 4. ML Bottleneck Predictor (`ml_model.py`)

**Predictive Engine:**
Predicts the binary risk classification (`is_high_risk`) of any given route based on underlying physical traits.

- **Dataset Augmentation:** The 32 base points are deterministically expanded to 640 training samples via a cross-join of (Region $\times$ Source Zone $\times$ Drug Type $\times$ Configuration $P_1-P_8$). All features are derived strictly from the IEEE dataset.
- **Algorithm Comparison:** The system trains both `XGBClassifier` (Gradient Boosted Trees) and `RandomForestClassifier` (Bagged Trees).
- **Cross-Validation Validation:** 5-Fold Stratified CV prevents overfitting.

**Model Evaluation Metrics:**
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 97.66% | 97.69% | 97.66% | 97.66% | **0.9982** |
| Random Forest | 93.75% | 94.00% | 93.75% | 93.74% | 0.9895 |

The system dynamically selects **XGBoost** as the champion model. Feature Importance analysis extracts `capacity_utilization` as the primary node-splitting feature (0.270 Gini importance).

---

## 5. Linear Programming Optimizer (`optimizer.py`)

**Mathematical Optimization:**
Formulated using `PuLP`, solving the classic Transportation Problem via the Simplex algorithm.

**Objective Function:**
Minimize Total Cost: $\min \sum (Cost_{r, c, d} \times X_{r, c, d})$
Where $X$ is the quantity of drug type $d$ allocated to region $r$ via transport configuration $c$.

**Hard Constraints:**
1. **Demand Satisfaction:** $\sum_c X_{r, c, d} = Demand_{r, d}$
2. **Capacity Ceiling:** $\sum_d X_{r, c, d} \leq Capacity_{r, c}$
3. **Cold-Chain Isolation:** $X_{r, c, cold} = 0$ if $c$ lacks cold-chain specs.

**Convergence:**
Solved perfectly across 60 active routes, establishing the absolute mathematical floor cost at ₺39,123,730.20.

---

## 6. Capacity Upgrader & Rerouting (`capacity_plan.py` / `reroute.py`)

**Optimization Delta Computation:**
Evaluates marginal utility. It iterates warehouse configurations $P \in [1, 8]$ and re-solves the LP constraint matrix. 
If $\Delta \text{Transport Savings} > \text{Upgrade Cost}$, an upgrade is triggered.

**Cross-Hauling Check:**
Scans for $Cost(W_A \rightarrow R_A) > Cost(W_B \rightarrow R_A) + Transit(W_B, R_A)$. 
The engine confirms zero suboptimal assignments exist currently. Annual savings opportunity is precisely 0.00, proving current LP allocation is globally optimal.

---

## 7. Dash Command Center (`dashboard.py`)

**Frontend Engineering:**
- **Framework:** Plotly Dash operating on a Flask WSGI server.
- **Data Hydration:** Bypasses live computation by hydrating purely from the `pipeline_outputs/` CSVs.
- **Layout Architecture:** Implements a CSS Flexbox grid system. 
- **Dynamic Injection:** The 'Executive Decision Banner' evaluates python conditionals at runtime to inject the highest threat vector (e.g., `Giresun 78.8`), the worst-case shortage (`9.2%`), and the dynamic ML model victor (`XGBoost`).
- **Aesthetics:** Completely custom-styled CSS bypassing standard Bootstrap. Implements FontAwesome scalable vector icons and a meticulously designed dark-mode color palette (`#0F172A` backgrounds, `#EF4444` critical alerts).
