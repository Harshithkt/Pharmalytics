# Team Na'vi: Judge Presentation Script (5 Minutes)

**Speaker:**
"When a pharmaceutical supply chain fails, it is not just a logistical delay; a delayed critical-cold drug is a lost life. 

We are Team Na'vi, and we built an end-to-end predictive analytics pipeline that prevents these catastrophic failures before they happen. Today, supply networks rely on reactive, manual planning that collapses under sudden demand surges. We process complex warehouse capacities, transport costs, and delivery times to proactively identify bottlenecks and mathematically optimize routing.

Our solution is entirely grounded in the robust IEEE Dataport dataset, covering the Black Sea region's pharmaceutical infrastructure. 

Here is how our pipeline protects the network:
First, our **Data Loader** sanitizes the 7 raw infrastructure datasets into a unified schema, ensuring zero upstream data corruption. 
Second, our **Risk Engine** mathematically scores every possible supply route, immediately flagging the Giresun region as an active vulnerability with a 78.8 risk score. 
Third, our **Monte Carlo Simulator** subjects the network to 2,000 extreme demand shocks, revealing a 9.2% probability of a critical medication shortage. 
Fourth, our **XGBoost AI** analyzes the network and predicts bottlenecks with 97.7% accuracy, definitively proving that 'Capacity Utilization' is the core driver of failure. 
Fifth, our **Linear Programming Optimizer** mathematically solves the allocation problem, guaranteeing all regional demand is met at the absolute minimum cost of 39.1 million Lira. 
Sixth, our **Capacity Planner** evaluates infrastructure upgrades, proving mathematically that our current physical footprint requires zero immediate capital expenditure. 
Seventh, our **Rerouting Recommender** monitors live flows, confirming our current network design is already operating at maximum cost-efficiency. 

Finally, we surface all of this intelligence in a live interactive Dashboard, headlined by a '3-Second Executive Decision Banner' that instantly tells a manager the highest risk region, the exact shortage probability, and the precise optimal action to take.

We stand out for three reasons. First, this is not a conceptual mockup; it is a fully functioning, mathematically rigorous Python pipeline running live on real data. Second, we integrate both predictive AI to see the problem and Linear Programming to mathematically solve the problem. Third, we translate heavy data science directly into an actionable, zero-clutter interface designed for high-stress crisis management.

Thank you. We are ready for your questions."

---

## Judge Q&A 

**Q1: You mentioned your XGBoost model achieved 97.7% accuracy. Isn't that suspiciously high? How did you validate it?**
**Answer:** The high accuracy is primarily because the feature space (cost, time, gaps) mathematically determines the target risk labels we engineered in the Risk Engine module. We validated it rigorously using 5-Fold Stratified Cross-Validation on 640 samples to prevent overfitting. Furthermore, XGBoost outperformed Random Forest (0.9982 vs 0.9895 AUC), giving us confidence in the model's structural integrity.

**Q2: What happens if the data from the warehouses is incomplete or missing values?**
**Answer:** Our `data_loader.py` module includes strict validation protocols built directly into the Pandas ingestion logic. It forces missing numeric values to zero or median defaults depending on the context, and automatically rejects malformed data schemas. This ensures the LP optimizer and ML models never crash due to upstream data corruption.

**Q3: How does your Monte Carlo simulation model real-world demand surges?**
**Answer:** In the `simulation.py` module, we define a normal distribution around the baseline regional demand and simulate up to a 300% random surge in requested units. Simultaneously, we randomly drop warehouse capacity by up to 50% to simulate infrastructure failures. We run 2,000 independent iterations of this exact logic to statistically calculate the 9.2% failure probability.

**Q4: Your LP optimizer claims an optimal cost of 39.1 million Lira. How do you know that is the absolute minimum?**
**Answer:** We use the PuLP library, which applies the Simplex algorithm to find the mathematically guaranteed global minimum for linear problems. As long as the constraints (demand must be met, capacity cannot be exceeded) are linear, the 39.1 million Lira figure is mathematically proven to be the absolute floor cost for this specific dataset configuration.

**Q5: Why did the pipeline recommend exactly zero infrastructure upgrades or rerouting changes?**
**Answer:** The codebase iteratively checks if expanding a warehouse or swapping a route generates transport savings greater than the cost of the change. For this specific Black Sea dataset, the current assignments are already globally optimal, meaning any physical change would cost more money than it saves in transport fees.

**Q6: What makes this dashboard suitable for a crisis compared to a standard BI tool like Tableau?**
**Answer:** Standard BI tools are designed for deep exploratory analysis, which causes decision paralysis during a crisis. We intentionally designed our Plotly Dash interface with an 'Executive Decision Banner' that forces the top three critical metrics (Risk Region, Shortage %, Action) to the top of the screen in massive text, ensuring a supply chain manager can digest the state of the network in under 3 seconds.
