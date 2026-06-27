import pandas as pd
import numpy as np
from risk_engine import WEIGHTS, RISK_CRITICAL_THRESHOLD
from data_loader import load_all_workbooks
import itertools

bundles = load_all_workbooks()

# 1. Base values
# We need time stats per region, source_zone
time_df = pd.read_csv('pipeline_outputs/risk_granular.csv')[['region', 'source_zone', 'mean_time', 'norm_time']]

# Cost per region
cost_df = pd.read_csv('pipeline_outputs/risk_region.csv')[['region', 'avg_cost', 'norm_cost']]

# Distance per region
from ml_model import _extract_distance_feature
dist_df = _extract_distance_feature(bundles['distance'].cleaned)
dist_df['distance_mean'] = dist_df['distance_mean'].fillna(dist_df['distance_mean'].median())

# Demand per region, drug_type
demand_raw = []
for k, v in bundles['demand'].cleaned.items():
    if isinstance(v, pd.DataFrame):
        demand_raw.append(v)
demand_df = pd.concat(demand_raw, ignore_index=True)
demand_grp = demand_df.groupby(['region', 'demand_type'])['demand'].sum().reset_index()

# Map demand_type to our 3 categories: total, cold, critical_cold
def map_dt(dt):
    dt_lower = str(dt).lower()
    if 'critical' in dt_lower and 'cold' in dt_lower: return 'critical_cold'
    if 'cold' in dt_lower: return 'cold'
    return 'normal'

demand_grp['dt'] = demand_grp['demand_type'].apply(map_dt)
demand_final = demand_grp.groupby(['region', 'dt'])['demand'].sum().reset_index()
# Also add 'total'
total_demand = demand_final.groupby('region')['demand'].sum().reset_index()
total_demand['dt'] = 'total'
demand_final = pd.concat([demand_final, total_demand], ignore_index=True)

# Capacity per region, config
cap_raw = []
for k, v in bundles['capacity'].cleaned.items():
    if isinstance(v, pd.DataFrame):
        cap_raw.append(v)
cap_df = pd.concat(cap_raw, ignore_index=True)

def map_cap(m):
    m_lower = str(m).lower()
    if 'crit' in m_lower and 'cold' in m_lower: return 'critical_cold'
    if 'cold' in m_lower: return 'cold'
    if 'norm' in m_lower: return 'normal'
    return 'total'

cap_df['dt'] = cap_df['metric'].apply(map_cap)
cap_final = cap_df.groupby(['region', 'configuration', 'dt'])['value'].max().reset_index()

# Cross join: region, source_zone, dt, configuration
regions = time_df['region'].unique()
zones = time_df['source_zone'].unique()
dts = ['total', 'normal', 'cold', 'critical_cold']
configs = cap_final['configuration'].unique()

rows = []
for r, z, dt, c in itertools.product(regions, zones, dts, configs):
    rows.append({'region': r, 'source_zone': z, 'drug_type': dt, 'configuration': c})

df = pd.DataFrame(rows)

# Merge time
df = df.merge(time_df, on=['region', 'source_zone'], how='left')
# Merge cost
df = df.merge(cost_df, on='region', how='left')
# Merge distance
df = df.merge(dist_df, on='region', how='left')
# Merge demand
df = df.merge(demand_final, left_on=['region', 'drug_type'], right_on=['region', 'dt'], how='left').drop(columns=['dt'])
# Merge capacity
df = df.merge(cap_final, left_on=['region', 'configuration', 'drug_type'], right_on=['region', 'configuration', 'dt'], how='left').drop(columns=['dt'])

# Now add some simulation results as extra features
sim_df = pd.read_csv('pipeline_outputs/simulation_results.csv')[['region', 'drug_type', 'shortage_probability', 'expected_shortage']]
df = df.merge(sim_df, on=['region', 'drug_type'], how='left')

# Compute gap
df['gap_ratio'] = ((df['demand'] - df['value']).clip(lower=0) / df['value'].replace(0, np.nan)).fillna(0)
# Compute cold_violation
# If drug_type is cold/critical, cold_violation = demand / value
df['cold_violation_ratio'] = 0.0
cold_mask = df['drug_type'].isin(['cold', 'critical_cold'])
df.loc[cold_mask, 'cold_violation_ratio'] = (df.loc[cold_mask, 'demand'] / df.loc[cold_mask, 'value'].replace(0, np.nan)).fillna(0).clip(upper=2)

def _minmax(s):
    lo, hi = s.min(), s.max()
    if hi == lo: return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)

df['norm_gap'] = _minmax(df['gap_ratio'])
df['norm_cold'] = _minmax(df['cold_violation_ratio'])
df['capacity_utilization'] = (df['demand'] / df['value'].replace(0, np.nan)).fillna(0).clip(upper=5)

df['risk_score'] = (
    WEIGHTS['cost'] * df['norm_cost'] * 100 +
    WEIGHTS['time'] * df['norm_time'] * 100 +
    WEIGHTS['gap'] * df['norm_gap'] * 100 +
    WEIGHTS['cold'] * df['norm_cold'] * 100
)

# Also bring in allocation unit cost if we want?
# Just use the 6 features for XGBoost:
# norm_cost, norm_time, norm_gap, norm_cold, distance_mean, capacity_utilization

print(f"Generated {len(df)} samples")
print(df[['region', 'source_zone', 'drug_type', 'configuration', 'risk_score']].head(10))

