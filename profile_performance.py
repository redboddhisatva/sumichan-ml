import pandas as pd
import time
import os
import json
import sys

# Mocking some parts to run the profiling script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core')))

from core.commute import best_commute
from core.ml_pipeline import train_xgboost_rent_model

# Load some data
stations_path = os.path.join("data", "stations.json")
with open(stations_path, encoding="utf-8") as f:
    stations = json.load(f)

# Create a mock DataFrame of size 10,000 for profiling
print("Creating mock data (WORST CASE FUZZY MATCH)...")
data = []
for i in range(10000):
    # Use a name that will NOT match anything to trigger full loop
    s1 = "NonExistentStation" 
    data.append({
        'total_rent': 100000 + (i % 50) * 1000,
        'size': str(20 + (i % 20)),
        'size_num': 20 + (i % 20),
        'access': f"[{{\"station\": \"{s1}\", \"walk_min\": 10}}]",
        'access_list': [{"station": s1, "walk_min": 10}],
        'area': 'Shinjuku',
        'layout': '1R',
        'density': 20000.0
    })

df = pd.DataFrame(data)

print("Profiling best_commute over 10,000 rows...")
start = time.time()
workplace = "Tokyo"
df["commute_min"] = df["access_list"].apply(
    lambda al: best_commute(al, workplace)
)
end = time.time()
print(f"best_commute took: {end - start:.4f} seconds for 10,000 rows")

# Fix dtypes for XGBoost
df['commute_min'] = pd.to_numeric(df['commute_min'], errors='coerce')
df['size_num'] = pd.to_numeric(df['size_num'], errors='coerce')
df['total_rent'] = pd.to_numeric(df['total_rent'], errors='coerce')

print("\nProfiling train_xgboost_rent_model over 10,000 rows...")
start = time.time()
model, mapping = train_xgboost_rent_model(df)
end = time.time()
print(f"train_xgboost_rent_model took: {end - start:.4f} seconds for 10,000 rows")
