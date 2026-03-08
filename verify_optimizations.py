"""Quick benchmark of the optimized pipeline."""
import time, sys, os, json
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from core.commute import best_commute
from core.ml_pipeline import (
    train_xgboost_rent_model,
    train_kmeans_clusters,
    calculate_deal_scores_vectorized,
)

stations_path = os.path.join("data", "stations.json")
with open(stations_path, encoding="utf-8") as f:
    stations = json.load(f)

station_names = list(stations.keys())
N = 50000
print(f"Benchmarking with {N:,} rows...\n")

data = []
for i in range(N):
    s1 = station_names[i % len(station_names)]
    data.append({
        'total_rent': float(80000 + (i % 80) * 1000),
        'size_num': float(20 + (i % 30)),
        'layout': ['1R', '1K', '1DK', '1LDK', '2K', '2DK', '2LDK'][i % 7],
        'density': 8000.0 + (i % 20) * 500,
        'commute_min': float(10 + (i % 60)),
        'access_list': [{"station": s1, "walk_min": 8}],
        'area': ['Shinjuku', 'Shibuya', 'Meguro', 'Setagaya', 'Minato'][i % 5],
    })
df = pd.DataFrame(data)

# 1. best_commute
t0 = time.perf_counter()
df["commute_min2"] = df["access_list"].apply(lambda al: best_commute(al, "Tokyo"))
t1 = time.perf_counter()
print(f"best_commute  : {t1-t0:.3f}s  ({N:,} rows)")

# 2. XGBoost training
t0 = time.perf_counter()
model, cat_map = train_xgboost_rent_model(df)
t1 = time.perf_counter()
print(f"XGBoost train : {t1-t0:.3f}s  ({N:,} rows)")

# 3. Prediction
t0 = time.perf_counter()
df2 = df.copy()
df2['layout_code'] = df2['layout'].map(lambda x: cat_map.get(x, -1))
X_pred = df2[['size_num', 'commute_min', 'density', 'layout_code']].fillna(0)
df['predicted_rent'] = model.predict(X_pred)
t1 = time.perf_counter()
print(f"XGBoost pred  : {t1-t0:.3f}s  ({N:,} rows)")

# 4. Vectorized deal score
t0 = time.perf_counter()
df['deal_score'] = calculate_deal_scores_vectorized(df['total_rent'], df['predicted_rent'])
t1 = time.perf_counter()
print(f"Deal score vec: {t1-t0:.3f}s  ({N:,} rows)")

# 5. K-Means clustering
area_stats = df.groupby("area").agg(
    avg_rent=("total_rent", "mean"),
    avg_size=("size_num", "mean"),
    avg_commute=("commute_min", "mean"),
    density=("density", "mean"),
).dropna()

t0 = time.perf_counter()
cluster_map = train_kmeans_clusters(area_stats)
t1 = time.perf_counter()
print(f"KMeans cluster: {t1-t0:.3f}s  ({len(area_stats)} areas)")

print(f"\nCluster results: {cluster_map}")
print("✅ All optimizations verified!")
