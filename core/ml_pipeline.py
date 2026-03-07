import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

@st.cache_resource(show_spinner="Training Rent Prediction Model...")
def train_xgboost_rent_model(df: pd.DataFrame) -> tuple[XGBRegressor, dict]:
    """
    Trains an XGBoost model to predict total_rent based on property features.
    Returns the trained model and a dict mapping layout strings to numeric codes.
    """
    train_df = df.copy()
    
    # We must have these features to train
    train_df = train_df.dropna(subset=['total_rent', 'size_num', 'commute_min', 'density'])
    
    # Filter out bizarre outliers to keep training stable
    train_df = train_df[(train_df['total_rent'] > 10000) & (train_df['total_rent'] < 1000000)]
    train_df = train_df[train_df['size_num'] > 5]
    
    # Encode 'layout' as categorical codes
    layout_cats = train_df['layout'].astype('category')
    train_df['layout_code'] = layout_cats.cat.codes
    
    # Save the mapping so we can encode new predictions identically
    cat_mapping = {val: idx for idx, val in enumerate(layout_cats.cat.categories)}
    
    X = train_df[['size_num', 'commute_min', 'density', 'layout_code']]
    y = train_df['total_rent']
    
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',      # histogram-based splitting — 3-10x faster
        n_jobs=-1,               # use all CPU cores
        early_stopping_rounds=10 # stop if no improvement for 10 rounds
    )
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    
    return model, cat_mapping


@st.cache_resource(show_spinner="Clustering Areas...")
def train_kmeans_clusters(area_stats: pd.DataFrame) -> dict[str, str]:
    """
    Trains K-Means to cluster areas into 4 distinct lifestyle groups.
    Uses MiniBatchKMeans for significantly faster convergence.
    Returns a dictionary mapping 'area' -> 'Cluster Label'.
    """
    # area_stats index is the area name
    train_df = area_stats.copy().dropna(subset=['avg_rent', 'avg_size', 'avg_commute', 'density'])
    
    if len(train_df) < 4:
        # Not enough data to cluster
        return {area: "Standard" for area in train_df.index}
        
    X = train_df[['avg_rent', 'avg_size', 'avg_commute', 'density']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = MiniBatchKMeans(
        n_clusters=4,
        random_state=42,
        n_init=3,           # fewer inits — fast convergence
        batch_size=256,     # mini-batch for speed
    )
    clusters = kmeans.fit_predict(X_scaled)
    
    train_df['cluster'] = clusters
    
    # Sort clusters by average rent to assign meaningful ascending names
    cluster_rents = train_df.groupby('cluster')['avg_rent'].mean().sort_values()
    rank_map = {old_id: new_rank for new_rank, old_id in enumerate(cluster_rents.index)}
    
    labels = [
        "Economy & Practical",    # 0 (Cheapest)
        "Balanced Commuter",      # 1
        "Spacious Living",        # 2
        "Premium Central"         # 3 (Most Expensive)
    ]
    
    mapping = {}
    for area, row in train_df.iterrows():
        c_id = int(row['cluster'])
        rank = rank_map[c_id]
        mapping[str(area)] = labels[rank]
        
    return mapping


def calculate_ml_deal_score(actual_rent: float, predicted_rent: float) -> int:
    """
    ML Deal Score (12-100) based on actual vs predicted rent.
    If actual is much lower than predicted, it's a fantastic deal (100).
    If actual is much higher, it's a bad deal (12).
    """
    if predicted_rent <= 0 or actual_rent <= 0:
        return 50
        
    ratio = actual_rent / predicted_rent
    
    if ratio <= 0.70: return 100
    if ratio <= 0.80: return 90
    if ratio <= 0.90: return 80
    if ratio <= 1.00: return 60
    if ratio <= 1.10: return 40
    if ratio <= 1.20: return 20 
    return 12


def calculate_deal_scores_vectorized(
    actual: pd.Series, predicted: pd.Series
) -> pd.Series:
    """
    Vectorized deal-score calculation — replaces row-by-row .apply().
    Returns a Series of int scores (12–100).
    """
    ratio = actual / predicted.replace(0, np.nan)
    conditions = [
        (actual <= 0) | (predicted <= 0),
        ratio <= 0.70,
        ratio <= 0.80,
        ratio <= 0.90,
        ratio <= 1.00,
        ratio <= 1.10,
        ratio <= 1.20,
    ]
    choices = [50, 100, 90, 80, 60, 40, 20]
    return pd.Series(
        np.select(conditions, choices, default=12),
        index=actual.index,
        dtype=int,
    )
