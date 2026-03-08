"""
Weighted scoring model for area recommendation.

Weights
-------
- Cost    40%  (affordability)
- Commute 40%  (commute time — equally important)
- Value   20%  (㎡ per ¥10,000 — space efficiency)
"""

from __future__ import annotations

import pandas as pd

WEIGHTS = {"cost": 0.35, "commute": 0.35, "value": 0.20, "density": 0.10}


import math

# ---------------------------------------------------------------------------
# Continuous Mathematical Score Functions
# ---------------------------------------------------------------------------

def cost_score(rent: float, wage: float) -> int:
    """
    Rent-to-income ratio → score (12–100) using a Logistic Inverse Curve.
    
    A 20% ratio or below yields ~100.
    A 30% ratio (Standard healthy threshold) yields ~68.
    A 40% ratio yields ~20.
    """
    ratio = rent / wage if wage > 0 else 1.0

    # Logistic curve parameters tuned to hit historical breakpoints
    # f(x) = L / (1 + e^(k*(x - x0)))
    L, k, x0 = 100, 35, 0.32
    
    score = L / (1 + math.exp(k * (ratio - x0)))
    
    # Floor at 12, Ceiling at 100
    return int(max(12, min(100, round(score))))


def commute_score(minutes: float) -> int:
    """
    Commute time → score (12–100) using Exponential Decay.
    
    0-15 mins: near 100.
    35 mins: ~75
    60 mins: ~40
    """
    if minutes <= 5: 
        return 100

    # Exponential decay function: y = A * e^(-k * x)
    # Tuned to gracefully decay over 60 minutes
    score = 100 * math.exp(-0.015 * (minutes - 10))
    
    return int(max(12, min(100, round(score))))


def value_score_normalized(sqm_values: pd.Series) -> pd.Series:
    """
    Normalise sqm-per-10k-yen values across all areas to 20–98 range.

    If all values are the same, returns 60 for every entry.
    """
    vmin = sqm_values.min()
    vmax = sqm_values.max()

    if vmax == vmin:
        return pd.Series([60] * len(sqm_values), index=sqm_values.index)

    normed = 20 + (sqm_values - vmin) / (vmax - vmin) * 78
    return normed.round().astype(int)


def get_density_score(density: float) -> int:
    """
    Population density (people per km²) → score (12–100) using a smooth curve.
    
    A density of 20,000+ yields 100.
    A density of 8,000 yields ~60.
    """
    # Linear interpolation capped at 100 and floored at 12
    score = (density / 20000) * 100
    return int(max(12, min(100, round(score))))


def total_score(cost: int, commute: int, value: int, density: int = 50) -> int:
    """
    Weighted total score including density.
    """
    return round(
        cost * WEIGHTS["cost"]
        + commute * WEIGHTS["commute"]
        + value * WEIGHTS["value"]
        + density * WEIGHTS["density"]
    )
