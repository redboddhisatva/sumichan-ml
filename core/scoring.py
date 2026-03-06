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


# ---------------------------------------------------------------------------
# Individual score functions
# ---------------------------------------------------------------------------

def cost_score(rent: float, wage: float) -> int:
    """
    Rent-to-income ratio → score (12–100).

    >>> cost_score(60_000, 300_000)
    100
    >>> cost_score(120_000, 300_000)
    26
    """
    ratio = rent / wage if wage > 0 else 1.0

    if ratio <= 0.20:
        return 100
    if ratio <= 0.25:
        return 90
    if ratio <= 0.28:
        return 80
    if ratio <= 0.30:
        return 68
    if ratio <= 0.33:
        return 54
    if ratio <= 0.36:
        return 40
    if ratio <= 0.40:
        return 26
    return 12


def commute_score(minutes: float) -> int:
    """
    Commute time → score (12–100).

    >>> commute_score(10)
    100
    >>> commute_score(50)
    40
    """
    if minutes <= 15:
        return 100
    if minutes <= 25:
        return 88
    if minutes <= 35:
        return 74
    if minutes <= 45:
        return 58
    if minutes <= 60:
        return 40
    if minutes <= 75:
        return 24
    return 12


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
    Population density (people per km²) → score (12–100).
    Higher density gets a higher score.
    """
    if density >= 20000:
        return 100
    if density >= 15000:
        return 90
    if density >= 12000:
        return 80
    if density >= 10000:
        return 70
    if density >= 8000:
        return 60
    if density >= 6000:
        return 50
    if density >= 4000:
        return 40
    if density >= 2000:
        return 24
    return 12


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
