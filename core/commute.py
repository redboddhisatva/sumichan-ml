"""
Travel-time estimation between stations using Haversine distance.

No external transit API required — estimates based on geographic distance
at an average speed of 28 km/h plus a 5-minute transfer buffer.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache

# ---------------------------------------------------------------------------
# Station coordinates
# ---------------------------------------------------------------------------

_STATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "stations.json"
)


@lru_cache(maxsize=1)
def _load_stations() -> dict[str, list[float]]:
    with open(_STATIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=2048)
def find_coords(name: str) -> tuple[float, float] | None:
    """
    Look up GPS coordinates for a station name.

    - Strips trailing "駅" if present
    - Falls back to partial/containment matching
    - Results are cached to avoid repeated lookups.
    """
    stations = _load_stations()
    clean = name.replace("駅", "").strip()

    # Exact match
    if clean in stations:
        coords = stations[clean]
        return tuple(coords)

    # Fuzzy: check if any key contains the name or vice-versa
    for key, coords in stations.items():
        if clean in key or key in clean:
            return tuple(coords)

    return None


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def haversine(coord1, coord2) -> float:
    """
    Return the distance in **km** between two (lat, lng) points.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Travel time
# ---------------------------------------------------------------------------

def travel_minutes(from_station: str, to_station: str) -> float | None:
    """
    Estimate train travel time between two stations.

    Formula: distance_km / 28 * 60 + 5  (avg 28 km/h + 5 min transfer)
    Minimum 5 minutes.  Returns None if either station has no coordinates.
    """
    c1 = find_coords(from_station)
    c2 = find_coords(to_station)

    if c1 is None or c2 is None:
        return None

    dist = haversine(c1, c2)
    minutes = dist / 28.0 * 60.0 + 5.0
    return max(5.0, minutes)


def best_commute(
    access_list: list[dict], workplace: str
) -> float | None:
    """
    Find the minimum total commute (walk + train) from a property's
    access list to the target workplace station.

    Returns None if no station could be evaluated.
    """
    best = None

    for entry in access_list:
        station = entry.get("station", "")
        walk = entry.get("walk_min", 0)

        train = travel_minutes(station, workplace)
        if train is None:
            continue

        total = walk + train
        if best is None or total < best:
            best = total

    return best
