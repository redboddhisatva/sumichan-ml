"""
Fetches CSV property data from GitHub and caches it for 1 hour.
"""

import io
import pandas as pd
import requests
import streamlit as st

_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "redboddhisatva/sumichan-property/main/"
)

_CSV_MAP = {
    "tokyo": "all_tokyo_stations.csv",
    "saitama": "saitama_stations.csv",
    "chiba": "chiba_stations.csv",
    "kanagawa": "kanagawa_stations.csv",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(regions: list[str]) -> pd.DataFrame:
    """
    Download one CSV per region, tag each row with its region,
    and return them all concatenated into a single DataFrame.

    Raises
    ------
    RuntimeError  if any CSV fetch fails.
    """
    frames: list[pd.DataFrame] = []

    for region in regions:
        filename = _CSV_MAP.get(region)
        if filename is None:
            continue

        url = _BASE_URL + filename
        resp = requests.get(url, timeout=30)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch {filename}: HTTP {resp.status_code}"
            )

        # The CSVs are UTF-8 with BOM (﻿) — pandas handles that with
        # encoding_errors="replace" isn't needed; read_csv strips the BOM
        # automatically when using utf-8-sig.
        df = pd.read_csv(
            io.StringIO(resp.content.decode("utf-8-sig")),
            dtype=str,           # keep everything as strings for parsing
        )
        df["region"] = region
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
