"""
Fetches property data from Supabase and caches it.
"""

import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(regions: list[str]) -> pd.DataFrame:
    """
    Fetch property data from Supabase for the specified regions.
    """
    if not regions:
        return pd.DataFrame()

    safe_regions = [r for r in regions if r.isalnum()]
    if not safe_regions:
        return pd.DataFrame()

    conn = st.connection("supabase", type="sql", autocommit=True)
    
    region_str = ", ".join(f"'{r}'" for r in safe_regions)
    query = f"SELECT * FROM properties WHERE region IN ({region_str})"
    
    df = conn.query(query)
    
    for col in df.columns:
        df[col] = df[col].astype(str)
        
    return df
