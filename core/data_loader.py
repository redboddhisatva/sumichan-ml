"""
Fetches property data from Supabase and caches it.
"""

import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(regions: list[str], layout: str = None) -> pd.DataFrame:
    """
    Fetch property data from Supabase for the specified regions.
    Filters by layout in the SQL query to optimize loading and downstream ML.
    """
    if not regions:
        return pd.DataFrame()

    safe_regions = [r for r in regions if r.isalnum()]
    if not safe_regions:
        return pd.DataFrame()

    conn = st.connection("supabase", type="sql", autocommit=True)
    
    region_str = ", ".join(f"'{r}'" for r in safe_regions)
    query = f"SELECT * FROM properties WHERE region IN ({region_str})"
    
    if layout:
        # Avoid SQL injection by replacing simple quotes, just in case
        safe_layout = layout.replace("'", "''")
        query += f" AND layout = '{safe_layout}'"
        
    df = conn.query(query)
    
    for col in df.columns:
        df[col] = df[col].astype(str)
        
    return df
