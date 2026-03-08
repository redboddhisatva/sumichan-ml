import os
import pandas as pd
import requests
import io
import streamlit as st
from sqlalchemy import create_engine

# URIs for the CSVs
BASE_URL = "https://raw.githubusercontent.com/redboddhisatva/sumichan-property/main/"
CSV_MAP = {
    "tokyo": "all_tokyo_stations.csv",
    "saitama": "saitama_stations.csv",
    "chiba": "chiba_stations.csv",
    "kanagawa": "kanagawa_stations.csv",
}

def upload_data():
    # 1. Get the database connection string from Streamlit secrets
    try:
        # Supabase connection string usually looks like:
        # postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
        db_url = st.secrets["connections"]["supabase"]["url"]
    except KeyError:
        print("❌ Error: Could not find Supabase URL in .streamlit/secrets.toml")
        print("Please add it like this:")
        print("[connections.supabase]")
        print('url = "postgresql://..."')
        return

    print("🔌 Connecting to Supabase...")
    engine = create_engine(db_url)
    
    # 2. Download and merge CSVs
    frames = []
    for region, filename in CSV_MAP.items():
        print(f"📥 Downloading {region} data...")
        url = BASE_URL + filename
        resp = requests.get(url, timeout=30)
        
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")), dtype=str)
            df["region"] = region
            frames.append(df)
            print(f"✅ {region} downloaded ({len(df)} rows)")
        else:
            print(f"❌ Failed to download {region}")

    if not frames:
        print("No data downloaded. Exiting.")
        return

    final_df = pd.concat(frames, ignore_index=True)
    print(f"\n📊 Total rows to upload: {len(final_df)}")

    # 3. Upload to Supabase
    print("🚀 Uploading to Supabase table 'properties' (this might take a minute)...")
    try:
        final_df.to_sql(
            "properties", 
            engine, 
            if_exists="replace", # Use 'replace' for the first run, 'append' later if needed
            index=False,
            method="multi", # Faster inserts
            chunksize=1000  # Upload in batches
        )
        print("🎉 Successfully uploaded all data to Supabase!")
    except Exception as e:
        print(f"❌ Error uploading to Supabase: {e}")

if __name__ == "__main__":
    upload_data()
