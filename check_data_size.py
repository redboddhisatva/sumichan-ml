import pandas as pd
import requests
import io

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

for region, filename in _CSV_MAP.items():
    url = _BASE_URL + filename
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")), dtype=str)
            print(f"{region}: {len(df)} rows")
        else:
            print(f"{region}: Failed {resp.status_code}")
    except Exception as e:
        print(f"{region}: Error {e}")
