import pandas as pd
import requests
import json
import re

urls = [
    "https://ja.wikipedia.org/wiki/%E6%9D%B1%E4%BA%AC%E9%83%BD%E3%81%AE%E5%8C%BA%E5%B8%82%E7%94%BA%E6%9D%91%E4%B8%80%E8%A6%A7",
    "https://ja.wikipedia.org/wiki/%E7%A5%9E%E5%A5%88%E5%B7%9D%E7%9C%8C%E3%81%AE%E5%B8%82%E7%94%BA%E6%9D%91%E4%B8%80%E8%A6%A7",
    "https://ja.wikipedia.org/wiki/%E5%9F%BC%E7%8E%89%E7%9C%8C%E3%81%AE%E5%B8%82%E7%94%BA%E6%9D%91%E4%B8%80%E8%A6%A7",
    "https://ja.wikipedia.org/wiki/%E5%8D%83%E8%91%89%E7%9C%8C%E3%81%AE%E5%B8%82%E7%94%BA%E6%9D%91%E4%B8%80%E8%A6%A7"
]

results = {}

for url in urls:
    try:
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        tables = pd.read_html(html)
        print(f"Loaded {len(tables)} tables from {url}")
        
        for df in tables:
            # Flatten columns if MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns.values]
            
            # Print columns for debug
            cols = [str(c) for c in df.columns]
            
            name_c = next((c for c in cols if '市町村' in c or '自治体' in c or '区町村' in c or '市区町村' in c), None)
            density_c = next((c for c in cols if '人口密度' in c), None)
            
            if name_c and density_c:
                print(f"Found table with {name_c} and {density_c}")
                for _, row in df.iterrows():
                    name = str(row[name_c]).split()[-1].strip() # in case of pref prefix
                    name = re.sub(r'\[.*?\]', '', name)
                    name = name.replace("\u3000", "").replace(" ", "")
                    # handle rows like 渋谷区 / 新宿区
                    
                    val_str = str(row[density_c]).replace(',', '')
                    try:
                        val = float(re.search(r'[\d\.]+', val_str).group())
                        results[name] = val
                    except:
                        pass
                break # Move to next URL
    except Exception as e:
        print(f"Error on {url}: {e}")

with open("data/density.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} areas to data/density.json")
