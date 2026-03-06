# 🗼 スミちゃん — Tokyo Living Finder

A bilingual (English / Japanese) Streamlit app that recommends the best Tokyo-area neighborhoods to live in, based on your salary, workplace station, and preferred apartment layout. It scores areas by affordability (45%), commute time (30%), and space efficiency (25%), then visualises the top picks with an interactive radar chart and sortable property listings.

## How to Use

1. **Set your preferences** in the sidebar: target regions, monthly take-home pay, nearest station to your workplace, and apartment layout.
2. **Click "🔍 Find Areas"** — the app fetches live property data from GitHub, scores each neighborhood, and shows the top 5 on a radar chart.
3. **Tap an area name** below the chart to browse individual listings with rent ratios, estimated commute times, and direct SUUMO links.

## Data Source

Property data is scraped from [SUUMO](https://suumo.jp/) and hosted at [redboddhisatva/sumichan-property](https://github.com/redboddhisatva/sumichan-property).

## Tech Stack

- **Python 3.11** · **Streamlit 1.35** · **Plotly 5.22** · **Pandas 2.2** · **NumPy 1.26**
- Haversine-based commute estimation (no external transit API)
- Bilingual UI via `core/i18n.py`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch `main`, main file path `app.py`
4. Click **Deploy**
