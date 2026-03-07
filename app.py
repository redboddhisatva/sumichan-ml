"""
🗼 スミちゃん · Home Sweet Sumika
Main Streamlit application — bilingual EN/JP.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from core.i18n import get_text
from core.data_loader import load_all_data
from core.parser import parse_rent, parse_fee, extract_area, parse_access
from core.commute import best_commute
from core.scoring import (
    cost_score,
    commute_score,
    get_density_score,
    total_score,
    WEIGHTS,
)
from core.ml_pipeline import (
    train_xgboost_rent_model,
    train_kmeans_clusters,
    calculate_deal_scores_vectorized,
)
from ui.radar import build_radar_chart, COLORS
from ui.listings import show_listings_table

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="スミちゃん · Home Sweet Sumika",
    page_icon="🗼",
    layout="wide",
)

# ── Custom CSS — matches JSX reference design ───────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=Noto+Sans+JP:wght@400;500;600;700&display=swap');

    /* ── Animations ────────────────────────────────────────────────── */
    @keyframes fade-in {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes slide-up {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in { animation: fade-in 0.6s ease-out; }
    .animate-slide-up { animation: slide-up 0.6s ease-out backwards; }

    /* ── Global typography ─────────────────────────────────────────── */
    html, body, [class*="css"], .stMarkdown, .stText,
    [data-testid="stSidebar"], [data-testid="stHeader"],
    input, select, textarea, button, label, p, span, div {
        font-family: 'Inter', 'Noto Sans JP', sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'DM Sans', 'Noto Sans JP', sans-serif !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }

    /* ── Page background ───────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 50%, #f1f5f9 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Sidebar ───────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.5rem !important;
        background: linear-gradient(to right, #0f172a, #1e3a8a, #0f172a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* ── Metric / stat cards ───────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #f3f4f6;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border-color: transparent;
    }
    [data-testid="stMetricValue"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* ── Primary buttons ───────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(to right, #1e3a8a, #1e40af, #334155) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 15px rgba(30,58,138,0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 30px rgba(30,58,138,0.4) !important;
        transform: scale(1.02) !important;
    }

    /* ── Secondary buttons ─────────────────────────────────────────── */
    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: transparent !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08) !important;
        transform: translateY(-2px) !important;
        color: #1e3a8a !important;
    }

    /* ── Radio buttons ──────────────────────────────────────────────── */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        gap: 8px;
        flex-wrap: wrap;
    }
    div[data-testid="stRadio"] > div > label {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 6px 16px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] > div > label:hover {
        border-color: #1e3a8a;
    }

    /* ── Section headers ───────────────────────────────────────────── */
    .main h2 { font-size: 1.8rem !important; margin-top: 2rem !important; }
    .main h3 { font-size: 1.5rem !important; }

    /* ── Dividers ───────────────────────────────────────────────────── */
    .main hr {
        border: none !important;
        border-top: 1px solid #e5e7eb !important;
        margin: 2rem 0 !important;
    }

    /* ── Inputs ─────────────────────────────────────────────────────── */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    .stNumberInput input {
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
    .stNumberInput input:focus {
        border-color: #1e3a8a !important;
        box-shadow: 0 0 0 3px rgba(30,58,138,0.1) !important;
    }

    /* ── Hero section ──────────────────────────────────────────────── */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem 2rem 2rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(to right, #f1f5f9, #eff6ff);
        border: 1px solid #e2e8f0;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    .hero-section h1 {
        font-size: 3rem !important;
        line-height: 1.1 !important;
        margin-bottom: 1rem !important;
    }
    .hero-gradient {
        background: linear-gradient(to right, #1e3a8a, #334155);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-section p.hero-sub {
        color: #6b7280 !important;
        font-size: 1.15rem !important;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.7;
    }

    /* ── Area cards ─────────────────────────────────────────────────── */
    .area-card {
        background: #ffffff;
        border: 1px solid #f3f4f6;
        border-radius: 16px;
        padding: 1.2rem;
        cursor: pointer;
        transition: all 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    .area-card:hover {
        box-shadow: 0 12px 35px rgba(0,0,0,0.08);
        border-color: transparent;
    }
    .area-card.active {
        border-color: #1e3a8a;
        box-shadow: 0 0 0 3px rgba(30,58,138,0.1), 0 12px 35px rgba(0,0,0,0.06);
    }
    .area-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .area-card .area-name {
        font-family: 'DM Sans', 'Noto Sans JP', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .area-score {
        font-size: 0.85rem;
        font-weight: 700;
        color: #1e3a8a;
        background: #eff6ff;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
    }
    .area-card .area-name-en {
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 500;
    }
    .area-card .area-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.4rem;
        margin-top: 0.8rem;
    }
    .area-stat-box {
        background: linear-gradient(135deg, #f9fafb, #f3f4f6);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.4rem;
        text-align: center;
    }
    .area-stat-box .stat-label {
        font-size: 0.65rem;
        color: #9ca3af;
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .area-stat-box .stat-value {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .area-card .score-bar {
        position: absolute;
        bottom: 0;
        left: 0;
        height: 3px;
        background: linear-gradient(to right, #1e3a8a, #334155);
        transition: all 1s ease;
        opacity: 0.7;
    }
    .area-card:hover .score-bar,
    .area-card.active .score-bar {
        opacity: 1;
    }
    .area-card .viewing-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: linear-gradient(to right, #1e3a8a, #334155);
        color: #ffffff;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
    }
    .area-card .area-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid #f3f4f6;
        font-size: 0.8rem;
    }
    .area-footer .prop-count { color: #6b7280; font-weight: 500; }
    .area-footer .view-link { color: #1e3a8a; font-weight: 600; }

    /* ── Category headers ──────────────────────────────────────────── */
    .category-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1.5rem 0 0.75rem 0;
    }
    .category-bar {
        width: 3rem;
        height: 4px;
        border-radius: 9999px;
    }
    .category-header h4 {
        font-family: 'DM Sans', 'Noto Sans JP', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        margin: 0 !important;
    }

    /* ── CTA footer ────────────────────────────────────────────────── */
    .cta-footer {
        background: linear-gradient(to right, #172554, #0f172a, #172554);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        color: #ffffff;
        position: relative;
        overflow: hidden;
        margin-top: 2rem;
    }
    .cta-footer h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 0.75rem !important;
    }
    .cta-footer p {
        color: rgba(255,255,255,0.8) !important;
        font-size: 1rem !important;
        max-width: 600px;
        margin: 0 auto 1.5rem auto !important;
    }

    /* ── Dataframe fallback ─────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load station list for dropdown ───────────────────────────────────────
@st.cache_data
def _load_station_names() -> list[str]:
    import pykakasi
    kks = pykakasi.kakasi()
    
    path = os.path.join(os.path.dirname(__file__), "data", "stations.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    stations = sorted(data.keys())
    bilingual = [""]  # Start with empty string as first option
    
    for s in stations:
        result = kks.convert(s)
        romaji = "".join([item['hepburn'].capitalize() for item in result])
        bilingual.append(f"{s} ({romaji})")
        
    return bilingual


STATION_NAMES = _load_station_names()


@st.cache_data
def _load_density_data() -> dict[str, float]:
    path = os.path.join(os.path.dirname(__file__), "data", "density.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

DENSITY_MAP = _load_density_data()


# ── Helper: language shortcode ───────────────────────────────────────────
def _lang_code(choice: str) -> str:
    return "en" if choice == "English" else "ja"


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.radio(
        "🌐 言語 / Language",
        ["日本語", "English"],
        horizontal=True,
        label_visibility="visible",
    )
    lang = _lang_code(lang_choice)
    t = lambda key: get_text(key, lang)

    st.markdown(f"# {t('app_title')}")
    st.caption(t("app_subtitle"))
    st.divider()

    # Region multi-select
    # Target Regions
    selected_regions = st.multiselect(
        t("region_label"),
        options=["tokyo", "saitama", "chiba", "kanagawa"],
        default=[],
        format_func=lambda x: t(x),
    )

    # Wage
    wage = st.number_input(
        t("wage_label"),
        min_value=0,
        max_value=2_000_000,
        value=0,
        step=10_000,
    )

    # Workplace — dropdown from stations.json
    workplace_full = st.selectbox(
        t("workplace_label"),
        options=STATION_NAMES,
        index=0, # This is the empty string ""
    )
    # Extract just the Kanji part for the backend
    workplace = workplace_full.split(" (")[0] if " (" in workplace_full else workplace_full

    # Layout filter
    layout_options = [
        t("layout_all"),
        "ワンルーム", "1K", "1DK", "1LDK",
        "2K", "2DK", "2LDK",
        "3K", "3DK", "3LDK",
    ]
    layout_filter = st.selectbox(t("layout_label"), layout_options)

    st.divider()
    search = st.button(t("search_btn"), use_container_width=True, type="primary")




# ── Session state: persist results across Streamlit re-runs ──────────────
# When the user clicks search, we store results in session_state.
# When they click an area radio, the page re-runs but results stay.

if search:
    st.session_state["search_triggered"] = True
    st.session_state["search_params"] = {
        "regions": selected_regions,
        "wage": wage,
        "workplace": workplace,
        "layout_filter": layout_filter,
        "lang": lang,
    }

# Check if we have a search to display
if not st.session_state.get("search_triggered"):
    # ── Hero welcome section ─────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-section animate-slide-up">
            <div class="hero-badge">🗼 AI-Powered Neighborhood Finder</div>
            <h1>Discover Where<br><span class="hero-gradient">You Belong</span></h1>
            <p class="hero-sub">
                Our ML engine analyzes rent, commute, space, and lifestyle
                to find your perfect neighborhood in Greater Tokyo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stat cards
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Areas Analyzed", "Kanto Area", help="Greater Tokyo coverage")
    with c2:
        st.metric("Data Points", "50K+", help="Real listing data")
    st.stop()

if st.session_state.get("search_triggered"):
    params = st.session_state["search_params"]
    _regions = params["regions"]
    _wage = params["wage"]
    _workplace = params["workplace"]
    _layout_filter = params["layout_filter"]
    _lang = lang  # always use current language for display

    t_disp = lambda key: get_text(key, _lang)

    # Determine layout_all for the stored language (might differ from current)
    _layout_all_ja = get_text("layout_all", "ja")
    _layout_all_en = get_text("layout_all", "en")

    if not _regions:
        st.info(" Please select at least one region to begin.")
        st.stop()
        
    if not _workplace:
        st.warning(t_disp("err_no_workplace"))
        st.stop()

    with st.spinner(t_disp("loading")):
        try:
            raw_df = load_all_data(_regions)
        except RuntimeError as e:
            st.error(t_disp("err_fetch_failed").format(str(e)))
            st.stop()

    if raw_df.empty:
        st.warning(t_disp("err_no_results"))
        st.stop()

    # ── Parse all rows ───────────────────────────────────────────────
    df = raw_df.copy()
    df["rent_val"] = df["rent"].apply(parse_rent)
    df["fee_val"] = df["management_fee"].apply(parse_fee)
    df["total_rent"] = df["rent_val"].fillna(0) + df["fee_val"].fillna(0)
    df["area"] = df["address"].apply(extract_area)
    df["size_num"] = pd.to_numeric(df["size"], errors="coerce")
    df["access_list"] = df["access"].apply(parse_access)
    df["commute_min"] = df["access_list"].apply(
        lambda al: best_commute(al, _workplace)
    )
    df["sqm_per_10k"] = np.where(
        df["total_rent"] > 0,
        df["size_num"] / (df["total_rent"] / 10_000),
        0,
    )

    # ── Train ML & Predict Deal Score ────────────────────────────────
    df["density"] = df["area"].map(lambda x: DENSITY_MAP.get(x, 8000.0))
    xgb_model, cat_mapping = train_xgboost_rent_model(df)
    
    # Predict fair rent and calculate deal score for each property
    df_pred = df.copy()
    df_pred['layout_code'] = df_pred['layout'].map(lambda x: cat_mapping.get(x, -1))
    X_pred = df_pred[['size_num', 'commute_min', 'density', 'layout_code']].fillna(0)
    
    df['predicted_rent'] = xgb_model.predict(X_pred)
    df['deal_score'] = calculate_deal_scores_vectorized(
        df['total_rent'], df['predicted_rent']
    )

    # ── Filter ───────────────────────────────────────────────────────
    is_all = _layout_filter in (_layout_all_ja, _layout_all_en)
    if not is_all:
        df = df[df["layout"] == _layout_filter]

    df = df[(df["total_rent"] > 0) & (df["area"].notna())]

    if df.empty:
        st.warning(t_disp("err_no_results"))
        st.stop()

    # ── Group by area (≥3 listings) ──────────────────────────────────
    area_counts = df.groupby("area").size()
    valid_areas = area_counts[area_counts >= 3].index
    df = df[df["area"].isin(valid_areas)]

    if df.empty:
        st.warning(t_disp("err_no_results"))
        st.stop()

    area_stats = df.groupby("area").agg(
        avg_rent=("total_rent", "mean"),
        avg_size=("size_num", "mean"),
        avg_sqm_per_10k=("sqm_per_10k", "mean"),
        avg_commute=("commute_min", "mean"),
        avg_deal_score=("deal_score", "mean"),
        count=("total_rent", "size"),
    ).dropna(subset=["avg_commute"])

    if area_stats.empty:
        st.warning(t_disp("err_no_results"))
        st.stop()

    # ── Score ────────────────────────────────────────────────────────
    area_stats["density"] = area_stats.index.map(lambda x: DENSITY_MAP.get(x, 8000.0))
    area_stats["density_score"] = area_stats["density"].apply(get_density_score)
    
    area_stats["cost_score"] = area_stats["avg_rent"].apply(
        lambda r: cost_score(r, _wage)
    )
    area_stats["commute_score"] = area_stats["avg_commute"].apply(
        commute_score
    )
    # The 'value_score' for the radar chart is now driven by the ML Deal Score
    area_stats["value_score"] = area_stats["avg_deal_score"].round().astype(int)
    
    # Cluster the areas using K-Means
    cluster_map = train_kmeans_clusters(area_stats)
    area_stats["cluster"] = area_stats.index.map(lambda x: cluster_map.get(x, "Standard"))
    area_stats["total"] = area_stats.apply(
        lambda row: total_score(
            row["cost_score"], row["commute_score"], row["value_score"], row["density_score"]
        ),
        axis=1,
    )

    top = area_stats.nlargest(10, "total").reset_index()

    # ── Metric chips ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t_disp("metric_listings"), f"{len(df):,}")
    c2.metric(t_disp("metric_areas"), len(valid_areas))
    c3.metric(t_disp("metric_layout"), _layout_filter)
    c4.metric(t_disp("metric_workplace"), _workplace)

    st.markdown("")  # spacer

    # ── Top areas section ────────────────────────────────────────────
    st.subheader(t_disp("top_areas_title").format(len(top)))

    # Initialize state for tracking the selected area
    if "sel_area" not in st.session_state or st.session_state.get("sel_search_trigger") != st.session_state["search_triggered"]:
        st.session_state["sel_area"] = top.iloc[0]["area"]
        st.session_state["sel_search_trigger"] = st.session_state["search_triggered"]

    selected_area = st.session_state["sel_area"]

    def set_selected_area(name: str):
        st.session_state["sel_area"] = name

    # Group the top 10 areas by their ML Lifestyle Cluster
    cluster_order = [
        "Premium Central",
        "Spacious Living",
        "Balanced Commuter",
        "Economy & Practical",
    ]

    # Cluster category colors
    cluster_colors = {
        "Premium Central": "#1e3a8a",
        "Spacious Living": "#065f46",
        "Balanced Commuter": "#92400e",
        "Economy & Practical": "#6b7280",
    }

    for cluster in cluster_order:
        cluster_df = top[top["cluster"] == cluster]
        if not cluster_df.empty:
            bar_color = cluster_colors.get(cluster, "#1e3a8a")
            st.markdown(
                f"""<div class="category-header">
                    <div class="category-bar" style="background:{bar_color}"></div>
                    <h4>{cluster}</h4>
                </div>""",
                unsafe_allow_html=True,
            )
            num_cols = min(3, len(cluster_df))
            cols = st.columns(num_cols) if num_cols > 0 else []
            for i, (_, row) in enumerate(cluster_df.iterrows()):
                a_name = row["area"]
                import pykakasi
                kks = pykakasi.kakasi()
                res = kks.convert(a_name)
                romaji = "".join([x['hepburn'].capitalize() for x in res])

                avg_rent_k = int(row.get("avg_rent", 0) / 1000) if row.get("avg_rent", 0) > 0 else "\u2014"
                avg_commute = int(row.get("avg_commute", 0)) if row.get("avg_commute", 0) > 0 else "\u2014"
                prop_count = int(row.get("total", 0))
                score_pct = min(100, int(row.get("total", 0) / max(1, top["total"].max()) * 100))
                is_active = (a_name == selected_area)
                active_cls = "active" if is_active else ""

                viewing_badge = '<span class="viewing-badge">Viewing</span>' if is_active else ''

                card_html = (
                    '<div class="area-card ' + active_cls + '" style="margin-bottom:0.75rem">'
                    + viewing_badge
                    + '<div class="area-header">'
                    + '<div class="area-name">' + str(a_name) + '</div>'
                    + '<div class="area-score">Score: ' + str(score_pct) + '</div>'
                    + '</div>'
                    + '<div class="area-name-en">' + romaji + '</div>'
                    + '<div class="area-stats">'
                    + '<div class="area-stat-box"><div class="stat-label">Avg Rent</div><div class="stat-value">\u00a5' + str(avg_rent_k) + 'K</div></div>'
                    + '<div class="area-stat-box"><div class="stat-label">Commute</div><div class="stat-value">' + str(avg_commute) + 'm</div></div>'
                    + '<div class="area-stat-box"><div class="stat-label">Listings</div><div class="stat-value">' + str(prop_count) + '</div></div>'
                    + '</div>'
                    + '<div class="area-footer"><span class="prop-count">' + str(prop_count) + ' properties</span><span class="view-link">View \u2192</span></div>'
                    + '<div class="score-bar" style="width:' + str(score_pct) + '%"></div>'
                    + '</div>'
                )
                with cols[i % num_cols]:
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.button(
                        f"Select {a_name}",
                        key=f"btn_{a_name}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                        on_click=set_selected_area,
                        args=(a_name,),
                    )
    st.markdown("---")

    # ── Radar chart (highlights selected area) ───────────────────────
    fig = build_radar_chart(top, _lang, selected_area=selected_area)
    st.plotly_chart(fig, use_container_width=True)

    # ── Listings table for selected area ─────────────────────────────
    if selected_area:
        area_df = df[df["area"] == selected_area].copy()
        show_listings_table(area_df, _wage, _lang)

    # ── CTA Footer ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="cta-footer animate-fade-in">
            <h3>Ready to Find Your Perfect Home?</h3>
            <p>
                Explore thousands of properties across Greater Tokyo.
                Our ML-powered analysis helps you discover hidden gems.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
