"""
🗼 スミちゃん · Tokyo Living Finder
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
    calculate_ml_deal_score
)
from ui.radar import build_radar_chart, COLORS
from ui.listings import show_listings_table

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="スミちゃん · Tokyo Living Finder",
    page_icon="🗼",
    layout="wide",
)

# ── Custom CSS for polish ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;600;700&display=swap');

    /* Global font override */
    html, body, [class*="css"], .stMarkdown, .stText,
    [data-testid="stSidebar"], [data-testid="stHeader"],
    input, select, textarea, button, label, p, span, div, h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', 'Noto Sans JP', sans-serif !important;
    }
    /* metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0d3be;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(77,54,22,0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
        color: #4d3616;
    }
    [data-testid="stMetricLabel"] {
        color: #715026;
    }
    /* radio buttons horizontal */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        gap: 8px;
        flex-wrap: wrap;
    }
    div[data-testid="stRadio"] > div > label {
        background: #ffffff;
        border: 1px solid #e0d3be;
        border-radius: 8px;
        padding: 6px 16px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] > div > label:hover {
        border-color: #a67739;
    }
    /* sidebar tweaks */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    /* hide the built-in sidebar collapse chevron (>>) */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load station list for dropdown ───────────────────────────────────────
@st.cache_data
def _load_station_names() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "data", "stations.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data.keys())


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
    region_keys = ["tokyo", "saitama", "chiba", "kanagawa"]
    region_labels = {k: t(k) for k in region_keys}
    selected_labels = st.multiselect(
        t("region_label"),
        options=list(region_labels.values()),
        default=[region_labels["tokyo"]],
    )
    label_to_key = {v: k for k, v in region_labels.items()}
    selected_regions = [label_to_key[lbl] for lbl in selected_labels]

    # Wage
    wage = st.number_input(
        t("wage_label"),
        min_value=100_000,
        max_value=2_000_000,
        value=300_000,
        step=10_000,
    )

    # Workplace — dropdown from stations.json
    default_idx = STATION_NAMES.index("新宿") if "新宿" in STATION_NAMES else 0
    workplace = st.selectbox(
        t("workplace_label"),
        options=STATION_NAMES,
        index=default_idx,
    )

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
    df['deal_score'] = df.apply(
        lambda row: calculate_ml_deal_score(row['total_rent'], row['predicted_rent']), axis=1
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

    for cluster in cluster_order:
        cluster_df = top[top["cluster"] == cluster]
        if not cluster_df.empty:
            st.markdown(f"##### {cluster}")
            num_cols = min(4, len(cluster_df))
            cols = st.columns(num_cols) if num_cols > 0 else []
            for i, (_, row) in enumerate(cluster_df.iterrows()):
                a_name = row["area"]
                score = int(row["total"])
                is_active = (a_name == selected_area)
                cols[i % num_cols].button(
                    f"{a_name} · {score}",
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
