"""
Bilingual string dictionary for English / Japanese UI.
All user-facing text lives here so the entire app can switch language
with a single sidebar toggle.
"""

STRINGS = {
    "ja": {
        # Sidebar
        "app_title": "🗼 スミちゃん",
        "app_subtitle": "あなたの理想の住まいを。",
        "lang_label": "🌐 言語 / Language",
        "region_label": "対象エリア",
        "wage_label": "手取り月収（円）",
        "workplace_label": "勤務先の最寄り駅",
        "workplace_placeholder": "例: 新宿",
        "layout_label": "間取り",
        "layout_all": "すべて",
        "search_btn": "🔍 エリアを探す",
        "scoring_note": "💴 家賃 40% · 🚃 通勤 40% · 📐 広さ 20%",

        # Region names
        "tokyo": "東京",
        "saitama": "埼玉",
        "chiba": "千葉",
        "kanagawa": "神奈川",

        # Main area
        "loading": "データを読み込んでいます…",
        "metric_listings": "物件数",
        "metric_areas": "エリア数",
        "metric_layout": "間取り",
        "metric_workplace": "勤務先",
        "top_areas_title": "おすすめエリア TOP {}",
        "select_area": "エリアを選択",
        "listings_header": "{} の物件一覧 · {}件",
        "summary_line": "平均家賃: {:.1f}万円 ｜ 平均広さ: {:.1f}㎡ ｜ 平均通勤: {:.0f}分",

        # Table columns
        "col_address": "住所",
        "col_layout": "間取り",
        "col_size": "広さ(㎡)",
        "col_rent": "家賃(万円)",
        "col_market_value": "市場価値(万円)",
        "col_deal_score": "割安スコア",
        "col_ratio": "家賃比率",
        "col_commute": "通勤推定(分)",
        "col_access": "アクセス",
        "table_cluster": "エリア分類",


        # Radar chart axes
        "axis_cost": "家賃",
        "axis_commute": "通勤",
        "axis_value": "割安度",
        "axis_density": "人口密度",

        # Errors
        "err_no_workplace": "⚠️ 勤務先の最寄り駅を入力してください。",
        "err_no_results": "該当するエリアが見つかりませんでした。条件を広げてみてください（間取り「すべて」、エリアを追加など）。",
        "err_fetch_failed": "⚠️ データの取得に失敗しました（HTTPステータス: {}）",
    },
    "en": {
        # Sidebar
        "app_title": "🗼 Sumi-chan",
        "app_subtitle": "Home Sweet Sumika",
        "lang_label": "🌐 言語 / Language",
        "region_label": "Target Regions",
        "wage_label": "Monthly Take-Home Pay (¥)",
        "workplace_label": "Nearest Station to Workplace",
        "workplace_placeholder": "e.g. Shinjuku (新宿)",
        "layout_label": "Layout",
        "layout_all": "All",
        "search_btn": "🔍 Find Areas",
        "scoring_note": "💴 Rent 40% · 🚃 Commute 40% · 📐 Space 20%",

        # Region names
        "tokyo": "Tokyo",
        "saitama": "Saitama",
        "chiba": "Chiba",
        "kanagawa": "Kanagawa",

        # Main area
        "loading": "Loading data…",
        "metric_listings": "Listings",
        "metric_areas": "Areas",
        "metric_layout": "Layout",
        "metric_workplace": "Workplace",
        "top_areas_title": "Recommended Areas TOP {}",
        "select_area": "Select area",
        "listings_header": "Listings in {} · {} properties",
        "summary_line": "Avg rent: ¥{:.1f}万 ｜ Avg size: {:.1f}㎡ ｜ Avg commute: {:.0f}min",

        # Table columns
        "col_address": "Address",
        "col_layout": "Layout",
        "col_size": "Size (㎡)",
        "col_rent": "Rent (万¥)",
        "col_market_value": "Market Value (万¥)",
        "col_deal_score": "Deal Score",
        "col_ratio": "Rent Ratio",
        "col_commute": "Est. Commute (min)",
        "col_access": "Access",


        # Radar chart axes
        "axis_cost": "Cost",
        "axis_commute": "Commute",
        "axis_value": "Deal Score",
        "axis_density": "Population Density",

        # Errors
        "err_no_workplace": "⚠️ Please enter your workplace station.",
        "err_no_results": "No matching areas found. Try broadening your filters (set layout to 'All', add more regions).",
        "err_fetch_failed": "⚠️ Failed to fetch data (HTTP status: {})",
    },
}


def get_text(key: str, lang: str = "ja") -> str:
    """Return the UI string for *key* in the given language."""
    return STRINGS.get(lang, STRINGS["ja"]).get(key, key)
