"""
Property listings table with colour-coded rent ratio.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.i18n import get_text


def show_listings_table(
    df: pd.DataFrame, wage: int, lang: str = "ja"
) -> None:
    """
    Display a styled property table for a single area, plus summary stats.
    """
    t = lambda key: get_text(key, lang)

    if df.empty:
        st.warning(t("err_no_results"))
        return

    area_name = df["area"].iloc[0] if "area" in df.columns else ""
    count = len(df)

    st.markdown(
        f"### {t('listings_header').format(area_name, count)}"
    )

    # --- Build display DataFrame ---
    display = pd.DataFrame()
    display[t("col_address")] = df["address"]
    display[t("col_layout")] = df["layout"]
    display[t("col_size")] = df["size"].astype(float).round(1)
    display[t("col_rent")] = (df["total_rent"] / 10_000).round(1)

    # Rent ratio
    if wage > 0:
        ratios = df["total_rent"] / wage * 100
        display[t("col_ratio")] = ratios.apply(lambda r: f"{r:.1f}%")
    else:
        display[t("col_ratio")] = "—"

    if "commute_min" in df.columns:
        display[t("col_commute")] = df["commute_min"].round(0).astype("Int64")

    display[t("col_access")] = df["access"]



    # Sort by rent ascending
    display = display.sort_values(by=t("col_rent"), ascending=True)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=min(400, 35 * len(display) + 38),
    )

    # --- Summary line ---
    avg_rent = df["total_rent"].mean() / 10_000
    avg_size = df["size"].astype(float).mean()
    avg_commute = (
        df["commute_min"].mean() if "commute_min" in df.columns else 0
    )

    st.caption(t("summary_line").format(avg_rent, avg_size, avg_commute))
