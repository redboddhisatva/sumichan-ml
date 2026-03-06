"""
Plotly radar chart for the top recommended areas.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from core.i18n import get_text

# Colour palette — Camel Fair
COLORS = [
    "#dda95b", "#a67739", "#715026", "#4d3616", "#efdcae",
    "#c49a4a", "#8b6230", "#5e411e", "#d6be8a", "#977548",
]


def build_radar_chart(
    top_areas: pd.DataFrame,
    lang: str = "ja",
    selected_area: str | None = None,
) -> go.Figure:
    """
    Build a 3-axis radar chart comparing the top areas.
    When *selected_area* is set, that trace is highlighted and the rest dimmed.

    Expected columns: area, cost_score, commute_score, value_score, total
    """
    categories = [
        get_text("axis_cost", lang),
        get_text("axis_commute", lang),
        get_text("axis_value", lang),
        get_text("axis_density", lang),
    ]

    fig = go.Figure()

    for i, (_, row) in enumerate(top_areas.iterrows()):
        area_name = row["area"]
        is_selected = (selected_area is None) or (area_name == selected_area)
        
        if not is_selected:
            continue

        values = [row["cost_score"], row["commute_score"], row["value_score"], row["density_score"]]
        # Close the polygon
        values_closed = values + [values[0]]
        cats_closed = categories + [categories[0]]

        color = COLORS[i % len(COLORS)]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                name=f'{area_name}  ({int(row["total"])})',
                line=dict(
                    color=color,
                    width=3,
                ),
                fill="toself",
                fillcolor=color,
                opacity=0.7,
                hovertemplate=f"<b>{area_name}</b><br>Score: {int(row['total'])}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#e0d3be",
                tickfont=dict(size=10, color="#715026"),
                ticksuffix="",
            ),
            angularaxis=dict(
                gridcolor="#e0d3be",
                tickfont=dict(size=12, color="#4d3616"),
            ),
        ),
        paper_bgcolor="#ffffff",
        font=dict(color="#4d3616"),
        showlegend=False,
        height=420,
        margin=dict(l=60, r=60, t=30, b=30),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#ffffff",
            font_size=13,
            font_family="'Montserrat', 'Noto Sans JP', sans-serif",
            font_color="#4d3616",
            bordercolor="#e0d3be"
        ),
    )

    return fig
