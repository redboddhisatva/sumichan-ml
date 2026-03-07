"""
Plotly radar chart for the top recommended areas.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from core.i18n import get_text

# Colour palette — JSX-matched Navy/Slate
COLORS = [
    "#1e3a8a", "#334155", "#1e40af", "#475569", "#0f172a",
    "#1e293b", "#3b82f6", "#64748b", "#2563eb", "#94a3b8",
]


def build_radar_chart(
    top_areas: pd.DataFrame,
    lang: str = "ja",
    selected_area: str | None = None,
) -> go.Figure:
    """
    Build a 4-axis radar chart comparing the top areas.
    When *selected_area* is set, that trace is highlighted and the rest dimmed.

    Expected columns: area, cost_score, commute_score, value_score, density_score, total
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
            bgcolor="#FAFBFC",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#E5E7EB",
                tickfont=dict(size=10, color="#6B7280"),
                ticksuffix="",
            ),
            angularaxis=dict(
                gridcolor="#E5E7EB",
                tickfont=dict(size=12, color="#1A1A2E", family="DM Sans"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1A1A2E", family="Inter"),
        showlegend=False,
        height=420,
        margin=dict(l=60, r=60, t=30, b=30),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=13,
            font_family="'Inter', 'Noto Sans JP', sans-serif",
            font_color="#1A1A2E",
            bordercolor="#E5E7EB"
        ),
    )

    return fig
