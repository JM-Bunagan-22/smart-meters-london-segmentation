"""
dashboard.py

Interactive Dash app for exploring the household consumption clusters
produced by run_daily_segmentation.py.

Run:
    python dashboard.py

Then open the URL it prints (usually http://127.0.0.1:8050) in your
browser. In a Codespace, VS Code will show a "port forwarded" popup with
an "Open in Browser" button instead.

Reads: ../data/processed/daily_cluster_features.csv
(produced automatically by run_daily_segmentation.py)
"""

import os

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# ---- EDIT IF NEEDED ----
DATA_PATH = "../data/processed/daily_cluster_features.csv"
# -------------------------

FEATURE_COLS = [
    "avg_daily_consumption", "std_daily_consumption", "avg_daily_mean",
    "avg_daily_peak", "avg_load_factor", "weekday_avg", "weekend_avg",
    "weekend_weekday_ratio", "coef_variation",
]


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run run_daily_segmentation.py first -- "
            "it generates this file automatically."
        )
    df = pd.read_csv(path)
    df["cluster"] = df["cluster"].astype(str)  # treat as categorical for coloring
    return df


df = load_data(DATA_PATH)
clusters = sorted(df["cluster"].unique(), key=int)

# z-scored version of features, used for the "relative profile" bar chart
zdf = df.copy()
for c in FEATURE_COLS:
    zdf[c] = (df[c] - df[c].mean()) / df[c].std()

app = Dash(__name__)
app.title = "Smart Meters London — Household Clusters"

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H2("London Smart Meters — Household Consumption Clusters"),
        html.P("Explore household segments based on daily consumption patterns."),

        html.Div([
            html.Label("Cluster:"),
            dcc.Dropdown(
                id="cluster-filter",
                options=[{"label": "All clusters", "value": "all"}] +
                        [{"label": f"Cluster {c}", "value": c} for c in clusters],
                value="all",
                clearable=False,
                style={"width": "300px"},
            ),
        ], style={"marginBottom": "20px"}),

        html.Div(id="summary-stats", style={"marginBottom": "20px", "fontSize": "16px"}),

        html.Div([
            html.Div([dcc.Graph(id="feature-profile-chart")], style={"flex": "1"}),
            html.Div([dcc.Graph(id="acorn-chart")], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px"}),

        html.H4("Household sample"),
        dcc.Graph(id="sample-table"),
    ],
)


@app.callback(
    Output("summary-stats", "children"),
    Output("feature-profile-chart", "figure"),
    Output("acorn-chart", "figure"),
    Output("sample-table", "figure"),
    Input("cluster-filter", "value"),
)
def update_dashboard(selected_cluster):
    if selected_cluster == "all":
        subset = df
        zsubset = zdf
    else:
        subset = df[df["cluster"] == selected_cluster]
        zsubset = zdf[zdf["cluster"] == selected_cluster]

    summary = f"{len(subset):,} households"
    if selected_cluster != "all":
        summary += f"  |  Cluster {selected_cluster}"

    # Feature profile: mean z-scored feature values (whole-df clusters if "all", else just this one vs 0)
    if selected_cluster == "all":
        profile = zdf.groupby("cluster")[FEATURE_COLS].mean().reset_index()
        profile_long = profile.melt(id_vars="cluster", var_name="feature", value_name="z_score")
        fig_profile = px.bar(
            profile_long, x="feature", y="z_score", color="cluster", barmode="group",
            title="Feature profile by cluster (z-scored)",
        )
    else:
        means = zsubset[FEATURE_COLS].mean().reset_index()
        means.columns = ["feature", "z_score"]
        fig_profile = px.bar(
            means, x="feature", y="z_score",
            title=f"Cluster {selected_cluster} feature profile (z-scored vs. overall average)",
        )
    fig_profile.update_layout(xaxis_tickangle=-40, height=420)

    # ACORN mix
    acorn_col = "Acorn_grouped" if "Acorn_grouped" in subset.columns else None
    if acorn_col:
        acorn_counts = subset[acorn_col].value_counts(normalize=True).mul(100).round(1).reset_index()
        acorn_counts.columns = ["Acorn_grouped", "pct"]
        fig_acorn = px.bar(
            acorn_counts, x="Acorn_grouped", y="pct",
            title="ACORN group mix (%)", labels={"pct": "% of households"},
        )
    else:
        fig_acorn = px.bar(title="ACORN data not available")
    fig_acorn.update_layout(height=420)

    # Sample table (first 10 rows)
    sample = subset.head(10)[["LCLid", "cluster"] + FEATURE_COLS[:4] +
                              ([acorn_col] if acorn_col else [])]
    sample = sample.round(2)
    fig_table = {
        "data": [{
            "type": "table",
            "header": {"values": list(sample.columns), "fill_color": "#2c3e50", "font": {"color": "white"}},
            "cells": {"values": [sample[c] for c in sample.columns]},
        }],
        "layout": {"height": 350, "margin": {"t": 10, "b": 10}},
    }

    return summary, fig_profile, fig_acorn, fig_table


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
