"""
dashboard.py
------------
Interactive Plotly Dash dashboard for pharmaceutical supply chain risk analytics.

Panels:
  1. Header KPI Cards
  2. Geospatial Risk Map (folium iframe)
  3. Cost Heatmap (transport costs by cluster and mode)
  4. Delivery Time Bottleneck Chart (source zone vs pharmacy point)
  5. Capacity Utilization Bar Chart (by region & configuration)
  6. Disruption Simulation – shortage probability by region/drug type
  7. Monte Carlo Scenario Sliders (surge_pct × failure_pct)
  8. ML Bottleneck Predictor – risk probability chart & feature importances
  9. LP Optimization Allocation sankey / bar

All data is sourced strictly from the 7 provided datasets.
"""

from __future__ import annotations

import base64
import tempfile
import warnings
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update
from dash.exceptions import PreventUpdate

from data_loader import load_all_workbooks, WorkbookBundle
from risk_engine   import compute_risk_scores
from simulation    import run_monte_carlo, sensitivity_sweep, shortage_summary
from ml_model      import build_feature_matrix, train_model, predict
from optimizer     import build_and_solve

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MAP_HTML    = BASE_DIR / "assets" / "risk_map.html"

# ── colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "bg":          "#0F1117",
    "card":        "#1A1D27",
    "border":      "#2D3148",
    "accent1":     "#6C63FF",
    "accent2":     "#00D4AA",
    "accent3":     "#FF6B6B",
    "accent4":     "#FFB347",
    "text":        "#E8EAF6",
    "muted":       "#8C8FA0",
    "critical":    "#FF4757",
    "high":        "#FFA502",
    "low":         "#2ED573",
    "gradient1":   "linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%)",
}

RISK_COLOUR_MAP = {"CRITICAL": COLOURS["critical"], "HIGH": COLOURS["high"], "LOW": COLOURS["low"]}


# ── data loader (cached once at startup) ─────────────────────────────────────

class AppData:
    """Singleton that loads and caches all derived data at startup."""

    def __init__(self, base_path: str | Path | None = None):
        print("[Dashboard] Loading workbooks …")
        self.bundles       = load_all_workbooks(base_path)
        print("[Dashboard] Computing risk scores …")
        self.region_risk, self.granular_risk = compute_risk_scores(self.bundles)
        print("[Dashboard] Running Monte Carlo …")
        self.sim_df        = run_monte_carlo(self.bundles, n_sims=1000)
        self.sim_summary   = shortage_summary(self.sim_df)
        print("[Dashboard] Training ML model …")
        self.features      = build_feature_matrix(self.bundles)
        self.ml_model, self.ml_metrics = train_model(self.features)
        self.predictions   = predict(self.ml_model, self.features)
        print("[Dashboard] Running LP optimizer …")
        self.alloc_df, self.total_cost, self.opt_report = build_and_solve(self.bundles)
        
        print("[Dashboard] Loading upgrader and rerouting plans …")
        try:
            self.cap_plan = pd.read_csv(Path("pipeline_outputs/capacity_plan.csv"))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.cap_plan = pd.DataFrame()
        try:
            self.reroute_plan = pd.read_csv(Path("pipeline_outputs/rerouting_recommendations.csv"))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.reroute_plan = pd.DataFrame()
            
        print("[Dashboard] Building folium map …")
        self.map_html_path = self._build_risk_map()
        print("[Dashboard] Ready.")

    # ── geo map ─────────────────────────────────────────────────────────────

    def _build_risk_map(self) -> str:
        """Build folium risk map; returns path to saved HTML."""
        MAP_HTML.parent.mkdir(parents=True, exist_ok=True)

        # Centre on Trabzon region
        fmap = folium.Map(location=[40.95, 39.5], zoom_start=8,
                          tiles="CartoDB dark_matter")

        risk_by_region = dict(zip(
            self.region_risk["region"].astype(str),
            self.region_risk["risk_score"].astype(float)
        ))

        region_colours = {
            "Trabzon": "#6C63FF",
            "Rize":    "#00D4AA",
            "Ordu":    "#FFB347",
            "Giresun": "#FF6B6B",
        }

        # Plot all pharmacy points from all geo sheets
        for sheet_name, frame in self.bundles["geo"].cleaned.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            if "lat" not in frame.columns:
                continue
            # Determine region from sheet name
            region = None
            for r in ["Trabzon", "Rize", "Ordu", "Giresun"]:
                if r.lower() in sheet_name.lower():
                    region = r
                    break

            rs = risk_by_region.get(region, 50.0)
            if rs >= 75:
                colour = COLOURS["critical"]
                label  = "CRITICAL"
            elif rs >= 50:
                colour = COLOURS["high"]
                label  = "HIGH"
            else:
                colour = COLOURS["low"]
                label  = "LOW"

            # Use region colour for source warehouses
            is_wh = "S-WH" in sheet_name or "P-WH" in sheet_name
            for _, row in frame.dropna(subset=["lat", "lon"]).iterrows():
                if is_wh:
                    folium.Marker(
                        location=[row["lat"], row["lon"]],
                        popup=folium.Popup(
                            f"<b>{sheet_name}</b><br>Warehouse point", max_width=200
                        ),
                        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                    ).add_to(fmap)
                else:
                    popup_html = (
                        f"<b>{row.get('point_id', 'Pharmacy')}</b><br>"
                        f"Region: {region or sheet_name}<br>"
                        f"Risk: {label} ({rs:.1f})<br>"
                        f"Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}"
                    )
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=5,
                        color=colour,
                        fill=True,
                        fill_color=colour,
                        fill_opacity=0.8,
                        popup=folium.Popup(popup_html, max_width=220),
                        tooltip=row.get("point_id", "pharmacy"),
                    ).add_to(fmap)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:999;
                    background:#1A1D27;padding:12px 18px;border-radius:8px;
                    border:1px solid #2D3148;color:#E8EAF6;font-size:13px;">
          <b style="color:#6C63FF;">Risk Level</b><br>
          <span style="color:#FF4757;">●</span> Critical (&ge;75)<br>
          <span style="color:#FFA502;">●</span> High (50–74)<br>
          <span style="color:#2ED573;">●</span> Low (&lt;50)<br>
          <span style="color:#4A90E2;">⌂</span> Warehouse
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))
        fmap.save(str(MAP_HTML))
        return str(MAP_HTML)

    # ── chart builders ──────────────────────────────────────────────────────

    def cost_heatmap(self) -> go.Figure:
        VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
        frames = []
        for frame in self.bundles["cost"].cleaned.values():
            if isinstance(frame, pd.DataFrame) and {"region", "mode", "value"}.issubset(frame.columns):
                frame = frame.copy()
                frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
                frame = frame[frame["region"].isin(VALID_REGIONS)].dropna(subset=["value"])
                # Keep only clean mode names (e.g. TC0, RC0, GC0, OC0)
                frame = frame[frame["mode"].str.match(r'^[A-Z]{2}\d$', na=False)]
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return go.Figure()
        df = pd.concat(frames, ignore_index=True)
        pivot = df.pivot_table(index="region", columns="mode", values="value", aggfunc="mean")
        fig = px.imshow(
            pivot, text_auto=".1f", aspect="auto",
            color_continuous_scale="Viridis",
            title="Transport Cost Matrix — Region × Mode",
        )
        _style_fig(fig)
        fig.update_layout(coloraxis_colorbar=dict(title="Cost"))
        return fig

    def time_bottleneck(self) -> go.Figure:
        VALID_REGIONS = {"Rize", "Trabzon", "Ordu", "Giresun"}
        VALID_ZONES   = {"A", "B", "C", "D", "E", "F", "G", "H"}
        frames = []
        for frame in self.bundles["time"].cleaned.values():
            if isinstance(frame, pd.DataFrame) and {"source_zone", "delivery_time", "region"}.issubset(frame.columns):
                frame = frame.copy()
                frame["delivery_time"] = pd.to_numeric(frame["delivery_time"], errors="coerce")
                frame = frame[
                    frame["region"].isin(VALID_REGIONS) &
                    frame["source_zone"].isin(VALID_ZONES) &
                    (frame["delivery_time"] > 0)
                ].dropna(subset=["delivery_time"])
                if not frame.empty:
                    frames.append(frame[["region", "source_zone", "delivery_time"]])
        if not frames:
            return go.Figure()
        df = pd.concat(frames, ignore_index=True)
        grp = df.groupby(["region", "source_zone"])["delivery_time"].agg(
            mean_time="mean", max_time="max"
        ).reset_index()
        fig = px.bar(
            grp, x="source_zone", y="mean_time", color="region",
            barmode="group",
            error_y=grp["max_time"] - grp["mean_time"],
            title="Delivery Time Bottleneck — Mean & Max per Source Zone",
            labels={"mean_time": "Mean Delivery Time (min)", "source_zone": "Source Zone"},
        )
        _style_fig(fig)
        return fig

    def capacity_utilization(self) -> go.Figure:
        frames = []
        for frame in self.bundles["capacity"].cleaned.values():
            if isinstance(frame, pd.DataFrame) and {"region", "configuration", "metric", "value"}.issubset(frame.columns):
                frames.append(frame.copy())
        if not frames:
            return go.Figure()
        cap_df = pd.concat(frames, ignore_index=True)
        demand_totals = {}
        for frame in self.bundles["demand"].cleaned.values():
            if isinstance(frame, pd.DataFrame) and "region" in frame.columns:
                for region, grp in frame.groupby("region"):
                    d = pd.to_numeric(grp["demand"], errors="coerce").fillna(0).sum()
                    demand_totals[region] = demand_totals.get(region, 0) + d

        base = (
            cap_df[cap_df["metric"] == "capacity"]
            .groupby(["region", "configuration"])["value"]
            .sum()
            .reset_index(name="total_capacity")
        )
        base["total_demand"] = base["region"].map(demand_totals).fillna(0)
        base["utilization"]  = (base["total_demand"] / base["total_capacity"]).clip(upper=2)

        fig = px.bar(
            base, x="configuration", y="utilization", color="region",
            barmode="group",
            title="Capacity Utilization — Demand / Capacity (per Configuration)",
            labels={"utilization": "Utilization Ratio", "configuration": "P= Configuration"},
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color=COLOURS["critical"],
                      annotation_text="Full capacity", annotation_position="top right")
        _style_fig(fig)
        return fig

    def shortage_chart(self, surge: float = 0.4, failure: float = 0.35) -> go.Figure:
        from simulation import run_monte_carlo
        sim = run_monte_carlo(self.bundles, n_sims=500, surge_pct=surge, failure_pct=failure, seed=42)
        fig = px.bar(
            sim, x="region", y="shortage_probability",
            color="drug_type", barmode="group",
            title=f"Shortage Probability — Surge={int(surge*100)}%, Failure={int(failure*100)}%",
            labels={"shortage_probability": "P(Shortage)", "region": "Region"},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color=COLOURS["accent3"],
                      annotation_text="50% threshold")
        fig.update_yaxes(range=[0, 1])
        _style_fig(fig)
        return fig

    def sweep_heatmap(self) -> go.Figure:
        from simulation import sensitivity_sweep
        sweep = sensitivity_sweep(self.bundles, n_sims=200)
        pivot = sweep.pivot(index="failure_pct", columns="surge_pct", values="mean_shortage_prob")
        fig = px.imshow(
            pivot, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Scenario Sweep: Mean Shortage Probability (Failure % × Surge %)",
            labels={"x": "Demand Surge %", "y": "Capacity Failure %", "color": "P(Shortage)"},
        )
        _style_fig(fig)
        return fig

    def ml_risk_chart(self) -> go.Figure:
        df = self.predictions.copy()
        df = df.sort_values("risk_probability", ascending=False)
        fig = px.bar(
            df, x="source_zone", y="risk_probability", color="region",
            barmode="group",
            title="ML-Predicted Bottleneck Risk Probability per Source Zone",
            labels={"risk_probability": "P(High Risk)", "source_zone": "Source Zone"},
        )
        fig.add_hline(y=0.5, line_dash="dot", line_color=COLOURS["accent3"],
                      annotation_text="Decision boundary")
        fig.update_yaxes(range=[0, 1])
        _style_fig(fig)
        return fig

    def feature_importance_chart(self) -> go.Figure:
        imp = self.ml_metrics.get("feature_importances", {})
        if not imp:
            return go.Figure()
        df  = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
        df  = df.sort_values("importance")
        fig = px.bar(
            df, x="importance", y="feature", orientation="h",
            title="XGBoost Feature Importances",
            labels={"importance": "Importance Score", "feature": "Feature"},
            color="importance", color_continuous_scale="Viridis",
        )
        _style_fig(fig)
        return fig

    def allocation_chart(self) -> go.Figure:
        if self.alloc_df.empty:
            return go.Figure().add_annotation(text="No LP solution available",
                                              showarrow=False, font=dict(size=18, color="white"))
        grp = self.alloc_df.groupby(["region", "warehouse"])["route_cost"].sum().reset_index()
        fig = px.bar(
            grp, x="warehouse", y="route_cost", color="region",
            barmode="stack",
            title=f"LP Optimal Allocation — Total Cost: {self.total_cost:,.2f}",
            labels={"route_cost": "Route Cost", "warehouse": "Warehouse"},
        )
        _style_fig(fig)
        return fig

    def region_risk_chart(self) -> go.Figure:
        df = self.region_risk.copy()
        df["colour"] = df["risk_label"].map(RISK_COLOUR_MAP)
        fig = go.Figure(go.Bar(
            x=df["region"],
            y=df["risk_score"],
            marker_color=df["colour"],
            text=df["risk_label"],
            textposition="outside",
        ))
        fig.update_layout(
            title="Composite Risk Score by Region",
            xaxis_title="Region",
            yaxis_title="Risk Score (0–100)",
            yaxis_range=[0, 105],
        )
        _style_fig(fig)
        fig.add_hline(y=75, line_dash="dot", line_color=COLOURS["critical"],
                      annotation_text="Critical threshold (75)")
        fig.add_hline(y=50, line_dash="dot", line_color=COLOURS["high"],
                      annotation_text="High threshold (50)")
        return fig

    def capacity_roadmap_chart(self) -> go.Figure:
        if self.cap_plan.empty:
            fig = go.Figure().add_annotation(text="No Capacity Plan available", showarrow=False, font=dict(size=18, color=COLOURS["muted"]))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            _style_fig(fig)
            return fig
        df = self.cap_plan.melt(id_vars=["region"], value_vars=["current_P", "recommended_P"],
                                var_name="Type", value_name="Configuration")
        # Ensure categorical ordering P=1 ... P=5
        df["Configuration"] = pd.Categorical(df["Configuration"], categories=["P=1", "P=2", "P=3", "P=4", "P=5"], ordered=True)
        # Convert to numeric for bar chart
        df["P_level"] = df["Configuration"].apply(lambda x: int(str(x).replace("P=", "")))
        fig = px.bar(
            df, x="region", y="P_level", color="Type", barmode="group",
            title="Capacity Upgrade Roadmap",
            labels={"P_level": "P= Configuration Level", "region": "Region"},
            color_discrete_map={"current_P": COLOURS["muted"], "recommended_P": COLOURS["accent2"]}
        )
        fig.update_yaxes(tickvals=[1, 2, 3, 4, 5], ticktext=["P=1", "P=2", "P=3", "P=4", "P=5"])
        _style_fig(fig)
        return fig
        
    def rerouting_savings_chart(self) -> go.Figure:
        if self.reroute_plan.empty:
            fig = go.Figure().add_annotation(text="No Suboptimal Routings Found", 
                                              showarrow=False, font=dict(size=18, color=COLOURS["low"]))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            _style_fig(fig)
            return fig
        df = self.reroute_plan.head(10).sort_values("annual_cost_saving", ascending=True)
        df["label"] = df["region"] + " (" + df["drug_type"] + ")<br>" + df["current_warehouse"] + " → " + df["recommended_warehouse"]
        fig = px.bar(
            df, x="annual_cost_saving", y="label", orientation="h",
            title="Top Rerouting Savings Opportunities",
            labels={"annual_cost_saving": "Annual Cost Saving ($)", "label": "Region (Drug Type) & Change"},
            color="annual_cost_saving", color_continuous_scale="Viridis",
        )
        _style_fig(fig)
        return fig


# ── style helper ─────────────────────────────────────────────────────────────

def _style_fig(fig: go.Figure) -> None:
    fig.update_layout(
        paper_bgcolor=COLOURS["card"],
        plot_bgcolor=COLOURS["card"],
        font=dict(color=COLOURS["text"], family="Inter, sans-serif", size=12),
        title_font=dict(size=15, color=COLOURS["text"]),
        legend=dict(bgcolor=COLOURS["bg"], bordercolor=COLOURS["border"], borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor=COLOURS["border"], zerolinecolor=COLOURS["border"])
    fig.update_yaxes(gridcolor=COLOURS["border"], zerolinecolor=COLOURS["border"])


# ── KPI card ─────────────────────────────────────────────────────────────────

def _kpi_card(title: str, value: str, colour: str, icon: str = "◈") -> html.Div:
    return html.Div([
        html.Div(icon, style={"fontSize": "28px", "color": colour, "marginBottom": "6px"}),
        html.Div(value, style={"fontSize": "26px", "fontWeight": "700",
                                "color": COLOURS["text"], "lineHeight": "1.1"}),
        html.Div(title, style={"fontSize": "12px", "color": COLOURS["muted"],
                                "marginTop": "4px", "textTransform": "uppercase",
                                "letterSpacing": "0.08em"}),
    ], style={
        "background":   COLOURS["card"],
        "border":       f"1px solid {colour}33",
        "borderLeft":   f"3px solid {colour}",
        "borderRadius": "10px",
        "padding":      "20px 18px",
        "flex":         "1",
        "minWidth":     "160px",
    })


# ── section header ───────────────────────────────────────────────────────────

def _section(title: str) -> html.Div:
    return html.Div([
        html.H3(title, style={"margin": "0", "color": COLOURS["text"],
                               "fontSize": "16px", "fontWeight": "600"}),
        html.Hr(style={"border": f"1px solid {COLOURS['accent1']}44",
                        "margin": "8px 0 20px 0"}),
    ])


# ── layout builder ───────────────────────────────────────────────────────────

def _build_layout(app_data: AppData) -> html.Div:
    sim_summary = app_data.sim_summary
    opt_report  = app_data.opt_report
    risk_df     = app_data.region_risk
    ml_metrics  = app_data.ml_metrics

    # KPI values
    max_risk  = f"{risk_df['risk_score'].max():.1f}"
    crit_cnt  = str(int((risk_df["risk_label"] == "CRITICAL").sum()))
    cv_auc    = f"{ml_metrics.get('cv_roc_auc_mean', 0):.3f}"
    total_c   = f"{opt_report.get('total_cost', 0):,.2f}"
    shortage_p= f"{sim_summary.get('max_shortage_probability', 0):.0%}"

    return html.Div([

        # ── top bar ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("⬡", style={"color": COLOURS["accent1"], "fontSize": "28px",
                                       "marginRight": "10px"}),
                html.Span("PharmaChain Risk Analytics",
                          style={"fontSize": "22px", "fontWeight": "700",
                                 "color": COLOURS["text"]}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div("Black Sea Region · Pharmaceutical Warehouse Supply Chain",
                     style={"color": COLOURS["muted"], "fontSize": "12px",
                            "marginTop": "4px"}),
        ], style={
            "background":   COLOURS["card"],
            "borderBottom": f"1px solid {COLOURS['border']}",
            "padding":      "18px 32px",
        }),

        # ── main content ─────────────────────────────────────────────────────
        html.Div([

            # ── KPI row ──────────────────────────────────────────────────────
            html.Div([
                _kpi_card("Max Risk Score",       max_risk,   COLOURS["accent1"], "⚠"),
                _kpi_card("Critical Routes",       crit_cnt,   COLOURS["critical"], "🔴"),
                _kpi_card("Max Shortage Prob.",    shortage_p, COLOURS["accent3"], "📉"),
                _kpi_card("ML Model AUC",          cv_auc,     COLOURS["accent2"], "🤖"),
                _kpi_card("Optimal Total Cost",    total_c,    COLOURS["accent4"], "💰"),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                      "marginBottom": "28px"}),

            # ── row 1: risk map + region risk bar ────────────────────────────
            _section("① Geospatial Risk Map & Regional Risk Scores"),
            html.Div([
                html.Div([
                    html.Iframe(
                        src="/assets/risk_map.html",
                        style={"width": "100%", "height": "480px",
                               "border": "none", "borderRadius": "10px"},
                    ),
                ], style={"flex": "1.4", "minWidth": "400px"}),
                html.Div([
                    dcc.Graph(id="region-risk-chart",
                              figure=app_data.region_risk_chart(),
                              style={"height": "480px"},
                              config={"displayModeBar": False}),
                ], style={"flex": "1", "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

            # ── row 2: cost heatmap + time bottleneck ────────────────────────
            _section("② Cost Heatmap & Delivery Time Bottleneck"),
            html.Div([
                dcc.Graph(id="cost-heatmap",
                          figure=app_data.cost_heatmap(),
                          style={"flex": "1", "height": "380px"},
                          config={"displayModeBar": False}),
                dcc.Graph(id="time-bottleneck",
                          figure=app_data.time_bottleneck(),
                          style={"flex": "1", "height": "380px"},
                          config={"displayModeBar": False}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

            # ── row 3: capacity utilization ──────────────────────────────────
            _section("③ Warehouse Capacity Utilization"),
            dcc.Graph(id="capacity-chart",
                      figure=app_data.capacity_utilization(),
                      style={"height": "380px", "marginBottom": "28px"},
                      config={"displayModeBar": False}),

            # ── row 4: disruption simulation ─────────────────────────────────
            _section("④ Disruption Simulation — Shortage Probability"),
            html.Div([
                html.Div([
                    html.Label("Demand Surge %", style={"color": COLOURS["muted"],
                                                         "fontSize": "13px"}),
                    dcc.Slider(id="surge-slider", min=0, max=50, step=5, value=40,
                               marks={i: f"{i}%" for i in range(0, 55, 10)},
                               tooltip={"placement": "bottom", "always_visible": False}),
                    html.Label("Capacity Failure %", style={"color": COLOURS["muted"],
                                                              "fontSize": "13px",
                                                              "marginTop": "12px"}),
                    dcc.Slider(id="failure-slider", min=0, max=50, step=5, value=35,
                               marks={i: f"{i}%" for i in range(0, 55, 10)},
                               tooltip={"placement": "bottom", "always_visible": False}),
                ], style={"flex": "0 0 260px", "background": COLOURS["card"],
                          "borderRadius": "10px", "padding": "24px 20px",
                          "border": f"1px solid {COLOURS['border']}"}),
                dcc.Graph(id="shortage-chart",
                          figure=app_data.shortage_chart(),
                          style={"flex": "1", "height": "380px"},
                          config={"displayModeBar": False}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

            # ── row 5: sweep heatmap ─────────────────────────────────────────
            _section("⑤ Monte Carlo Scenario Sweep"),
            dcc.Graph(id="sweep-heatmap",
                      figure=app_data.sweep_heatmap(),
                      style={"height": "360px", "marginBottom": "28px"},
                      config={"displayModeBar": False}),

            # ── row 6: ML predictions + feature importance ───────────────────
            _section("⑥ ML Bottleneck Predictor (XGBoost)"),
            html.Div([
                html.Div([
                    html.Div(f"CV ROC-AUC: {ml_metrics.get('cv_roc_auc_mean','N/A')} "
                             f"± {ml_metrics.get('cv_roc_auc_std','N/A')}",
                             style={"color": COLOURS["accent2"], "fontSize": "14px",
                                    "fontWeight": "600", "marginBottom": "12px"}),
                    dcc.Graph(id="ml-risk-chart",
                              figure=app_data.ml_risk_chart(),
                              style={"height": "340px"},
                              config={"displayModeBar": False}),
                ], style={"flex": "1.4"}),
                dcc.Graph(id="feature-importance",
                          figure=app_data.feature_importance_chart(),
                          style={"flex": "1", "height": "380px"},
                          config={"displayModeBar": False}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

            # ── row 7: LP allocation ─────────────────────────────────────────
            _section("⑦ LP Optimal Warehouse Allocation"),
            html.Div([
                html.Div([
                    html.Div(f"Solver Status: {opt_report.get('status','N/A')}",
                             style={"color": COLOURS["accent2"], "fontSize": "13px",
                                    "marginBottom": "4px"}),
                    html.Div(f"Variables: {opt_report.get('n_decision_vars','N/A')} | "
                             f"Constraints: {opt_report.get('n_constraints','N/A')}",
                             style={"color": COLOURS["muted"], "fontSize": "12px",
                                    "marginBottom": "16px"}),
                    dcc.Graph(id="allocation-chart",
                              figure=app_data.allocation_chart(),
                              style={"height": "360px"},
                              config={"displayModeBar": False}),
                ], style={"flex": "1.4"}),
                html.Div([
                    _section("Cost Breakdown by Drug Type"),
                    *[
                        html.Div([
                            html.Span(dtype.replace("_", " ").title(),
                                      style={"color": COLOURS["muted"], "fontSize": "13px",
                                             "flex": "1"}),
                            html.Span(f"{cost:,.2f}",
                                      style={"color": COLOURS["text"], "fontWeight": "600"}),
                        ], style={"display": "flex", "justifyContent": "space-between",
                                  "padding": "10px 0",
                                  "borderBottom": f"1px solid {COLOURS['border']}"})
                        for dtype, cost in opt_report.get("cost_by_drug_type", {}).items()
                    ],
                ], style={"flex": "0 0 280px", "background": COLOURS["card"],
                          "borderRadius": "10px", "padding": "24px 20px",
                          "border": f"1px solid {COLOURS['border']}"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "40px"}),

            # ── row 8: upgrades & rerouting ──────────────────────────────────
            _section("⑧ Capacity Upgrades & Rerouting Roadmap"),
            html.Div([
                dcc.Graph(id="capacity-roadmap",
                          figure=app_data.capacity_roadmap_chart(),
                          style={"flex": "1", "height": "380px"},
                          config={"displayModeBar": False}),
                dcc.Graph(id="rerouting-savings",
                          figure=app_data.rerouting_savings_chart(),
                          style={"flex": "1", "height": "380px"},
                          config={"displayModeBar": False}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

        ], style={"padding": "24px 32px"}),

        # ── footer ───────────────────────────────────────────────────────────
        html.Div(
            "PharmaChain Risk Analytics · IEEE Dataport Dataset · "
            "Black Sea Pharmaceutical Warehouse Centralisation Study",
            style={"textAlign": "center", "color": COLOURS["muted"],
                   "fontSize": "11px", "padding": "16px",
                   "borderTop": f"1px solid {COLOURS['border']}"},
        ),

    ], style={
        "background":   COLOURS["bg"],
        "minHeight":    "100vh",
        "fontFamily":   "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "color":        COLOURS["text"],
    })


# ── callback registration ────────────────────────────────────────────────────

def register_callbacks(app: Dash, app_data: AppData) -> None:
    @app.callback(
        Output("shortage-chart", "figure"),
        Input("surge-slider",   "value"),
        Input("failure-slider", "value"),
    )
    def update_shortage(surge_pct: int, failure_pct: int) -> go.Figure:
        return app_data.shortage_chart(
            surge=surge_pct / 100,
            failure=failure_pct / 100,
        )


# ── app factory ──────────────────────────────────────────────────────────────

def create_app(base_path: str | Path | None = None) -> tuple[Dash, AppData]:
    """Create and configure the Dash application."""
    # Inject Google Fonts
    external_stylesheets = [
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ]
    app = Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        assets_folder=str(BASE_DIR / "assets"),
        suppress_callback_exceptions=True,
    )
    app.title = "PharmaChain Risk Analytics"

    app_data = AppData(base_path)
    app.layout = _build_layout(app_data)
    register_callbacks(app, app_data)
    return app, app_data


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app, _ = create_app()
    app.run(debug=False, host="0.0.0.0", port=8050)
