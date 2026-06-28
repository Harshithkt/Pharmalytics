from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_and_validate, print_quality_summary


OUTPUT_DIR = Path(__file__).resolve().parent / "eda_outputs"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_plotly_figure(fig: go.Figure, stem: str) -> None:
    html_path = OUTPUT_DIR / f"{stem}.html"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path))
    except Exception as exc:  # pragma: no cover - export fallback
        print(f"PNG export failed for {stem}: {exc}")


def build_demand_bar(demand_sheets: dict[str, pd.DataFrame]) -> go.Figure:
    demand_frames = []
    for frame in demand_sheets.values():
        if not isinstance(frame, pd.DataFrame) or "demand_type" not in frame.columns:
            continue
        demand_frames.append(frame.copy())
    combined = pd.concat(demand_frames, ignore_index=True)
    mapping = {
        "demandNorm": "normal",
        "demandCold": "cold",
        "demandColdCrit": "critical-cold",
    }
    filtered = combined[combined["demand_type"].isin(mapping)].copy()
    filtered["drug_type"] = filtered["demand_type"].map(mapping)
    summary = filtered.groupby("drug_type", as_index=False)["demand"].sum().sort_values("drug_type")
    fig = px.bar(
        summary,
        x="drug_type",
        y="demand",
        color="drug_type",
        text_auto=True,
        title="Demand distribution by drug type across clusters",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Drug type", yaxis_title="Total demand")
    return fig


def build_capacity_heatmap(demand_sheets: dict[str, pd.DataFrame], capacity_sheets: dict[str, pd.DataFrame]) -> go.Figure:
    demand_totals = []
    for sheet_name, frame in demand_sheets.items():
        if isinstance(frame, pd.DataFrame) and "demand" in frame.columns:
            demand_totals.append(
                {
                    "region": sheet_name.replace("Demand ", ""),
                    "demand_total": float(frame["demand"].fillna(0).sum()),
                }
            )
    demand_lookup = pd.DataFrame(demand_totals)

    capacity_frames = []
    for sheet_name, frame in capacity_sheets.items():
        if isinstance(frame, pd.DataFrame) and {"configuration", "metric", "value", "region"}.issubset(frame.columns):
            base = frame[frame["metric"].astype(str) == "capacity"].copy()
            if not base.empty:
                capacity_frames.append(
                    base.groupby(["region", "configuration"], as_index=False)["value"].sum().rename(columns={"value": "capacity_total"})
                )
    capacity_df = pd.concat(capacity_frames, ignore_index=True)
    merged = capacity_df.merge(demand_lookup, on="region", how="left")
    merged["utilization"] = merged["demand_total"] / merged["capacity_total"]
    pivot = merged.pivot(index="region", columns="configuration", values="utilization").sort_index()
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Demand-to-capacity utilization by warehouse configuration",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Configuration", yaxis_title="Region")
    return fig


def build_time_boxplot(time_sheets: dict[str, pd.DataFrame]) -> go.Figure:
    frames = []
    for frame in time_sheets.values():
        if isinstance(frame, pd.DataFrame) and {"source_zone", "delivery_time"}.issubset(frame.columns):
            frames.append(frame[["source_zone", "delivery_time"]].copy())
    combined = pd.concat(frames, ignore_index=True)
    combined["source_zone"] = combined["source_zone"].astype(str)
    order = [zone for zone in ["A", "B", "C", "D", "E", "F", "G", "H"] if zone in combined["source_zone"].unique()]
    fig = px.box(
        combined,
        x="source_zone",
        y="delivery_time",
        category_orders={"source_zone": order},
        points="outliers",
        title="Delivery time distribution across pharmacy points by source zone",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Source zone", yaxis_title="Delivery time")
    return fig


def build_cost_heatmap(cost_sheets: dict[str, pd.DataFrame]) -> go.Figure:
    target = None
    for frame in cost_sheets.values():
        if isinstance(frame, pd.DataFrame) and {"cluster", "mode", "value"}.issubset(frame.columns):
            modes = frame["mode"].dropna().astype(str).unique().tolist()
            if any(mode.startswith("TC") for mode in modes):
                target = frame.copy()
                break
    if target is None:
        raise ValueError("No TC0-TC3 cost matrix found in the cost workbook")
    target["mode"] = target["mode"].astype(str)
    pivot = target.pivot_table(index="cluster", columns="mode", values="value", aggfunc="mean")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Transport cost matrix across clusters and modes",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Mode", yaxis_title="Cluster")
    return fig


def build_correlation_heatmap(bundles: dict[str, pd.DataFrame]) -> go.Figure:
    summary_rows = []
    regions = sorted({frame["region"].iloc[0] for frame in bundles["demand"].values() if isinstance(frame, pd.DataFrame) and "region" in frame.columns})
    for region in regions:
        demand_frame = pd.concat(
            [frame for frame in bundles["demand"].values() if isinstance(frame, pd.DataFrame) and "region" in frame.columns and frame["region"].iloc[0] == region],
            ignore_index=True,
        )
        time_frame = pd.concat(
            [frame for frame in bundles["time"].values() if isinstance(frame, pd.DataFrame) and "region" in frame.columns and frame["region"].iloc[0] == region],
            ignore_index=True,
        )
        distance_frame = pd.concat(
            [frame for frame in bundles["distance"].values() if isinstance(frame, pd.DataFrame) and "region" in frame.columns and frame["region"].iloc[0] == region],
            ignore_index=True,
        )
        cost_frame = pd.concat(
            [frame for frame in bundles["cost"].values() if isinstance(frame, pd.DataFrame) and "region" in frame.columns and frame["region"].iloc[0] == region],
            ignore_index=True,
        )
        summary_rows.append(
            {
                "region": region,
                "demand": pd.to_numeric(demand_frame.get("demand"), errors="coerce").fillna(0).sum(),
                "time": pd.to_numeric(time_frame.get("delivery_time"), errors="coerce").fillna(0).mean(),
                "distance": pd.to_numeric(distance_frame.get("value"), errors="coerce").fillna(0).mean(),
                "cost": pd.to_numeric(cost_frame.get("value"), errors="coerce").fillna(0).mean(),
            }
        )

    correlation_frame = pd.DataFrame(summary_rows).drop(columns=["region"]).fillna(0)
    corr = correlation_frame.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation matrix of demand, time, distance, and cost features",
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_geo_map(geo_sheets: dict[str, pd.DataFrame]) -> tuple[folium.Map, pd.DataFrame]:
    frame = geo_sheets.get("Rize")
    if frame is None:
        frame = next(iter(geo_sheets.values()))
    center = [frame["lat"].mean(), frame["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    points = frame.copy()
    for _, row in points.iterrows():
        color = "red" if bool(row.get("outlier", False)) else "blue"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=row.get("point_id", "pharmacy"),
        ).add_to(fmap)
    return fmap, points


def build_geo_png(points: pd.DataFrame) -> go.Figure:
    fig = px.scatter_geo(
        points,
        lat="lat",
        lon="lon",
        hover_name="point_id",
        color="outlier",
        title="Pharmacy geospatial scatter plot",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(template="plotly_white")
    return fig


def main() -> None:
    ensure_output_dir()
    bundles, report = load_and_validate()
    print_quality_summary(report)

    demand_fig = build_demand_bar(bundles["demand"].cleaned)
    save_plotly_figure(demand_fig, "01_demand_distribution")

    capacity_fig = build_capacity_heatmap(bundles["demand"].cleaned, bundles["capacity"].cleaned)
    save_plotly_figure(capacity_fig, "02_capacity_utilization")

    time_fig = build_time_boxplot(bundles["time"].cleaned)
    save_plotly_figure(time_fig, "03_delivery_time_distribution")

    cost_fig = build_cost_heatmap(bundles["cost"].cleaned)
    save_plotly_figure(cost_fig, "04_cost_matrix_heatmap")

    corr_fig = build_correlation_heatmap(
        {
            "demand": bundles["demand"].cleaned,
            "time": bundles["time"].cleaned,
            "distance": bundles["distance"].cleaned,
            "cost": bundles["cost"].cleaned,
        }
    )
    save_plotly_figure(corr_fig, "05_feature_correlation")

    folium_map, geo_points = build_geo_map(bundles["geo"].cleaned)
    folium_html = OUTPUT_DIR / "06_geospatial_scatter.html"
    folium_map.save(str(folium_html))
    geo_fig = build_geo_png(geo_points)
    save_plotly_figure(geo_fig, "06_geospatial_scatter")


if __name__ == "__main__":
    main()
