"""
Data and figures shared by the Bangladesh district explorer apps (Notebook 08_07).

The three apps (Streamlit, Panel, Solara) are thin user interfaces over this module:
it loads the prepared data and builds the map, charts and summary cards. Keeping the
logic here makes it testable without any web framework and lets the apps share it.

``CALLS`` counts how often each function runs, which the notebook uses to compare how
much work each framework repeats per user interaction.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

import branca.colormap as bcm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA = Path(os.environ.get(
    "BD_APP_DATA",
    Path(__file__).resolve().parents[4] / "Data" / "00_shared" / "data_web_cloud_gis" / "08_07_outputs"))

VARIABLES = {
    "poverty_pct": "Poverty (% of population)",
    "density": "Population density (per km²)",
    "rice_t_km2": "Rice production 2018-19 (t per km²)",
    "ndvi_max_mean": "Mean annual peak NDVI, 2001-2024",
    "ndvi_trend": "Trend of annual peak NDVI (per decade)",
}
COLORS = {"poverty_pct": "YlOrRd", "density": "PuBu", "rice_t_km2": "YlGn",
          "ndvi_max_mean": "Greens", "ndvi_trend": "RdYlGn"}
CALLS: Counter = Counter()


def load_districts() -> gpd.GeoDataFrame:
    """District table with geometry (simplified) and the map variables."""
    CALLS["load_districts"] += 1
    return gpd.read_parquet(DATA / "districts.parquet")


def load_ndvi() -> pd.DataFrame:
    """Mean NDVI of every district on every MODIS date (long format)."""
    CALLS["load_ndvi"] += 1
    return pd.read_parquet(DATA / "district_ndvi.parquet")


def district_list(gdf: gpd.GeoDataFrame, division: Optional[str] = None) -> list[str]:
    d = gdf if not division or division == "All" else gdf[gdf.division == division]
    return sorted(d.district)


def district_card(gdf: gpd.GeoDataFrame, district: str) -> dict:
    """Headline numbers for a district, with its rank among the 64 districts (1 = highest)."""
    CALLS["district_card"] += 1
    row = gdf.set_index("district").loc[district]
    rank = lambda col: int(gdf[col].rank(ascending=False, method="min")[gdf.district == district].iloc[0]) \
        if pd.notna(row[col]) else None
    return {"division": row.division,
            "poverty_pct": None if pd.isna(row.poverty_pct) else float(row.poverty_pct), "poverty_rank": rank("poverty_pct"),
            "population": None if pd.isna(row.population) else int(row.population),
            "density": None if pd.isna(row.density) else float(row.density),
            "rice_kt": float(row.rice_t) / 1e3, "rice_rank": rank("rice_t_km2"),
            "ndvi_trend": float(row.ndvi_trend), "ndvi_trend_rank": rank("ndvi_trend")}


def choropleth(gdf: gpd.GeoDataFrame, variable: str, highlight: Optional[str] = None, height: int = 480) -> folium.Map:
    """Folium map of one variable, with the selected district outlined."""
    CALLS["choropleth"] += 1
    values = gdf[variable]
    lo, hi = np.nanpercentile(values, [2, 98])
    if variable == "ndvi_trend":
        lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
    cmap = bcm.LinearColormap(_ramp(COLORS[variable]), vmin=float(lo), vmax=float(hi), caption=VARIABLES[variable])
    m = folium.Map(location=[23.7, 90.3], zoom_start=6, tiles=None, height=height, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri").add_to(m)
    folium.GeoJson(
        gdf[["district", "division", variable, "geometry"]],
        style_function=lambda f: {"fillColor": "#cccccc" if f["properties"][variable] is None else cmap(f["properties"][variable]),
                                  "color": "white", "weight": 0.6, "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(["district", "division", variable], aliases=["District", "Division", VARIABLES[variable]]),
    ).add_to(m)
    if highlight:
        folium.GeoJson(gdf[gdf.district == highlight][["district", "geometry"]], interactive=False,
                       style_function=lambda f: {"color": "#111111", "weight": 3, "fill": False}).add_to(m)
    cmap.add_to(m)
    return m


def ndvi_figure(ndvi: pd.DataFrame, district: str, height: int = 330) -> go.Figure:
    """16-day NDVI of a district against the national median, with annual peaks."""
    CALLS["ndvi_figure"] += 1
    d = ndvi[ndvi.district == district].set_index("date").ndvi
    national = ndvi.groupby("date").ndvi.median()
    peaks = d.groupby(d.index.year).max()
    fig = go.Figure()
    fig.add_scatter(x=national.index, y=national.values, name="National median", line=dict(color="#bbbbbb", width=1))
    fig.add_scatter(x=d.index, y=d.values, name=district, line=dict(color="#2c7fb8", width=1.2))
    fig.add_scatter(x=pd.to_datetime([f"{y}-07-01" for y in peaks.index]), y=peaks.values, name="Annual peak",
                    mode="markers+lines", line=dict(color="#006d2c", width=1, dash="dot"), marker=dict(size=5))
    fig.update_layout(title=f"NDVI, {district} (MODIS 16-day, district mean)", height=height,
                      margin=dict(l=40, r=10, t=40, b=30), legend=dict(orientation="h", y=-0.15), yaxis_title="NDVI")
    return fig


def seasonal_figure(ndvi: pd.DataFrame, district: str, height: int = 280) -> go.Figure:
    """Mean NDVI by month: the district's crop and flood calendar against the country."""
    CALLS["seasonal_figure"] += 1
    month = ndvi.assign(month=ndvi.date.dt.month)
    d = month[month.district == district].groupby("month").ndvi.mean()
    nat = month.groupby("month").ndvi.mean()
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    fig.add_bar(x=labels, y=d.reindex(range(1, 13)).values, name=district, marker_color="#2c7fb8")
    fig.add_scatter(x=labels, y=nat.reindex(range(1, 13)).values, name="Bangladesh", mode="lines+markers",
                    line=dict(color="#888888"))
    fig.update_layout(title="Mean NDVI by month, 2000-2024", height=height, margin=dict(l=40, r=10, t=40, b=30),
                      legend=dict(orientation="h", y=-0.2), yaxis_title="NDVI")
    return fig


def _ramp(name: str) -> list[str]:
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return [matplotlib.colors.to_hex(cmap(x)) for x in np.linspace(0.1, 0.95, 7)]
