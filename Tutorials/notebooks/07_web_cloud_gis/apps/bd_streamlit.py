"""
Bangladesh district explorer, Streamlit version (Notebook 08_07).

Run:  streamlit run bd_streamlit.py
"""

import streamlit as st
from streamlit_folium import st_folium

import bd_dashboard as bd

st.set_page_config(page_title="Bangladesh district explorer", page_icon="🌾", layout="wide")


@st.cache_data                     # read once per server process, shared by every session
def districts():
    return bd.load_districts()


@st.cache_data
def ndvi():
    return bd.load_ndvi()


gdf, nd = districts(), ndvi()

with st.sidebar:
    st.header("Select")
    division = st.selectbox("Division", ["All"] + sorted(gdf.division.unique()), key="division")
    names = bd.district_list(gdf, division)
    district = st.selectbox("District", names, index=names.index("Bogra") if "Bogra" in names else 0, key="district")
    variable = st.radio("Map variable", list(bd.VARIABLES), format_func=bd.VARIABLES.get, key="variable")
    st.caption("Data: BBS poverty and census tables, DAE rice statistics, MODIS NDVI 2000-2024 (Notebooks 08_03, 08_06).")

st.title("Bangladesh district explorer")
card = bd.district_card(gdf, district)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Poverty", f"{card['poverty_pct']:.1f} %" if card["poverty_pct"] is not None else "n/a",
          f"rank {card['poverty_rank']} of 64" if card["poverty_rank"] else None, delta_color="off")
c2.metric("Population density", f"{card['density']:,.0f} per km²" if card["density"] else "n/a")
c3.metric("Rice 2018-19", f"{card['rice_kt']:,.0f} kt", f"rank {card['rice_rank']} of 64 per km²", delta_color="off")
c4.metric("NDVI trend", f"{card['ndvi_trend']:+.3f} / decade", f"rank {card['ndvi_trend_rank']} of 64", delta_color="off")

left, right = st.columns([1, 1])
with left:
    st_folium(bd.choropleth(gdf, variable, district), height=480, use_container_width=True, returned_objects=[])
with right:
    st.plotly_chart(bd.ndvi_figure(nd, district), width="stretch")
    st.plotly_chart(bd.seasonal_figure(nd, district), width="stretch")
