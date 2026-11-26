"""PyRSense Flood Mapper: Sentinel-1 RTC flood extent for any bounding box (Planetary Computer).

Run:  streamlit run flood_mapper.py      (needs geolattice[rsense-access])
"""

import datetime as dt

import streamlit as st

import geolattice.rsense as rs

st.set_page_config(page_title="PyRSense Flood Mapper", layout="wide")
st.title("Sentinel-1 Flood Mapper")

c = st.sidebar
west = c.number_input("West", value=90.30, format="%.4f")
south = c.number_input("South", value=23.60, format="%.4f")
east = c.number_input("East", value=90.50, format="%.4f")
north = c.number_input("North", value=23.80, format="%.4f")
event = c.date_input("Flood date window start", dt.date(2026, 7, 1))
days = c.slider("Window (days)", 6, 30, 12)
method = c.selectbox("Threshold method", ["split_otsu", "otsu", "fixed"])
speckle = c.selectbox("Speckle filter", ["refined_lee", "lee", "gamma_map", "none"])
go = c.button("Map flood")

if go:
    aoi = (west, south, east, north)
    with st.spinner("Loading Sentinel-1 RTC..."):
        s1 = rs.sar.load_rtc(aoi, (str(event), str(event + dt.timedelta(days=days))),
                             polarizations=("vv",), resolution=20)
        vv = s1["vv"].isel(time=0).compute()
    if speckle != "none":
        vv = getattr(rs.sar, speckle)(vv)
    res = rs.sar.flood_map(vv, method=method)
    st.write(f"Threshold: {res.attrs['threshold_db']:.2f} dB · water fraction "
             f"{float(res.water.mean()):.1%}")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rs.viz.plot_raster(rs.sar.to_db(vv), ax=axes[0], cmap="gray", title="VV (dB)")
    rs.viz.plot_raster(res.water.astype(int), ax=axes[1], categorical={0: ("land", "#d9d9d9"),
                       1: ("water", "#2171b5")}, title="Water")
    st.pyplot(fig)
