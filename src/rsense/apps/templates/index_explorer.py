"""PyRSense Index Explorer: upload a multiband GeoTIFF, map bands, compute any spectral index.

Run:  streamlit run index_explorer.py
"""

import tempfile

import numpy as np
import streamlit as st

import geolattice.rsense as rs

st.set_page_config(page_title="PyRSense Index Explorer", layout="wide")
st.title("Spectral Index Explorer")

up = st.sidebar.file_uploader("Multiband GeoTIFF (reflectance)", type=["tif", "tiff"])
if up is None:
    st.info("Upload a GeoTIFF to begin. Bands should be surface reflectance (0-1) or scaled DNs.")
    st.stop()

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
    f.write(up.read())
    path = f.name
da = rs.read(path)
n = da.sizes.get("band", 1)
st.sidebar.write(f"{n} bands, {da.sizes['y']}×{da.sizes['x']} px, CRS {da.rio.crs}")

defaults = ["blue", "green", "red", "nir", "swir16", "swir22", "rededge1", "rededge2"]
names = [st.sidebar.selectbox(f"Band {i + 1}", ["(skip)"] + rs.optical.bands.CANONICAL[:13],
                              index=(rs.optical.bands.CANONICAL.index(defaults[i]) + 1)
                              if i < len(defaults) else 0) for i in range(n)]
scale = st.sidebar.number_input("Divide values by (e.g. 10000 for S2 DN)", value=1.0, min_value=1.0)
ds = (da / scale).assign_coords(band=[nm if nm != "(skip)" else f"b{i}" for i, nm in enumerate(names)])
ds = ds.to_dataset(dim="band")

catalog = rs.optical.list_indices()
available = [r.name for r in catalog.itertuples()
             if all(b in ds.data_vars for b in r.bands.split(", "))]
domain = st.sidebar.selectbox("Domain", ["all"] + sorted(catalog.domain.unique()))
choices = [i for i in available if domain == "all" or
           catalog.set_index("name").loc[i, "domain"] == domain]
name = st.sidebar.selectbox("Index", choices or ["NDVI"])
vi = rs.to_cpu(rs.optical.index(ds, name))

c1, c2 = st.columns([3, 1])
with c1:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    rs.viz.plot_raster(vi, ax=ax, cmap="RdYlGn" if catalog.set_index("name").loc[name, "domain"]
                       in ("vegetation", "rededge") else "viridis")
    st.pyplot(fig)
with c2:
    v = vi.values[np.isfinite(vi.values)]
    st.metric("Mean", f"{v.mean():.3f}")
    st.metric("Median", f"{np.median(v):.3f}")
    st.metric("Std", f"{v.std():.3f}")
    st.caption(catalog.set_index("name").loc[name, "long_name"])
    out = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    rs.write(vi, out)
    st.download_button("Download GeoTIFF", open(out, "rb"), file_name=f"{name}.tif")
