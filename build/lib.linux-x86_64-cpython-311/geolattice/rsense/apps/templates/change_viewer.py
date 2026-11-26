"""PyRSense Change Viewer: compare two co-registered rasters with a swipe map and CVA stats.

Run:  streamlit run change_viewer.py     (needs leafmap, localtileserver)
"""

import tempfile

import leafmap.foliumap as leafmap
import streamlit as st

import geolattice.rsense as rs

st.set_page_config(page_title="PyRSense Change Viewer", layout="wide")
st.title("Before / After Change Viewer")
a = st.sidebar.file_uploader("Before (GeoTIFF)", type=["tif", "tiff"])
b = st.sidebar.file_uploader("After (GeoTIFF)", type=["tif", "tiff"])
if not (a and b):
    st.info("Upload two single- or multi-band rasters on the same grid.")
    st.stop()


def save(f):
    p = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    open(p, "wb").write(f.read())
    return p


pa, pb = save(a), save(b)
m = leafmap.Map()
m.split_map(pa, pb, left_label="Before", right_label="After")
m.to_streamlit(height=600)

da, db = rs.read(pa), rs.read(pb)
res = rs.classify.cva(da.to_dataset(dim="band"), db.to_dataset(dim="band"))
st.metric("Changed area", f"{float(res.change.mean()):.1%}")
st.caption(f"CVA magnitude threshold (Otsu): {res.attrs['threshold']:.4f}")
