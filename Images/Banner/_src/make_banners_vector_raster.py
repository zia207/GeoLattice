"""Render the Vector & Raster series banners (HTML + inline SVG -> PNG with headless Chrome).

    python Images/Banner/_src/make_banners_vector_raster.py
"""

import base64
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BANNER = HERE.parent
LOGO = base64.b64encode((BANNER.parent / "geolattice_logo.png").read_bytes()).decode()
FOOT = "VECTOR &amp; RASTER DATA WITH GEOLATTICE · UPATTA ANALYTICS"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 1000px; height: 250px; overflow: hidden; }
body { font-family: 'Plus Jakarta Sans', 'Noto Sans', 'DejaVu Sans', sans-serif; color: #fff;
  background:
    linear-gradient(rgba(255,255,255,0.045) 1px, transparent 1px) 0 0 / 22px 22px,
    linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px) 0 0 / 22px 22px,
    radial-gradient(120% 140% at 85% 20%, #2a56a8 0%, #1d3f86 45%, #152f6b 100%); }
.bar { position: absolute; left: 0; top: 0; bottom: 0; width: 7px; background: #f5c518; }
.num { position: absolute; left: 560px; top: 40px; font-size: 230px; font-weight: 800; line-height: 1;
  color: rgba(255,255,255,0.045); }
.wrap { position: absolute; left: 40px; top: 30px; width: 600px; }
.kick { display: flex; align-items: center; gap: 12px; }
.logo { width: 34px; height: 34px; border-radius: 8px; background: #fff; padding: 2px; }
.logo img { width: 30px; height: 30px; border-radius: 50%; }
.k { font-size: 10.5px; font-weight: 800; letter-spacing: 2.6px; color: #f5c518; }
h1 { margin-top: 14px; font-size: 38px; font-weight: 800; line-height: 1.04; letter-spacing: -0.6px; }
h1 span { color: #f5c518; }
.sub { margin-top: 9px; font-size: 13.5px; font-weight: 500; color: #dbe5ff; line-height: 1.35; }
.chips { margin-top: 13px; display: flex; flex-wrap: nowrap; gap: 6px; }
.chip { white-space: nowrap; font-size: 10px; font-weight: 700; padding: 4px 9px; border-radius: 999px; color: #fff;
  background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.22); }
.chip.hl { background: #f5c518; color: #13306a; border-color: #f5c518; }
.foot { position: absolute; left: 40px; bottom: 13px; font-size: 8.5px; font-weight: 700; letter-spacing: 1.4px;
  color: rgba(219,229,255,0.55); }
.art { position: absolute; right: 26px; top: 14px; width: 340px; height: 222px; }
.art text { font-family: 'Plus Jakarta Sans', 'Noto Sans', sans-serif; font-weight: 700; }
"""


def page(kicker, title, sub, chips, art, number="", tall=False):
    chips_html = "".join(f'<span class="chip{" hl" if i == 0 and hl else ""}">{c}</span>'
                         for i, (c, hl) in enumerate(chips))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<div class="bar"></div><div class="num">{number}</div>
{'<style>h1{font-size:32px;margin-top:9px} .sub{margin-top:6px;font-size:12.5px} .chips{margin-top:9px}</style>' if tall else ''}
<div class="wrap" style="top:{20 if tall else 34}px">
  <div class="kick"><div class="logo"><img src="data:image/png;base64,{LOGO}"></div><div class="k">{kicker}</div></div>
  <h1>{title}</h1><div class="sub">{sub}</div><div class="chips">{chips_html}</div>
</div>
<div class="art">{art}</div>
<div class="foot">{FOOT}</div></body></html>"""


# ----------------------------------------------------------------------------- illustrations
Y, B, L, D = "#f5c518", "#8fb3ff", "#c9d8ff", "#0f2557"


def vector_art(x0=0, y0=0, w=340, h=222, labels=True):
    """Points, lines and polygons on a coordinate frame."""
    s = [f'<g transform="translate({x0},{y0})">',
         f'<rect x="6" y="6" width="{w - 12}" height="{h - 30}" rx="10" fill="{D}" fill-opacity="0.55" '
         f'stroke="{B}" stroke-opacity="0.55"/>']
    for gx in range(30, w - 20, 40):  # graticule
        s.append(f'<line x1="{gx}" y1="12" x2="{gx}" y2="{h - 30}" stroke="#fff" stroke-opacity="0.07"/>')
    for gy in range(30, h - 30, 36):
        s.append(f'<line x1="12" y1="{gy}" x2="{w - 12}" y2="{gy}" stroke="#fff" stroke-opacity="0.07"/>')
    s += [  # polygons
        f'<path d="M40 60 L110 38 L150 78 L128 132 L62 124 Z" fill="{B}" fill-opacity="0.28" stroke="{B}" stroke-width="2"/>',
        f'<path d="M190 120 L262 104 L300 150 L240 176 L196 160 Z" fill="{Y}" fill-opacity="0.22" stroke="{Y}" stroke-width="2"/>',
        # line (river / road)
        f'<path d="M24 170 C 80 150, 110 180, 160 150 S 240 70, 318 52" fill="none" stroke="{L}" stroke-width="3" '
        f'stroke-linecap="round"/>',
    ]
    for px, py in [(214, 46), (246, 72), (282, 92), (176, 56), (96, 158)]:  # points
        s.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{Y}" stroke="{D}" stroke-width="2"/>')
    for vx, vy in [(40, 60), (110, 38), (150, 78), (128, 132), (62, 124)]:  # vertices
        s.append(f'<circle cx="{vx}" cy="{vy}" r="3" fill="#fff"/>')
    if labels:
        for t, tx, ty in [("polygon", 70, 92), ("point", 222, 34), ("line", 300, 44), ("polygon", 226, 146)]:
            s.append(f'<text x="{tx}" y="{ty}" font-size="9" fill="#fff" fill-opacity="0.8">{t}</text>')
        s.append(f'<text x="{w / 2}" y="{h - 10}" font-size="8.5" fill="{L}" fill-opacity="0.8" text-anchor="middle">'
                 f'points · lines · polygons · attributes · CRS</text>')
    s.append("</g>")
    return "".join(s)


def raster_art(x0=0, y0=0, w=340, h=222, labels=True, cell_size=20):
    """A stack of three bands plus one classified grid."""
    import math

    s = [f'<g transform="translate({x0},{y0})">']
    cols, rows, cell = 9, 6, cell_size
    def grid(ox, oy, fn, opacity=1.0):
        out = []
        for r in range(rows):
            for c in range(cols):
                v = fn(r, c)
                col = Y if v > 0.78 else (B if v > 0.5 else ("#4f7fd6" if v > 0.3 else "#2b4f9c"))
                out.append(f'<rect x="{ox + c * cell}" y="{oy + r * cell}" width="{cell - 2}" height="{cell - 2}" rx="2" '
                           f'fill="{col}" fill-opacity="{opacity}"/>')
        return "".join(out)
    f = lambda r, c: 0.5 + 0.5 * math.sin(r * 0.9 + c * 0.55) * math.cos(c * 0.35 - r * 0.2)
    for k, (dx, dy, op) in enumerate([(56, 10, 0.35), (36, 30, 0.6), (16, 50, 1.0)]):  # band stack
        s.append(f'<rect x="{dx - 4}" y="{dy - 4}" width="{cols * cell + 6}" height="{rows * cell + 6}" rx="6" '
                 f'fill="{D}" fill-opacity="0.85" stroke="{B}" stroke-opacity="0.6"/>')
        s.append(grid(dx, dy, lambda r, c, k=k: f(r + k * 0.7, c - k * 0.4), op))
        if labels:
            s.append(f'<text x="{56 + cols * cell + 10}" y="{dy + 8}" font-size="8.5" fill="#fff" '
                     f'fill-opacity="0.75">band {3 - k}</text>')
    if labels:
        s.append(f'<text x="{w / 2}" y="{h - 10}" font-size="8.5" fill="{L}" fill-opacity="0.8" text-anchor="middle">'
                 f'cells · bands · resolution · nodata · CRS</text>')
    s.append("</g>")
    return "".join(s)


def series_art():
    """Vector (left) and raster (right) linked by rasterize / polygonize arrows."""
    s = ['<svg width="340" height="222" viewBox="0 0 340 222">',
         f'<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{Y}"/></marker>'
         f'<marker id="b" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{L}"/></marker></defs>']
    s.append(f'<g transform="translate(0,34) scale(0.43)">{vector_art(0, 0, 330, 300, labels=False)}</g>')
    s.append(f'<g transform="translate(188,30) scale(0.62)">{raster_art(0, 0, 240, 200, labels=False)}</g>')
    s += [f'<path d="M146 80 C 160 66, 172 66, 186 74" fill="none" stroke="{Y}" stroke-width="2" marker-end="url(#a)"/>',
          f'<path d="M186 140 C 172 152, 160 152, 146 144" fill="none" stroke="{L}" stroke-width="2" marker-end="url(#b)"/>',
          f'<text x="166" y="58" font-size="9" fill="{Y}" text-anchor="middle">rasterize</text>',
          f'<text x="166" y="168" font-size="9" fill="{L}" text-anchor="middle">polygonize</text>',
          f'<text x="70" y="186" font-size="9.5" fill="#fff" fill-opacity="0.85" text-anchor="middle">Part 1 · Vector</text>',
          f'<text x="262" y="186" font-size="9.5" fill="#fff" fill-opacity="0.85" text-anchor="middle">Part 2 · Raster</text>',
          "</svg>"]
    return "".join(s)


BANNERS = {
    "01_02_banner_vector_raster_series.png": page(
        "TUTORIAL SERIES · DR. ZIA U. AHMED", "Vector &amp; Raster Data<br><span>with GeoLattice</span>",
        "The two data models of GIS, features and grids of cells: read, projected,<br>"
        "mapped and analysed with <b>geolattice.vector</b> and <b>geolattice.raster</b>.",
        [("2 parts · 10 notebooks", True), ("Python 3.11+", False), ("GeoPandas", False), ("Shapely", False),
         ("Rasterio", False), ("xarray", False), ("GDAL", False)],
        series_art(), tall=True),
    "01_banner_vector_geolattice.png": page(
        "PART 1 OF 2", "Working with Vector Data",
        "Points, lines and polygons: read, project, map and analyse features",
        [("Read &amp; write", False), ("GeoParquet · FlatGeobuf", False), ("CRS", False),
         ("Visualization", False), ("Geoprocessing", False), ("Geometry ops", False)],
        f'<svg width="340" height="222" viewBox="0 0 340 222">{vector_art()}</svg>', number="1"),
    "02_banner_raster_geolattice.png": page(
        "PART 2 OF 2", "Working with Raster Data",
        "Grids of cells: read, reproject and compute on imagery and surfaces",
        [("Read &amp; write", False), ("GeoTIFF · COG", False), ("CRS &amp; warping", False),
         ("Raster algebra", False), ("Resample · mosaic", False), ("Zonal stats", False)],
        f'<svg width="340" height="222" viewBox="0 0 340 222"><g transform="translate(30,8)">{raster_art()}</g></svg>', number="2"),
}

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for name, html in BANNERS.items():
            src = Path(tmp) / (name + ".html")
            src.write_text(html)
            out = BANNER / name
            subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=2", "--window-size=1000,250",
                            "--virtual-time-budget=4000", f"--screenshot={out}", src.as_uri()],
                           check=True, capture_output=True)
            print("wrote", out)
