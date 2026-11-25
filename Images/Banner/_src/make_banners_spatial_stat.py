"""Render the Spatial Statistics module banners (HTML + inline SVG -> PNG with headless Chrome).

Reuses the page layout and palette of make_banners_vector_raster.py. The module overview banner
(04_00) is hand-built in 04_spatial_stat/04_00_banner_spatial_stat_geolattice.html.

    python Images/Banner/_src/make_banners_spatial_stat.py
"""

import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_banners_vector_raster as base  # noqa: E402

base.FOOT = "SPATIAL STATISTICS WITH GEOLATTICE · UPATTA ANALYTICS"
OUT = base.BANNER / "04_spatial_stat"
Y, B, L, D = base.Y, base.B, base.L, base.D
M1, M2 = "#4f7fd6", "#2b4f9c"  # mid and deep blues used for raster classes


def page(kicker, title, sub, chips, art, number="", h1=None):
    html = base.page(kicker, title, sub, [(c, False) for c in chips], art, number=number)
    if h1:
        html = html.replace("</head>", f"<style>h1{{font-size:{h1}px}}</style></head>")
    return html


# ----------------------------------------------------------------------------- helpers
def svg(body):
    return f'<svg width="340" height="222" viewBox="0 0 340 222">{body}</svg>'


def frame(w=340, h=222):
    s = [f'<rect x="6" y="6" width="{w - 12}" height="{h - 30}" rx="10" fill="{D}" fill-opacity="0.55" '
         f'stroke="{B}" stroke-opacity="0.55"/>']
    for gx in range(30, w - 20, 40):
        s.append(f'<line x1="{gx}" y1="12" x2="{gx}" y2="{h - 30}" stroke="#fff" stroke-opacity="0.06"/>')
    for gy in range(30, h - 30, 36):
        s.append(f'<line x1="12" y1="{gy}" x2="{w - 12}" y2="{gy}" stroke="#fff" stroke-opacity="0.06"/>')
    return "".join(s)


def caption(text):
    return (f'<text x="170" y="212" font-size="8.5" fill="{L}" fill-opacity="0.8" text-anchor="middle">'
            f'{text}</text>')


def label(t, x, y, size=9, fill="#fff", op=0.8, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" fill-opacity="{op}" '
            f'text-anchor="{anchor}">{t}</text>')


def cls(v):
    return Y if v > 0.78 else (B if v > 0.55 else (M1 if v > 0.32 else M2))


def ramp(v):
    """Continuous deep blue -> light blue -> yellow."""
    stops = [(0.0, (0x2b, 0x4f, 0x9c)), (0.55, (0x8f, 0xb3, 0xff)), (1.0, (0xf5, 0xc5, 0x18))]
    v = min(max(v, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))
    return "#f5c518"


def field(r, c, k=0.0):
    return 0.5 + 0.5 * math.sin(r * 0.8 + c * 0.5 + k) * math.cos(c * 0.33 - r * 0.25 + k * 0.5)


def squares(ox, oy, rows, cols, cell, fn, gap=2, op=1.0):
    return "".join(f'<rect x="{ox + c * cell}" y="{oy + r * cell}" width="{cell - gap}" height="{cell - gap}" rx="2" '
                   f'fill="{fn(r, c)}" fill-opacity="{op}"/>' for r in range(rows) for c in range(cols))


def hexes(ox, oy, rows, cols, r, color):
    s, w = [], math.sqrt(3) * r
    for i in range(rows):
        for j in range(cols):
            cx, cy = ox + j * w + (i % 2) * w / 2, oy + i * 1.5 * r
            pts = " ".join(f"{cx + (r - 1) * math.cos(math.radians(60 * k - 30)):.1f},"
                           f"{cy + (r - 1) * math.sin(math.radians(60 * k - 30)):.1f}" for k in range(6))
            s.append(f'<polygon points="{pts}" fill="{color(i, j)}"/>')
    return "".join(s)


def axes(x0, y0, x1, y1):
    return (f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{L}" stroke-opacity="0.6"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{L}" stroke-opacity="0.6"/>')


ARROW = (f'<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">'
         f'<path d="M0 0 L8 4 L0 8 Z" fill="{Y}"/></marker></defs>')


# ----------------------------------------------------------------------------- illustrations
def art_point_pattern():
    rng = random.Random(3)
    s = [frame()]
    pts = []
    for cx, cy, n in [(52, 54, 14), (120, 130, 12), (150, 50, 8)]:
        pts += [(cx + rng.gauss(0, 11), cy + rng.gauss(0, 11)) for _ in range(n)]
    pts += [(rng.uniform(20, 176), rng.uniform(20, 184)) for _ in range(12)]
    for cx, cy in [(52, 54), (120, 130)]:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="26" fill="{Y}" fill-opacity="0.12"/>')
    s += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{Y}" stroke="{D}" stroke-width="1"/>' for x, y in pts]
    s.append(f'<line x1="186" y1="16" x2="186" y2="192" stroke="{B}" stroke-opacity="0.35"/>')
    x0, y0, x1, y1 = 204, 30, 322, 168
    s.append(axes(x0, y0, x1, y1))
    env_hi = [(x0 + t * (x1 - x0), y1 - t * 112 - 12 * math.sin(t * math.pi)) for t in [i / 10 for i in range(11)]]
    env_lo = [(x0 + t * (x1 - x0), y1 - t * 112 + 12 * math.sin(t * math.pi)) for t in [i / 10 for i in range(11)]]
    s.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in env_hi + env_lo[::-1])
             + f'" fill="{B}" fill-opacity="0.25"/>')
    s.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1 - 112}" stroke="{L}" stroke-dasharray="4 3"/>')
    obs = [(x0 + t * (x1 - x0), y1 - 112 * t - 34 * math.sin(t * math.pi * 0.9)) for t in [i / 20 for i in range(21)]]
    s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in obs)
             + f'" fill="none" stroke="{Y}" stroke-width="2.5"/>')
    s.append(label("L(r)", 208, 26, 9))
    s.append(label("r", 316, 182, 9))
    s.append(label("CSR envelope", 262, 160, 8, L, 0.85, "middle"))
    s.append(caption("intensity · K, L, G and F functions · CSR tests · envelopes"))
    return svg("".join(s))


def art_autocorrelation():
    s = [frame()]
    s.append(squares(22, 24, 7, 7, 23, lambda r, c: cls(field(r, c))))
    x0, y0, x1, y1, cx, cy = 202, 22, 322, 172, 262, 97
    s.append(f'<line x1="{x0}" y1="{cy}" x2="{x1}" y2="{cy}" stroke="{L}" stroke-opacity="0.45"/>'
             f'<line x1="{cx}" y1="{y0}" x2="{cx}" y2="{y1}" stroke="{L}" stroke-opacity="0.45"/>')
    rng = random.Random(11)
    for _ in range(34):
        x = rng.uniform(-1, 1)
        y = 0.7 * x + rng.uniform(-0.32, 0.32)
        col = Y if x > 0 and y > 0 else (B if x < 0 and y < 0 else "#ffffff")
        s.append(f'<circle cx="{cx + x * 48:.1f}" cy="{cy - y * 52:.1f}" r="2.8" fill="{col}" fill-opacity="0.9"/>')
    s.append(f'<line x1="{cx - 58}" y1="{cy + 42}" x2="{cx + 58}" y2="{cy - 42}" stroke="{Y}" stroke-width="2"/>')
    for t, x, y in [("HH", 314, 30), ("LL", 210, 170), ("LH", 210, 30), ("HL", 314, 170)]:
        s.append(label(t, x, y, 9, anchor="middle"))
    s.append(label("x", 324, 94, 8, L))
    s.append(label("Wx", 266, 20, 8, L))
    s.append(caption("spatial weights · Moran's I · Geary's C · LISA"))
    return svg("".join(s))


def art_regression():
    s = [ARROW, frame()]
    cell, ox, oy = 30, 24, 30

    def col(r, c):
        if (r, c) == (2, 2):
            return Y
        return B if abs(r - 2) <= 1 and abs(c - 2) <= 1 else M2
    s.append(squares(ox, oy, 5, 5, cell, col))
    cx, cy = ox + 2 * cell + 14, oy + 2 * cell + 14
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                sx, sy = cx + dc * cell, cy + dr * cell
                ex, ey = cx + dc * 9, cy + dr * 9
                s.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{D}" stroke-width="2" '
                         f'marker-end="url(#a)"/>')
    s.append(label("queen neighbours → Wy", 98, 196 - 12, 8.5, L, 0.9, "middle"))
    s.append(f'<text x="252" y="70" font-size="17" fill="#fff" text-anchor="middle">'
             f'y = <tspan fill="{Y}">ρWy</tspan> + Xβ + u</text>')
    s.append(f'<text x="252" y="96" font-size="13" fill="{L}" text-anchor="middle">'
             f'u = <tspan fill="{Y}">λWu</tspan> + ε</text>')
    for i, t in enumerate(["OLS", "SAR", "SEM", "SDM"]):
        x = 196 + i * 30
        fill = Y if t == "OLS" else "none"
        s.append(f'<rect x="{x}" y="122" width="26" height="18" rx="9" fill="{fill}" fill-opacity="0.15" '
                 f'stroke="{B}" stroke-opacity="0.7"/>')
        s.append(label(t, x + 13, 135, 8.5, anchor="middle", op=0.95))
    s.append(label("LM tests → model choice", 252, 162, 8.5, L, 0.85, "middle"))
    s.append(caption("OLS diagnostics · LM tests · SAR · SEM · SLX · SDM"))
    return svg("".join(s))


def art_cluster():
    def gi(i, j):
        d1, d2 = math.hypot(i - 3, j - 4), math.hypot(i - 6, j - 13)
        if d1 < 1.5:
            return Y
        if d1 < 2.6:
            return "#c9a93a"
        if d2 < 1.6:
            return B
        if d2 < 2.4:
            return M1
        return M2
    s = [frame(), hexes(24, 24, 9, 16, 11, gi)]
    s.append(f'<circle cx="{24 + 4.5 * 19:.0f}" cy="{24 + 3 * 16.5:.0f}" r="44" fill="none" stroke="#fff" '
             f'stroke-width="1.6" stroke-dasharray="5 4"/>')
    s.append(label("scan window", 126, 22 + 3, 8.5, "#fff", 0.9))
    s.append(label("hot spot", 52, 172, 9, Y, 0.95))
    s.append(label("cold spot", 262, 182, 9, B, 0.95))
    s.append(caption("Getis-Ord Gi* · DBSCAN · scan statistic · regionalisation"))
    return svg("".join(s))


def art_disease():
    rng = random.Random(5)
    raw = {(r, c): min(1, max(0, field(r, c, 0.4) + rng.uniform(-0.45, 0.45))) for r in range(6) for c in range(6)}
    s = [ARROW, frame()]
    s.append(squares(22, 36, 6, 6, 20, lambda r, c: cls(raw[r, c])))
    s.append(squares(202, 36, 6, 6, 20, lambda r, c: cls(0.1 + 0.8 * field(r, c, 0.4))))
    s.append(label("raw SMR", 80, 28, 9, anchor="middle"))
    s.append(label("smoothed (BYM)", 260, 28, 9, anchor="middle"))
    s.append(f'<path d="M146 92 C 164 78, 178 78, 194 92" fill="none" stroke="{Y}" stroke-width="2" '
             f'marker-end="url(#a)"/>')
    s.append(label("shrinkage", 170, 112, 9, Y, 0.95, "middle"))
    s.append(label("small counts, noisy rates", 80, 174, 8, L, 0.85, "middle"))
    s.append(label("borrow strength from neighbours", 260, 174, 8, L, 0.85, "middle"))
    s.append(caption("SMR · empirical Bayes · BYM · MCMC · reporting"))
    return svg("".join(s))


def art_gw():
    rng = random.Random(9)
    s = [frame()]
    for cx, cy in [(96, 92), (244, 112)]:
        for rad, op in [(62, 0.08), (44, 0.12), (26, 0.18)]:
            s.append(f'<circle cx="{cx}" cy="{cy}" r="{rad}" fill="{Y}" fill-opacity="{op}"/>')
        s.append(f'<circle cx="{cx}" cy="{cy}" r="62" fill="none" stroke="{Y}" stroke-width="1.4" '
                 f'stroke-dasharray="5 4"/>')
    for _ in range(70):
        x, y = rng.uniform(18, 322), rng.uniform(18, 186)
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{ramp(x / 340)}" stroke="{D}" stroke-width="1"/>')
    for cx, cy in [(96, 92), (244, 112)]:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#fff" stroke="{D}" stroke-width="2"/>')
    s.append(f'<line x1="96" y1="92" x2="158" y2="92" stroke="#fff" stroke-width="1.4"/>')
    s.append(f'<text x="127" y="84" font-size="8.5" fill="#fff" text-anchor="middle" stroke="{D}" stroke-width="3" '
             f'paint-order="stroke">bandwidth</text>')
    s.append(label("local β (low → high)", 316, 24, 8.5, L, 0.9, "end"))
    s.append(caption("kernels · bandwidth · GWR · MGWR · GW machine learning"))
    return svg("".join(s))


def art_interpolation():
    s = [frame()]
    cell = 12
    s.append(squares(18, 18, 14, 25, cell, lambda r, c: ramp(field(r * 0.45, c * 0.45, 0.2)), gap=-1, op=0.9))
    rng = random.Random(2)
    for _ in range(18):
        x, y = rng.uniform(28, 310), rng.uniform(28, 176)
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#fff" stroke="{D}" stroke-width="1.6"/>')
    s.append(f'<rect x="186" y="78" width="24" height="24" fill="none" stroke="#fff" stroke-width="2"/>')
    s.append(f'<text x="198" y="95" font-size="13" fill="#fff" text-anchor="middle">?</text>')
    s.append(label("samples", 24, 194, 8.5, "#fff", 0.9))
    s.append(caption("IDW · splines · covariates · cross-validation · areal"))
    return svg("".join(s))


def art_sampling():
    s = [frame()]
    s.append(f'<line x1="170" y1="14" x2="170" y2="192" stroke="{B}" stroke-opacity="0.4"/>')
    for i in range(4):
        for j in range(5):
            s.append(f'<circle cx="{36 + j * 28}" cy="{44 + i * 38}" r="4" fill="{Y}" stroke="{D}" stroke-width="1.4"/>')
    rng = random.Random(8)
    strata = [(180, 16, 70, 88, M1), (252, 16, 70, 88, M2), (180, 106, 70, 86, M2), (252, 106, 70, 86, M1)]
    for x, y, w, h, col in strata:
        s.append(f'<rect x="{x}" y="{y}" width="{w - 2}" height="{h - 2}" rx="4" fill="{col}" fill-opacity="0.7"/>')
        for _ in range(4):
            s.append(f'<circle cx="{rng.uniform(x + 10, x + w - 12):.1f}" cy="{rng.uniform(y + 10, y + h - 24):.1f}" '
                     f'r="4" fill="{Y}" stroke="{D}" stroke-width="1.4"/>')
    s.append(label("systematic", 88, 194 - 2, 9, anchor="middle", op=0.9))
    s.append(label("stratified random", 250, 194 - 2, 9, anchor="middle", op=0.9))
    s.append(caption("sampling designs · estimation variance · monitoring networks"))
    return svg("".join(s))


def art_spacetime():
    s = [ARROW]
    for k, (dx, dy, op) in enumerate([(120, 14, 0.45), (80, 50, 0.7), (40, 86, 1.0)]):
        hot = (1 + k, 2 + 2 * (2 - k))
        s.append(f'<g transform="translate({dx},{dy}) skewX(-28)">')
        s.append(f'<rect x="-4" y="-4" width="{7 * 18 + 6}" height="{4 * 18 + 6}" rx="5" fill="{D}" '
                 f'fill-opacity="0.9" stroke="{B}" stroke-opacity="0.6"/>')
        s.append(squares(0, 0, 4, 7, 18, lambda r, c, hot=hot: Y if (r, c) == hot else
                         ("#c9a93a" if abs(r - hot[0]) + abs(c - hot[1]) == 1 else M2), op=op))
        s.append("</g>")
        s.append(label(f"t{3 - k}", dx + 7 * 18 + 8, dy + 12, 9, anchor="start", op=0.85))
    s.append(f'<path d="M314 168 L314 40" stroke="{Y}" stroke-width="2" marker-end="url(#a)"/>')
    s.append(label("time", 322, 110, 9, Y, 0.95))
    s.append(caption("space-time weights · emerging hot spots · ST interpolation · ST regression"))
    return svg("".join(s))


def art_capstone():
    s = [frame()]
    s.append(hexes(26, 28, 7, 7, 12, lambda i, j: cls(field(i, j, 1.1))))
    x0, y0, x1, y1 = 196, 30, 320, 160
    s.append(axes(x0, y0, x1, y1))
    for i, v in enumerate([0.35, 0.55, 0.48, 0.8, 0.66, 0.92]):
        h = v * 110
        s.append(f'<rect x="{x0 + 8 + i * 19}" y="{y1 - h:.1f}" width="13" height="{h:.1f}" rx="2" '
                 f'fill="{Y if v > 0.75 else B}" fill-opacity="0.9"/>')
    s.append(f'<polyline points="203,128 222,110 241,116 260,80 279,92 298,58 314,48" fill="none" '
             f'stroke="#fff" stroke-width="1.8"/>')
    s.append(label("map", 70, 184, 9, anchor="middle", op=0.9))
    s.append(label("model · evidence", 258, 178, 9, anchor="middle", op=0.9))
    s.append(caption("health · elections · overdose · crime · deprivation"))
    return svg("".join(s))


BANNERS = {
    "04_01_banner_point_pattern_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 1", "Point Pattern Analysis",
        "Where events happen: intensity, interaction and tests against randomness",
        ["Fundamentals", "Intensity · KDE", "K · L · G · F", "Hypothesis tests", "Areal pitfalls"],
        art_point_pattern(), number="1"),
    "04_02_banner_autocorrelation_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 2", "Spatial Autocorrelation",
        "Is near more alike than far? Weights, global indices and local clusters",
        ["Concepts", "Spatial weights", "Exploratory maps", "Moran's I · Geary's C", "LISA", "Pitfalls"],
        art_autocorrelation(), number="2"),
    "04_03_banner_spatial_regression_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 3", "Spatial Regression",
        "When residuals are not independent: diagnose, choose and fit spatial models",
        ["OLS diagnostics", "Model selection", "SAR · SEM · SDM", "Reporting", "Limits", "Exercises"],
        art_regression(), number="3"),
    "04_04_banner_spatial_cluster_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 4", "Spatial Clustering",
        "Find hot spots, dense groups and regions, then check that they are real",
        ["Hot spots · Gi*", "DBSCAN", "Scan statistic", "Regionalisation", "Validation"],
        art_cluster(), number="4"),
    "04_05_banner_disease_mapping_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 5", "Disease Mapping",
        "Stable small-area rates: smoothing, Bayesian models and careful reporting",
        ["Small-area problem", "Raw SMR", "Empirical Bayes", "BYM · MCMC", "Shrinkage", "Ethics"],
        art_disease(), number="5"),
    "04_06_banner_gw_models_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 6", "Geographically Weighted Models",
        "Relationships that change over space: kernels, bandwidths, GWR and beyond",
        ["Kernels · bandwidth", "GWSS", "GWR inference", "MGWR", "GW-ML · deep", "Space-time"],
        art_gw(), number="6", h1=33),
    "04_07_banner_spatial_interpolation_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 7", "Spatial Interpolation",
        "From samples to surfaces: deterministic, covariate-assisted and areal methods",
        ["Data prep", "IDW · splines", "Validation", "Covariates", "Areal interpolation"],
        art_interpolation(), number="7"),
    "04_08_banner_spatial_sampling_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 8", "Spatial Sampling Design",
        "Where to measure: sampling designs, estimation variance and monitoring networks",
        ["Foundations", "Designs", "Estimation variance", "Network design", "Exercises"],
        art_sampling(), number="8"),
    "04_09_banner_spatio_temporal_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 9", "Spatio-temporal Analysis",
        "Patterns that move: space-time weights, emerging hot spots and models",
        ["ST weights", "Emerging hot spots", "ST point patterns", "ST interpolation", "ST regression"],
        art_spacetime(), number="9"),
    "04_10_banner_spatial_stat_capstone_geolattice.png": page(
        "SPATIAL STATISTICS · TOPIC 10", "Capstone Projects",
        "Spatial statistics on real problems, from health and elections to crime",
        ["Health disparities", "Voting patterns", "Drug overdose", "NYC crime", "Deprivation"],
        art_capstone(), number="10"),
}

if __name__ == "__main__":
    only = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        for name, html in BANNERS.items():
            if only and not any(k in name for k in only):
                continue
            src = Path(tmp) / (name + ".html")
            src.write_text(html)
            out = OUT / name
            subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=2", "--window-size=1000,250",
                            "--virtual-time-budget=4000", f"--screenshot={out}", src.as_uri()],
                           check=True, capture_output=True)
            print("wrote", out)
