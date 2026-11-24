"""
Bangladesh district explorer, Solara version (Notebook 08_07).

Run:  solara run bd_solara.py
"""

import solara

import bd_dashboard as bd

GDF, NDVI = bd.load_districts(), bd.load_ndvi()     # module level: loaded once per server process

# Reactive state; Solara keeps a separate copy for every browser session.
division = solara.reactive("All")
district = solara.reactive("Bogra")
variable = solara.reactive("poverty_pct")


@solara.component
def Cards(name: str):
    c = solara.use_memo(lambda: bd.district_card(GDF, name), dependencies=[name])
    pov = f"{c['poverty_pct']:.1f} % (rank {c['poverty_rank']} of 64)" if c["poverty_pct"] is not None else "n/a"
    solara.Markdown(f"### {name}, {c['division']} division\n"
                    f"**Poverty** {pov} · **Density** {c['density']:,.0f} per km² · "
                    f"**Rice** {c['rice_kt']:,.0f} kt · **NDVI trend** {c['ndvi_trend']:+.3f} per decade")


@solara.component
def Page():
    names = solara.use_memo(lambda: bd.district_list(GDF, division.value), dependencies=[division.value])
    if district.value not in names:
        district.set(names[0])
    # use_memo: recompute only when the listed inputs change
    m = solara.use_memo(lambda: bd.choropleth(GDF, variable.value, district.value)._repr_html_(),
                        dependencies=[variable.value, district.value])
    ts = solara.use_memo(lambda: bd.ndvi_figure(NDVI, district.value), dependencies=[district.value])
    season = solara.use_memo(lambda: bd.seasonal_figure(NDVI, district.value), dependencies=[district.value])

    solara.Title("Bangladesh district explorer")
    labels = {label: key for key, label in bd.VARIABLES.items()}
    with solara.Columns([1, 2, 2]):
        with solara.Column():                       # controls (a plain column works at any screen width)
            solara.Markdown("## Bangladesh district explorer")
            solara.Select("Division", values=["All"] + sorted(GDF.division.unique()), value=division)
            solara.Select("District", values=names, value=district)
            solara.Select("Map variable", values=list(labels), value=bd.VARIABLES[variable.value],
                          on_value=lambda label: variable.set(labels[label]))
            solara.Markdown("Data: BBS, DAE, MODIS NDVI 2000-2024 (Notebooks 08_03, 08_06).")
        with solara.Column():
            Cards(district.value)
            solara.HTML(tag="div", unsafe_innerHTML=m)
        with solara.Column():
            solara.FigurePlotly(ts)
            solara.FigurePlotly(season)
