"""
Bangladesh district explorer, Panel version (Notebook 08_07).

Run:  panel serve bd_panel.py
"""

import panel as pn

import bd_dashboard as bd

pn.extension("plotly", sizing_mode="stretch_width")


@pn.cache                          # read once per server process, shared by every session
def data():
    return bd.load_districts(), bd.load_ndvi()


def build():
    """Create the widgets, panes and template; return (template, widgets) so tests can drive them."""
    gdf, nd = data()
    division = pn.widgets.Select(name="Division", options=["All"] + sorted(gdf.division.unique()), value="All")
    district = pn.widgets.Select(name="District", options=bd.district_list(gdf), value="Bogra")
    variable = pn.widgets.RadioButtonGroup(name="Map variable", options={v: k for k, v in bd.VARIABLES.items()},
                                           value="poverty_pct", orientation="vertical", button_type="light")

    cards = pn.pane.Markdown()
    map_pane = pn.pane.plot.Folium(height=480)
    ts_pane = pn.pane.Plotly(height=330)
    season_pane = pn.pane.Plotly(height=280)

    # Each output is recomputed only when one of its own inputs changes.
    @pn.depends(division, watch=True)
    def _filter(div):
        names = bd.district_list(gdf, div)
        district.options = names
        if district.value not in names:
            district.value = names[0]

    @pn.depends(district, watch=True)
    def _district(name):
        c = bd.district_card(gdf, name)
        pov = f"{c['poverty_pct']:.1f} % (rank {c['poverty_rank']} of 64)" if c["poverty_pct"] is not None else "n/a"
        cards.object = (f"### {name}, {c['division']} division\n"
                        f"**Poverty** {pov} · **Density** {c['density']:,.0f} per km² · "
                        f"**Rice** {c['rice_kt']:,.0f} kt · **NDVI trend** {c['ndvi_trend']:+.3f} per decade")
        ts_pane.object = bd.ndvi_figure(nd, name)
        season_pane.object = bd.seasonal_figure(nd, name)

    @pn.depends(variable, district, watch=True)
    def _map(var, name):
        map_pane.object = bd.choropleth(gdf, var, name)

    _district(district.value)              # first render
    _map(variable.value, district.value)
    template = pn.template.FastListTemplate(
        title="Bangladesh district explorer",
        sidebar=[division, district, pn.pane.Markdown("**Map variable**"), variable,
                 pn.pane.Markdown("Data: BBS, DAE, MODIS NDVI 2000-2024 (Notebooks 08_03, 08_06).")],
        main=[cards, pn.Row(map_pane, pn.Column(ts_pane, season_pane))],
        accent="#2c7fb8")
    return template, {"division": division, "district": district, "variable": variable,
                      "cards": cards, "map": map_pane, "timeseries": ts_pane, "seasonal": season_pane}


if __name__.startswith("bokeh"):    # `panel serve` runs this file once per browser session
    build()[0].servable()
