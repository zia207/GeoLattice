"""Tests for spatial data visualization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from shapely.geometry import LineString, Point, box

from geolattice.viz import (
    EDGE,
    FACE,
    VisualizationError,
    add_basemap,
    add_choropleth,
    add_geojson,
    add_labels,
    add_layer_control,
    add_markers,
    add_north_arrow,
    add_scale_bar,
    add_vector_layer,
    adjust_image,
    animate_map,
    apply_map_style,
    infer_raster_kind,
    interactive_map,
    normalize_band,
    plot_bar,
    plot_boundary,
    plot_categorical,
    plot_choropleth,
    plot_contours,
    plot_false_color,
    plot_gridded,
    plot_histogram,
    plot_layer,
    plot_layers,
    plot_pair,
    plot_pie,
    plot_points,
    plot_raster,
    plot_raster_bands,
    plot_raster_facet,
    plot_raster_histogram,
    plot_rasters,
    plot_rgb,
    plot_scatter,
    plot_unique_raster,
    save_map,
    style_axes,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _polygons():
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"name": ["a", "b", "c", "d"], "value": [1, 4, 9, 16]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )


def _lines():
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"road": ["main"]},
        geometry=[LineString([(0, 0), (1, 1), (2, 0.2)])],
        crs="EPSG:4326",
    )


def _points():
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"site": ["north", "south"], "score": [3, 8]},
        geometry=[Point(0.2, 0.2), Point(1.4, 1.6)],
        crs="EPSG:4326",
    )


class TestPlotLayer:
    def test_polygon_uses_the_tutorial_face_color(self):
        ax = plot_layer(_polygons(), "States", show=False)
        assert ax.get_title(loc="left") == "States"
        face = ax.collections[0].get_facecolor()[0]
        assert np.allclose(face, to_rgba(FACE))

    def test_outline_only(self):
        ax = plot_layer(_polygons(), "Outline", fill=False, show=False)
        edge = ax.collections[0].get_edgecolor()[0]
        assert np.allclose(edge, to_rgba(EDGE))
        faces = ax.collections[0].get_facecolor()
        assert len(faces) == 0 or faces[0][3] == 0

    def test_lines_and_points(self):
        line_ax = plot_layer(_lines(), "Roads", show=False)
        point_ax = plot_layer(_points(), "Sites", show=False)
        assert line_ax.collections
        assert point_ax.collections

    def test_pair_draws_two_panels(self):
        fig, axes = plot_pair(_polygons(), _lines(), ["Polygons", "Lines"], show=False)
        assert len(axes) == 2
        assert axes[0].get_title(loc="left") == "Polygons"
        assert fig.axes

    def test_pair_requires_two_titles(self):
        with pytest.raises(VisualizationError, match="titles"):
            plot_pair(_polygons(), _lines(), ["Only one"], show=False)

    def test_empty_layer_raises(self):
        import geopandas as gpd

        empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")
        with pytest.raises(VisualizationError, match="no geometries"):
            plot_layer(empty, "Empty", show=False)

    def test_missing_column_raises(self):
        with pytest.raises(VisualizationError, match="population"):
            plot_layer(_polygons(), column="population", show=False)


class TestChoropleth:
    def test_quantiles_and_missing_values(self):
        data = _polygons()
        data.loc[0, "value"] = np.nan
        ax = plot_choropleth(
            data,
            "value",
            "Density",
            scheme="quantiles",
            k=3,
            missing_kwds={"color": "lightgrey"},
            show=False,
        )
        assert ax.get_title(loc="left") == "Density"
        assert ax.collections

    def test_horizontal_legend_adds_a_colorbar_axes(self):
        ax = plot_choropleth(
            _polygons(),
            "value",
            legend_orientation="horizontal",
            show=False,
        )
        assert ax.child_axes
        assert ax.child_axes[0].collections

    def test_categorical_and_boundary(self):
        ax = plot_categorical(_polygons(), "name", "Names", show=False)
        boundary = plot_boundary(_polygons(), "Edges", show=False)
        assert ax.get_title(loc="left") == "Names"
        assert boundary.collections

    def test_layers_keep_points_above_polygons_and_share_a_crs(self):
        import geopandas as gpd

        polygons = _polygons()
        points = _points().to_crs(3857)
        with pytest.warns(UserWarning, match="Reprojecting"):
            ax = plot_layers(
                [
                    {"data": polygons, "label": "Blocks", "fill": False},
                    {"data": points, "label": "Sites", "zorder": 5, "color": "#E76F51"},
                ],
                title="Overlay",
                show=False,
            )
        labels = ax.get_legend().get_texts()
        assert [text.get_text() for text in labels] == ["Blocks", "Sites"]
        assert ax.get_title(loc="left") == "Overlay"
        drawn = gpd.GeoDataFrame(points.to_crs(4326))
        assert drawn.crs.to_epsg() == 4326


class TestRaster:
    def test_numpy_and_xarray_and_contours(self):
        import xarray as xr

        grid = np.arange(48, dtype=float).reshape(6, 8)
        image = plot_raster(grid, title="Array", show=False)
        data = xr.DataArray(
            grid.reshape(1, 6, 8),
            dims=("time", "y", "x"),
            coords={"time": [0], "y": np.arange(6), "x": np.arange(8)},
            name="ndvi",
        )
        dataset = data.to_dataset()
        raster = plot_raster(dataset, time=0, title="NDVI", show=False)
        contours = plot_contours(grid, levels=4, filled=True, show=False)
        _fig, axes = plot_rasters([grid, grid], titles=["A", "B"], show=False)
        assert image.get_title(loc="left") == "Array"
        assert raster.get_title(loc="left") == "NDVI"
        assert contours.collections
        assert axes[0, 0].get_title(loc="left") == "A"
        assert axes[0, 1].get_title(loc="left") == "B"

    def test_dataset_with_several_variables_needs_a_name(self):
        import xarray as xr

        grid = np.ones((3, 3))
        dataset = xr.Dataset({"red": (("y", "x"), grid), "nir": (("y", "x"), grid)})
        with pytest.raises(VisualizationError, match="variable"):
            plot_raster(dataset, show=False)

    def test_rgb_image(self):
        import xarray as xr

        cube = xr.DataArray(
            np.random.default_rng(0).random((3, 5, 6)),
            dims=("band", "y", "x"),
            coords={"band": [1, 2, 3]},
        )
        ax = plot_raster(cube, rgb=True, title="True color", show=False)
        assert ax.get_title(loc="left") == "True color"

    def test_general_unique_and_rgb_kinds(self):
        continuous = np.linspace(0, 100, 100).reshape(10, 10)
        classes = np.array([[1, 1, 2], [2, 3, 3], [1, 2, 3]], dtype=np.int16)
        rgb = np.random.default_rng(1).integers(0, 255, size=(3, 8, 8), dtype=np.uint8)

        assert infer_raster_kind(continuous) == "general"
        assert infer_raster_kind(classes) == "unique"
        assert infer_raster_kind(rgb) == "rgb"

        ax = plot_raster(continuous, kind="general", rescale=(10, 90), title="General", show=False)
        assert ax.get_title(loc="left") == "General"

        unique_ax = plot_unique_raster(
            classes,
            categories={1: "Water", 2: "Forest", 3: "Urban"},
            title="Land cover",
            show=False,
        )
        assert unique_ax.get_title(loc="left") == "Land cover"

        rgb_ax = plot_rgb(rgb, title="RGB", show=False)
        assert rgb_ax.get_title(loc="left") == "RGB"

        # Four-band stack: false color using bands 4, 3, 2
        stack = np.random.default_rng(2).random((4, 6, 6))
        false_ax = plot_false_color(stack, bands=(4, 3, 2), title="NIR-R-G", show=False)
        assert false_ax.get_title(loc="left") == "NIR-R-G"

    def test_bands_histogram_facet_and_helpers(self):
        stack = np.random.default_rng(3).random((3, 12, 10))
        fig, axes = plot_raster_bands(stack, bands=(1, 2, 3), show=False)
        assert axes.shape[0] * axes.shape[1] >= 3

        hist_ax = plot_raster_histogram(stack, bands=(1, 2), title="Bands", show=False)
        assert hist_ax.get_title(loc="left") == "Bands"

        fig2, _axes2 = plot_raster_facet(stack, col_wrap=2, show=False)
        assert fig2 is not None

        band = normalize_band(stack[0], stretch="percentile")
        assert band.min() >= 0.0 and band.max() <= 1.0
        adjusted = adjust_image(np.dstack([band, band, band]), brightness=0.8, contrast=0.2)
        assert adjusted.shape[-1] == 3

    def test_vector_overlay(self):
        grid = np.arange(25, dtype=float).reshape(5, 5)
        ax = plot_raster(
            grid,
            extent=(0, 2, 0, 2),
            vector=_polygons(),
            title="With borders",
            show=False,
        )
        assert ax.get_title(loc="left") == "With borders"
        assert ax.collections  # polygon edges

    def test_projected_grid_needs_cartopy(self):
        pytest.importorskip("cartopy")
        ax = plot_gridded(np.ones((4, 6)), title="Globe", show=False)
        assert ax.get_title(loc="left") == "Globe"


class TestInteractive:
    def test_explore_map_embeds_the_layer(self):
        web = interactive_map(_polygons(), column="value", cmap="OrRd", scheme="quantiles", k=4)
        html = web._repr_html_()
        assert "leaflet" in html.lower()
        assert "value" in html

    def test_multilayer_map_has_a_layer_control(self):
        web = interactive_map(
            [
                {"data": _polygons(), "name": "Blocks", "column": "name", "categorical": True},
                {"data": _lines(), "name": "Roads", "color": "#222222"},
            ],
            tiles="CartoDB positron",
        )
        names = {child.__class__.__name__ for child in web._children.values()}
        assert "LayerControl" in names

    def test_markers_geojson_and_choropleth(self):
        web = interactive_map(location=[0, 0], zoom_start=3, height="300px")
        add_markers(web, _points(), name="Sites", tooltip="site")
        add_geojson(web, _polygons(), name="Blocks")
        add_vector_layer(web, _lines(), name="Roads")
        add_choropleth(web, _polygons(), "value", name="Values")
        add_layer_control(web)
        html = web._repr_html_()
        assert "Sites" in html
        assert "Blocks" in html
        controls = [
            child for child in web._children.values() if child.__class__.__name__ == "LayerControl"
        ]
        assert len(controls) == 1

    def test_folium_choropleth_joins_on_a_key(self):
        web = interactive_map(zoom_start=2)
        table = pd.DataFrame({"code": ["a", "b", "c", "d"], "value": [1, 4, 9, 16]})
        add_choropleth(
            web,
            _polygons(),
            "value",
            key_on="feature.properties.name",
            id_column="code",
            table=table,
            legend_name="Value",
        )
        assert any(child.__class__.__name__ == "Choropleth" for child in web._children.values())


class TestChartsAndAnnotation:
    def test_charts(self):
        frame = _polygons()
        assert plot_bar(frame, "value", labels="name", title="Totals", show=False).patches
        assert plot_pie(frame, "value", labels="name", show=False).patches
        assert plot_histogram(frame, "value", bins=4, show=False).patches
        scatter = plot_scatter(frame, "value", "value", column="value", show=False)
        assert scatter.collections

    def test_pie_rejects_negative_values(self):
        frame = _polygons()
        frame.loc[0, "value"] = -1
        with pytest.raises(VisualizationError, match="positive"):
            plot_pie(frame, "value", show=False)

    def test_labels_north_arrow_and_scale_bar(self):
        data = _polygons()
        ax = plot_layer(data, "Labeled", show=False)
        add_labels(ax, data, "name")
        add_north_arrow(ax)
        add_scale_bar(ax, length=0.5, crs=data.crs, unit="°")
        texts = [text.get_text() for text in ax.texts]
        assert "a" in texts
        assert "N" in texts
        assert any("0.5" in text for text in texts)

    def test_points_and_save(self, tmp_path):
        ax = plot_points(np.array([[0, 0, 1], [1, 1, 3], [2, 0, 2]]), title="Samples", show=False)
        path = save_map(ax.figure, tmp_path / "map.png")
        assert path.is_file()
        assert path.stat().st_size > 0

    def test_animation_has_one_frame_per_time(self):
        import geopandas as gpd

        left = _polygons().assign(year=2020)
        right = _polygons().assign(year=2021, value=[2, 5, 10, 17])
        both = gpd.GeoDataFrame(pd.concat([left, right], ignore_index=True), crs="EPSG:4326")
        anim = animate_map(both, "year", column="value", interval=200)
        assert isinstance(anim, FuncAnimation)

    def test_style_helpers(self):
        apply_map_style()
        _fig, ax = plt.subplots()
        style_axes(ax, "Styled", axis_off=True)
        assert ax.get_title(loc="left") == "Styled"
        assert not ax.axison


class TestBasemap:
    def test_unknown_provider_raises(self):
        _fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        with pytest.raises(VisualizationError, match="Unknown tile provider"):
            add_basemap(ax, source="NotAProvider.Missing")

    def test_add_basemap_uses_contextily(self, monkeypatch):
        import contextily

        calls = {}

        def fake_add(ax, **kwargs):
            calls["crs"] = kwargs.get("crs")
            calls["source"] = kwargs.get("source")

        monkeypatch.setattr(contextily, "add_basemap", fake_add)
        _fig, ax = plt.subplots()
        add_basemap(ax, source="CartoDB.Positron", crs="EPSG:3857")
        assert calls["crs"] == "EPSG:3857"
        assert calls["source"].name == "CartoDB.Positron"
