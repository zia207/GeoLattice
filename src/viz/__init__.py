"""Spatial data visualization.

Static maps, choropleths, raster and gridded plots, Folium maps, attribute
charts, and a few cartographic extras. The vector helpers match the
geoprocessing tutorial (``plot_layer``, ``plot_pair``). Choropleths, legends,
classification schemes, missing values, layer order, and tile basemaps follow
`GeoPandas mapping <https://geopandas.org/en/stable/docs/user_guide/mapping.html>`_.
Interactive maps use
`Folium <https://github.com/python-visualization/folium>`_ through
``GeoDataFrame.explore``.

Raster visualization follows the three GeoHub layer types (general, unique
value, and true-color RGB), Spatial Thoughts xarray selection patterns, and
Python GIS RGB / false-color / histogram workflows.

Install the extras with::

    pip install "geolattice[viz]"

Cartopy is imported only by ``plot_gridded``.
"""

from geolattice.viz._imports import VisualizationError
from geolattice.viz.annotate import (
    add_labels,
    add_north_arrow,
    add_scale_bar,
    cartographic_elements,
)
from geolattice.viz.basemap import add_basemap
from geolattice.viz.charts import plot_bar, plot_histogram, plot_pie, plot_scatter
from geolattice.viz.interactive import (
    add_choropleth,
    add_geojson,
    add_layer_control,
    add_markers,
    add_vector_layer,
    interactive_map,
)
from geolattice.viz.raster import (
    adjust_image,
    infer_raster_kind,
    normalize_band,
    plot_contours,
    plot_false_color,
    plot_gridded,
    plot_raster,
    plot_raster_bands,
    plot_raster_facet,
    plot_raster_histogram,
    plot_rasters,
    plot_rgb,
    plot_unique_raster,
)
from geolattice.viz.style import (
    CATEGORICAL,
    DIVERGING,
    EDGE,
    FACE,
    SEQUENTIAL,
    apply_map_style,
    style_axes,
)
from geolattice.viz.vector import (
    animate_map,
    plot_boundary,
    plot_categorical,
    plot_choropleth,
    plot_layer,
    plot_layers,
    plot_pair,
    plot_points,
    save_map,
)

__all__ = [
    "FACE",
    "EDGE",
    "SEQUENTIAL",
    "DIVERGING",
    "CATEGORICAL",
    "VisualizationError",
    "apply_map_style",
    "style_axes",
    "plot_layer",
    "plot_pair",
    "plot_layers",
    "plot_choropleth",
    "plot_categorical",
    "plot_boundary",
    "plot_points",
    "animate_map",
    "add_basemap",
    "plot_raster",
    "plot_rasters",
    "plot_rgb",
    "plot_false_color",
    "plot_unique_raster",
    "plot_raster_bands",
    "plot_raster_facet",
    "plot_raster_histogram",
    "infer_raster_kind",
    "normalize_band",
    "adjust_image",
    "plot_contours",
    "plot_gridded",
    "interactive_map",
    "add_vector_layer",
    "add_geojson",
    "add_markers",
    "add_choropleth",
    "add_layer_control",
    "add_labels",
    "add_north_arrow",
    "add_scale_bar",
    "cartographic_elements",
    "plot_bar",
    "plot_pie",
    "plot_histogram",
    "plot_scatter",
    "save_map",
]
