"""Helpers used by the optical-imagery tutorial."""

import numpy as np
import pytest
import xarray as xr

from geolattice.network import is_online
from geolattice.raster.open_imagery import clip_to_bbox, raster_ndvi, read_raster
from geolattice.stac import get_clearest_item
from geolattice.vector import VectorBBox, to_crs


def test_imports_match_the_tutorial():
    from geolattice.network import is_online as online
    from geolattice.raster import clip_to_bbox as clip
    from geolattice.raster import raster_ndvi as ndvi
    from geolattice.raster import read_raster as read
    from geolattice.stac import get_clearest_item as clearest
    from geolattice.stac import search_stac
    from geolattice.vector import VectorBBox as Box
    from geolattice.vector import to_crs as project

    assert all(callable(fn) for fn in (online, clip, ndvi, read, clearest, search_stac, project))
    assert Box is VectorBBox


def test_is_online_rejects_a_closed_port():
    assert is_online("http://127.0.0.1:1", timeout=0.2) is False


def test_vector_bbox_reprojects_to_lonlat():
    local = VectorBBox((400_000, 4_700_000, 401_000, 4_701_000), 32618)
    geographic = to_crs(local, 4326)
    assert geographic.bounds == geographic.total_bounds
    assert -78 < geographic.bounds[0] < -74
    assert 42 < geographic.bounds[1] < 43
    back = to_crs(geographic, "EPSG:32618")
    # A lon/lat box is larger than the original projected rectangle, so the
    # return trip grows by a few metres.
    np.testing.assert_allclose(back.bounds, local.bounds, atol=20.0)


def test_clearest_item_prefers_a_covering_scene():
    class Item:
        def __init__(self, scene_id, cloud, geometry):
            self.id = scene_id
            self.properties = {"eo:cloud_cover": cloud}
            self.geometry = geometry

    inside = {
        "type": "Polygon",
        "coordinates": [
            [[-76.7, 42.3], [-76.3, 42.3], [-76.3, 42.6], [-76.7, 42.6], [-76.7, 42.3]]
        ],
    }
    outside = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    cloudy = Item("cloudy", 4, inside)
    clear_elsewhere = Item("elsewhere", 0, outside)
    clear = Item("clear", 8, inside)
    area = VectorBBox((-76.6, 42.4, -76.4, 42.5), 4326)
    chosen = get_clearest_item("sentinel-2-l2a", area, items=[cloudy, clear_elsewhere, clear])
    assert chosen.id == "cloudy"


def test_raster_ndvi_masks_null_red():
    red = xr.DataArray([0.1, np.nan], dims="x")
    nir = xr.DataArray([0.5, 0.4], dims="x")
    ndvi = raster_ndvi(nir, red)
    assert ndvi.values[0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))
    assert np.isnan(ndvi.values[1])


def test_clip_to_bbox_uses_the_raster_crs():
    pytest.importorskip("rioxarray")
    from affine import Affine

    values = np.arange(16, dtype="float32").reshape(4, 4)
    raster = xr.DataArray(
        values, dims=("y", "x"), coords={"y": [3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]}
    )
    raster = raster.rio.write_crs(4326)
    raster = raster.rio.write_transform(Affine(1, 0, 0, 0, -1, 4))
    clipped = clip_to_bbox(raster, VectorBBox((1, 1, 3, 3), 4326))
    assert clipped.sizes["x"] < raster.sizes["x"]
    assert clipped.rio.crs.to_epsg() == 4326
    assert read_raster is not None
