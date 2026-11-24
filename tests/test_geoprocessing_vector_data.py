"""Tests for vector geoprocessing tools."""

import math
import warnings

import pandas as pd
import pytest
import pyproj
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform

from geolattice.crs import CRS
from geolattice.vector import (
    GeoprocessError,
    buffer,
    clip,
    clipping,
    convex_hull,
    dissolve,
    erase,
    intersect,
    union,
)
from geolattice.vector.read_vector_data import VectorFeature


def _frame(rows, geometries, crs="EPSG:32618"):
    import geopandas as gpd

    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)


def _squares():
    left = _frame(
        {"left_id": [1, 2]},
        [box(0, 0, 2, 2), box(2, 2, 4, 4)],
    )
    right = _frame(
        {"right_id": [1, 2]},
        [box(1, 1, 3, 3), box(3, 3, 5, 5)],
    )
    return left, right


class TestClip:
    """Clip keeps the part of each input feature that falls inside the mask."""

    def test_clip_cuts_edges_and_drops_features_outside(self):
        data = _frame(
            {"name": ["inside", "edge", "outside"]},
            [box(1, 1, 2, 2), box(4, 4, 6, 6), box(10, 10, 11, 11)],
        )
        result = clip(data, box(0, 0, 5, 5))
        assert set(result["name"]) == {"inside", "edge"}
        areas = dict(zip(result["name"], result.geometry.area))
        assert areas["inside"] == pytest.approx(1)
        assert areas["edge"] == pytest.approx(1)
        assert result.crs.to_epsg() == 32618

    def test_clipping_alias_accepts_a_bounding_box(self):
        data = _frame({"name": ["a", "b"]}, [box(0, 0, 2, 2), box(3, 3, 4, 4)])
        result = clipping(data, (0, 0, 1, 1))
        assert list(result["name"]) == ["a"]
        assert result.geometry.iloc[0].area == pytest.approx(1)

    def test_clip_does_not_change_the_input(self):
        data = _frame({"name": ["a"]}, [box(0, 0, 2, 2)])
        original = data.geometry.iloc[0].wkt
        clip(data, box(0, 0, 1, 1))
        assert data.geometry.iloc[0].wkt == original

    def test_reprojects_mask_into_the_input_crs(self):
        data = _frame({"name": ["parcel"]}, [box(-0.2, -0.2, 0.2, 0.2)], crs="EPSG:4326")
        forward = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform
        mask = _frame({}, [transform(forward, box(-1, -1, 1, 1))], crs="EPSG:3857")
        with pytest.warns(UserWarning, match="Reprojecting the mask"):
            result = clip(data, mask)
        assert len(result) == 1
        assert result.geometry.iloc[0].area == pytest.approx(0.16)
        assert result.crs.to_epsg() == 4326

    def test_same_crs_does_not_reproject(self):
        data = _frame({"name": ["a"]}, [box(0, 0, 1, 1)], crs="EPSG:32618")
        mask = _frame({}, [box(0, 0, 2, 2)], crs=32618)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = clip(data, mask)
        assert result.geometry.iloc[0].area == pytest.approx(1)


class TestUnion:
    """Unary union merges one layer. Two layers build an overlay union."""

    def test_unary_union_merges_every_geometry(self):
        data = _frame({"name": ["a", "b"]}, [box(0, 0, 2, 2), box(2, 0, 4, 2)])
        result = union(data)
        assert len(result) == 1
        assert result["n_features"].iloc[0] == 2
        assert result.geometry.iloc[0].area == pytest.approx(8)
        assert "name" not in result.columns

    def test_overlay_union_splits_overlaps_and_keeps_both_sides(self):
        from shapely import union_all

        left, right = _squares()
        result = union(left, right)
        expected = union_all([*left.geometry, *right.geometry]).area
        assert len(result) == 7
        assert result.geometry.area.sum() == pytest.approx(expected)
        both = result.dropna(subset=["left_id", "right_id"])
        assert len(both) == 3
        assert set(both["left_id"]) == {1, 2}

    def test_shared_attribute_names_get_suffixes(self):
        left = _frame({"zone": ["a"]}, [box(0, 0, 2, 2)])
        right = _frame({"zone": ["b"]}, [box(1, 1, 3, 3)])
        result = intersect(left, right)
        assert list(result.columns) == ["zone_1", "zone_2", "geometry"]
        assert result["zone_1"].iloc[0] == "a"
        assert result["zone_2"].iloc[0] == "b"


class TestDissolve:
    """Dissolve groups features and unions each group."""

    def test_dissolve_by_attribute_sums_values(self):
        features = [
            VectorFeature(1, box(0, 0, 1, 1), {"zone": "a", "value": 1}),
            VectorFeature(2, box(1, 0, 2, 1), {"zone": "a", "value": 4}),
            VectorFeature(3, box(0, 1, 1, 2), {"zone": "b", "value": 10}),
        ]
        result = dissolve(features, by="zone", aggfunc={"value": "sum"})
        assert set(result["zone"]) == {"a", "b"}
        values = dict(zip(result["zone"], result["value"]))
        areas = dict(zip(result["zone"], result.geometry.area))
        assert values == {"a": 5, "b": 10}
        assert areas["a"] == pytest.approx(2)
        assert areas["b"] == pytest.approx(1)
        assert "fid" not in result.columns

    def test_dissolve_without_by_aggregates_the_whole_layer(self):
        data = _frame(
            {"population": [10, 25]},
            [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        )
        result = dissolve(data, aggfunc="sum")
        assert len(result) == 1
        assert result["population"].iloc[0] == 35
        assert result.geometry.iloc[0].area == pytest.approx(2)

    def test_missing_dissolve_field_raises(self):
        data = _frame({"name": ["a"]}, [box(0, 0, 1, 1)])
        with pytest.raises(GeoprocessError, match="state"):
            dissolve(data, by="state")


class TestIntersect:
    """Intersect returns one feature for each overlapping pair."""

    def test_pairwise_overlap_keeps_both_attributes(self):
        left, right = _squares()
        result = intersect(left, right)
        assert len(result) == 3
        assert result.geometry.area.sum() == pytest.approx(3)
        pairs = set(zip(result["left_id"].tolist(), result["right_id"].tolist()))
        assert pairs == {(1, 1), (2, 1), (2, 2)}

    def test_disjoint_layers_return_an_empty_frame(self):
        left = _frame({"left_id": [1]}, [box(0, 0, 1, 1)])
        right = _frame({"right_id": [2]}, [box(5, 5, 6, 6)])
        result = intersect(left, right)
        assert len(result) == 0
        assert list(result.columns) == ["left_id", "right_id", "geometry"]
        assert result.crs.to_epsg() == 32618

    def test_boundary_touch_is_dropped_when_geometry_type_is_kept(self):
        left = _frame({}, [box(0, 0, 1, 1)])
        right = _frame({}, [box(1, 0, 2, 1)])
        with pytest.warns(UserWarning, match="keep_geom_type"):
            kept = intersect(left, right)
        assert len(kept) == 0
        shared = intersect(left, right, keep_geom_type=False)
        assert len(shared) == 1
        assert shared.geometry.iloc[0].geom_type == "LineString"

    def test_only_candidate_pairs_are_compared(self):
        left = _frame(
            {"name": [f"l{i}" for i in range(5)]},
            [box(i * 10, 0, i * 10 + 1, 1) for i in range(5)],
        )
        right = _frame(
            {"name": [f"r{i}" for i in range(5)]},
            [box(i * 10 + 0.5, 0, i * 10 + 1.5, 1) for i in range(5)],
        )
        result = intersect(left, right)
        assert len(result) == 5
        assert result.geometry.area.sum() == pytest.approx(2.5)


class TestErase:
    """Erase subtracts one layer from another."""

    def test_erase_removes_the_overlapping_area(self):
        data = _frame({"name": ["block"]}, [box(0, 0, 4, 4)])
        result = erase(data, box(1, 1, 2, 2))
        assert list(result["name"]) == ["block"]
        assert result.geometry.iloc[0].area == pytest.approx(15)

    def test_fully_covered_feature_is_dropped(self):
        data = _frame({"name": ["gone", "kept"]}, [box(0, 0, 1, 1), box(10, 10, 11, 11)])
        result = erase(data, box(-1, -1, 2, 2))
        assert list(result["name"]) == ["kept"]
        assert result.geometry.iloc[0].area == pytest.approx(1)

    def test_erase_can_split_a_feature_without_adding_rows(self):
        data = _frame({"name": ["road"]}, [box(0, 0, 4, 2)])
        result = erase(data, box(1, -1, 2, 3))
        assert len(result) == 1
        assert result.geometry.iloc[0].area == pytest.approx(6)


class TestConvexHull:
    """Convex hull can be built per feature or for a whole layer."""

    def test_each_feature_keeps_its_attributes(self):
        polygon = Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)])
        data = _frame({"name": ["lot"]}, [polygon])
        result = convex_hull(data)
        assert list(result["name"]) == ["lot"]
        assert result.geometry.iloc[0].area == pytest.approx(polygon.convex_hull.area)
        assert result.geometry.iloc[0].area > polygon.area

    def test_as_one_hull_covers_every_feature(self):
        points = [Point(0, 0), Point(4, 0), Point(0, 3)]
        result = convex_hull(points, as_one=True)
        assert len(result) == 1
        assert result["n_features"].iloc[0] == 3
        assert result.geometry.iloc[0].area == pytest.approx(6)

    def test_wkt_and_geojson_geometries_are_accepted(self):
        frame = pd.DataFrame({"geometry": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"]})
        mapping = {"type": "Point", "coordinates": [2.0, 3.0]}
        wkt_result = convex_hull(frame)
        point_result = convex_hull(mapping)
        assert wkt_result.geometry.iloc[0].geom_type == "Polygon"
        assert point_result.geometry.iloc[0].geom_type == "Point"


class TestBuffer:
    """Buffer offsets features by a distance in CRS units."""

    def test_round_buffer_area(self):
        data = _frame({}, [Point(0, 0)])
        result = buffer(data, 10)
        assert result.geometry.iloc[0].area == pytest.approx(Point(0, 0).buffer(10).area)
        assert result.geometry.iloc[0].area == pytest.approx(math.pi * 100, rel=1e-2)
        assert result.crs.to_epsg() == 32618

    def test_square_caps_and_distance_field(self):
        data = _frame({"radius_m": [1.0, 2.0]}, [Point(0, 0), Point(10, 10)])
        result = buffer(data, "radius_m", cap_style="square")
        assert result.geometry.iloc[0].area == pytest.approx(4)
        assert result.geometry.iloc[1].area == pytest.approx(16)

    def test_dissolve_unions_buffers(self):
        data = _frame({}, [Point(0, 0), Point(1, 0)])
        separate = buffer(data, 1)
        merged = buffer(data, 1, dissolve=True)
        assert len(separate) == 2
        assert len(merged) == 1
        assert merged.geometry.iloc[0].area < separate.geometry.area.sum()
        assert merged["n_features"].iloc[0] == 2

    def test_single_sided_line_buffer_is_smaller_than_a_full_buffer(self):
        line = LineString([(0, 0), (10, 0)])
        full = buffer([line], 2)
        half = buffer([line], 2, single_sided=True, cap_style="flat")
        assert half.geometry.iloc[0].area < full.geometry.iloc[0].area

    def test_geographic_crs_warns_that_distance_is_in_degrees(self):
        data = _frame({}, [Point(0, 0)], crs="EPSG:4326")
        with pytest.warns(UserWarning, match="degrees"):
            result = buffer(data, 0.1)
        assert result.crs.is_geographic

    def test_resolution_changes_the_approximation(self):
        data = _frame({}, [Point(0, 0)])
        coarse = buffer(data, 1, resolution=2)
        fine = buffer(data, 1, quad_segs=16)
        assert coarse.geometry.iloc[0].area != pytest.approx(fine.geometry.iloc[0].area)

    def test_invalid_style_and_distance_length_raise(self):
        data = _frame({}, [Point(0, 0), Point(1, 1)])
        with pytest.raises(GeoprocessError, match="cap_style"):
            buffer(data, 1, cap_style="oval")
        with pytest.raises(GeoprocessError, match="2 feature"):
            buffer(data, [1, 2, 3])


class TestInputs:
    """Layers from this package and plain tables are both accepted."""

    def test_open_layer_reader_keeps_crs_and_attributes(self):
        class VectorLayer:
            def __init__(self):
                self.crs = CRS(32618)

            def features(self):
                return iter(())

            def to_dataframe(self):
                return pd.DataFrame({"name": ["parcel"], "geometry": [box(0, 0, 2, 2)]})

        result = buffer(VectorLayer(), 0)
        assert result["name"].iloc[0] == "parcel"
        assert result.geometry.iloc[0].area == pytest.approx(4)
        assert result.crs.to_epsg() == 32618

    def test_invalid_polygon_is_repaired(self):
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        assert not bowtie.is_valid
        data = _frame({"name": ["x"]}, [bowtie])
        result = clip(data, box(-1, -1, 2, 2))
        assert len(result) == 1
        assert result.geometry.iloc[0].is_valid
        assert result.geometry.iloc[0].area == pytest.approx(0.5)
        with pytest.raises(GeoprocessError, match="invalid geometry"):
            clip(data, box(-1, -1, 2, 2), make_valid=False)

    def test_unsupported_input_raises_type_error(self):
        with pytest.raises(TypeError):
            buffer(42, 1)
