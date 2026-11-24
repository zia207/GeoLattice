"""Tests for plot geometry construction and cleanup."""

import math
import warnings

import pandas as pd
import pytest
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

from geolattice.vector import (
    GeoprocessError,
    add_geometry_attributes,
    affine_transform,
    build_sections,
    centroids,
    check_validity,
    collect_geometries,
    delete_holes,
    diamonds,
    fix_geometries,
    offset_lines,
    oriented_minimum_bounding_box,
    orthogonalize,
    ovals,
    point_on_surface,
    points_along_geometry,
    points_displacement,
    rectangles,
    remove_duplicate_vertices,
    rotate,
    snap_geometries_to_layer,
    snap_points_to_grid,
    transect,
    translate,
)


def _frame(rows, geometries, crs="EPSG:32618"):
    import geopandas as gpd

    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)


def _corner_angles(polygon):
    coords = list(polygon.exterior.coords)[:-1]
    angles = []
    count = len(coords)
    for index in range(count):
        previous = coords[(index - 1) % count]
        current = coords[index]
        nxt = coords[(index + 1) % count]
        first = math.atan2(previous[1] - current[1], previous[0] - current[0])
        second = math.atan2(nxt[1] - current[1], nxt[0] - current[0])
        difference = abs(math.degrees(second - first)) % 360
        if difference > 180:
            difference = 360 - difference
        angles.append(difference)
    return angles


class TestPlotShapes:
    """Points become fixed-size plot polygons."""

    def test_rectangle_is_centered_on_the_grave(self):
        result = rectangles([Point(10, 20)], 1.2, 2.5)
        geom = result.geometry.iloc[0]
        assert geom.area == pytest.approx(1.2 * 2.5)
        assert geom.centroid.x == pytest.approx(10)
        assert geom.centroid.y == pytest.approx(20)
        minx, miny, maxx, maxy = geom.bounds
        assert maxx - minx == pytest.approx(1.2)
        assert maxy - miny == pytest.approx(2.5)

    def test_rotation_field_turns_the_width(self):
        graves = _frame({"rotation": [90]}, [Point(10, 20)])
        result = rectangles(graves, 2, 4, angle="rotation")
        minx, miny, maxx, maxy = result.geometry.iloc[0].bounds
        assert maxx - minx == pytest.approx(4)
        assert maxy - miny == pytest.approx(2)
        assert result.geometry.iloc[0].centroid.x == pytest.approx(10)

    def test_multipoint_becomes_one_polygon_per_part(self):
        result = rectangles([MultiPoint([(0, 0), (10, 0)])], 2, 2)
        assert len(result) == 2
        assert list(result.geometry.area) == pytest.approx([4, 4])

    def test_diamond_and_oval_sizes(self):
        diamond = diamonds([Point(0, 0)], 2, 4).geometry.iloc[0]
        oval = ovals([Point(0, 0)], 2, 4, quad_segs=32).geometry.iloc[0]
        assert diamond.area == pytest.approx(4)
        assert oval.area == pytest.approx(math.pi * 1 * 2, rel=0.02)

    def test_does_not_change_the_input_and_rejects_polygons(self):
        graves = _frame({"name": ["a"]}, [Point(0, 0)])
        original = graves.geometry.iloc[0].wkt
        rectangles(graves, 1, 2)
        assert graves.geometry.iloc[0].wkt == original
        with pytest.raises(GeoprocessError, match="expects points"):
            rectangles([box(0, 0, 1, 1)], 1, 2)

    def test_geographic_size_warns(self):
        graves = _frame({"name": ["a"]}, [Point(0, 0)], crs="EPSG:4326")
        with pytest.warns(UserWarning, match="geographic"):
            rectangles(graves, 1, 2)


class TestOrientedBox:
    """The long-side angle lines later plots up with the layer."""

    def test_long_side_of_an_axis_aligned_box(self):
        result = oriented_minimum_bounding_box([box(0, 0, 4, 1)])
        assert result["angle"].iloc[0] == pytest.approx(0)
        assert result["width"].iloc[0] == pytest.approx(4)
        assert result["height"].iloc[0] == pytest.approx(1)
        assert result["n_features"].iloc[0] == 1

    def test_angle_feeds_plot_rotation(self):
        cemetery = shapely_rotate(box(0, 0, 40, 10), 30, origin=(0, 0))
        angle = oriented_minimum_bounding_box([cemetery])["angle"].iloc[0]
        assert angle == pytest.approx(30)
        grave = rectangles([Point(0, 0)], 4, 1, angle=angle).geometry.iloc[0]
        grave_angle = oriented_minimum_bounding_box([grave])["angle"].iloc[0]
        assert grave_angle == pytest.approx(30)

    def test_per_feature_keeps_attributes(self):
        plots = _frame({"name": ["a", "b"]}, [box(0, 0, 2, 1), box(5, 0, 6, 3)])
        result = oriented_minimum_bounding_box(plots, per_feature=True)
        assert list(result["name"]) == ["a", "b"]
        assert result["width"].iloc[1] == pytest.approx(3)


class TestAffine:
    """Scale, then rotation, then translation."""

    def test_operation_order_around_the_origin(self):
        result = affine_transform(
            [Point(1, 0)],
            scale=2,
            angle=90,
            translation=(5, 1),
            origin=(0, 0),
        )
        point = result.geometry.iloc[0]
        assert point.x == pytest.approx(5)
        assert point.y == pytest.approx(3)

    def test_non_uniform_scale_happens_before_rotation(self):
        result = affine_transform(
            [Point(1, 1)],
            scale=(2, 3),
            angle=90,
            origin=(0, 0),
        )
        point = result.geometry.iloc[0]
        assert point.x == pytest.approx(-3)
        assert point.y == pytest.approx(2)

    def test_rotate_turns_a_grid_about_its_center(self):
        result = rotate([LineString([(0, 0), (2, 0)])], 180)
        coords = list(result.geometry.iloc[0].coords)
        assert coords[0][0] == pytest.approx(2)
        assert coords[1][0] == pytest.approx(0)
        assert coords[0][1] == pytest.approx(0)

    def test_translate_shifts_the_section(self):
        source = _frame({"name": ["north"]}, [box(0, 0, 1, 1)])
        original = source.geometry.iloc[0].wkt
        result = translate(source, xoff=1.5, yoff=-0.4)
        assert source.geometry.iloc[0].wkt == original
        assert result.geometry.iloc[0].bounds == pytest.approx((1.5, -0.4, 2.5, 0.6))
        assert result["name"].iloc[0] == "north"


class TestRowsAndPoints:
    """Parallel rows, grave spacing, and plot-width transects."""

    def test_offset_lines_follow_the_left_of_the_survey(self):
        surveyed = [LineString([(0, 0), (10, 0)])]
        result = offset_lines(surveyed, 2.5, count=2, side="both", keep_original=True)
        assert list(result["offset_index"]) == [-2, -1, 0, 1, 2]
        heights = [
            geom.interpolate(0.5, normalized=True).y for geom in result.geometry
        ]
        assert heights == pytest.approx([-5, -2.5, 0, 2.5, 5])

    def test_points_are_spaced_along_the_row(self):
        result = points_along_geometry([LineString([(0, 0), (10, 0)])], 2.5)
        assert [point.x for point in result.geometry] == pytest.approx([0, 2.5, 5, 7.5, 10])
        assert list(result["distance_along"]) == pytest.approx([0, 2.5, 5, 7.5, 10])

    def test_offsets_skip_the_ends_of_the_row(self):
        result = points_along_geometry(
            [LineString([(0, 0), (10, 0)])],
            2.5,
            start_offset=2.5,
            end_offset=2.5,
        )
        assert [point.x for point in result.geometry] == pytest.approx([2.5, 5, 7.5])

    def test_transects_split_a_path_into_plot_widths(self):
        result = transect([LineString([(0, 0), (10, 0)])], length=4, interval=5)
        assert list(result["distance_along"]) == pytest.approx([0, 5, 10])
        middle = result.geometry.iloc[1]
        assert middle.interpolate(0.5, normalized=True).x == pytest.approx(5)
        assert middle.bounds[1] == pytest.approx(-2)
        assert middle.bounds[3] == pytest.approx(2)

    def test_surveyed_row_places_rectangular_plots(self):
        rows = offset_lines(
            [LineString([(0, 0), (10, 0)])],
            2.5,
            count=1,
            side="left",
            keep_original=True,
        )
        graves = points_along_geometry(rows, 5)
        plots = rectangles(graves, 1.2, 2.5)
        assert len(plots) == len(graves)
        assert plots.geometry.area.iloc[0] == pytest.approx(1.2 * 2.5)


class TestOrthogonalize:
    """Hand-digitized corners move onto right angles."""

    def test_squares_a_skewed_plot(self):
        skewed = Polygon([(0, 0), (4, 0.15), (4.1, 2.1), (0.05, 2)])
        result = orthogonalize([skewed], angle_tolerance=20)
        angles = _corner_angles(result.geometry.iloc[0])
        assert angles == pytest.approx([90, 90, 90, 90], abs=0.2)

    def test_leaves_a_rectangle_in_place(self):
        rectangle = box(0, 0, 4, 2)
        result = orthogonalize([rectangle])
        assert result.geometry.iloc[0].bounds == pytest.approx(rectangle.bounds)
        assert result.geometry.iloc[0].area == pytest.approx(8)


class TestSnap:
    """Grid rounding and snapping edges so plots meet."""

    def test_rounds_half_away_from_zero(self):
        result = snap_points_to_grid([Point(1.2, 2.7), Point(0.5, -1.5)], 1)
        assert result.geometry.iloc[0].x == pytest.approx(1)
        assert result.geometry.iloc[0].y == pytest.approx(3)
        assert result.geometry.iloc[1].x == pytest.approx(1)
        assert result.geometry.iloc[1].y == pytest.approx(-2)

    def test_origin_shifts_the_grid(self):
        result = snap_points_to_grid([Point(1.2, 2.7)], 1, origin_x=0.5, origin_y=0.5)
        assert result.geometry.iloc[0].x == pytest.approx(1.5)
        assert result.geometry.iloc[0].y == pytest.approx(2.5)

    def test_self_snap_closes_a_gap_between_plots(self):
        plots = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1.05, 0), (2, 0), (2, 1), (1.05, 1)]),
        ]
        result = snap_geometries_to_layer(plots, plots, 0.1)
        assert result.geometry.iloc[1].bounds[0] == pytest.approx(1)
        assert result.geometry.iloc[0].bounds[2] == pytest.approx(1)

    def test_vertex_inside_tolerance_beats_a_closer_edge(self):
        result = snap_geometries_to_layer(
            [Point(0.2, 0.2)],
            [LineString([(0, 0), (10, 0)])],
            0.5,
        )
        assert result.geometry.iloc[0].x == pytest.approx(0)
        assert result.geometry.iloc[0].y == pytest.approx(0)

    def test_segment_snap_pulls_a_point_onto_a_line(self):
        result = snap_geometries_to_layer(
            [Point(5, 0.2)],
            [LineString([(0, 0), (10, 0)])],
            0.5,
            mode="segments",
        )
        assert result.geometry.iloc[0].x == pytest.approx(5)
        assert result.geometry.iloc[0].y == pytest.approx(0)


class TestSectionsAndAttributes:
    """Section grids, measurements, and multipart family plots."""

    def test_grid_is_labeled_from_the_southwest(self):
        result = build_sections(origin=(0, 0), width=10, height=5, rows=2, columns=3)
        assert len(result) == 6
        assert list(result["section"]) == ["R1C1", "R1C2", "R1C3", "R2C1", "R2C2", "R2C3"]
        assert result.geometry.iloc[0].bounds == pytest.approx((0, 0, 10, 5))
        assert result.geometry.iloc[5].bounds == pytest.approx((20, 5, 30, 10))
        assert result.geometry.area.iloc[0] == pytest.approx(50)

    def test_grid_covers_a_layer_extent(self):
        result = build_sections([box(0, 0, 30, 10)], width=10, height=10)
        assert len(result) == 3
        assert list(result["column"]) == [1, 2, 3]

    def test_attribute_mode_unions_a_section(self):
        plots = _frame(
            {"section": ["A", "A", "B"], "plots": [1, 1, 1]},
            [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2)],
        )
        result = build_sections(plots, by="section", aggfunc={"plots": "sum"})
        areas = dict(zip(result["section"], result.geometry.area))
        counts = dict(zip(result["section"], result["plots"]))
        assert areas["A"] == pytest.approx(2)
        assert areas["B"] == pytest.approx(1)
        assert counts == {"A": 2, "B": 1}
        assert dict(zip(result["section"], result["n_features"])) == {"A": 2, "B": 1}

    def test_add_area_perimeter_and_coordinates(self):
        features = _frame(
            {"name": ["plot", "grave"]},
            [box(0, 0, 2, 2), Point(3, 4)],
        )
        result = add_geometry_attributes(features)
        plot = result[result["name"] == "plot"].iloc[0]
        grave = result[result["name"] == "grave"].iloc[0]
        assert plot["area"] == pytest.approx(4)
        assert plot["perimeter"] == pytest.approx(8)
        assert grave["x"] == pytest.approx(3)
        assert grave["y"] == pytest.approx(4)

    def test_line_length(self):
        result = add_geometry_attributes([LineString([(0, 0), (3, 4)])])
        assert result["length"].iloc[0] == pytest.approx(5)
        assert "area" not in result.columns

    def test_collect_keeps_overlapping_parts(self):
        plots = _frame(
            {"family_id": ["smith", "smith"]},
            [box(0, 0, 1, 1), box(0, 0, 1, 1)],
        )
        result = collect_geometries(plots, by="family_id")
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "MultiPolygon"
        assert result.geometry.iloc[0].area == pytest.approx(2)
        assert result["n_features"].iloc[0] == 2


class TestMarkersAndDisplacement:
    """Label points, and burials stacked on one coordinate."""

    def test_centroid_and_point_on_surface(self):
        plot = Polygon([(0, 0), (6, 0), (6, 1), (1, 1), (1, 4), (6, 4), (6, 5), (0, 5)])
        center = centroids([plot]).geometry.iloc[0]
        marker = point_on_surface([plot]).geometry.iloc[0]
        assert marker.within(plot)
        assert not center.within(plot)

    def test_displacement_spreads_a_stack(self):
        burials = _frame({"name": ["a", "b", "c"]}, [Point(5, 5), Point(5, 5), Point(9, 9)])
        result = points_displacement(burials, distance=1, start_angle=0)
        moved = result[result["name"].isin(["a", "b"])]
        assert set(moved["displacement_group"]) == {1}
        offsets = sorted((point.x - 5, point.y - 5) for point in moved.geometry)
        assert offsets[0] == pytest.approx((-1, 0))
        assert offsets[1] == pytest.approx((1, 0))
        alone = result[result["name"] == "c"].iloc[0]
        assert alone.geometry.x == pytest.approx(9)
        assert alone.geometry.y == pytest.approx(9)
        assert pd.isna(alone["displacement_group"])


class TestValidityAndCleanup:
    """Repair geometry before area is published."""

    def test_check_and_fix_a_bowtie(self):
        valid = box(0, 0, 1, 1)
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
        report = check_validity([valid, bowtie])
        assert list(report["is_valid"]) == [True, False]
        assert report["validity_reason"].iloc[0] == "Valid Geometry"
        assert "Self-intersection" in report["validity_reason"].iloc[1]
        invalid = check_validity([valid, bowtie], invalid_only=True)
        assert len(invalid) == 1
        repaired = fix_geometries([bowtie])
        assert repaired.geometry.iloc[0].is_valid
        assert repaired.geometry.iloc[0].area == pytest.approx(0.5)

    def test_remove_duplicate_vertices(self):
        line = LineString([(0, 0), (1, 0), (1, 0), (2, 0)])
        result = remove_duplicate_vertices([line])
        assert list(result.geometry.iloc[0].coords) == [(0, 0), (1, 0), (2, 0)]

    def test_delete_holes_by_area(self):
        plot = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [
                [(1, 1), (2, 1), (2, 2), (1, 2)],
                [(3, 3), (8, 3), (8, 8), (3, 8)],
            ],
        )
        assert len(delete_holes([plot]).geometry.iloc[0].interiors) == 0
        kept = delete_holes([plot], minimum_area=5).geometry.iloc[0]
        assert len(kept.interiors) == 1
        assert Polygon(kept.interiors[0]).area == pytest.approx(25)


class TestRejections:
    """Bad sizes and modes name the problem."""

    def test_rejects_bad_values(self):
        with pytest.raises(GeoprocessError, match="greater than zero"):
            rectangles([Point(0, 0)], 0, 1)
        with pytest.raises(GeoprocessError, match="tolerance"):
            snap_geometries_to_layer([Point(0, 0)], [Point(1, 1)], -1)
        with pytest.raises(GeoprocessError, match="mode"):
            snap_geometries_to_layer([Point(0, 0)], [Point(1, 1)], 1, mode="nearest")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            rectangles([Point(0, 0)], 1, 2)
