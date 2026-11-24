"""Tests for vector analysis tools."""

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from geolattice.vector import (
    GeoprocessError,
    aggregate_attributes,
    basic_statistics_for_fields,
    calculate_fields,
    climb_along_line,
    count_points_in_polygon,
    dbscan_clustering,
    distance_matrix,
    distance_to_nearest_hub_lines,
    distance_to_nearest_hub_points,
    drop_geometry,
    filter_features,
    hub_lines,
    join_attributes,
    join_by_lines,
    kmeans_clustering,
    list_unique_values,
    mean_coordinates,
    nearest_neighbor_analysis,
    nearest_neighbour_analysis,
    overlap_analysis,
    rename_fields,
    reshape_attributes,
    select_fields,
    shortest_line_between_features,
    sort_features,
    statistics_by_categories,
    stdbscan_clustering,
    string_field,
    sum_line_lengths,
    transform_field,
    update_where,
)


def _frame(rows, geometries, crs="EPSG:32618"):
    import geopandas as gpd

    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)


def _points():
    return _frame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "cat": ["A", "A", "B", "B", "A", "C"],
            "weight": [1.0, 2.0, 1.0, 3.0, 1.0, 1.0],
        },
        [
            Point(0, 0),
            Point(10, 0),
            Point(0, 10),
            Point(10, 10),
            Point(5, 5),
            Point(100, 100),
        ],
    )


def _cluster_points():
    coords = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0.5, 0.5),
        (50, 50),
        (51, 50),
        (50, 51),
        (51, 51),
        (50.5, 50.5),
        (200, 200),
    ]
    return _frame(
        {
            "id": list(range(len(coords))),
            "ts": [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(len(coords))],
        },
        [Point(x, y) for x, y in coords],
    )


def _polygons():
    return _frame(
        {"name": ["west", "east"]},
        [box(0, 0, 20, 20), box(40, 40, 80, 80)],
    )


def _lines():
    return _frame(
        {"name": ["a", "b"]},
        [LineString([(0, 0), (20, 0)]), LineString([(50, 50), (70, 70)])],
    )


def _hubs():
    return _frame(
        {"hub_id": ["H1", "H2"], "name": ["school", "clinic"]},
        [Point(0, 0), Point(50, 50)],
    )


class TestBasicStatisticsForFields:
    def test_numeric_field_statistics(self):
        result = basic_statistics_for_fields(_points(), "val")
        assert len(result) == 1
        assert result["COUNT"].iloc[0] == 6
        assert result["SUM"].iloc[0] == pytest.approx(210)
        assert result["MEAN"].iloc[0] == pytest.approx(35)
        assert result["MIN"].iloc[0] == pytest.approx(10)
        assert result["MAX"].iloc[0] == pytest.approx(60)
        assert result.attrs["field"] == "val"

    def test_string_field_statistics(self):
        result = basic_statistics_for_fields(_points(), "cat")
        assert result["COUNT"].iloc[0] == 6
        assert result["UNIQUE"].iloc[0] == 3
        assert result["MIN"].iloc[0] == "A"
        assert "MEAN_LENGTH" in result.columns

    def test_missing_field_raises(self):
        with pytest.raises(GeoprocessError, match="missing"):
            basic_statistics_for_fields(_points(), "missing")


class TestClimbAlongLine:
    def test_climb_and_descent_from_z_vertices(self):
        lines = _frame(
            {"name": ["trail"]},
            [LineString([(0, 0, 0), (10, 0, 5), (20, 0, 2), (30, 0, 8)])],
        )
        result = climb_along_line(lines)
        assert result["climb"].iloc[0] == pytest.approx(11)  # 5 + 6
        assert result["descent"].iloc[0] == pytest.approx(3)
        assert result["minelev"].iloc[0] == pytest.approx(0)
        assert result["maxelev"].iloc[0] == pytest.approx(8)
        assert result.attrs["total_climb"] == pytest.approx(11)

    def test_requires_z_values(self):
        with pytest.raises(GeoprocessError, match="Z values"):
            climb_along_line(_lines())


class TestCountPointsInPolygon:
    def test_counts_points_inside_each_polygon(self):
        result = count_points_in_polygon(_polygons(), _points())
        by_name = dict(zip(result["name"], result["NUMPOINTS"]))
        assert by_name["west"] == 5
        assert by_name["east"] == 0

    def test_weight_field_sums_values(self):
        result = count_points_in_polygon(
            _polygons(),
            _points(),
            weight_field="weight",
        )
        by_name = dict(zip(result["name"], result["NUMPOINTS"]))
        assert by_name["west"] == pytest.approx(8.0)

    def test_class_field_counts_unique_classes(self):
        result = count_points_in_polygon(
            _polygons(),
            _points(),
            class_field="cat",
        )
        by_name = dict(zip(result["name"], result["NUMPOINTS"]))
        assert by_name["west"] == 2  # A and B


class TestDbscanClustering:
    def test_finds_two_clusters_and_noise(self):
        result = dbscan_clustering(_cluster_points(), min_size=3, eps=2)
        labels = [label for label in result["CLUSTER_ID"] if pd.notna(label)]
        assert len(set(labels)) == 2
        assert (result["CLUSTER_ID"].isna()).sum() == 1
        assert set(result["CLUSTER_SIZE"].unique()) <= {0, 5}


class TestDistanceMatrix:
    def test_linear_matrix(self):
        origins = _frame({"oid": ["a", "b"]}, [Point(0, 0), Point(3, 4)])
        targets = _frame({"tid": ["x", "y"]}, [Point(0, 0), Point(3, 4)])
        result = distance_matrix(
            origins,
            targets,
            input_field="oid",
            target_field="tid",
            matrix_type="linear",
        )
        assert set(result.columns) >= {"InputID", "TargetID", "Distance", "geometry"}
        assert len(result) == 4
        zero = result.loc[(result["InputID"] == "a") & (result["TargetID"] == "x")]
        assert zero["Distance"].iloc[0] == pytest.approx(0)

    def test_summary_matrix_nearest(self):
        origins = _frame({"oid": [1]}, [Point(0, 0)])
        targets = _frame(
            {"tid": [1, 2, 3]},
            [Point(3, 4), Point(0, 10), Point(100, 0)],
        )
        result = distance_matrix(
            origins,
            targets,
            matrix_type="summary",
            nearest_points=2,
        )
        assert result["MIN"].iloc[0] == pytest.approx(5)
        assert result["MAX"].iloc[0] == pytest.approx(10)

    def test_standard_matrix(self):
        origins = _frame({"oid": ["a"]}, [Point(0, 0)])
        targets = _frame({"tid": ["x", "y"]}, [Point(3, 4), Point(0, 0)])
        result = distance_matrix(
            origins,
            targets,
            input_field="oid",
            target_field="tid",
            matrix_type="standard",
        )
        assert "Target_x" in result.columns
        assert result["Target_x"].iloc[0] == pytest.approx(5)


class TestDistanceToNearestHub:
    def test_points_report_nearest_hub_name_and_distance(self):
        sources = _frame({"sid": [1, 2]}, [Point(1, 0), Point(48, 50)])
        result = distance_to_nearest_hub_points(
            sources,
            _hubs(),
            hub_field="name",
            unit="layer",
        )
        assert list(result["HubName"]) == ["school", "clinic"]
        assert result["HubDist"].iloc[0] == pytest.approx(1)
        assert result.geometry.iloc[0].geom_type == "Point"

    def test_lines_connect_to_hubs(self):
        sources = _frame({"sid": [1]}, [Point(3, 4)])
        result = distance_to_nearest_hub_lines(
            sources,
            _hubs(),
            hub_field="name",
            unit="layer",
        )
        assert result.geometry.iloc[0].geom_type == "LineString"
        assert result["HubName"].iloc[0] == "school"
        assert result["HubDist"].iloc[0] == pytest.approx(5)


class TestJoinByLines:
    def test_matches_hubs_to_spokes_by_id(self):
        spokes = _frame(
            {"hub_id": ["H1", "H2", "H1"], "sid": [1, 2, 3]},
            [Point(2, 0), Point(52, 50), Point(0, 2)],
        )
        result = join_by_lines(
            _hubs(),
            spokes,
            hub_field="hub_id",
            spoke_field="hub_id",
        )
        assert len(result) == 3
        assert all(geom.geom_type == "LineString" for geom in result.geometry)
        assert hub_lines is join_by_lines

    def test_unmatched_spokes_are_skipped(self):
        spokes = _frame(
            {"hub_id": ["H9"], "sid": [1]},
            [Point(2, 0)],
        )
        result = join_by_lines(
            _hubs(),
            spokes,
            hub_field="hub_id",
            spoke_field="hub_id",
        )
        assert len(result) == 0


class TestKmeansClustering:
    def test_partitions_into_requested_clusters(self):
        result = kmeans_clustering(_cluster_points().iloc[:10], n_clusters=2)
        assert set(result["CLUSTER_ID"]) == {1, 2}
        assert result.attrs["num_clusters"] == 2
        for cluster_id, group in result.groupby("CLUSTER_ID"):
            assert int(group["CLUSTER_SIZE"].iloc[0]) == len(group)
            assert (group["CLUSTER_SIZE"] == len(group)).all()

    def test_rejects_too_many_clusters(self):
        with pytest.raises(GeoprocessError, match="cannot exceed"):
            kmeans_clustering(_points().iloc[:3], n_clusters=5)


class TestListUniqueValues:
    def test_counts_unique_combinations(self):
        result = list_unique_values(_points(), "cat")
        counts = dict(zip(result["cat"], result["COUNT"]))
        assert counts == {"A": 3, "B": 2, "C": 1}
        assert result.attrs["total_values"] == 3

    def test_multi_field_groups(self):
        data = _frame(
            {"a": ["x", "x", "y"], "b": [1, 1, 1]},
            [Point(0, 0), Point(1, 1), Point(2, 2)],
        )
        result = list_unique_values(data, ["a", "b"])
        assert len(result) == 2
        assert "COUNT" in result.columns


class TestMeanCoordinates:
    def test_unweighted_mean(self):
        data = _frame({}, [Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)])
        result = mean_coordinates(data)
        assert len(result) == 1
        assert result.geometry.iloc[0].x == pytest.approx(5)
        assert result.geometry.iloc[0].y == pytest.approx(5)

    def test_weighted_and_grouped_mean(self):
        data = _frame(
            {"uid": ["a", "a", "b"], "w": [1.0, 3.0, 1.0]},
            [Point(0, 0), Point(10, 0), Point(100, 100)],
        )
        result = mean_coordinates(data, weight_field="w", uid_field="uid")
        by_uid = {row.uid: row.geometry for row in result.itertuples()}
        assert by_uid["a"].x == pytest.approx(7.5)
        assert by_uid["b"].x == pytest.approx(100)


class TestNearestNeighbourAnalysis:
    def test_regular_grid_index_near_one(self):
        coords = [(0, 0), (10, 0), (0, 10), (10, 10)]
        data = _frame({}, [Point(x, y) for x, y in coords])
        result = nearest_neighbour_analysis(data)
        assert result["POINT_COUNT"].iloc[0] == 4
        assert result["OBSERVED_MD"].iloc[0] == pytest.approx(10)
        # Bounding box of the points is 10x10, so expected MD is 2.5 and
        # the Clark–Evans index is 4 for this regular lattice.
        assert result["EXPECTED_MD"].iloc[0] == pytest.approx(2.5)
        assert result["NN_INDEX"].iloc[0] == pytest.approx(4.0)
        assert nearest_neighbor_analysis is nearest_neighbour_analysis

    def test_requires_at_least_two_points(self):
        with pytest.raises(GeoprocessError, match="at least two"):
            nearest_neighbour_analysis(_frame({}, [Point(0, 0)]))


class TestOverlapAnalysis:
    def test_overlap_area_and_percentage(self):
        data = _frame({"name": ["p"]}, [box(0, 0, 10, 10)])
        overlay = _frame({}, [box(5, 0, 15, 10)])
        result = overlap_analysis(data, overlay)
        area_col = [c for c in result.columns if c.endswith("_area")][0]
        pct_col = [c for c in result.columns if c.endswith("_pc")][0]
        assert result[area_col].iloc[0] == pytest.approx(50)
        assert result[pct_col].iloc[0] == pytest.approx(50)

    def test_multiple_overlays(self):
        data = _frame({"name": ["p"]}, [box(0, 0, 10, 10)])
        overlays = [
            _frame({}, [box(0, 0, 5, 10)]),
            _frame({}, [box(0, 0, 10, 2)]),
        ]
        result = overlap_analysis(data, overlays)
        area_cols = [c for c in result.columns if c.endswith("_area")]
        assert len(area_cols) == 2
        assert result[area_cols[0]].iloc[0] == pytest.approx(50)
        assert result[area_cols[1]].iloc[0] == pytest.approx(20)


class TestShortestLineBetweenFeatures:
    def test_nearest_point_method(self):
        source = _frame({"sid": [1]}, [Point(0, 0)])
        destination = _frame(
            {"did": [1, 2]},
            [Point(3, 4), Point(10, 0)],
        )
        result = shortest_line_between_features(source, destination, neighbors=1)
        assert len(result) == 1
        assert result["distance"].iloc[0] == pytest.approx(5)
        assert result.geometry.iloc[0].geom_type == "LineString"

    def test_centroid_method_and_max_distance(self):
        source = _frame({"sid": [1]}, [box(0, 0, 2, 2)])
        destination = _frame(
            {"did": [1, 2]},
            [box(10, 0, 12, 2), box(3, 0, 5, 2)],
        )
        result = shortest_line_between_features(
            source,
            destination,
            method="centroid",
            neighbors=2,
            max_distance=5,
        )
        assert len(result) == 1
        assert result["distance"].iloc[0] == pytest.approx(3)


class TestStdbscanClustering:
    def test_clusters_nearby_points_in_time(self):
        result = stdbscan_clustering(
            _cluster_points(),
            "ts",
            min_size=3,
            eps=2,
            eps2=12,
            time_unit="hours",
        )
        labels = [label for label in result["CLUSTER_ID"] if pd.notna(label)]
        assert len(set(labels)) >= 1
        assert "CLUSTER_SIZE" in result.columns

    def test_invalid_time_unit_raises(self):
        with pytest.raises(GeoprocessError, match="time_unit"):
            stdbscan_clustering(_cluster_points(), "ts", time_unit="years")


class TestStatisticsByCategories:
    def test_count_only_when_no_values_field(self):
        result = statistics_by_categories(_points(), "cat")
        counts = dict(zip(result["cat"], result["COUNT"]))
        assert counts == {"A": 3, "B": 2, "C": 1}

    def test_field_statistics_within_categories(self):
        result = statistics_by_categories(
            _points(),
            "cat",
            values_field="val",
        )
        by_cat = result.set_index("cat")
        assert by_cat.loc["A", "COUNT"] == 3
        assert by_cat.loc["A", "MEAN"] == pytest.approx(80 / 3)
        assert by_cat.loc["B", "SUM"] == pytest.approx(70)


class TestSumLineLengths:
    def test_sums_lengths_inside_polygons(self):
        result = sum_line_lengths(_polygons(), _lines())
        by_name = dict(zip(result["name"], zip(result["LENGTH"], result["COUNT"])))
        assert by_name["west"][0] == pytest.approx(20)
        assert by_name["west"][1] == 1
        assert by_name["east"][0] == pytest.approx(math.sqrt(20**2 + 20**2))
        assert by_name["east"][1] == 1


class TestInputsAndAliases:
    def test_does_not_mutate_input(self):
        data = _points()
        original = data.geometry.iloc[0].wkt
        mean_coordinates(data)
        count_points_in_polygon(_polygons(), data)
        assert data.geometry.iloc[0].wkt == original

    def test_geographic_crs_warns_for_planar_distance(self):
        data = _frame({}, [Point(0, 0), Point(1, 0), Point(0, 1)], crs="EPSG:4326")
        with pytest.warns(UserWarning, match="degrees"):
            nearest_neighbour_analysis(data)


class TestAttributeManipulation:
    def test_filter_features_by_query_and_columns(self):
        result = filter_features(_points(), "cat == 'A'", columns=["id", "cat"])
        assert list(result["cat"]) == ["A", "A", "A"]
        assert list(result.columns) == ["id", "cat", "geometry"]

    def test_filter_features_by_mask_callable(self):
        result = filter_features(_points(), mask=lambda df: df["val"] >= 50)
        assert list(result["id"]) == [5, 6]

    def test_select_and_rename_fields(self):
        selected = select_fields(_points(), ["id", "val"])
        assert list(selected.columns) == ["id", "val", "geometry"]
        renamed = rename_fields(_points(), {"cat": "class", "val": "value"})
        assert "class" in renamed.columns and "value" in renamed.columns
        assert "cat" not in renamed.columns

    def test_calculate_fields_expression_and_callable(self):
        result = calculate_fields(
            _points(),
            {
                "double": "val * 2",
                "high": lambda df: df["val"] > 40,
            },
        )
        assert result["double"].iloc[0] == pytest.approx(20)
        assert list(result["high"]) == [False, False, False, False, True, True]

    def test_aggregate_attributes_with_and_without_geometry(self):
        dissolved = aggregate_attributes(_points(), "cat", {"val": "sum"})
        assert set(dissolved["cat"]) == {"A", "B", "C"}
        assert dissolved.geometry.notna().all()
        table = aggregate_attributes(
            _points(),
            "cat",
            {"val": "sum"},
            keep_geometry=False,
        )
        assert isinstance(table, pd.DataFrame)
        assert "geometry" not in table.columns
        assert table.set_index("cat").loc["A", "val"] == pytest.approx(80)

    def test_sort_features(self):
        result = sort_features(_points(), ["cat", "val"], ascending=[True, False])
        assert list(result["cat"]) == ["A", "A", "A", "B", "B", "C"]
        assert list(result.loc[result["cat"] == "A", "val"]) == [50.0, 20.0, 10.0]

    def test_join_attributes_left_and_inner(self):
        left = _points()
        table = pd.DataFrame({"cat": ["A", "B"], "label": ["alpha", "beta"]})
        joined = join_attributes(left, table, on="cat", how="left")
        assert list(joined["label"][:2]) == ["alpha", "alpha"]
        assert pd.isna(joined.loc[joined["cat"] == "C", "label"].iloc[0])
        inner = join_attributes(left, table, on="cat", how="inner")
        assert set(inner["cat"]) == {"A", "B"}

    def test_transform_and_string_field(self):
        scaled = transform_field(_points(), "val", lambda v: v / 10, output_field="tenths")
        assert scaled["tenths"].iloc[0] == pytest.approx(1)
        upper = string_field(_points(), "cat", "upper", output_field="CAT")
        assert list(upper["CAT"]) == ["A", "A", "B", "B", "A", "C"]
        codes = string_field(
            _frame({"name": ["Moffat", "Mesa"]}, [Point(0, 0), Point(1, 1)]),
            "name",
            "slice",
            start=0,
            stop=3,
            output_field="code",
        )
        assert list(codes["code"]) == ["Mof", "Mes"]

    def test_update_where(self):
        result = update_where(_points(), "cat == 'A'", {"weight": 9.0, "tag": "keep"})
        assert list(result.loc[result["cat"] == "A", "weight"]) == [9.0, 9.0, 9.0]
        assert result.loc[result["cat"] == "B", "weight"].iloc[0] == pytest.approx(1.0)
        assert list(result.loc[result["cat"] == "A", "tag"]) == ["keep", "keep", "keep"]
        assert result.loc[result["cat"] == "B", "tag"].isna().all()

    def test_reshape_attributes_long_and_wide(self):
        rice = _frame(
            {
                "DIST": ["a", "b"],
                "Aus": [1.0, 2.0],
                "Aman": [3.0, 4.0],
                "Boro": [5.0, 6.0],
            },
            [Point(0, 0), Point(1, 1)],
        )
        long = reshape_attributes(
            rice,
            to="long",
            id_vars="DIST",
            value_vars=["Aus", "Aman", "Boro"],
        )
        assert len(long) == 6
        assert set(long["variable"]) == {"Aus", "Aman", "Boro"}
        wide = reshape_attributes(
            long,
            to="wide",
            index="DIST",
            columns="variable",
            values="value",
        )
        assert set(wide["DIST"]) == {"a", "b"}
        assert wide.loc[wide["DIST"] == "a", "Boro"].iloc[0] == pytest.approx(5)

    def test_drop_geometry(self):
        table = drop_geometry(_points())
        assert isinstance(table, pd.DataFrame)
        assert "geometry" not in table.columns
        assert list(table.columns) == ["id", "val", "cat", "weight"]
