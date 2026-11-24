"""Tests for geolattice.spatial_analysis.network_analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

networkx = pytest.importorskip("networkx")
nx = networkx

from geolattice.spatial_analysis.network_analysis import (
    NetworkAnalysisError,
    arrival_times,
    circuity_sample,
    describe_spatial_graph,
    diffusion_centrality,
    gaussian_distance_decay,
    graph_component_summary,
    graph_from_edgelist,
    graph_positions,
    haversine_km,
    is_planar_graph,
    kansky_indices,
    lines_to_graph,
    link_probability_by_distance,
    lorenz_curve_gini,
    metapopulation_sir,
    morans_i,
    node_flow_balance,
    od_to_digraph,
    participation_coefficient,
    population_coverage,
    pseudo_r2,
    row_standardize,
    select_seeds,
    set_edge_lengths_euclidean,
    set_edge_lengths_haversine,
    simulate_contagion,
    spatial_lag,
    symmetric_flow_graph,
    two_sfca,
)


DATA = Path(__file__).resolve().parents[1] / "Data" / "00_shared" / "data_spatial_net"


def _toy_grid(n: int = 4) -> nx.Graph:
    G = nx.grid_2d_graph(n, n)
    # relabel to simple ints with lon/lat-ish coords
    mapping = {node: i for i, node in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping)
    for n_id, (i, j) in enumerate([(a, b) for a in range(n) for b in range(n)]):
        G.nodes[n_id]["x"] = -74.0 + 0.01 * i
        G.nodes[n_id]["y"] = 40.0 + 0.01 * j
    return G


def test_haversine_km_scalar_and_vector():
    d = haversine_km(-74.0, 40.7, -73.9, 40.7)
    assert 7.5 < d < 9.5
    arr = haversine_km(
        np.array([-74.0, -74.0]),
        np.array([40.7, 40.7]),
        np.array([-73.9, -74.0]),
        np.array([40.7, 40.8]),
    )
    assert arr.shape == (2,)
    assert arr[0] > 0 and arr[1] > 0


def test_kansky_and_planarity():
    G = nx.grid_2d_graph(5, 5)
    idx = kansky_indices(G)
    assert idx["nodes"] == 25
    assert idx["edges"] == 40
    assert 0 < idx["beta"] < 2
    assert is_planar_graph(G)


def test_edge_lengths_and_positions():
    G = _toy_grid(3)
    set_edge_lengths_haversine(G, attr="length_km")
    set_edge_lengths_euclidean(G, attr="length_m", scale=111_320)
    pos = graph_positions(G)
    assert len(pos) == 9
    assert all("length_km" in d for *_, d in G.edges(data=True))


def test_component_summary_and_describe():
    G = _toy_grid(3)
    set_edge_lengths_euclidean(G)
    summary = graph_component_summary(G)
    assert summary["largest component %"] == 100.0
    desc = describe_spatial_graph(G, length_unit_scale=1.0)
    assert desc["links"] == G.number_of_edges()
    assert desc["components"] == 1


def test_link_probability_and_circuity():
    G = _toy_grid(4)
    set_edge_lengths_euclidean(G, attr="length")
    pos = graph_positions(G)
    bins = np.linspace(0, 0.05, 6)
    s = link_probability_by_distance(G, pos, bins)
    assert len(s) == 5
    circ = circuity_sample(G, n_pairs=50, weight="length", seed=0)
    assert {"network", "euclidean", "circuity"}.issubset(circ.columns)
    assert (circ.circuity >= 0.99).all()


def test_morans_i_and_spatial_lag():
    W = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    Wr = row_standardize(W)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lag = spatial_lag(y, Wr)
    assert lag.shape == (4,)
    I = morans_i(y, Wr)
    assert isinstance(I, float)
    result = morans_i(y, Wr, permutations=99, seed=1)
    assert "I" in result and 0 <= result["pseudo_p"] <= 1


def test_lines_to_graph():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        {
            "highway": ["primary", "secondary"],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(10, 0), (10, 10)]),
            ],
        },
        crs="EPSG:32618",
    )
    G = lines_to_graph(gdf, snap=1.0)
    assert G.number_of_edges() == 2
    assert G.number_of_nodes() == 3


def test_population_coverage_two_sfca_lorenz():
    df = pd.DataFrame(
        {
            "net_min": [5, 12, 18, 35],
            "population": [100, 200, 150, 50],
            "pop65": [10, 20, 30, 5],
        }
    )
    cov = population_coverage(
        df, time_col="net_min", pop_cols=("population", "pop65"), thresholds=(10, 20)
    )
    assert cov.loc[10, "% population"] == pytest.approx(20.0)
    T = np.array([[5, 25], [12, 8], [18, 15], [30, 40]], dtype=float)
    P = df.population.to_numpy(float)
    S = np.array([20.0, 30.0])
    A = two_sfca(T, P, S, d0=20)
    assert A.shape == (4,)
    A2 = two_sfca(T, P, S, d0=20, decay=lambda t: gaussian_distance_decay(t, d0=20))
    assert A2.shape == (4,)
    _, _, gini = lorenz_curve_gini(A, P)
    assert 0 <= gini <= 1


def test_od_flow_graph_and_balance():
    od = pd.DataFrame(
        {
            "origin": ["A", "A", "B"],
            "destination": ["B", "C", "C"],
            "commuters": [10, 5, 7],
            "Pi": [100, 100, 80],
            "Pj": [80, 60, 60],
        }
    )
    G = od_to_digraph(od)
    assert G.number_of_edges() == 3
    bal = node_flow_balance(G)
    assert bal.loc["A", "outflow"] == 15
    U = symmetric_flow_graph(od)
    assert U.number_of_edges() == 3
    assert pseudo_r2([1, 2, 3], [1.1, 1.9, 3.2]) > 0.9


def test_metapopulation_sir_and_arrival():
    N = np.array([1000.0, 800.0, 600.0])
    C = np.array(
        [
            [0, 50, 20],
            [40, 0, 30],
            [10, 25, 0],
        ],
        dtype=float,
    )
    H = metapopulation_sir(C, N, seed_index=0, days=50, coupling=0.05)
    assert H.shape == (50, 3)
    arr = arrival_times(H, threshold=1e-3, index=["A", "B", "C"])
    assert arr["A"] == 0 or arr["A"] >= 0


def test_social_network_helpers():
    ties = pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 0]})
    nodes = pd.DataFrame(
        {
            "village": ["v1", "v1", "v2"],
            "named_by": [2, 1, 0],
            "extension_contact": [True, False, False],
            "participation": [0.5, 0.2, 0.8],
            "diffusion_centrality": [1.0, 0.5, 0.2],
        },
        index=[0, 1, 2],
    )
    G = graph_from_edgelist(ties, nodes, directed=True, node_attrs=("village",))
    assert G.number_of_edges() == 3
    part = participation_coefficient(nx.Graph(G), nodes.village.to_dict())
    assert set(part) == {0, 1, 2}
    A = nx.to_numpy_array(G, nodelist=[0, 1, 2])
    dc = diffusion_centrality(A.T, q=0.25, T=3)
    assert dc.shape == (3,)
    seeds = select_seeds(nodes, "Top in-degree", k=2)
    assert len(seeds) == 2
    hist, t_adopt = simulate_contagion(A, seeds, [0, 1, 2], model="simple", periods=5, rng=np.random.default_rng(0))
    assert hist.shape == (6,)
    assert t_adopt.shape == (3,)


def test_kansky_too_small_raises():
    with pytest.raises(NetworkAnalysisError):
        kansky_indices(nx.path_graph(2))


@pytest.mark.skipif(not (DATA / "roads.graphml").exists(), reason="spatial net data missing")
def test_load_road_graph_optional():
    osmnx = pytest.importorskip("osmnx")
    from geolattice.spatial_analysis.network_analysis import load_road_graph

    G = load_road_graph(DATA / "roads.graphml")
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
