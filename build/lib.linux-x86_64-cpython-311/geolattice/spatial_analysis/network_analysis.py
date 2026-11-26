"""
Spatial network analysis for GeoLattice.

Implements the workflows taught in
``Tutorials/notebooks/03_spatial_analysis/03_06_spatial_network``:

1. Geographic embedding — haversine lengths, circuity, Kansky α/β/γ,
   component summaries, link-probability vs distance
2. Road routing — OSMnx GraphML load, speeds / travel times, OD matrices,
   isochrones, cumulative accessibility, closure impact
3. Spatial weights graphs — libpysal contiguity / k-NN / distance-band /
   Delaunay / Gabriel / RNG → NetworkX, percolation, Moran's I, lines → graph
4. Facility access — nearest facility, coverage, 2SFCA / E2SFCA, Lorenz / Gini
5. Flow networks — OD Digraph, gravity fit, dominant-flow hierarchy,
   Louvain regions, metapopulation SIR
6. Relational / social networks — distance decay, participation, diffusion
   centrality, seeding strategies, simple / complex contagion

Optional extras (imported lazily)::

    pip install networkx geopandas osmnx libpysal scikit-learn

or ``pip install "geolattice[network]"`` when the optional group is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, Sequence[float]]
NodeId = Any
GraphLike = Any  # networkx Graph / DiGraph / MultiDiGraph

_EARTH_RADIUS_KM = 6371.0088


class NetworkAnalysisError(Exception):
    """Raised when a spatial-network operation cannot be completed."""


# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------


def _require_networkx():
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "network_analysis requires networkx. Install with: pip install networkx"
        ) from exc
    return nx


def _require_osmnx():
    try:
        import osmnx as ox
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This function requires osmnx. Install with: pip install osmnx"
        ) from exc
    return ox


def _require_libpysal():
    try:
        from libpysal import weights
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This function requires libpysal. Install with: pip install libpysal"
        ) from exc
    return weights


def _require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This function requires geopandas. Install with: pip install geopandas"
        ) from exc
    return gpd


def _require_sklearn_linear():
    try:
        from sklearn.linear_model import LinearRegression, PoissonRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This function requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc
    return LinearRegression, PoissonRegressor


# ---------------------------------------------------------------------------
# 1 · Geographic embedding & topology
# ---------------------------------------------------------------------------


def haversine_km(
    lon1: ArrayLike,
    lat1: ArrayLike,
    lon2: ArrayLike,
    lat2: ArrayLike,
    *,
    R: float = _EARTH_RADIUS_KM,
) -> Union[float, np.ndarray]:
    """Great-circle distance in kilometres (vectorized)."""
    lon1, lat1, lon2, lat2 = (
        np.radians(lon1),
        np.radians(lat1),
        np.radians(lon2),
        np.radians(lat2),
    )
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def graph_positions(
    G: GraphLike,
    *,
    x: str = "x",
    y: str = "y",
) -> Dict[NodeId, Tuple[float, float]]:
    """Return ``{node: (x, y)}`` from node attributes."""
    missing = [n for n in G.nodes if x not in G.nodes[n] or y not in G.nodes[n]]
    if missing:
        raise NetworkAnalysisError(
            f"{len(missing)} node(s) missing '{x}'/'{y}' attributes "
            f"(e.g. {missing[:3]})"
        )
    return {n: (float(G.nodes[n][x]), float(G.nodes[n][y])) for n in G.nodes}


def set_edge_lengths_haversine(
    G: GraphLike,
    *,
    attr: str = "length_km",
    x: str = "x",
    y: str = "y",
) -> GraphLike:
    """Fill edge lengths from lon/lat node attributes via :func:`haversine_km`."""
    for u, v, key, data in _edge_iter(G):
        a, b = G.nodes[u], G.nodes[v]
        length = float(haversine_km(a[x], a[y], b[x], b[y]))
        _set_edge_attr(G, u, v, key, attr, length)
    return G


def set_edge_lengths_euclidean(
    G: GraphLike,
    *,
    attr: str = "length",
    x: str = "x",
    y: str = "y",
    scale: float = 1.0,
) -> GraphLike:
    """Fill edge lengths from projected node coordinates (``scale`` converts units)."""
    for u, v, key, data in _edge_iter(G):
        a, b = G.nodes[u], G.nodes[v]
        length = float(np.hypot(a[x] - b[x], a[y] - b[y]) * scale)
        _set_edge_attr(G, u, v, key, attr, length)
    return G


def kansky_indices(G: GraphLike) -> Dict[str, float]:
    """Planar connectivity indices α, β, γ plus node / edge counts."""
    nx = _require_networkx()
    Gu = nx.Graph(G)
    v = Gu.number_of_nodes()
    e = Gu.number_of_edges()
    p = nx.number_connected_components(Gu)
    if v < 3:
        raise NetworkAnalysisError("kansky_indices needs at least 3 nodes")
    return {
        "nodes": float(v),
        "edges": float(e),
        "beta": e / v,
        "alpha": (e - v + p) / (2 * v - 5),
        "gamma": e / (3 * (v - 2)),
    }


def graph_component_summary(G: GraphLike) -> Dict[str, float]:
    """Edges, clustering, LCC share, and average path length on the LCC."""
    nx = _require_networkx()
    Gu = nx.Graph(G)
    if Gu.number_of_nodes() == 0:
        raise NetworkAnalysisError("graph is empty")
    components = list(nx.connected_components(Gu))
    lcc = Gu.subgraph(max(components, key=len)).copy()
    return {
        "edges": float(Gu.number_of_edges()),
        "avg clustering": float(nx.average_clustering(Gu)),
        "largest component %": 100.0 * lcc.number_of_nodes() / Gu.number_of_nodes(),
        "avg path length (LCC)": float(nx.average_shortest_path_length(lcc)),
    }


def describe_spatial_graph(
    G: GraphLike,
    pos: Optional[Mapping[NodeId, Sequence[float]]] = None,
    *,
    length_unit_scale: float = 1000.0,
) -> Dict[str, float]:
    """Links, degree, components, clustering, and mean / max link length."""
    nx = _require_networkx()
    Gu = nx.Graph(G)
    n = Gu.number_of_nodes()
    e = Gu.number_of_edges()
    if n == 0:
        raise NetworkAnalysisError("graph is empty")
    if pos is None:
        pos = graph_positions(Gu)
    lengths = [
        float(np.hypot(*(np.asarray(pos[u]) - np.asarray(pos[v])))) / length_unit_scale
        for u, v in Gu.edges
    ]
    degrees = dict(Gu.degree())
    return {
        "links": float(e),
        "mean degree": 2.0 * e / n,
        "max degree": float(max(degrees.values()) if degrees else 0),
        "components": float(nx.number_connected_components(Gu)),
        "clustering": float(nx.average_clustering(Gu)),
        "mean link km": float(np.mean(lengths)) if lengths else 0.0,
        "max link km": float(np.max(lengths)) if lengths else 0.0,
    }


def link_probability_by_distance(
    G: GraphLike,
    pos: Mapping[NodeId, Sequence[float]],
    bins: ArrayLike,
) -> pd.Series:
    """Empirical P(edge) by distance bin for an undirected simple graph."""
    nx = _require_networkx()
    Gu = nx.Graph(G)
    keys = list(Gu.nodes)
    xy = np.asarray([pos[k] for k in keys], dtype=float)
    D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    A = nx.to_numpy_array(Gu, nodelist=keys)
    iu = np.triu_indices(len(keys), 1)
    d, a = D[iu], A[iu]
    bins_arr = np.asarray(bins, dtype=float)
    cat = np.digitize(d, bins_arr)
    values = [float(a[cat == k].mean()) if np.any(cat == k) else np.nan for k in range(1, len(bins_arr))]
    return pd.Series(values, index=bins_arr[1:], name="P(link)")


def circuity_sample(
    G: GraphLike,
    n_pairs: int = 400,
    *,
    weight: str = "length",
    x: str = "x",
    y: str = "y",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Sample OD pairs: Euclidean vs network distance and circuity ratio."""
    nx = _require_networkx()
    Gu = nx.Graph(G)
    nodes = list(Gu.nodes)
    if len(nodes) < 2:
        raise NetworkAnalysisError("need at least 2 nodes for circuity_sample")
    rng = np.random.default_rng(seed)
    pairs = rng.choice(nodes, size=(n_pairs, 2), replace=True)
    rows = []
    for u, v in pairs:
        if u == v:
            continue
        try:
            net = nx.shortest_path_length(Gu, u, v, weight=weight)
        except nx.NetworkXNoPath:
            continue
        eu = float(np.hypot(Gu.nodes[u][x] - Gu.nodes[v][x], Gu.nodes[u][y] - Gu.nodes[v][y]))
        if eu <= 0:
            continue
        rows.append(
            {
                "origin": u,
                "destination": v,
                "network": net,
                "euclidean": eu,
                "circuity": net / eu,
            }
        )
    return pd.DataFrame(rows)


def is_planar_graph(G: GraphLike) -> bool:
    """Return whether the undirected simple graph is planar."""
    nx = _require_networkx()
    return bool(nx.check_planarity(nx.Graph(G))[0])


# ---------------------------------------------------------------------------
# 2 · Road routing & network accessibility (OSMnx)
# ---------------------------------------------------------------------------


def load_road_graph(path: PathLike) -> GraphLike:
    """Load an OSMnx-compatible GraphML road network.

    Prefers ``osmnx.load_graphml`` (correct typed attributes). Falls back to
    NetworkX ``read_graphml`` when osmnx is not installed.
    """
    try:
        ox = _require_osmnx()
    except ImportError:
        nx = _require_networkx()
        return nx.read_graphml(str(path))
    return ox.load_graphml(str(path))


def prepare_travel_times(
    G: GraphLike,
    *,
    hwy_speeds: Optional[Mapping[str, float]] = None,
    minutes_attr: Optional[str] = "minutes",
) -> GraphLike:
    """Add edge speeds and travel times; optionally store minutes."""
    ox = _require_osmnx()
    if hwy_speeds is None:
        ox.add_edge_speeds(G)
    else:
        ox.add_edge_speeds(G, hwy_speeds=dict(hwy_speeds))
    ox.add_edge_travel_times(G)
    if minutes_attr:
        for u, v, key, data in _edge_iter(G):
            tt = data.get("travel_time")
            if tt is not None:
                _set_edge_attr(G, u, v, key, minutes_attr, float(tt) / 60.0)
    return G


def snap_points_to_nodes(
    gdf,
    G: GraphLike,
    *,
    return_dist: bool = False,
):
    """Nearest network nodes for point geometries (osmnx)."""
    ox = _require_osmnx()
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    if return_dist:
        nodes, dists = ox.nearest_nodes(G, xs, ys, return_dist=True)
        return nodes, dists
    return ox.nearest_nodes(G, xs, ys)


def route_summary(
    G: GraphLike,
    route: Sequence[NodeId],
    *,
    highway_classes: Sequence[str] = ("primary", "secondary"),
) -> Dict[str, float]:
    """Summarise a node route: km, minutes, segments, % on given highways."""
    ox = _require_osmnx()
    gdf = ox.routing.route_to_gdf(G, list(route))
    total = float(gdf.length.sum())
    on_class = float(gdf[gdf.highway.isin(list(highway_classes))].length.sum()) if total else 0.0
    return {
        "km": total / 1000.0,
        "minutes": float(gdf.travel_time.sum()) / 60.0,
        "segments": float(len(gdf)),
        "% on primary/secondary": 100.0 * on_class / total if total else 0.0,
    }


def shortest_routes(
    G: GraphLike,
    origin: NodeId,
    destination: NodeId,
    *,
    weights: Sequence[str] = ("length", "travel_time"),
) -> Dict[str, Sequence[NodeId]]:
    """Shortest paths under one or more edge weight attributes."""
    ox = _require_osmnx()
    out = {}
    for w in weights:
        out[w] = ox.routing.shortest_path(G, origin, destination, weight=w)
    return out


def od_travel_time_matrix(
    G: GraphLike,
    places,
    *,
    node_col: str = "node",
    id_col: Optional[str] = None,
    weight: str = "travel_time",
    unit: str = "minutes",
) -> pd.DataFrame:
    """Place × place travel-time (or length) matrix via Dijkstra."""
    nx = _require_networkx()
    if id_col is None:
        labels = list(places.index) if places.index.name or not isinstance(places.index, pd.RangeIndex) else places.index.astype(str).tolist()
        # Prefer an explicit name column when index is default RangeIndex
        for candidate in ("town", "name", "id", "tract_id"):
            if candidate in places.columns:
                labels = places[candidate].tolist()
                break
        else:
            labels = list(places.index)
    else:
        labels = places[id_col].tolist()
    nodes = places[node_col].tolist()
    scale = 60.0 if unit == "minutes" and weight in {"travel_time"} else 1.0
    out: Dict[Any, Dict[Any, float]] = {}
    for lab_i, node_i in zip(labels, nodes):
        dist = nx.single_source_dijkstra_path_length(G, node_i, weight=weight)
        out[lab_i] = {
            lab_j: float(dist.get(node_j, np.inf)) / scale
            for lab_j, node_j in zip(labels, nodes)
        }
    return pd.DataFrame(out).T


def isochrone_polygons(
    G: GraphLike,
    center: NodeId,
    budgets_min: Sequence[float],
    *,
    weight: str = "travel_time",
    buffer_m: float = 1500.0,
    erode_m: float = 700.0,
    crs: Optional[Any] = None,
):
    """Network-buffer isochrones for travel-time budgets (minutes)."""
    nx = _require_networkx()
    gpd = _require_geopandas()
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    scale = 60.0 if weight == "travel_time" else 1.0
    rows = []
    for budget in budgets_min:
        cutoff = float(budget) * scale
        lengths = nx.single_source_dijkstra_path_length(G, center, cutoff=cutoff, weight=weight)
        pts = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in lengths]
        if not pts:
            continue
        geom = MultiPoint(pts).buffer(buffer_m)
        if erode_m > 0:
            geom = geom.buffer(-erode_m).buffer(erode_m)
        rows.append({"minutes": float(budget), "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, crs=crs or G.graph.get("crs"))
    return gdf


def cumulative_accessibility(
    G: GraphLike,
    opportunities,
    *,
    cutoff: float,
    weight: str = "travel_time",
    opportunity_col: str = "population",
    node_col: str = "node",
) -> pd.Series:
    """Opportunity mass reachable within ``cutoff`` from each opportunity node."""
    nx = _require_networkx()
    node_to_mass = {
        getattr(r, node_col): float(getattr(r, opportunity_col))
        for r in opportunities.itertuples()
    }
    out = {}
    for origin in node_to_mass:
        dist = nx.single_source_dijkstra_path_length(G, origin, cutoff=cutoff, weight=weight)
        out[origin] = sum(m for n, m in node_to_mass.items() if n in dist)
    return pd.Series(out, name="accessibility")


def edge_travel_time_betweenness(
    G: GraphLike,
    *,
    k: int = 200,
    weight: str = "travel_time",
    seed: int = 42,
    attr: str = "tt_betweenness",
) -> GraphLike:
    """Attach sample-k travel-time edge betweenness as an edge attribute."""
    nx = _require_networkx()
    bc = nx.edge_betweenness_centrality(G, k=k, weight=weight, seed=seed)
    for edge, value in bc.items():
        if len(edge) == 3:
            u, v, key = edge
            G.edges[u, v, key][attr] = value
        else:
            u, v = edge
            G.edges[u, v][attr] = value
    return G


def closure_impact(
    G: GraphLike,
    edges_to_remove: Sequence[Tuple],
    places,
    *,
    node_col: str = "node",
    id_col: Optional[str] = None,
    weight: str = "travel_time",
    unit: str = "minutes",
    bidirectional: bool = True,
) -> pd.DataFrame:
    """Δ OD travel times after removing edges (road closure scenario)."""
    nx = _require_networkx()
    baseline = od_travel_time_matrix(
        G, places, node_col=node_col, id_col=id_col, weight=weight, unit=unit
    )
    Gc = G.copy()
    remove = []
    for edge in edges_to_remove:
        if len(edge) == 2:
            u, v = edge
            remove.append((u, v, 0) if Gc.is_multigraph() else (u, v))
            if bidirectional:
                remove.append((v, u, 0) if Gc.is_multigraph() else (v, u))
        else:
            remove.append(tuple(edge))
            if bidirectional and len(edge) >= 2:
                u, v = edge[0], edge[1]
                key = edge[2] if len(edge) > 2 else 0
                remove.append((v, u, key) if Gc.is_multigraph() else (v, u))
    Gc.remove_edges_from(remove)
    after = od_travel_time_matrix(
        Gc, places, node_col=node_col, id_col=id_col, weight=weight, unit=unit
    )
    return after - baseline


# ---------------------------------------------------------------------------
# 3 · Spatial weights → graphs (libpysal)
# ---------------------------------------------------------------------------


def weights_to_graph(
    w,
    labels: Optional[Sequence] = None,
    *,
    undirected: bool = True,
) -> GraphLike:
    """Convert a libpysal ``W`` object to NetworkX (symmetrise k-NN)."""
    nx = _require_networkx()
    g = nx.Graph(w.to_networkx()) if undirected else nx.DiGraph(w.to_networkx())
    if labels is not None:
        mapping = dict(enumerate(labels))
        g = nx.relabel_nodes(g, mapping)
    return g


def build_spatial_graphs(
    gdf,
    *,
    kinds: Sequence[str] = (
        "queen",
        "rook",
        "knn",
        "distance_band",
        "delaunay",
        "gabriel",
        "rng",
    ),
    k: int = 4,
    threshold: Optional[float] = None,
    id_col: Optional[str] = None,
) -> Dict[str, GraphLike]:
    """Build common adjacency / distance graphs from polygons or points."""
    weights = _require_libpysal()
    geom = gdf.geometry
    if geom.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        cent = geom.centroid
    else:
        cent = geom
    xy = np.c_[cent.x.to_numpy(), cent.y.to_numpy()]
    labels = gdf[id_col].tolist() if id_col else list(gdf.index)

    builders = {
        "queen": lambda: weights.Queen.from_dataframe(gdf, use_index=False),
        "rook": lambda: weights.Rook.from_dataframe(gdf, use_index=False),
        "knn": lambda: weights.KNN.from_array(xy, k=k),
        "distance_band": lambda: weights.DistanceBand.from_array(
            xy,
            threshold=threshold if threshold is not None else weights.min_threshold_distance(xy),
            binary=True,
        ),
        "delaunay": lambda: weights.Delaunay(xy),
        "gabriel": lambda: weights.Gabriel(xy),
        "rng": lambda: weights.Relative_Neighborhood(xy),
    }
    out = {}
    for kind in kinds:
        key = kind.lower().replace(" ", "_")
        if key not in builders:
            raise NetworkAnalysisError(f"unknown spatial graph kind: {kind!r}")
        out[kind] = weights_to_graph(builders[key](), labels=labels)
    return out


def min_threshold_distance(xy: ArrayLike) -> float:
    """Smallest distance-band threshold with no islands."""
    weights = _require_libpysal()
    return float(weights.min_threshold_distance(np.asarray(xy, dtype=float)))


def percolation_curve(
    xy: ArrayLike,
    thresholds: Sequence[float],
    *,
    labels: Optional[Sequence] = None,
) -> pd.DataFrame:
    """LCC %, components, and mean degree vs distance threshold."""
    nx = _require_networkx()
    weights = _require_libpysal()
    xy_arr = np.asarray(xy, dtype=float)
    rows = []
    for thr in thresholds:
        w = weights.DistanceBand.from_array(xy_arr, threshold=float(thr), binary=True)
        g = weights_to_graph(w, labels=labels)
        n = g.number_of_nodes()
        comps = list(nx.connected_components(g))
        lcc = max((len(c) for c in comps), default=0)
        rows.append(
            {
                "threshold": float(thr),
                "lcc_pct": 100.0 * lcc / n if n else 0.0,
                "components": len(comps),
                "mean_degree": 2.0 * g.number_of_edges() / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def row_standardize(W: np.ndarray) -> np.ndarray:
    """Row-standardise an adjacency matrix."""
    W = np.asarray(W, dtype=float)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return W / row_sums


def spatial_lag(y: ArrayLike, Wr: np.ndarray) -> np.ndarray:
    """Spatial lag ``Wr @ y``."""
    return np.asarray(Wr, dtype=float) @ np.asarray(y, dtype=float)


def morans_i(
    y: ArrayLike,
    Wr: np.ndarray,
    *,
    permutations: int = 0,
    seed: Optional[int] = None,
) -> Union[float, Dict[str, float]]:
    """Moran's I given a row-standardised weights matrix."""
    y_arr = np.asarray(y, dtype=float)
    Wr_arr = np.asarray(Wr, dtype=float)
    z = y_arr - y_arr.mean()
    denom = float(z @ z)
    if denom == 0:
        raise NetworkAnalysisError("morans_i: variable has zero variance")
    I = float(len(y_arr) / Wr_arr.sum() * (z @ Wr_arr @ z) / denom)
    if permutations <= 0:
        return I
    rng = np.random.default_rng(seed)
    perms = np.empty(permutations, dtype=float)
    for i in range(permutations):
        yp = rng.permutation(y_arr)
        zp = yp - yp.mean()
        perms[i] = len(yp) / Wr_arr.sum() * (zp @ Wr_arr @ zp) / (zp @ zp)
    p = (np.sum(perms >= I) + 1) / (permutations + 1)
    return {
        "I": I,
        "E[I]": -1.0 / (len(y_arr) - 1),
        "pseudo_p": float(p),
    }


def lines_to_graph(
    gdf,
    *,
    snap: float = 1.0,
    length_attr: str = "length",
    keep_attrs: Sequence[str] = ("highway",),
) -> GraphLike:
    """Build an undirected graph from a line GeoDataFrame (snap endpoints)."""
    nx = _require_networkx()
    g = nx.Graph()
    g.graph["crs"] = str(gdf.crs) if getattr(gdf, "crs", None) is not None else None
    for idx, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        a = tuple(np.round(np.asarray(coords[0]) / snap) * snap)
        b = tuple(np.round(np.asarray(coords[-1]) / snap) * snap)
        if a == b:
            continue
        length = float(row.geometry.length)
        if not g.has_edge(a, b) or g.edges[a, b][length_attr] > length:
            attrs = {length_attr: length, "feature": idx}
            for name in keep_attrs:
                if name in row.index:
                    attrs[name] = row[name]
            g.add_edge(a, b, **attrs)
    for n in g.nodes:
        g.nodes[n]["x"], g.nodes[n]["y"] = n
    return g


def similarity_weighted_communities(
    G: GraphLike,
    attrs: Mapping[NodeId, ArrayLike],
    *,
    resolution: float = 1.0,
    seed: Optional[int] = 42,
) -> List[set]:
    """Louvain communities on edges weighted by attribute similarity."""
    nx = _require_networkx()
    Gw = nx.Graph()
    Gw.add_nodes_from(G.nodes)
    for u, v in G.edges:
        du = np.asarray(attrs[u], dtype=float)
        dv = np.asarray(attrs[v], dtype=float)
        w = float(np.exp(-np.linalg.norm(du - dv)))
        Gw.add_edge(u, v, weight=w)
    return list(nx.community.louvain_communities(Gw, weight="weight", resolution=resolution, seed=seed))


# ---------------------------------------------------------------------------
# 4 · Facility access & equity
# ---------------------------------------------------------------------------


def nearest_facility_network(
    G: GraphLike,
    demand_nodes: Sequence[NodeId],
    facility_nodes: Sequence[NodeId],
    *,
    weight: str = "minutes",
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-source Dijkstra: distance and nearest facility per demand node."""
    nx = _require_networkx()
    # Invert search: run from all facilities once
    sources = list(facility_nodes)
    dist_map, path_map = nx.multi_source_dijkstra(G, sources, weight=weight)
    facility_set = set(sources)
    # Find which facility by walking predecessors — use reverse search from facilities
    # Rebuild with parent tracking via separate single-source from each facility is clearer
    best_dist = np.full(len(demand_nodes), np.inf)
    best_fac = np.full(len(demand_nodes), None, dtype=object)
    for fac in sources:
        d = nx.single_source_dijkstra_path_length(G, fac, weight=weight)
        for i, dem in enumerate(demand_nodes):
            val = d.get(dem, np.inf)
            if val < best_dist[i]:
                best_dist[i] = val
                best_fac[i] = fac
    return best_dist, best_fac


def nearest_facility_euclidean(
    demand_gdf,
    facility_gdf,
    *,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Straight-line nearest facility and distance (map units / 1000 → km)."""
    rows = []
    for idx, geom in demand_gdf.geometry.items():
        dists = facility_gdf.distance(geom)
        j = dists.idxmin()
        fac_id = facility_gdf.loc[j, id_col] if id_col else j
        rows.append(
            {
                "demand": idx,
                "facility": fac_id,
                "euclid_km": float(dists.loc[j]) / 1000.0,
            }
        )
    return pd.DataFrame(rows).set_index("demand")


def travel_time_matrix(
    G: GraphLike,
    origins: Sequence[NodeId],
    destinations: Sequence[NodeId],
    *,
    weight: str = "minutes",
    access_leg: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Origin × destination network times (+ optional access legs on origins)."""
    nx = _require_networkx()
    T = np.full((len(origins), len(destinations)), np.inf)
    for j, dest in enumerate(destinations):
        d = nx.single_source_dijkstra_path_length(G, dest, weight=weight)
        for i, orig in enumerate(origins):
            T[i, j] = d.get(orig, np.inf)
    if access_leg is not None:
        T = T + np.asarray(access_leg, dtype=float)[:, None]
    return T


def population_coverage(
    df: pd.DataFrame,
    *,
    time_col: str = "net_min",
    pop_cols: Sequence[str] = ("population",),
    thresholds: Sequence[float] = (10, 15, 20, 30),
) -> pd.DataFrame:
    """Share of each population column within travel-time thresholds."""
    rows = []
    for t in thresholds:
        m = df[time_col] <= t
        row = {"≤ minutes": t}
        for col in pop_cols:
            total = float(df[col].sum())
            row[f"% {col}"] = 100.0 * float(df.loc[m, col].sum()) / total if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("≤ minutes")


def gaussian_distance_decay(t: ArrayLike, *, d0: float = 20.0) -> np.ndarray:
    """Gaussian impedance with weight ≈ 0.14 at ``d0``."""
    t_arr = np.asarray(t, dtype=float)
    return np.exp(-0.5 * (t_arr / (d0 / 2.0)) ** 2)


def two_sfca(
    T: np.ndarray,
    P: ArrayLike,
    S: ArrayLike,
    *,
    d0: float = 20.0,
    decay: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    scale: float = 1000.0,
) -> np.ndarray:
    """Classic or enhanced 2SFCA accessibility (supply per ``scale`` demand)."""
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    if decay is None:
        W = (T <= d0).astype(float)
    else:
        W = decay(T) * (T <= d0)
    denom = (W * P[:, None]).sum(axis=0)
    denom = np.where(denom == 0, np.nan, denom)
    R = S / denom
    return (W * R[None, :]).sum(axis=1) * scale


def lorenz_curve_gini(
    values: ArrayLike,
    weights: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Population-weighted Lorenz curve and Gini coefficient."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    cv = np.cumsum(v * w) / (v * w).sum()
    cw, cv = np.r_[0.0, cw], np.r_[0.0, cv]
    gini = 1.0 - float(np.sum((cw[1:] - cw[:-1]) * (cv[1:] + cv[:-1])))
    return cw, cv, gini


def evaluate_candidate_sites(
    G: GraphLike,
    demand_nodes: Sequence[NodeId],
    demand_pop: ArrayLike,
    current_times: ArrayLike,
    candidates: Sequence[NodeId],
    *,
    weight: str = "minutes",
    coverage_threshold: float = 15.0,
) -> pd.DataFrame:
    """p-median (pop-weighted mean time) and coverage gain per candidate site."""
    nx = _require_networkx()
    P = np.asarray(demand_pop, dtype=float)
    current = np.asarray(current_times, dtype=float)
    rows = []
    for cand in candidates:
        d = nx.single_source_dijkstra_path_length(G, cand, weight=weight)
        cand_times = np.array([d.get(n, np.inf) for n in demand_nodes], dtype=float)
        new_times = np.minimum(current, cand_times)
        rows.append(
            {
                "candidate": cand,
                "mean_time": float(np.average(new_times, weights=P)),
                "coverage_pct": 100.0 * float(P[new_times <= coverage_threshold].sum()) / float(P.sum()),
                "mean_time_delta": float(np.average(current, weights=P) - np.average(new_times, weights=P)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_time")


# ---------------------------------------------------------------------------
# 5 · OD flows, gravity, hierarchy, contagion
# ---------------------------------------------------------------------------


def od_to_digraph(
    od: pd.DataFrame,
    *,
    source: str = "origin",
    target: str = "destination",
    weight: str = "commuters",
    attrs: Sequence[str] = (),
    min_flow: float = 1.0,
) -> GraphLike:
    """Build a directed weighted flow graph from an OD table."""
    nx = _require_networkx()
    G = nx.DiGraph()
    for r in od.itertuples(index=False):
        row = r._asdict() if hasattr(r, "_asdict") else dict(zip(od.columns, r))
        w = float(row[weight])
        if w < min_flow:
            continue
        data = {weight: w}
        for a in attrs:
            if a in row:
                data[a] = row[a]
        u, v = row[source], row[target]
        if G.has_edge(u, v):
            G[u][v][weight] += w
        else:
            G.add_edge(u, v, **data)
    return G


def node_flow_balance(
    G: GraphLike,
    *,
    weight: str = "commuters",
    population: Optional[Mapping] = None,
) -> pd.DataFrame:
    """Inflow, outflow, net flow, and in/out ratio by node."""
    rows = []
    for n in G.nodes:
        inflow = float(sum(d.get(weight, 0) for _, _, d in G.in_edges(n, data=True)))
        outflow = float(sum(d.get(weight, 0) for _, _, d in G.out_edges(n, data=True)))
        row = {
            "node": n,
            "inflow": inflow,
            "outflow": outflow,
            "net": inflow - outflow,
            "in_out_ratio": inflow / outflow if outflow else np.nan,
        }
        if population is not None:
            row["population"] = population.get(n)
        rows.append(row)
    return pd.DataFrame(rows).set_index("node")


def attach_od_impedance(
    od: pd.DataFrame,
    G: GraphLike,
    places,
    *,
    node_col: str = "node",
    origin_col: str = "origin",
    destination_col: str = "destination",
    weight: str = "travel_time",
    minutes_col: str = "minutes",
    euclid_col: Optional[str] = "euclid_km",
) -> pd.DataFrame:
    """Attach network minutes (and optional Euclidean km) to OD rows."""
    nx = _require_networkx()
    out = od.copy()
    node_of = places[node_col] if node_col in places.columns else places.index.to_series()
    # build lookup from place id → node / geometry
    if places.index.name or origin_col in list(places.index):
        place_index = places
    else:
        for c in (origin_col, "town", "name", "id"):
            if c in places.columns:
                place_index = places.set_index(c)
                break
        else:
            place_index = places

    minutes = []
    euclids = []
    for r in out.itertuples(index=False):
        row = r._asdict() if hasattr(r, "_asdict") else dict(zip(out.columns, r))
        o, d = row[origin_col], row[destination_col]
        on = place_index.loc[o, node_col] if node_col in place_index.columns else place_index.loc[o]
        dn = place_index.loc[d, node_col] if node_col in place_index.columns else place_index.loc[d]
        try:
            tt = nx.shortest_path_length(G, on, dn, weight=weight)
            minutes.append(float(tt) / (60.0 if weight == "travel_time" else 1.0))
        except Exception:
            minutes.append(np.inf)
        if euclid_col:
            go = place_index.loc[o].geometry if hasattr(place_index.loc[o], "geometry") else None
            gd = place_index.loc[d].geometry if hasattr(place_index.loc[d], "geometry") else None
            if go is not None and gd is not None:
                euclids.append(float(go.distance(gd)) / 1000.0)
            else:
                euclids.append(np.nan)
    out[minutes_col] = minutes
    if euclid_col:
        out[euclid_col] = euclids
    return out


def pseudo_r2(y: ArrayLike, yhat: ArrayLike) -> float:
    """Squared Pearson correlation as a gravity-model fit diagnostic."""
    y_arr = np.asarray(y, dtype=float)
    yhat_arr = np.asarray(yhat, dtype=float)
    if y_arr.size < 2:
        return float("nan")
    return float(np.corrcoef(y_arr, yhat_arr)[0, 1] ** 2)


def fit_gravity_model(
    od: pd.DataFrame,
    *,
    mass_o: str = "Pi",
    mass_d: str = "Pj",
    impedance: str = "minutes",
    flow_col: str = "commuters",
    method: str = "ppml",
):
    """Fit a gravity model by OLS-on-log or PPML (Poisson)."""
    LinearRegression, PoissonRegressor = _require_sklearn_linear()
    df = od.loc[(od[mass_o] > 0) & (od[mass_d] > 0) & (od[impedance] > 0)].copy()
    X = np.column_stack(
        [
            np.log(df[mass_o].to_numpy(float)),
            np.log(df[mass_d].to_numpy(float)),
            np.log(df[impedance].to_numpy(float)),
        ]
    )
    names = ["log_mass_o", "log_mass_d", "log_impedance"]
    y = df[flow_col].to_numpy(float)
    method = method.lower()
    if method == "ppml":
        model = PoissonRegressor(alpha=0.0, max_iter=1000)
        model.fit(X, y)
        coef = pd.Series(model.coef_, index=names, name="ppml")
        intercept = float(model.intercept_)
    elif method == "ols":
        mask = y > 0
        model = LinearRegression()
        model.fit(X[mask], np.log(y[mask]))
        coef = pd.Series(model.coef_, index=names, name="ols")
        intercept = float(model.intercept_)
    else:
        raise NetworkAnalysisError("method must be 'ppml' or 'ols'")
    return model, pd.concat([pd.Series({"intercept": intercept}), coef])


def dominant_flow_hierarchy(
    od: pd.DataFrame,
    populations: Mapping,
    *,
    origin_col: str = "origin",
    destination_col: str = "destination",
    flow_col: str = "commuters",
) -> Tuple[GraphLike, List]:
    """Nystuen–Dacey dominant-flow hierarchy tree and central places."""
    nx = _require_networkx()
    # For each origin, dominant outflow to a larger (or equal) place
    tree = nx.DiGraph()
    centers = set(populations)
    for origin, grp in od.groupby(origin_col):
        if origin not in populations:
            continue
        candidates = grp[grp[destination_col].map(lambda d: populations.get(d, 0) >= populations[origin])]
        if candidates.empty:
            continue
        top = candidates.loc[candidates[flow_col].idxmax()]
        dest = top[destination_col]
        tree.add_edge(origin, dest, weight=float(top[flow_col]))
        if dest != origin:
            centers.discard(origin)
    return tree, sorted(centers)


def symmetric_flow_graph(
    od: pd.DataFrame,
    *,
    normalise: bool = False,
    origin_col: str = "origin",
    destination_col: str = "destination",
    flow_col: str = "commuters",
    mass_o: str = "Pi",
    mass_d: str = "Pj",
) -> GraphLike:
    """Undirected Louvain-ready flow graph (optionally relative flows)."""
    nx = _require_networkx()
    U = nx.Graph()
    for r in od.itertuples(index=False):
        row = r._asdict() if hasattr(r, "_asdict") else dict(zip(od.columns, r))
        w = float(row[flow_col])
        if normalise:
            w = w / min(float(row[mass_o]), float(row[mass_d]))
        u, v = row[origin_col], row[destination_col]
        if u == v:
            continue
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)
    return U


def flow_communities(
    od: pd.DataFrame,
    *,
    normalise: bool = False,
    resolution: float = 1.0,
    seed: Optional[int] = 3,
    **sym_kwargs,
) -> Tuple[List[set], float]:
    """Louvain regions on a symmetric flow graph; returns communities and Q."""
    nx = _require_networkx()
    U = symmetric_flow_graph(od, normalise=normalise, **sym_kwargs)
    comms = list(nx.community.louvain_communities(U, weight="weight", resolution=resolution, seed=seed))
    Q = float(nx.community.modularity(U, comms, weight="weight"))
    return comms, Q


def flow_pagerank(
    G: GraphLike,
    *,
    weight: str = "commuters",
) -> pd.Series:
    """Attraction ranking from weighted PageRank on a flow Digraph."""
    nx = _require_networkx()
    pr = nx.pagerank(G, weight=weight)
    return pd.Series(pr, name="pagerank").sort_values(ascending=False)


def metapopulation_sir(
    C: np.ndarray,
    N: ArrayLike,
    seed_index: int,
    *,
    beta: float = 0.35,
    gamma: float = 0.2,
    days: int = 300,
    coupling: float = 0.1,
) -> np.ndarray:
    """Commuting-coupled metapopulation SIR; returns prevalence history ``(days, n)``."""
    N = np.asarray(N, dtype=float)
    C = np.asarray(C, dtype=float)
    m = coupling * C / N[:, None]
    M = m.copy()
    np.fill_diagonal(M, 1.0 - m.sum(axis=1))
    S, I = N.copy(), np.zeros_like(N)
    S[seed_index] -= 1.0
    I[seed_index] += 1.0
    hist = []
    for _ in range(days):
        new_inf = S * (1.0 - np.exp(-beta * (M @ (I / N))))
        S = S - new_inf
        I = I + new_inf - gamma * I
        hist.append(I / N)
    return np.asarray(hist)


def arrival_times(
    H: np.ndarray,
    *,
    threshold: float = 1e-4,
    index: Optional[Sequence] = None,
) -> pd.Series:
    """First time step where prevalence ≥ ``threshold`` for each patch."""
    times = [int(np.argmax(H[:, i] >= threshold)) if np.any(H[:, i] >= threshold) else -1 for i in range(H.shape[1])]
    return pd.Series(times, index=index, name="arrival_day")


# ---------------------------------------------------------------------------
# 6 · Relational / social spatial networks
# ---------------------------------------------------------------------------


def graph_from_edgelist(
    ties: pd.DataFrame,
    nodes=None,
    *,
    source: str = "source",
    target: str = "target",
    directed: bool = True,
    node_attrs: Sequence[str] = (),
) -> GraphLike:
    """Build a (Di)Graph from an edgelist; optionally attach node attributes."""
    nx = _require_networkx()
    G = nx.DiGraph() if directed else nx.Graph()
    for r in ties.itertuples(index=False):
        row = r._asdict() if hasattr(r, "_asdict") else dict(zip(ties.columns, r))
        G.add_edge(row[source], row[target])
    if nodes is not None:
        for idx, row in nodes.iterrows():
            attrs = {a: row[a] for a in node_attrs if a in row.index}
            if hasattr(row, "geometry"):
                attrs.setdefault("x", float(row.geometry.x))
                attrs.setdefault("y", float(row.geometry.y))
            if idx in G:
                G.nodes[idx].update(attrs)
            else:
                G.add_node(idx, **attrs)
    return G


def pairwise_distance_matrix(gdf) -> np.ndarray:
    """Euclidean pairwise distance matrix from point geometries."""
    xy = np.c_[gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()]
    return np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))


def tie_distance_decay(
    pairs_km: ArrayLike,
    ties: ArrayLike,
    *,
    same_group: Optional[ArrayLike] = None,
    bins: Optional[ArrayLike] = None,
) -> pd.DataFrame:
    """P(tie) vs distance, optionally stratified by same-group membership."""
    d = np.asarray(pairs_km, dtype=float)
    a = np.asarray(ties, dtype=float)
    if bins is None:
        bins = np.linspace(0, np.nanpercentile(d, 95), 20)
    bins = np.asarray(bins, dtype=float)
    cat = np.digitize(d, bins)

    def _curve(mask):
        return [
            float(a[mask & (cat == k)].mean()) if np.any(mask & (cat == k)) else np.nan
            for k in range(1, len(bins))
        ]

    out = {"distance": bins[1:], "P(tie)": _curve(np.ones(len(d), bool))}
    if same_group is not None:
        sg = np.asarray(same_group, dtype=bool)
        out["P(tie)|same"] = _curve(sg)
        out["P(tie)|diff"] = _curve(~sg)
    return pd.DataFrame(out)


def participation_coefficient(
    G: GraphLike,
    communities: Mapping[NodeId, Any],
) -> Dict[NodeId, float]:
    """Guimerà–Amaral participation coefficient by community assignment."""
    out = {}
    for v in G:
        k = G.degree(v)
        if k == 0:
            out[v] = 0.0
            continue
        counts = pd.Series([communities[u] for u in G.neighbors(v)]).value_counts()
        out[v] = float(1.0 - ((counts / k) ** 2).sum())
    return out


def diffusion_centrality(
    A: np.ndarray,
    *,
    q: float = 0.25,
    T: int = 4,
) -> np.ndarray:
    """Banerjee et al. finite-horizon diffusion centrality."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    ones = np.ones(n)
    total = np.zeros(n)
    M = q * A
    power = np.eye(n)
    for _ in range(1, T + 1):
        power = power @ M
        total = total + power @ ones
    return total


def community_label_agreement(labels_a: ArrayLike, labels_b: ArrayLike) -> float:
    """Adjusted Rand index between two labelings."""
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "community_label_agreement requires scikit-learn"
        ) from exc
    return float(adjusted_rand_score(labels_a, labels_b))


def select_seeds(
    nodes_df: pd.DataFrame,
    strategy: str,
    k: int,
    *,
    rng: Optional[np.random.Generator] = None,
    extension_col: str = "extension_contact",
    group_col: str = "village",
    indegree_col: str = "named_by",
    participation_col: str = "participation",
    diffusion_col: str = "diffusion_centrality",
) -> List:
    """Select ``k`` seeds by strategy name (notebook 06 seeding policies)."""
    rs = rng or np.random.default_rng()
    strategy = strategy.strip()
    order_in = nodes_df.sort_values(
        [c for c in (indegree_col, diffusion_col) if c in nodes_df.columns],
        ascending=False,
    )

    if strategy in {"Random", "random"}:
        return list(rs.choice(nodes_df.index, k, replace=False))
    if strategy in {"Extension contacts", "extension"}:
        ext_ids = nodes_df.index[nodes_df[extension_col].astype(bool)].tolist()
        return list(rs.choice(ext_ids, min(k, len(ext_ids)), replace=False))
    if strategy in {"Top in-degree", "in-degree", "indegree"}:
        return order_in.index[:k].tolist()
    if strategy in {"Leader per village", "leader_per_group"}:
        return order_in.groupby(group_col).head(1).index[:k].tolist()
    if strategy in {"Bridges", "bridges"}:
        cols = [c for c in (participation_col, indegree_col) if c in nodes_df.columns]
        return nodes_df.sort_values(cols, ascending=False).index[:k].tolist()
    if strategy in {"Pairs per village", "pairs_per_group"}:
        vs = (
            order_in.groupby(group_col)[indegree_col]
            .sum()
            .sort_values(ascending=False)
            .index[: k // 2]
        )
        return order_in[order_in[group_col].isin(vs)].groupby(group_col).head(2).index.tolist()
    raise NetworkAnalysisError(f"unknown seeding strategy: {strategy!r}")


def simulate_contagion(
    A: np.ndarray,
    seed_ids: Sequence,
    index: Sequence,
    *,
    model: str = "simple",
    periods: int = 20,
    q: float = 0.12,
    p: float = 0.5,
    theta: Optional[ArrayLike] = None,
    spont: float = 0.002,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple or complex contagion on an adjacency matrix.

    Returns ``(adoption_history, time_of_adoption)``.
    """
    rs = rng or np.random.default_rng()
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    idx = {node: i for i, node in enumerate(index)}
    k_out = np.maximum(A.sum(axis=1), 1.0)
    adopted = np.zeros(n, dtype=bool)
    for s in seed_ids:
        adopted[idx[s]] = True
    t_adopt = np.full(n, np.nan)
    t_adopt[adopted] = 0
    hist = [float(adopted.mean())]
    theta_arr = None if theta is None else np.asarray(theta, dtype=float)
    for t in range(1, periods + 1):
        k_a = A @ adopted.astype(float)
        if model == "simple":
            prob = 1.0 - (1.0 - q) ** k_a
        elif model == "complex":
            if theta_arr is None:
                raise NetworkAnalysisError("complex contagion requires theta")
            prob = np.where(k_a / k_out >= theta_arr, p, 0.0)
        else:
            raise NetworkAnalysisError("model must be 'simple' or 'complex'")
        prob = 1.0 - (1.0 - prob) * (1.0 - spont)
        new = (~adopted) & (rs.random(n) < prob)
        adopted |= new
        t_adopt[new] = t
        hist.append(float(adopted.mean()))
    return np.asarray(hist), t_adopt


def monte_carlo_contagion(
    A: np.ndarray,
    nodes_df: pd.DataFrame,
    strategy: str,
    *,
    model: str = "simple",
    reps: int = 200,
    k: int = 14,
    seed: int = 11,
    **simulate_kw,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repeated seeding + diffusion; returns histories ``(reps, periods+1)`` and adoption times."""
    rs = np.random.default_rng(seed)
    histories = []
    tadopts = []
    for _ in range(reps):
        seeds = select_seeds(nodes_df, strategy, k, rng=rs)
        h, ta = simulate_contagion(A, seeds, list(nodes_df.index), model=model, rng=rs, **simulate_kw)
        histories.append(h)
        tadopts.append(ta)
    return np.asarray(histories), np.asarray(tadopts)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _edge_iter(G: GraphLike):
    if G.is_multigraph():
        for u, v, key, data in G.edges(keys=True, data=True):
            yield u, v, key, data
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, None, data


def _set_edge_attr(G: GraphLike, u, v, key, attr: str, value) -> None:
    if key is None:
        G.edges[u, v][attr] = value
    else:
        G.edges[u, v, key][attr] = value


__all__ = [
    "NetworkAnalysisError",
    # embedding
    "haversine_km",
    "graph_positions",
    "set_edge_lengths_haversine",
    "set_edge_lengths_euclidean",
    "kansky_indices",
    "graph_component_summary",
    "describe_spatial_graph",
    "link_probability_by_distance",
    "circuity_sample",
    "is_planar_graph",
    # routing
    "load_road_graph",
    "prepare_travel_times",
    "snap_points_to_nodes",
    "route_summary",
    "shortest_routes",
    "od_travel_time_matrix",
    "isochrone_polygons",
    "cumulative_accessibility",
    "edge_travel_time_betweenness",
    "closure_impact",
    # spatial weights
    "weights_to_graph",
    "build_spatial_graphs",
    "min_threshold_distance",
    "percolation_curve",
    "row_standardize",
    "spatial_lag",
    "morans_i",
    "lines_to_graph",
    "similarity_weighted_communities",
    # access
    "nearest_facility_network",
    "nearest_facility_euclidean",
    "travel_time_matrix",
    "population_coverage",
    "gaussian_distance_decay",
    "two_sfca",
    "lorenz_curve_gini",
    "evaluate_candidate_sites",
    # flows
    "od_to_digraph",
    "node_flow_balance",
    "attach_od_impedance",
    "pseudo_r2",
    "fit_gravity_model",
    "dominant_flow_hierarchy",
    "symmetric_flow_graph",
    "flow_communities",
    "flow_pagerank",
    "metapopulation_sir",
    "arrival_times",
    # social
    "graph_from_edgelist",
    "pairwise_distance_matrix",
    "tie_distance_decay",
    "participation_coefficient",
    "diffusion_centrality",
    "community_label_agreement",
    "select_seeds",
    "simulate_contagion",
    "monte_carlo_contagion",
]
