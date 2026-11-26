"""Spatial neighborhoods: turn polygons/points into a sparse weights matrix that
``local`` and ``multiscale`` use to pool group counts across nearby units before
computing a *spatial* (rather than purely non-spatial) segregation measure.

Four kinds, all returning the same :class:`Neighborhood`:

- ``kernel`` / ``distance_band`` -- plain Euclidean distance between centroids,
  weighted by a Gaussian/triangular/uniform kernel or thresholded at a fixed radius.
- ``knn`` -- k-nearest centroids (via libpysal).
- ``network`` -- shortest-path network distance along a street graph (OSMnx), via
  ``scipy.sparse.csgraph.dijkstra`` rather than a pandana contraction-hierarchy
  table -- slower per query than pandana but has no extra compiled dependency, and
  is fast enough for the thousands-of-nodes graphs a segregation study needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse


def _require_libpysal():
    try:
        import libpysal
    except ImportError as exc:
        raise ImportError(
            "this neighborhood kind needs libpysal: pip install libpysal (or geolattice[network])"
        ) from exc
    return libpysal


def _require_osmnx():
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "neighborhood='network' needs osmnx: pip install osmnx (or geolattice[network])"
        ) from exc
    return ox


@dataclass
class Neighborhood:
    """A sparse unit-by-unit weights matrix plus the centroid coordinates it was built
    from. ``weights`` is ``(n, n)``; ``weights[i, j]`` is how much unit ``j`` counts toward
    unit ``i``'s neighborhood (0 on unvisited pairs, including, by convention, the
    diagonal -- callers add a unit's own counts back in explicitly, since whether "self"
    belongs in its own neighborhood differs by use)."""

    weights: sparse.csr_matrix
    ids: list

    @property
    def n(self) -> int:
        return self.weights.shape[0]

    def pooled_counts(self, counts: np.ndarray, *, include_self: bool = True,
                       row_standardize: bool = True) -> np.ndarray:
        """Neighborhood-pooled group counts: each unit's own counts plus its weighted
        neighbors' counts. ``row_standardize`` rescales each row's neighbor weights to sum
        to 1 first (so pooling is a weighted average of neighbor composition, not a raw
        sum that depends on how many neighbors a unit happens to have)."""
        w = self.weights
        if row_standardize:
            row_sum = np.asarray(w.sum(axis=1)).ravel()
            scale = np.where(row_sum > 0, 1.0 / row_sum, 0.0)
            w = sparse.diags(scale) @ w
        pooled = np.asarray(w @ counts)
        if include_self:
            pooled = pooled + counts
        return pooled


def _centroids(gdf) -> np.ndarray:
    geom = gdf.geometry
    cent = geom.centroid if geom.geom_type.isin(["Polygon", "MultiPolygon"]).any() else geom
    return np.c_[cent.x.to_numpy(), cent.y.to_numpy()]


def kernel_weights(gdf, *, bandwidth: float, kernel: Literal["gaussian", "triangular", "uniform"]
                    = "gaussian", id_col: str | None = None) -> Neighborhood:
    """Distance-decay weights: every pair within a few bandwidths gets a smooth weight
    (Gaussian) or a hard cutoff (triangular tapers to 0 at ``bandwidth``; uniform is 1 inside,
    0 outside). ``bandwidth`` is in the GeoDataFrame's own CRS units (projected CRS
    recommended)."""
    xy = _centroids(gdf)
    ids = gdf[id_col].tolist() if id_col else list(gdf.index)
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    radius = bandwidth if kernel != "gaussian" else bandwidth * 3
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    n = len(xy)
    if pairs.size == 0:
        return Neighborhood(sparse.csr_matrix((n, n)), ids)
    d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    if kernel == "gaussian":
        w = np.exp(-0.5 * (d / bandwidth) ** 2)
    elif kernel == "triangular":
        w = np.clip(1 - d / bandwidth, 0, None)
    else:
        w = (d <= bandwidth).astype("float64")
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    vals = np.concatenate([w, w])
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return Neighborhood(mat, ids)


def distance_band_weights(gdf, *, threshold: float, id_col: str | None = None) -> Neighborhood:
    """All unit pairs within ``threshold`` distance, weighted 1 (equivalent to
    ``kernel_weights(..., kernel="uniform")``, kept separate as the more common name)."""
    return kernel_weights(gdf, bandwidth=threshold, kernel="uniform", id_col=id_col)


def knn_weights(gdf, *, k: int = 5, id_col: str | None = None) -> Neighborhood:
    """The ``k`` nearest centroids to each unit, weighted 1 (asymmetric: ``i`` having ``j``
    as a nearest neighbor does not guarantee the reverse, matching libpysal's own ``KNN``)."""
    weights_mod = _require_libpysal()
    xy = _centroids(gdf)
    ids = gdf[id_col].tolist() if id_col else list(gdf.index)
    w = weights_mod.weights.KNN(xy, k=k)
    w.transform = "B"  # binary, not row-standardized -- pooled_counts handles standardizing
    n = w.n
    rows, cols, vals = [], [], []
    # libpysal's W stores neighbors/weights dicts keyed by id; build the sparse matrix from those
    id_index = {i: pos for pos, i in enumerate(w.id_order)}
    for i, neighbors in w.neighbors.items():
        wi = w.weights[i]
        ri = id_index[i]
        for j, wij in zip(neighbors, wi):
            rows.append(ri)
            cols.append(id_index[j])
            vals.append(float(wij))
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return Neighborhood(mat, ids)


def network_weights(gdf, graph: Any | None = None, *, radius: float, network_type: str = "walk",
                    id_col: str | None = None) -> Neighborhood:
    """Shortest-path network distance (in the graph's own edge-length units, typically
    meters) between each unit's nearest node, thresholded at ``radius``. Pass an existing
    OSMnx/NetworkX graph via ``graph``, or leave it ``None`` to download one covering the
    GeoDataFrame's bounds (needs ``osmnx``, and network access the first time).

    Distances come from ``scipy.sparse.csgraph.dijkstra`` on the graph's sparse adjacency
    matrix -- no pandana contraction-hierarchy table is built, trading some per-query speed
    on very large graphs for one fewer compiled dependency.
    """
    import networkx as nx

    if graph is None:
        ox = _require_osmnx()
        west, south, east, north = gdf.to_crs(4326).total_bounds
        graph = ox.graph_from_bbox((north, south, east, west), network_type=network_type)
        graph = ox.add_edge_lengths(graph) if not nx.get_edge_attributes(graph, "length") else graph
        node_xy_crs = 4326
    else:
        node_xy_crs = None

    nodes = list(graph.nodes)
    node_xy = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in nodes])

    gdf_pts = gdf.to_crs(node_xy_crs) if node_xy_crs else gdf
    xy = _centroids(gdf_pts)
    from scipy.spatial import cKDTree

    nearest = cKDTree(node_xy).query(xy)[1]

    adjacency = nx.to_scipy_sparse_array(graph, weight="length", format="csr")
    dist_matrix = sparse.csgraph.dijkstra(adjacency, directed=False, indices=np.unique(nearest))
    row_lookup = {node: i for i, node in enumerate(np.unique(nearest))}

    n = len(gdf)
    ids = gdf[id_col].tolist() if id_col else list(gdf.index)
    rows, cols, vals = [], [], []
    # Build node -> gdf-unit membership once, then expand node-to-node distances to unit-to-unit.
    node_to_units: dict[int, list[int]] = {}
    for i, node_pos in enumerate(nearest):
        node_to_units.setdefault(node_pos, []).append(i)
    for i in range(n):
        d_row = dist_matrix[row_lookup[nearest[i]]]
        within_nodes = np.where((d_row <= radius) & np.isfinite(d_row))[0]
        for node_pos in within_nodes:
            for j in node_to_units.get(node_pos, []):
                if j == i:
                    continue
                d = float(d_row[node_pos])
                rows.append(i)
                cols.append(j)
                vals.append(max(0.0, 1.0 - d / radius))  # linear decay to 0 at radius
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n)) if rows else sparse.csr_matrix((n, n))
    return Neighborhood(mat, ids)
