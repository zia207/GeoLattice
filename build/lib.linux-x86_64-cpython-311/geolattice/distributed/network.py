"""Network analysis on line data: cuGraph on GPU, SciPy / NetworkX on CPU.

:func:`graph_from_lines` turns a line GeoDataFrame into a :class:`Graph` whose nodes are
line end points (snapped to ``tolerance``) and whose edges carry a weight (length by
default). The network must already be noded: lines are joined only at shared end points.

* :func:`shortest_path_lengths` from many sources at once (SciPy's multi-source Dijkstra
  in C, or cuGraph SSSP per source on the GPU).
* :func:`shortest_path` returns the node sequence, cost and route geometry.
* :func:`centrality` (betweenness, PageRank, degree) and :func:`components`.
* :func:`nearest_nodes` snaps points to the graph with :func:`~geolattice.distributed.vector.knn`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from ._backend import Compute
from ._registry import dispatch, register


@dataclass
class Graph:
    """Nodes ``xy`` ``(n, 2)``; edges ``src``, ``dst``, ``weight``; ``edge_ids`` are the
    source line labels. Undirected graphs store each edge once."""

    xy: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    edge_ids: np.ndarray
    directed: bool = False
    crs: Any = None

    @property
    def n_nodes(self) -> int:
        return len(self.xy)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def to_scipy(self) -> sparse.csr_matrix:
        """CSR adjacency (both directions for undirected graphs; parallel edges keep the
        cheapest weight)."""
        s, d, w = self.src, self.dst, self.weight
        if not self.directed:
            s, d, w = np.concatenate([s, d]), np.concatenate([d, s]), np.concatenate([w, w])
        order = np.lexsort((w, d, s))
        s, d, w = s[order], d[order], w[order]
        first = np.ones(len(s), dtype=bool)
        first[1:] = (s[1:] != s[:-1]) | (d[1:] != d[:-1])
        return sparse.csr_matrix((w[first], (s[first], d[first])), shape=(self.n_nodes,) * 2)

    def to_cugraph(self) -> Any:
        from ._backend import rapids

        cudf, cugraph = rapids("cudf"), rapids("cugraph")
        # keep the cheapest of parallel edges, as to_scipy does; cuGraph would otherwise keep
        # an arbitrary one (measured: travel times up to 0.3 % longer than SciPy's)
        s, d, w = self.src, self.dst, self.weight.astype("float64")
        if not self.directed:
            s, d = np.minimum(s, d), np.maximum(s, d)
        order = np.lexsort((w, d, s))
        s, d, w = s[order], d[order], w[order]
        first = np.ones(len(s), dtype=bool)
        first[1:] = (s[1:] != s[:-1]) | (d[1:] != d[:-1])
        df = cudf.DataFrame({"src": s[first], "dst": d[first], "w": w[first]})
        g = cugraph.Graph(directed=self.directed)
        g.from_cudf_edgelist(df, source="src", destination="dst", weight="w", renumber=False,
                             vertices=cudf.Series(np.arange(self.n_nodes)))
        return g

    def to_networkx(self) -> Any:
        import networkx as nx

        g = nx.DiGraph() if self.directed else nx.Graph()
        g.add_nodes_from(range(self.n_nodes))
        g.add_weighted_edges_from(zip(self.src.tolist(), self.dst.tolist(), self.weight.tolist()))
        return g

    def nodes_gdf(self) -> Any:
        import geopandas as gpd

        return gpd.GeoDataFrame({"node": np.arange(self.n_nodes)},
                                geometry=gpd.points_from_xy(self.xy[:, 0], self.xy[:, 1]),
                                crs=self.crs)


def graph_from_lines(lines: Any, *, weight: str | None = None, directed: bool = False,
                     tolerance: float = 0.0) -> Graph:
    """Build a :class:`Graph` from (multi)line geometries.

    ``weight`` names a column of edge costs (default: line length). With ``tolerance``,
    end points closer than about ``tolerance`` are merged (grid snapping). For directed
    graphs each line runs from its first to its last vertex.
    """
    import shapely

    lines = lines.explode(index_parts=False)
    geoms = lines.geometry.values
    t = shapely.get_type_id(np.asarray(geoms, dtype=object))
    if not np.all(t == 1):
        raise TypeError("graph_from_lines needs LineString or MultiLineString geometries")
    start = shapely.get_coordinates(shapely.get_point(geoms, 0))
    end = shapely.get_coordinates(shapely.get_point(geoms, -1))
    pts = np.vstack([start, end])
    key = np.round(pts / tolerance) * tolerance if tolerance > 0 else pts
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    m = len(geoms)
    w = np.asarray(lines[weight], dtype=float) if weight else shapely.length(geoms)
    return Graph(uniq, inv[:m].astype(np.int64), inv[m:].astype(np.int64), w.astype(np.float64),
                 np.asarray(lines.index), directed, lines.crs)


# --------------------------------------------------------------------------- shortest paths
@register("network.sssp", "cpu")
def _sssp_cpu(g: Graph, sources: np.ndarray, cutoff: float | None,
              predecessors: bool) -> tuple[np.ndarray, np.ndarray | None]:
    from scipy.sparse.csgraph import dijkstra

    out = dijkstra(g.to_scipy(), directed=g.directed, indices=sources,
                   limit=np.inf if cutoff is None else cutoff, return_predecessors=predecessors)
    return out if predecessors else (out, None)


@register("network.sssp", "gpu")
def _sssp_gpu(g: Graph, sources: np.ndarray, cutoff: float | None,
              predecessors: bool) -> tuple[np.ndarray, np.ndarray | None]:
    from ._backend import rapids

    cugraph = rapids("cugraph")
    cg = g.to_cugraph()
    dist = np.full((len(sources), g.n_nodes), np.inf)
    pred = np.full((len(sources), g.n_nodes), -9999, dtype=np.int64) if predecessors else None
    for i, s in enumerate(sources):
        df = cugraph.sssp(cg, source=int(s), cutoff=cutoff).to_pandas()
        v = df["vertex"].to_numpy()
        d = df["distance"].to_numpy()
        ok = d < np.finfo(d.dtype).max / 2
        dist[i, v[ok]] = d[ok]
        if predecessors:
            p = df["predecessor"].to_numpy()
            pred[i, v] = np.where(p < 0, -9999, p)
    return dist, pred


def shortest_path_lengths(g: Graph, sources: Sequence[int] | int, *, cutoff: float | None = None,
                          compute: Compute | None = None) -> np.ndarray:
    """Cost from each source node to every node, ``(len(sources), n_nodes)``; ``inf`` where
    unreachable or beyond ``cutoff``."""
    src = np.atleast_1d(np.asarray(sources, dtype=np.int64))
    impl, _ = dispatch("network.sssp", compute)
    dist, _ = impl(g, src, cutoff, False)
    return np.asarray(dist).reshape(len(src), g.n_nodes)


def shortest_path(g: Graph, source: int, target: int, *,
                  compute: Compute | None = None) -> dict[str, Any]:
    """Cheapest route: ``{"nodes": [...], "cost": float, "geometry": LineString}`` through
    node positions (``nodes`` is empty and ``cost`` inf if unreachable)."""
    import shapely

    impl, _ = dispatch("network.sssp", compute)
    dist, pred = impl(g, np.array([source]), None, True)
    dist, pred = np.asarray(dist).reshape(-1), np.asarray(pred).reshape(-1)
    if not np.isfinite(dist[target]):
        return {"nodes": [], "cost": float("inf"), "geometry": None}
    path = [int(target)]
    while path[-1] != source:
        path.append(int(pred[path[-1]]))
        if path[-1] < 0 or len(path) > g.n_nodes:
            raise RuntimeError("predecessor chain broken")
    path.reverse()
    geom = shapely.LineString(g.xy[path]) if len(path) > 1 else shapely.Point(g.xy[path[0]])
    return {"nodes": path, "cost": float(dist[target]), "geometry": geom}


# --------------------------------------------------------------------------- centrality, components
@register("network.centrality", "cpu")
def _centrality_cpu(g: Graph, kind: str, k: int | None, normalized: bool, seed: int) -> np.ndarray:
    import networkx as nx

    if kind == "degree":
        return np.asarray((g.to_scipy() > 0).sum(axis=1)).ravel().astype(float)
    nxg = g.to_networkx()
    if kind == "betweenness":
        res = nx.betweenness_centrality(nxg, k=k, normalized=normalized, seed=seed)
    else:
        res = nx.pagerank(nxg, weight="weight")
    return np.array([res[i] for i in range(g.n_nodes)])


_PATH_COUNT_LIMIT = 1e18  # cuGraph counts shortest paths in int64 (overflows past ~9.2e18)


def _bfs_counts(a: sparse.csr_matrix, source: int) -> tuple[np.ndarray, np.ndarray]:
    """Hop distance and number of shortest paths (float64) from ``source``."""
    n = a.shape[0]
    dist = np.full(n, -1, dtype=np.int64)
    sigma = np.zeros(n)
    dist[source], sigma[source] = 0, 1.0
    frontier = np.array([source])
    level = 0
    while frontier.size:
        sub = a[frontier]
        reach = np.unique(sub.indices)
        new = reach[dist[reach] < 0]
        if new.size == 0:
            break
        level += 1
        dist[new] = level
        contrib = sub.T @ sigma[frontier]
        sigma[new] = contrib[new]
        frontier = new
    return dist, sigma


def max_shortest_path_count(g: Graph) -> float:
    """Largest number of equal-hop shortest paths from two far-apart nodes (a cheap probe
    of whether cuGraph's int64 path counters can overflow)."""
    a = g.to_scipy()
    a.data[:] = 1.0
    dist, sigma = _bfs_counts(a, 0)
    far = int(np.argmax(dist))
    _, sigma2 = _bfs_counts(a, far)
    return float(max(sigma.max(), sigma2.max()))


@register("network.centrality", "gpu")
def _centrality_gpu(g: Graph, kind: str, k: int | None, normalized: bool, seed: int) -> np.ndarray:
    import warnings

    from ._backend import rapids

    cugraph = rapids("cugraph")

    if kind == "degree":
        return _centrality_cpu(g, kind, k, normalized, seed)
    if kind == "betweenness" and max_shortest_path_count(g) > _PATH_COUNT_LIMIT:
        warnings.warn("Graph has more than 1e18 equal-length shortest paths between some nodes "
                      "(lattice-like network); cuGraph's int64 path counts would overflow, so "
                      "betweenness runs on CPU.", RuntimeWarning, stacklevel=3)
        return _centrality_cpu(g, kind, k, normalized, seed)
    cg = g.to_cugraph()
    if kind == "betweenness":
        df = cugraph.betweenness_centrality(cg, k=k, normalized=normalized, random_state=seed)
        col = "betweenness_centrality"
    else:
        df = cugraph.pagerank(cg)
        col = "pagerank"
    df = df.to_pandas()
    out = np.zeros(g.n_nodes)
    out[df["vertex"].to_numpy()] = df[col].to_numpy()
    return out


def centrality(g: Graph, kind: Literal["betweenness", "pagerank", "degree"] = "betweenness", *,
               k: int | None = None, normalized: bool = True, seed: int = 0,
               compute: Compute | None = None) -> pd.Series:
    """Node centrality.

    ``betweenness`` counts shortest paths by hops (unweighted); ``k`` samples that many
    source nodes for an approximation on large graphs. On GPU, graphs whose shortest-path
    counts could overflow cuGraph's int64 counters (large lattices) run on CPU with a
    warning. ``pagerank`` uses edge weights.
    """
    if kind not in ("betweenness", "pagerank", "degree"):
        raise ValueError("kind must be 'betweenness', 'pagerank' or 'degree'")
    impl, _ = dispatch("network.centrality", compute)
    return pd.Series(impl(g, kind, k, normalized, seed), name=kind)


@register("network.components", "cpu")
def _components_cpu(g: Graph) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components

    return connected_components(g.to_scipy(), directed=g.directed, connection="weak")[1]


@register("network.components", "gpu")
def _components_gpu(g: Graph) -> np.ndarray:
    from ._backend import rapids

    cugraph = rapids("cugraph")

    # weak components of a directed graph are the components of its undirected version
    # (cuGraph's component functions take undirected graphs only)
    und = Graph(g.xy, g.src, g.dst, g.weight, g.edge_ids, directed=False, crs=g.crs)
    df = cugraph.connected_components(und.to_cugraph()).to_pandas()
    out = np.zeros(g.n_nodes, dtype=np.int64)
    out[df["vertex"].to_numpy()] = pd.factorize(df["labels"].to_numpy())[0]
    return out


def components(g: Graph, *, compute: Compute | None = None) -> np.ndarray:
    """(Weakly) connected component label per node, numbered from 0."""
    impl, _ = dispatch("network.components", compute)
    labels = np.asarray(impl(g))
    return pd.factorize(labels)[0]


def nearest_nodes(g: Graph, points: Any, *, compute: Compute | None = None) -> np.ndarray:
    """Node id nearest to each point (Euclidean)."""
    from .vector import knn

    nb = knn(points, g.nodes_gdf(), k=1, compute=compute)
    return nb["index_right"].to_numpy()
