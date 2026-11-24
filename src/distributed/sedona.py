"""Apache Sedona (Spark) engine for spatial joins that outgrow one machine.

:func:`spark` starts (or reuses) a Spark session with Sedona registered. The Sedona and
GeoTools JARs are fetched from Maven on first use, matched to the installed ``apache-sedona``
and ``pyspark`` versions; set ``GEOLATTICE_SEDONA_PACKAGES`` to override the coordinates (for
example on an air-gapped cluster with a local repository). Spark workers are pointed at
the current Python interpreter so driver and worker versions match.

:func:`sjoin` runs a spatial SQL join (``ST_Intersects``, ``ST_Within``, ``ST_DWithin``
...). GeoPandas inputs come back as a GeoDataFrame with the same columns as
``geopandas.sjoin``; Spark DataFrame inputs return a lazy Spark DataFrame. It is also
reachable as ``vector.sjoin(..., engine="sedona")``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_SESSION: Any = None

# Sedona release -> GeoTools wrapper build published alongside it on Maven Central.
_GEOTOOLS = {"1.9": "33.5", "1.8": "33.1", "1.7": "28.5", "1.6": "28.2"}

_PREDICATES = {
    "intersects": "ST_Intersects(l.{lg}, r.{rg})",
    "within": "ST_Within(l.{lg}, r.{rg})",
    "contains": "ST_Contains(l.{lg}, r.{rg})",
    "covers": "ST_Covers(l.{lg}, r.{rg})",
    "covered_by": "ST_CoveredBy(l.{lg}, r.{rg})",
    "touches": "ST_Touches(l.{lg}, r.{rg})",
    "crosses": "ST_Crosses(l.{lg}, r.{rg})",
    "overlaps": "ST_Overlaps(l.{lg}, r.{rg})",
    "dwithin": "ST_DWithin(l.{lg}, r.{rg}, {distance})",
}


def packages() -> str:
    """Maven coordinates for the installed Sedona and Spark versions."""
    env = os.environ.get("GEOLATTICE_SEDONA_PACKAGES")
    if env:
        return env
    import pyspark
    import sedona

    sv = sedona.version
    spark_mm = ".".join(pyspark.__version__.split(".")[:2])
    scala = "2.13" if int(spark_mm.split(".")[0]) >= 4 else "2.12"
    gt = _GEOTOOLS.get(".".join(sv.split(".")[:2]))
    if gt is None:
        raise RuntimeError(f"No known GeoTools wrapper for Sedona {sv}; set GEOLATTICE_SEDONA_PACKAGES")
    return (f"org.apache.sedona:sedona-spark-shaded-{spark_mm}_{scala}:{sv},"
            f"org.datasyslab:geotools-wrapper:{sv}-{gt}")


def spark(master: str = "local[*]", *, app_name: str = "geolattice", config: dict | None = None) -> Any:
    """Start or reuse a Sedona-enabled Spark session (``SedonaContext``)."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    try:
        from sedona.spark import SedonaContext
    except ImportError as e:
        raise ImportError("The Sedona engine needs apache-sedona and pyspark: "
                          "pip install 'geolattice[distributed-spark]'") from e
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    builder = (SedonaContext.builder().master(master).appName(app_name)
               .config("spark.jars.packages", packages())
               .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
    for k, v in (config or {}).items():
        builder = builder.config(k, v)
    _SESSION = SedonaContext.create(builder.getOrCreate())
    return _SESSION


def stop() -> None:
    """Stop the Spark session started by :func:`spark`."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.stop()
        _SESSION = None


def is_spark(obj: Any) -> bool:
    return type(obj).__module__.startswith("pyspark.sql")


def to_spark(gdf: Any, session: Any = None) -> Any:
    """GeoDataFrame to a Spark DataFrame (GeoArrow transfer when available)."""
    s = session or spark()
    try:
        from sedona.spark import create_spatial_dataframe

        return create_spatial_dataframe(s, gdf)
    except Exception:
        return s.createDataFrame(gdf)


def to_geopandas(sdf: Any, *, geometry: str = "geometry", crs: Any = None) -> Any:
    """Spark DataFrame with a geometry column back to a GeoDataFrame."""
    import geopandas as gpd

    pdf = sdf.toPandas()
    return gpd.GeoDataFrame(pdf, geometry=geometry, crs=crs)


def sjoin(left: Any, right: Any, *, how: str = "inner", predicate: str = "intersects",
          distance: float | None = None, lsuffix: str = "left", rsuffix: str = "right",
          session: Any = None) -> Any:
    """Spatial join with Sedona; see the module docstring for inputs and outputs."""
    if how not in ("inner", "left"):
        raise ValueError("how must be 'inner' or 'left'")
    if predicate not in _PREDICATES:
        raise ValueError(f"Unsupported predicate {predicate!r}")
    if predicate == "dwithin" and distance is None:
        raise ValueError("predicate='dwithin' needs distance=")
    s = session or spark()
    lazy = is_spark(left) or is_spark(right)
    if lazy:
        lsdf = left if is_spark(left) else to_spark(left, s)
        rsdf = right if is_spark(right) else to_spark(right, s)
        lg = rg = "geometry"
        lsdf.createOrReplaceTempView("geolattice_l")
        rsdf.createOrReplaceTempView("geolattice_r")
        cond = _PREDICATES[predicate].format(lg=lg, rg=rg, distance=float(distance or 0))
        join = "JOIN" if how == "inner" else "LEFT JOIN"
        rcols = [c for c in rsdf.columns if c != rg]
        lcols = set(lsdf.columns)
        sel = ", ".join([f"r.`{c}` AS `{c}_{rsuffix}`" if c in lcols else f"r.`{c}`" for c in rcols])
        return s.sql(f"SELECT l.*{', ' + sel if sel else ''} FROM geolattice_l l {join} geolattice_r r ON {cond}")

    # GeoPandas in and out: join positions in Spark, then build the frame with GeoPandas'
    # own join builder so columns, suffixes and index_right match geopandas.sjoin.
    from geopandas.tools.sjoin import _frame_join

    if left.crs is not None and right.crs is not None and left.crs != right.crs:
        raise ValueError(f"CRS mismatch: left {left.crs} vs right {right.crs}; reproject first")
    import geopandas as gpd

    lg = gpd.GeoDataFrame({"_l": np.arange(len(left))}, geometry=left.geometry.values)
    rg = gpd.GeoDataFrame({"_r": np.arange(len(right))}, geometry=right.geometry.values)
    to_spark(lg, s).createOrReplaceTempView("geolattice_l")
    to_spark(rg, s).createOrReplaceTempView("geolattice_r")
    cond = _PREDICATES[predicate].format(lg="geometry", rg="geometry", distance=float(distance or 0))
    pairs = s.sql(f"SELECT l._l, r._r FROM geolattice_l l JOIN geolattice_r r ON {cond}").toPandas()
    li = pairs["_l"].to_numpy(dtype=np.int64)
    ri = pairs["_r"].to_numpy(dtype=np.int64)
    order = np.lexsort((ri, li))
    joined, _ = _frame_join(left, right, (li[order], ri[order]), None, how, lsuffix, rsuffix,
                            predicate)
    return joined
