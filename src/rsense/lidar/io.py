"""Point cloud I/O: LAS/LAZ (laspy), COPC (laspy CopcReader), EPT and anything else via PDAL."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np

from ..core.pointcloud import PointCloud

_DEFAULT_FIELDS = ("intensity", "return_number", "number_of_returns", "classification",
                   "scan_angle_rank", "scan_angle", "gps_time", "red", "green", "blue",
                   "user_data", "point_source_id")


def _crs_from_header(header) -> Any:
    try:
        crs = header.parse_crs()
        return crs.to_epsg() or crs.to_wkt() if crs is not None else None
    except Exception:  # noqa: BLE001  (malformed VLRs should not block reading)
        return None


def read_las(path: str | PathLike, *, fields: Sequence[str] | None = None,
             bounds: tuple[float, float, float, float] | None = None,
             resolution: float | None = None, device: str | None = None) -> PointCloud:
    """Read LAS/LAZ (or COPC with ``bounds``/``resolution`` for spatial/level-of-detail queries).

    ``fields`` limits the attributes loaded (default: common LAS dimensions present).
    ``device="cuda"`` moves the result to GPU.
    """
    try:
        import laspy
    except ImportError as e:
        raise ImportError("Reading LAS needs `pip install laspy[lazrs]` (geolattice[rsense-lidar])") from e
    path = str(path)
    if path.endswith(".copc.laz") and (bounds is not None or resolution is not None):
        from laspy import Bounds, CopcReader

        with CopcReader.open(path) as rdr:
            b = None
            if bounds is not None:
                b = Bounds(mins=np.array(bounds[:2]), maxs=np.array(bounds[2:]))
            pts = rdr.query(bounds=b, resolution=resolution)
            header = rdr.header
    else:
        las = laspy.read(path)
        pts, header = las.points, las.header
        if bounds is not None:
            m = ((pts.x >= bounds[0]) & (pts.x <= bounds[2]) & (pts.y >= bounds[1])
                 & (pts.y <= bounds[3]))
            pts = pts[m]
    names = set(pts.point_format.dimension_names)
    wanted = [f for f in (fields or _DEFAULT_FIELDS) if f in names]
    attrs = {f: np.asarray(pts[f]) for f in wanted}
    pc = PointCloud(np.asarray(pts.x, dtype="float64"), np.asarray(pts.y, dtype="float64"),
                    np.asarray(pts.z, dtype="float64"), attrs, _crs_from_header(header))
    return pc.to_device(device) if device else pc


def write_las(pc: PointCloud, path: str | PathLike, *, point_format: int = 6,
              version: str = "1.4", scale: float = 0.001) -> None:
    """Write a PointCloud to LAS/LAZ (``.laz`` compresses with lazrs/laszip)."""
    import laspy

    pc = pc.to_cpu()
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = np.array([scale] * 3)
    header.offsets = np.array([np.floor(np.min(a)) for a in (pc.x, pc.y, pc.z)])
    if pc.crs is not None:
        from pyproj import CRS

        header.add_crs(CRS.from_user_input(pc.crs))
    las = laspy.LasData(header)
    las.x, las.y, las.z = pc.x, pc.y, pc.z
    dims = set(las.point_format.dimension_names)
    for k, v in pc.attrs.items():
        if k in dims:
            las[k] = v
    las.write(str(path))


_PDAL_NAMES = {"Intensity": "intensity", "ReturnNumber": "return_number",
               "NumberOfReturns": "number_of_returns", "Classification": "classification",
               "Red": "red", "Green": "green", "Blue": "blue", "GpsTime": "gps_time",
               "ScanAngleRank": "scan_angle_rank", "ScanAngle": "scan_angle",
               "PointSourceId": "point_source_id", "UserData": "user_data",
               "HeightAboveGround": "height"}


def _from_pdal_array(arr, srs=None) -> PointCloud:
    attrs = {_PDAL_NAMES[k]: np.asarray(arr[k]) for k in arr.dtype.names if k in _PDAL_NAMES}
    return PointCloud(np.asarray(arr["X"], float), np.asarray(arr["Y"], float),
                      np.asarray(arr["Z"], float), attrs, srs or None)


def _to_pdal_array(pc: PointCloud) -> np.ndarray:
    p = pc.to_cpu()
    inv = {v: k for k, v in _PDAL_NAMES.items()}
    cols = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
    data = {"X": p.x, "Y": p.y, "Z": p.z}
    for k, v in p.attrs.items():
        name = inv.get(k, k)
        cols.append((name, np.asarray(v).dtype.str))
        data[name] = v
    out = np.empty(len(p), dtype=cols)
    for k, v in data.items():
        out[k] = v
    return out


def _require_pdal():
    try:
        import pdal

        return pdal
    except ImportError as e:
        raise ImportError("PDAL Python bindings are needed (conda install -c conda-forge "
                          "python-pdal)") from e


def read_pdal(source: str, *, pipeline: list[dict] | None = None,
              bounds: tuple[float, float, float, float] | None = None,
              bounds_crs: Any = None, resolution: float | None = None,
              out_crs: Any = None) -> PointCloud:
    """Read anything PDAL can (EPT URLs such as USGS 3DEP, E57, PLY, COPC...).

    ``pipeline`` stages are appended after the reader (e.g. ``[{"type": "filters.outlier"}]``).
    ``bounds`` are in the source CRS unless ``bounds_crs`` is given; USGS 3DEP EPT data are in
    EPSG:3857, so ``bounds_crs=4326`` lets you pass a lon/lat box. ``out_crs`` reprojects the
    points (``filters.reprojection``).
    """
    pdal = _require_pdal()
    reader: dict[str, Any] = {"filename": source}
    if source.endswith("ept.json"):
        reader["type"] = "readers.ept"
    if bounds is not None:
        if bounds_crs is not None:
            from pyproj import Transformer

            dst = _source_srs(source) or "EPSG:3857"
            t = Transformer.from_crs(bounds_crs, dst, always_xy=True)
            bounds = t.transform_bounds(*bounds)
        reader["bounds"] = f"([{bounds[0]},{bounds[2]}],[{bounds[1]},{bounds[3]}])"
    if resolution is not None:
        reader["resolution"] = resolution
    stages = [reader, *(pipeline or [])]
    if out_crs is not None:
        from pyproj import CRS

        stages.append({"type": "filters.reprojection",
                       "out_srs": CRS.from_user_input(out_crs).to_string()})
    p = pdal.Pipeline(__import__("json").dumps(stages))
    p.execute()
    srs = p.srswkt2 if hasattr(p, "srswkt2") else None
    return _from_pdal_array(p.arrays[0], srs)


def _source_srs(source: str):
    """CRS of an EPT dataset (from ept.json), else None."""
    if not source.endswith("ept.json"):
        return None
    try:
        import requests

        meta = requests.get(source, timeout=30).json()
        srs = meta.get("srs", {})
        if "authority" in srs and "horizontal" in srs:
            return f"{srs['authority']}:{srs['horizontal']}"
        return srs.get("wkt")
    except Exception:  # noqa: BLE001
        return None


def pdal_pipeline(stages: list[dict] | str, pc: PointCloud | None = None) -> PointCloud:
    """Run a PDAL pipeline (list of stage dicts or JSON string) and return the first output.

    With ``pc`` the cloud is fed in as the input array (no reader stage needed), so any
    PDAL filter (``filters.smrf``, ``filters.hag_nn``, ``filters.outlier``, ``filters.csf``,
    ``filters.elm``...) can run on an in-memory PointCloud. Writers in the pipeline still
    write their files.
    """
    pdal = _require_pdal()
    js = stages if isinstance(stages, str) else __import__("json").dumps(stages)
    p = pdal.Pipeline(js, arrays=[_to_pdal_array(pc)]) if pc is not None else pdal.Pipeline(js)
    p.execute()
    srs = (p.srswkt2 if hasattr(p, "srswkt2") else None) or (pc.crs if pc is not None else None)
    return _from_pdal_array(p.arrays[0], srs)


def read_las_chunks(path: str | PathLike, chunk_size: int = 5_000_000, *,
                    fields: Sequence[str] | None = None):
    """Iterate over a large LAS/LAZ file in ``chunk_size`` point chunks (PointClouds)."""
    import laspy

    with laspy.open(str(path)) as f:
        crs = _crs_from_header(f.header)
        for pts in f.chunk_iterator(chunk_size):
            names = set(pts.point_format.dimension_names)
            wanted = [k for k in (fields or _DEFAULT_FIELDS) if k in names]
            yield PointCloud(np.asarray(pts.x, dtype="float64"), np.asarray(pts.y, dtype="float64"),
                             np.asarray(pts.z, dtype="float64"),
                             {k: np.asarray(pts[k]) for k in wanted}, crs)


def las_info(path: str | PathLike) -> dict[str, Any]:
    """Header summary without reading the points: version, point format and dimensions,
    point count, points by return, scales/offsets, bounds, CRS and VLR descriptions."""
    import laspy

    with laspy.open(str(path)) as f:
        h = f.header
        return {"version": str(h.version), "point_format": h.point_format.id,
                "dimensions": list(h.point_format.dimension_names),
                "point_count": int(h.point_count),
                "points_by_return": [int(v) for v in h.number_of_points_by_return],
                "scales": h.scales.tolist(), "offsets": h.offsets.tolist(),
                "mins": h.mins.tolist(), "maxs": h.maxs.tolist(),
                "crs": _crs_from_header(h),
                "creation_date": str(h.creation_date) if h.creation_date else None,
                "generating_software": h.generating_software,
                "vlrs": [f"{v.user_id}:{v.record_id} {v.description}" for v in h.vlrs],
                "compressed": str(path).endswith(".laz")}


def write_ply(pc: PointCloud, path: str | PathLike, *, binary: bool = True,
              fields: Sequence[str] | None = None, offset: Any = "auto") -> None:
    """Write a PLY file (CloudCompare, MeshLab, Open3D). ``offset="auto"`` subtracts the
    minimum x/y so coordinates fit float32 viewers; pass ``offset=None`` to keep them."""
    p = pc.to_cpu()
    o = np.array([np.floor(np.min(p.x)), np.floor(np.min(p.y)), 0.0]) if offset == "auto" else \
        (np.zeros(3) if offset is None else np.asarray(offset, float))
    cols = [("x", "f8"), ("y", "f8"), ("z", "f8")]
    data = {"x": np.asarray(p.x) - o[0], "y": np.asarray(p.y) - o[1], "z": np.asarray(p.z) - o[2]}
    for k in (fields if fields is not None else p.attrs):
        a = np.asarray(p[k])
        cols.append((k, "f4" if a.dtype.kind == "f" else ("u1" if k in ("red", "green", "blue")
                                                          and a.max() <= 255 else "i4")))
        data[k] = a
    arr = np.empty(len(p), dtype=cols)
    for k, v in data.items():
        arr[k] = v
    tmap = {"f8": "double", "f4": "float", "u1": "uchar", "i4": "int"}
    fmt = "binary_little_endian" if binary else "ascii"
    hdr = ["ply", f"format {fmt} 1.0", f"comment offset {o[0]} {o[1]} {o[2]}",
           f"element vertex {len(p)}"]
    hdr += [f"property {tmap[t]} {n}" for n, t in cols] + ["end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(hdr) + "\n").encode())
        if binary:
            fh.write(arr.astype(arr.dtype.newbyteorder("<")).tobytes())
        else:
            np.savetxt(fh, np.column_stack([arr[n] for n, _ in cols]), fmt="%.6g")


def usgs_3dep_ept(bounds_4326: tuple[float, float, float, float]) -> list[str]:
    """Find USGS 3DEP Entwine (EPT) dataset URLs intersecting a lon/lat bbox."""
    import requests
    from shapely.geometry import box, shape

    resources = requests.get("https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/"
                             "boundaries/resources.geojson", timeout=60).json()
    aoi = box(*bounds_4326)
    return [f["properties"]["url"] for f in resources["features"]
            if shape(f["geometry"]).intersects(aoi)]
