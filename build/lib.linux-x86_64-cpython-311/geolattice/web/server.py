"""
A small web-GIS server: XYZ raster tiles, OGC API – Features and a map viewer.

:func:`create_app` turns rasters and vector layers into an ASGI application
built on Starlette:

=====================================================  =========================================
Endpoint                                               Returns
=====================================================  =========================================
``/``                                                  Landing page (OGC API links)
``/api``                                               OpenAPI 3.0 definition of the service
``/conformance``                                       Conformance classes
``/collections``                                       Vector collections
``/collections/{id}``                                  One collection with its extent
``/collections/{id}/items``                            GeoJSON features (``bbox``, ``datetime``,
                                                       ``limit``, ``offset``, attribute filters)
``/collections/{id}/items/{featureId}``                One feature
``/tiles``                                             Raster tile layers
``/tiles/{name}/tilejson.json``                        TileJSON for a layer
``/tiles/{name}/{z}/{x}/{y}.png``                      A rendered tile (204 when empty)
``/viewer``                                            Leaflet map of every layer
=====================================================  =========================================

Feature requests are encoded in Starlette's thread pool, so a large ``/items``
response does not hold up tile requests.

``datetime`` (an RFC 3339 instant or ``start/end`` interval, open ends as ``..``)
filters on a collection's datetime column when it has exactly one; for
collections without one it is accepted and ignored.

Run it with :func:`serve` (uvicorn) or any ASGI server; mount it inside a
FastAPI application with ``app.mount("/gis", create_app(...))``.

Example:
    from geolattice.web import create_app, serve

    app = create_app(
        rasters={"ndvi": {"path": "ndvi_cog.tif", "colormap": "RdYlGn", "rescale": (-0.2, 0.9)}},
        vectors={"counties": "counties.parquet"},
        title="Onondaga County",
    )
    serve(app, port=8000)          # then open http://127.0.0.1:8000/viewer
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

CONFORMANCE = [
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core",
    "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/geojson",
]
CRS84 = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
OPENAPI_TYPE = "application/vnd.oai.openapi+json;version=3.0"


def create_app(
    *,
    rasters: Optional[Mapping[str, Union[str, Path, Mapping[str, Any]]]] = None,
    vectors: Optional[Mapping[str, Any]] = None,
    title: str = "GeoLattice web-GIS",
    description: str = "Tiles and features served with geolattice.web",
    default_limit: int = 100,
    max_limit: int = 10_000,
    cors: bool = True,
) -> Any:
    """
    Build an ASGI app that serves raster tiles and vector features.

    Args:
        rasters: ``{name: path}`` or ``{name: {"path": ..., "bands": [...], "rescale": ...,
            "colormap": ..., "nodata": ..., "resampling": ..., "minzoom": ..., "maxzoom": ...}}``.
            Paths may be local files or URLs (COGs are read with range requests).
            When ``rescale`` is omitted it is computed once from the whole raster.
        vectors: ``{name: GeoDataFrame | path}``; paths are read with GeoPandas
            (``.parquet`` with :func:`geolattice.vector.read_geoparquet`). Features are
            served in WGS 84 longitude/latitude (CRS84), as OGC API – Features requires.
        title, description: Shown on the landing page and the viewer.
        default_limit, max_limit: Paging of ``/items``.
        cors: Allow cross-origin requests, so other web pages can use the service.

    Returns:
        A ``starlette.applications.Starlette`` application.
    """
    starlette = _require("starlette", "starlette")
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, Response
    from starlette.routing import Route

    from geolattice.web.tiles import raster_web_info, render_tile, suggest_rescale, tilejson
    from geolattice.web.viewer import leaflet_html

    raster_layers = {name: _raster_config(spec) for name, spec in (rasters or {}).items()}
    for name, cfg in raster_layers.items():
        info = raster_web_info(cfg["path"])
        cfg.setdefault("minzoom", info["minzoom"])
        cfg.setdefault("maxzoom", info["maxzoom"])
        cfg["info"] = info
        if cfg.get("rescale") is None:
            cfg["rescale"] = suggest_rescale(cfg["path"], cfg.get("bands"))
        _check_raster_config(name, cfg, info["count"])
    vector_layers = {name: _vector_frame(data) for name, data in (vectors or {}).items()}
    temporal = {name: _temporal_column(frame) for name, frame in vector_layers.items()}

    def base_url(request: Request) -> str:
        return str(request.base_url).rstrip("/")

    def not_found(message: str) -> JSONResponse:
        return JSONResponse({"code": "NotFound", "description": message}, status_code=404)

    def bad_request(message: str) -> JSONResponse:
        return JSONResponse({"code": "InvalidParameterValue", "description": message}, status_code=400)

    async def landing(request: Request) -> JSONResponse:
        root = base_url(request)
        return JSONResponse({
            "title": title, "description": description,
            "links": [
                {"href": f"{root}/", "rel": "self", "type": "application/json", "title": "This document"},
                {"href": f"{root}/api", "rel": "service-desc", "type": OPENAPI_TYPE,
                 "title": "API definition"},
                {"href": f"{root}/conformance", "rel": "conformance", "type": "application/json"},
                {"href": f"{root}/collections", "rel": "data", "type": "application/json"},
                {"href": f"{root}/tiles", "rel": "tiles", "type": "application/json"},
                {"href": f"{root}/viewer", "rel": "alternate", "type": "text/html", "title": "Map viewer"},
            ],
        })

    async def conformance(request: Request) -> JSONResponse:
        return JSONResponse({"conformsTo": CONFORMANCE})

    async def api(request: Request) -> JSONResponse:
        return JSONResponse(_openapi(base_url(request), title, description, list(vector_layers),
                                     list(raster_layers), default_limit, max_limit),
                            media_type=OPENAPI_TYPE)

    def collection_doc(root: str, cid: str) -> Dict[str, Any]:
        frame = vector_layers[cid]
        extent: Dict[str, Any] = {"spatial": {"bbox": [_bbox(frame)], "crs": CRS84}}
        if temporal[cid]:
            times = _utc(frame[temporal[cid]]).dropna()
            if len(times):
                extent["temporal"] = {"interval": [[times.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                    times.max().strftime("%Y-%m-%dT%H:%M:%SZ")]]}
        return {
            "id": cid, "title": cid, "itemType": "feature", "crs": [CRS84],
            "extent": extent,
            "links": [
                {"href": f"{root}/collections/{cid}", "rel": "self", "type": "application/json"},
                {"href": f"{root}/collections/{cid}/items", "rel": "items", "type": "application/geo+json"},
            ],
        }

    async def collections(request: Request) -> JSONResponse:
        root = base_url(request)
        return JSONResponse({"collections": [collection_doc(root, cid) for cid in vector_layers],
                             "links": [{"href": f"{root}/collections", "rel": "self", "type": "application/json"}]})

    async def collection(request: Request) -> JSONResponse:
        cid = request.path_params["cid"]
        if cid not in vector_layers:
            return not_found(f"No collection {cid!r}")
        return JSONResponse(collection_doc(base_url(request), cid))

    def items(request: Request) -> JSONResponse:      # sync: encoding runs in the thread pool
        cid = request.path_params["cid"]
        if cid not in vector_layers:
            return not_found(f"No collection {cid!r}")
        frame = vector_layers[cid]
        params = dict(request.query_params)
        try:
            limit = min(int(params.pop("limit", default_limit)), max_limit)
            offset = int(params.pop("offset", 0))
            if limit < 1 or offset < 0:
                raise ValueError
        except ValueError:
            return bad_request("limit must be a positive integer and offset a non-negative integer")
        bbox_text = params.pop("bbox", None)
        datetime_text = params.pop("datetime", None)
        params.pop("f", None)
        if datetime_text:
            try:
                start, end = _parse_datetime(datetime_text)
            except ValueError:
                return bad_request("datetime must be an RFC 3339 instant or interval, e.g. "
                                   "2024-01-01T00:00:00Z or 2024-01-01T00:00:00Z/..")
            if temporal[cid]:
                times = _utc(frame[temporal[cid]])
                keep = times.notna()
                if start is not None:
                    keep &= times >= start
                if end is not None:
                    keep &= times <= end
                frame = frame[keep.values]
        if bbox_text:
            try:
                box = [float(v) for v in bbox_text.split(",")]
                if len(box) != 4:
                    raise ValueError
            except ValueError:
                return bad_request("bbox must be minLon,minLat,maxLon,maxLat")
            from shapely.geometry import box as shapely_box

            hits = frame.sindex.query(shapely_box(*box), predicate="intersects")
            frame = frame.iloc[sorted(hits)]
        for key, value in params.items():
            if key not in frame.columns:
                return bad_request(f"Unknown property {key!r}")
            frame = frame[frame[key].astype(str) == value]
        matched = len(frame)
        page = frame.iloc[offset:offset + limit]
        root = base_url(request)
        query = request.url.query
        links = [{"href": str(request.url), "rel": "self", "type": "application/geo+json"},
                 {"href": f"{root}/collections/{cid}", "rel": "collection", "type": "application/json"}]
        if offset + limit < matched:
            nxt = _set_query(query, offset=offset + limit, limit=limit)
            links.append({"href": f"{root}/collections/{cid}/items?{nxt}", "rel": "next",
                          "type": "application/geo+json"})
        members = {"numberMatched": matched, "numberReturned": len(page), "timeStamp": _now(), "links": links}
        body = '{"type":"FeatureCollection","features":[' + ",".join(_features_json(page)) + "]," \
            + json.dumps(members)[1:]
        return Response(body, media_type="application/geo+json")

    def item(request: Request) -> JSONResponse:
        cid, fid = request.path_params["cid"], request.path_params["fid"]
        if cid not in vector_layers:
            return not_found(f"No collection {cid!r}")
        frame = vector_layers[cid]
        match = frame[frame.index.astype(str) == fid]
        if match.empty:
            return not_found(f"No feature {fid!r} in {cid!r}")
        links = [{"href": str(request.url), "rel": "self", "type": "application/geo+json"}]
        body = _features_json(match.iloc[:1])[0][:-1] + ',"links":' + json.dumps(links) + "}"
        return Response(body, media_type="application/geo+json")

    async def tile_layers(request: Request) -> JSONResponse:
        root = base_url(request)
        return JSONResponse({"tiles": [
            {"name": name, "tilejson": f"{root}/tiles/{name}/tilejson.json",
             "url": f"{root}/tiles/{name}/{{z}}/{{x}}/{{y}}.png", "bounds": cfg["info"]["bounds"],
             "minzoom": cfg["minzoom"], "maxzoom": cfg["maxzoom"]} for name, cfg in raster_layers.items()]})

    async def tile_json(request: Request) -> JSONResponse:
        name = request.path_params["name"]
        if name not in raster_layers:
            return not_found(f"No tile layer {name!r}")
        cfg = raster_layers[name]
        return JSONResponse(tilejson(name, f"{base_url(request)}/tiles/{name}/{{z}}/{{x}}/{{y}}.png",
                                     bounds=cfg["info"]["bounds"], minzoom=cfg["minzoom"],
                                     maxzoom=cfg["maxzoom"], center=cfg["info"]["center"]))

    def tile(request: Request) -> Response:          # sync: runs in Starlette's thread pool
        name = request.path_params["name"]
        if name not in raster_layers:
            return not_found(f"No tile layer {name!r}")
        z, x, y = (int(request.path_params[k]) for k in ("z", "x", "y"))
        if not (0 <= z <= 24 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
            return bad_request("Tile index out of range")
        cfg = raster_layers[name]
        image = render_tile(cfg["path"], x, y, z, bands=cfg.get("bands"), rescale=cfg["rescale"],
                            colormap=cfg.get("colormap"), nodata=cfg.get("nodata"),
                            resampling=cfg.get("resampling", "bilinear"))
        if image is None:
            return Response(status_code=204)
        return Response(image, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

    async def viewer(request: Request) -> HTMLResponse:
        root = base_url(request)
        return HTMLResponse(leaflet_html(
            title=title,
            rasters=[{"name": n, "url": f"{root}/tiles/{n}/{{z}}/{{x}}/{{y}}.png", "bounds": c["info"]["bounds"],
                      "minzoom": c["minzoom"], "maxzoom": c["maxzoom"]} for n, c in raster_layers.items()],
            vectors=[{"name": n, "url": f"{root}/collections/{n}/items?limit={max_limit}"} for n in vector_layers],
        ))

    routes = [
        Route("/", landing), Route("/api", api), Route("/conformance", conformance), Route("/collections", collections),
        Route("/collections/{cid}", collection), Route("/collections/{cid}/items", items),
        Route("/collections/{cid}/items/{fid}", item), Route("/tiles", tile_layers),
        Route("/tiles/{name}/tilejson.json", tile_json), Route("/tiles/{name}/{z:int}/{x:int}/{y:int}.png", tile),
        Route("/viewer", viewer),
    ]
    middleware = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])] if cors else []
    app = Starlette(routes=routes, middleware=middleware)
    app.state.rasters = raster_layers
    app.state.vectors = vector_layers
    return app


def serve(app: Any, *, host: str = "127.0.0.1", port: int = 8000, **kwargs: Any) -> None:
    """Run an ASGI app with uvicorn (blocking). ``host="0.0.0.0"`` exposes it on the network."""
    uvicorn = _require("uvicorn", "uvicorn")
    uvicorn.run(app, host=host, port=port, **kwargs)


class BackgroundServer:
    """An ASGI app served by uvicorn in a background thread; see :func:`start_server`."""

    def __init__(self, app: Any, host: str, port: int, log_level: str, timeout: float) -> None:
        import threading
        import time

        uvicorn = _require("uvicorn", "uvicorn")

        class _Server(uvicorn.Server):
            def install_signal_handlers(self) -> None:   # not the main thread
                pass

        self._server = _Server(uvicorn.Config(app, host=host, port=port, log_level=log_level))
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"The server did not start on {host}:{port}.")
            time.sleep(0.02)
        sock = self._server.servers[0].sockets[0]
        self.host, self.port = sock.getsockname()[:2]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    def __enter__(self) -> "BackgroundServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"BackgroundServer({self.url})"


def start_server(app: Any, *, host: str = "127.0.0.1", port: int = 0, log_level: str = "warning",
                 timeout: float = 30.0) -> BackgroundServer:
    """
    Serve an ASGI app (``create_app``, FastAPI, Starlette) in a background thread and return at once.

    Meant for notebooks and tests, where :func:`serve` would block. ``port=0`` picks a
    free port; read it from ``server.url``. Stop with ``server.stop()`` or use a ``with`` block.
    """
    return BackgroundServer(app, host, port, log_level, timeout)


# ----------------------------------------------------------------------------- helpers
def _raster_config(spec: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(spec, (str, Path)):
        return {"path": os.fspath(spec)}
    cfg = dict(spec)
    if "path" not in cfg:
        raise ValueError("A raster layer mapping needs a 'path' key.")
    cfg["path"] = os.fspath(cfg["path"])
    return cfg


def _vector_frame(data: Any) -> Any:
    gpd = _require("geopandas", "geopandas")
    if isinstance(data, (str, Path)):
        location = os.fspath(data)
        if location.lower().endswith((".parquet", ".geoparquet")):
            from geolattice.vector.write_cloud_native_vector_data import read_geoparquet

            frame = read_geoparquet(location)
        else:
            frame = gpd.read_file(location)
    elif isinstance(data, gpd.GeoDataFrame):
        frame = data
    else:
        raise TypeError("Vector layers must be GeoDataFrames or file paths.")
    if frame.crs is None:
        raise ValueError("A vector layer has no CRS; set one before serving it.")
    frame = frame.to_crs(4326)
    return frame


def _features_json(frame: Any, chunk: int = 50) -> list:
    """
    GeoJSON text of each feature (``id`` from the index, as GeoPandas writes it).

    Geometry is encoded by GEOS (``shapely.to_geojson``), in chunks, instead of
    through Python dictionaries and ``json``: about ten times faster for large
    polygons, and the GIL is handed over between chunks so other requests (tiles) keep
    flowing. Threads still share one interpreter, so for many concurrent users run several
    worker processes (``uvicorn module:app --workers 4``).
    """
    import time

    import shapely

    props = frame.drop(columns=frame.geometry.name)
    props = props.astype(object).where(props.notna(), None).to_dict("records")
    ids = [json.dumps(str(i)) for i in frame.index]
    geoms = frame.geometry.values
    out = []
    for start in range(0, len(frame), chunk):
        texts = shapely.to_geojson(geoms[start:start + chunk])
        for i, text in enumerate(texts, start):
            out.append('{"type":"Feature","id":%s,"geometry":%s,"properties":%s}'
                       % (ids[i], "null" if text is None else text,
                          json.dumps(props[i], default=_json_value)))
        time.sleep(0)                     # hand the GIL to waiting requests (tiles) between chunks
    return out


def _json_value(value: Any) -> Any:
    """JSON encoding for property values json cannot handle (timestamps, dates, NumPy scalars)."""
    if hasattr(value, "isoformat"):
        text = value.isoformat()
        return text.replace("+00:00", "Z")
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _check_raster_config(name: str, cfg: Dict[str, Any], count: int) -> None:
    """Fail at start-up, not on the first tile request, when bands or rescale do not fit the raster."""
    bands = list(cfg.get("bands") or (range(1, 4) if count >= 3 else [1]))
    if len(bands) not in (1, 3) or not all(1 <= int(b) <= count for b in bands):
        raise ValueError(f"Raster layer {name!r}: bands {bands} do not fit a {count}-band raster "
                         "(give one band or three)")
    rescale = cfg["rescale"]
    n_pairs = 1 if len(rescale) == 2 and not any(isinstance(v, (list, tuple)) for v in rescale) else len(rescale)
    if n_pairs not in (1, len(bands)):
        raise ValueError(f"Raster layer {name!r}: rescale has {n_pairs} pairs for {len(bands)} bands")


def _temporal_column(frame: Any) -> Optional[str]:
    """The collection's datetime column when it has exactly one, else ``None``."""
    import pandas as pd

    names = [c for c in frame.columns if c != frame.geometry.name
             and pd.api.types.is_datetime64_any_dtype(frame[c])]
    return names[0] if len(names) == 1 else None


def _utc(series: Any) -> Any:
    import pandas as pd

    times = pd.to_datetime(series)
    return times.dt.tz_localize("UTC") if times.dt.tz is None else times.dt.tz_convert("UTC")


def _parse_datetime(text: str) -> tuple:
    """Parse an OGC API ``datetime`` value into (start, end) UTC timestamps; ``None`` = open."""
    import pandas as pd

    def one(value: str) -> Any:
        if value in ("..", ""):
            return None
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    parts = text.split("/")
    if len(parts) == 1:
        if parts[0] in ("..", ""):
            raise ValueError(text)
        instant = one(parts[0])
        return instant, instant
    if len(parts) != 2 or parts[0] in ("..", "") and parts[1] in ("..", ""):
        raise ValueError(text)
    start, end = one(parts[0]), one(parts[1])
    if start is not None and end is not None and start > end:
        raise ValueError(text)
    return start, end


def _openapi(root: str, title: str, description: str, collections: list, tile_layers: list,
             default_limit: int, max_limit: int) -> Dict[str, Any]:
    """A minimal OpenAPI 3.0 document for the service (OGC API – Features, /req/core/api-definition)."""
    def ok(desc: str, media: str = "application/json") -> Dict[str, Any]:
        return {"200": {"description": desc, "content": {media: {"schema": {"type": "object"}}}}}

    param = {
        "bbox": {"name": "bbox", "in": "query", "required": False, "style": "form", "explode": False,
                 "schema": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                 "description": "minLon,minLat,maxLon,maxLat (CRS84)"},
        "datetime": {"name": "datetime", "in": "query", "required": False, "schema": {"type": "string"},
                     "description": "RFC 3339 instant or interval (open ends as ..)"},
        "limit": {"name": "limit", "in": "query", "required": False,
                  "schema": {"type": "integer", "minimum": 1, "maximum": max_limit, "default": default_limit}},
        "offset": {"name": "offset", "in": "query", "required": False,
                   "schema": {"type": "integer", "minimum": 0, "default": 0}},
    }
    paths: Dict[str, Any] = {
        "/": {"get": {"summary": "Landing page", "operationId": "getLandingPage", "responses": ok("Links")}},
        "/api": {"get": {"summary": "This API definition", "operationId": "getAPI",
                         "responses": ok("OpenAPI document", OPENAPI_TYPE)}},
        "/conformance": {"get": {"summary": "Conformance classes", "operationId": "getConformance",
                                 "responses": ok("Conformance")}},
        "/collections": {"get": {"summary": "Feature collections", "operationId": "getCollections",
                                 "responses": ok("Collections")}},
    }
    for cid in collections:
        paths[f"/collections/{cid}"] = {"get": {"summary": f"The {cid} collection",
                                                "operationId": f"describe_{cid}", "responses": ok("Collection")}}
        paths[f"/collections/{cid}/items"] = {"get": {
            "summary": f"Features of {cid}", "operationId": f"getFeatures_{cid}",
            "parameters": list(param.values()), "responses": ok("Features", "application/geo+json")}}
        paths[f"/collections/{cid}/items/{{featureId}}"] = {"get": {
            "summary": f"One feature of {cid}", "operationId": f"getFeature_{cid}",
            "parameters": [{"name": "featureId", "in": "path", "required": True, "schema": {"type": "string"}}],
            "responses": ok("Feature", "application/geo+json")}}
    for name in tile_layers:
        paths[f"/tiles/{name}/{{z}}/{{x}}/{{y}}.png"] = {"get": {
            "summary": f"XYZ tile of {name}", "operationId": f"getTile_{name}",
            "parameters": [{"name": k, "in": "path", "required": True, "schema": {"type": "integer"}}
                           for k in ("z", "x", "y")],
            "responses": {"200": {"description": "PNG tile", "content": {"image/png": {}}},
                          "204": {"description": "Tile outside the raster"}}}}
    return {"openapi": "3.0.3", "info": {"title": title, "description": description, "version": "1.0"},
            "servers": [{"url": root}], "paths": paths}


def _bbox(frame: Any) -> list:
    bounds = frame.total_bounds
    return [float(v) if math.isfinite(v) else None for v in bounds]


def _set_query(query: str, **updates: Any) -> str:
    from urllib.parse import parse_qsl, urlencode

    params = dict(parse_qsl(query, keep_blank_values=True))
    params.update({k: str(v) for k, v in updates.items()})
    return urlencode(params)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require(module: str, package: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"geolattice.web.server needs {package}: pip install 'geolattice[web]'") from exc


__all__ = ["create_app", "serve", "start_server", "BackgroundServer", "CONFORMANCE"]
