"""Python download API for the PyGIS Explorer catalogues."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from geolattice.rsense import get_images as gi


class _Scene:
    def __init__(self, scene_id, cloud, platform="sentinel-2a"):
        self.id = scene_id
        self.datetime = datetime(2024, 3, 2, tzinfo=timezone.utc)
        self.properties = {"eo:cloud_cover": cloud, "platform": platform}


class _Stac:
    def __init__(self, items=None):
        self.calls = []
        self.items = items if items is not None else [_Scene("cloudy", 40.0), _Scene("clear", 3.0)]

    def stac_search(self, *args):
        self.calls.append(args)
        return list(self.items)

    def stac_download(self, items, source, collection, aoi, bands, mode, out, stack, workers):
        self.calls.append(
            ("download", [item.id for item in items], list(bands), mode, stack, workers)
        )
        scene = out / "clear"
        scene.mkdir(parents=True, exist_ok=True)
        (scene / "red.tif").write_bytes(b"tif")
        return [("cloudy", "blue", "missing")] if any(item.id == "cloudy" for item in items) else []

    def retry(self, fn, **_kwargs):
        return fn()


BBOX = (-76.6, 42.4, -76.4, 42.5)
DATES = ("2024-03-01", "2024-03-31")


def test_lazy_module_name():
    import geolattice.rsense as rs

    assert rs.get_images.__name__ == "geolattice.rsense.get_images"


def test_collections_and_default_bands():
    assert gi.list_collections("usgs") == ["lidar-lpc", "lidar-projects"]
    assert "sentinel-2-l2a" in gi.list_collections("earth-search")
    assert gi.default_bands("sentinel-2-l2a") == ["red", "green", "blue", "nir", "scl"]
    assert gi.default_bands("modis-13Q1-061", "planetary") == ["ndvi", "evi", "red", "nir"]
    assert gi.default_bands("sentinel-1-rtc", "planetary", s1_polarization="HH") == ["hh"]
    assert gi.default_bands("noaa-rgb8", "noaa") == []
    assert "occurrence" in gi.bands_for("jrc-gsw")
    with pytest.raises(ValueError, match="source"):
        gi.list_collections("google")
    assert gi.list_collections("rcm-ard") == ["rcm-ard"]
    assert gi.default_bands("rcm-ard", "rcm-ard") == ["rr", "rl", "data_mask"]
    assert set(gi.bands_for("rcm-ard")) >= {"rr", "rl", "rrrl", "data_mask"}


def test_load_aoi_bbox_and_geojson(tmp_path):
    area = gi.load_aoi(bbox="-76.6, 42.4, -76.4, 42.5")
    assert area.bbox == BBOX
    path = tmp_path / "aoi.geojson"
    path.write_text(
        '{"type":"Polygon","coordinates":[[[-76.6,42.4],[-76.4,42.4],'
        "[-76.4,42.5],[-76.6,42.5],[-76.6,42.4]]]}"
    )
    from_file = gi.load_aoi(aoi=path)
    assert from_file.bbox == pytest.approx(BBOX)
    assert "POLYGON" in from_file.wkt()
    with pytest.raises(ValueError, match="not both"):
        gi.load_aoi(bbox=BBOX, aoi=path)
    with pytest.raises(ValueError, match="Pass bbox"):
        gi.load_aoi()
    with pytest.raises(FileNotFoundError):
        gi.load_aoi(aoi=tmp_path / "missing.shp")
    with pytest.raises(ValueError, match="positive area"):
        gi.load_aoi(bbox=(-76.0, 42.0, -76.5, 42.5))


def test_load_aoi_shapefile_and_geodataframe(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    lonlat = gpd.GeoDataFrame(geometry=[box(*BBOX)], crs="EPSG:4326")
    shp = tmp_path / "field.shp"
    lonlat.to_file(shp)
    loaded = gi.load_aoi(aoi=shp)
    assert loaded.bbox == pytest.approx(BBOX)

    projected = gpd.GeoDataFrame(
        geometry=[box(400_000, 4_700_000, 401_000, 4_701_000)], crs="EPSG:32618"
    )
    projected.to_file(tmp_path / "utm.geojson")
    utm = gi.load_aoi(aoi=tmp_path / "utm.geojson")
    assert -78 < utm.bbox[0] < -74
    assert 42 < utm.bbox[1] < 43

    memory = gi.load_aoi(aoi=lonlat)
    assert memory.bbox == pytest.approx(BBOX)
    shape = gi.load_aoi(aoi=box(*BBOX))
    assert shape.bbox == pytest.approx(BBOX)
    with pytest.raises(ValueError, match="longitude"):
        gi.load_aoi(aoi=box(400_000, 4_700_000, 401_000, 4_701_000))


def test_rejects_bad_requests():
    with pytest.raises(ValueError, match="dates"):
        gi.search_images(bbox=BBOX, source="earth-search", quiet=True)
    with pytest.raises(ValueError, match="after end"):
        gi.search_images(bbox=BBOX, dates=("2024-04-01", "2024-03-01"), quiet=True)
    with pytest.raises(ValueError, match="Level-2"):
        gi.search_images(
            bbox=BBOX,
            dates=("2000-01-01", "2005-01-01"),
            source="earth-search",
            collection="landsat-c2-l2",
            landsat_product="l1tp",
            quiet=True,
        )
    with pytest.raises(ValueError, match="Unknown noaa"):
        gi.search_images(bbox=BBOX, dates=DATES, source="noaa", collection="noaa-rgb", quiet=True)
    with pytest.raises(ValueError, match="product type"):
        gi.search_images(
            bbox=BBOX,
            dates=DATES,
            source="cdse",
            collection="SENTINEL-2",
            product_type="NOT_A_TYPE",
            quiet=True,
        )


def test_search_passes_explorer_filters(monkeypatch):
    fake = _Stac()
    monkeypatch.setattr(gi, "_explorer", lambda: fake)
    found = gi.search_images(
        bbox=BBOX,
        dates=DATES,
        source="planetary",
        collection="landsat-c2-l2",
        landsat_product="l1tp",
        platforms="Landsat 5",
        quiet=True,
    )
    assert found.collection == "landsat-c2-l1"
    assert found.scenes[0]["id"] == "cloudy"
    source, collection, _area, start, end, cloud, limit, platforms, extra = fake.calls[0]
    assert (source, collection, start, end, cloud, limit) == (
        "planetary",
        "landsat-c2-l1",
        "2024-03-01",
        "2024-03-31",
        None,
        50,
    )
    assert platforms == ["landsat-5"]
    assert extra["landsat:correction"] == {"eq": "L1TP"}


def test_sentinel1_and_clearest_download(monkeypatch, tmp_path):
    fake = _Stac()
    monkeypatch.setattr(gi, "_explorer", lambda: fake)
    found = gi.search_images(
        aoi=None,
        bbox=BBOX,
        dates=DATES,
        source="planetary",
        collection="sentinel-1-rtc",
        s1_mode="IW",
        s1_polarization="VV+VH",
        s1_orbit="descending",
        quiet=True,
    )
    extra = fake.calls[0][8]
    assert extra["sar:instrument_mode"] == {"eq": "IW"}
    assert extra["sar:polarizations"] == {"eq": ["VV", "VH"]}
    assert extra["sat:orbit_state"] == {"eq": "descending"}

    saved = gi.get_images(
        tmp_path / "s2", search=found, clearest=True, bands=["vv", "vh"], quiet=True
    )
    download = fake.calls[-1]
    assert download[0] == "download"
    assert download[1] == ["clear"]
    assert download[2] == ["vv", "vh"]
    assert saved.mode == "clip"
    assert (saved.out / "aoi.geojson").is_file()
    assert (saved.out / "request.json").is_file()
    assert any(path.name == "red.tif" for path in saved.files)
    assert saved.scenes[0]["cloud"] == 3.0


class _RcmScene:
    def __init__(self, scene_id, orbit, day):
        self.id = scene_id
        self.datetime = datetime(2025, 1, day, tzinfo=timezone.utc)
        self.properties = {
            "sat:orbit_state": orbit, "title": scene_id, "platform": "RCM2",
            # no eo:cloud_cover -- RCM-ARD is SAR, "clearest" must still degrade gracefully
        }


def test_rcm_ard_orbit_filter_and_download(monkeypatch, tmp_path):
    scenes = [
        _RcmScene("asc-early", "Ascending", 5),
        _RcmScene("desc-mid", "Descending", 10),
        _RcmScene("asc-late", "Ascending", 15),
    ]
    fake = _Stac(items=scenes)
    monkeypatch.setattr(gi, "_explorer", lambda: fake)

    # The EODMS STAC API does not support the `query` extension, so the orbit filter has to
    # be applied client-side, not as a server-side sat:orbit_state query -- confirm the mock
    # (which ignores query params entirely, like the real server) still ends up orbit-filtered.
    found = gi.search_rcm_ard(bbox=BBOX, dates=DATES, orbit="ascending", quiet=True)
    assert [s["id"] for s in found.scenes] == ["asc-early", "asc-late"]
    assert found.source == "rcm-ard" and found.collection == "rcm-ard"

    unfiltered = gi.search_rcm_ard(bbox=BBOX, dates=DATES, quiet=True)
    assert len(unfiltered.scenes) == 3

    with pytest.raises(ValueError, match="orbit"):
        gi.search_rcm_ard(bbox=BBOX, dates=DATES, orbit="sideways", quiet=True)

    # clearest has no cloud-cover meaning for SAR; it should still pick one scene, not crash
    one = gi.search_rcm_ard(bbox=BBOX, dates=DATES, orbit="ascending", clearest=True, quiet=True)
    assert len(one.scenes) == 1 and one.scenes[0]["id"] in ("asc-early", "asc-late")

    saved = gi.get_rcm_ard(tmp_path / "rcm", bbox=BBOX, dates=DATES, orbit="ascending", quiet=True)
    download = fake.calls[-1]
    assert download[0] == "download"
    assert sorted(download[1]) == ["asc-early", "asc-late"]
    assert download[2] == ["rr", "rl", "data_mask"]  # RCM_ARD_DEFAULT_BANDS
    assert saved.source == "rcm-ard" and saved.mode == "clip"


def test_ids_and_naip_dates_optional(monkeypatch):
    fake = _Stac()
    monkeypatch.setattr(gi, "_explorer", lambda: fake)

    def naip_search(area, limit):
        fake.calls.append(("naip", limit, area.bbox))
        return [_Scene("naip-2022", None)]

    fake.naip_search = naip_search
    found = gi.search_images(
        bbox=BBOX, source="earth-search", collection="naip", ids=["naip-2022"], quiet=True
    )
    assert found.collection == "naip"
    assert [scene["id"] for scene in found.scenes] == ["naip-2022"]
    assert found.notes == []


def test_noaa_download_note(monkeypatch, tmp_path):
    class _NOAA:
        def search(self, collection, area, start, end, limit):
            return [
                {
                    "id": "mission-2016",
                    "properties": {"title": "Charleston", "datetime": "2016-01-01T00:00:00Z"},
                    "noaa": {"name": "Charleston"},
                }
            ]

        def download(self, scene, area, out):
            path = out / "charleston.tif"
            path.write_bytes(b"tif")
            return path, "resampled to about 2 m"

    class _Explorer:
        def NOAA(self):
            return _NOAA()

        def retry(self, fn, **_kwargs):
            return fn()

    monkeypatch.setattr(gi, "_explorer", lambda: _Explorer())
    saved = gi.get_images(
        tmp_path / "coast",
        bbox=BBOX,
        dates=("2016-01-01", "2020-12-31"),
        source="noaa",
        collection="noaa-rgb8",
        quiet=True,
    )
    assert saved.mode == "full"
    assert saved.scenes[0]["title"] == "Charleston"
    assert saved.notes == ["resampled to about 2 m"]
    assert (tmp_path / "coast" / "charleston.tif").is_file()


def test_system_exit_becomes_runtime_error():
    def _exit():
        raise SystemExit("Set CDSE_USER and CDSE_PASSWORD")

    with pytest.raises(RuntimeError, match="CDSE_USER"):
        gi._call(_exit)


def test_explorer_script_matches_band_tables():
    pytest.importorskip("requests")
    downloader = gi._explorer()
    assert gi.DEFAULT_COLLECTION == downloader.DEFAULT_COLLECTION
    assert gi.PLANETARY_BANDS == downloader.PLANETARY_BANDS
    assert gi.PLANETARY_DEFAULT_BANDS == downloader.PLANETARY_DEFAULT_BANDS
    assert gi._explorer_path().name in {"image_downloader.py", "_image_downloader.py"}
    bundled = Path(gi.__file__).resolve().with_name("_image_downloader.py")
    assert bundled.is_file()
    live = next(
        (
            parent / "PyGIS-Explorer" / "image_downloader.py"
            for parent in Path(gi.__file__).resolve().parents
            if (parent / "PyGIS-Explorer" / "image_downloader.py").is_file()
        ),
        None,
    )
    if live is not None:
        assert bundled.read_bytes() == live.read_bytes()
