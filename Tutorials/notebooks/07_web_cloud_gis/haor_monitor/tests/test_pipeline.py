"""Pipeline tests with a fake catalogue: no network, so they run anywhere, including CI."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from haor_monitor.config import Settings
from haor_monitor.pipeline import load_table, run


def fake_items(days):
    """Two neighbouring tiles per overpass, like the real catalogue."""
    return [
        SimpleNamespace(
            id=f"S2_{tile}_{d:%Y%m%d}",
            datetime=datetime(d.year, d.month, d.day, 4, 40, tzinfo=timezone.utc),
            properties={"s2:processing_baseline": "05.10"},
        )
        for d in days
        for tile in ("A", "B")
    ]


ALL_DAYS = [date(2024, 1, 5), date(2024, 1, 15), date(2024, 2, 4), date(2024, 3, 5)]


def fake_search(settings, start, end):
    return fake_items([d for d in ALL_DAYS if start <= d <= end])


def fake_load(items, settings):
    water = 3 if items[0].datetime.month < 3 else 6
    scl = np.full((10, 10), 4, "uint8")
    scl[:water] = 6
    nir = np.full((10, 10), 0.4, "float32")
    nir[:water] = 0.02
    return {
        "green": np.full((10, 10), 0.06, "float32"),
        "red": np.full((10, 10), 0.04, "float32"),
        "nir": nir,
        "scl": scl,
        "baseline": "05.10",
        "scenes": [i.id for i in items],
    }


def test_run_is_incremental_and_idempotent(tmp_path):
    settings = Settings(data_dir=tmp_path)
    first = run(settings, date(2024, 1, 1), date(2024, 1, 31), search=fake_search, load=fake_load)
    assert first["processed"] == 2 and first["rows_total"] == 2
    again = run(settings, date(2024, 1, 1), date(2024, 1, 31), search=fake_search, load=fake_load)
    assert again["processed"] == 0 and again["rows_total"] == 2  # nothing duplicated
    later = run(settings, date(2024, 1, 1), date(2024, 3, 31), search=fake_search, load=fake_load)
    assert later["days_found"] == 4 and later["processed"] == 2 and later["rows_total"] == 4
    table = load_table(settings)
    assert list(pd.to_datetime(table["date"]).dt.date) == ALL_DAYS
    assert (table["scenes"] == 2).all()  # two tiles mosaicked per day
    assert table.loc[3, "water_fraction"] > table.loc[0, "water_fraction"]


def test_a_failing_day_does_not_stop_the_run(tmp_path):
    def flaky_load(items, settings):
        if items[0].datetime.day == 15:
            raise OSError("network timeout")
        return fake_load(items, settings)

    summary = run(
        Settings(data_dir=tmp_path), date(2024, 1, 1), date(2024, 3, 31), search=fake_search, load=flaky_load
    )
    assert summary["failed"] == ["2024-01-15"] and summary["processed"] == 3
    retry = run(
        Settings(data_dir=tmp_path), date(2024, 1, 1), date(2024, 3, 31), search=fake_search, load=fake_load
    )
    assert retry["processed"] == 1 and retry["rows_total"] == 4  # the retry fills the gap


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HAOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HAOR_MAX_CLOUD", "25")
    monkeypatch.setenv("HAOR_AOI", "91.0,25.0,91.1,25.1")
    s = Settings.from_env()
    assert s.data_dir == tmp_path and s.max_cloud == 25.0 and s.aoi == (91.0, 25.0, 91.1, 25.1)
