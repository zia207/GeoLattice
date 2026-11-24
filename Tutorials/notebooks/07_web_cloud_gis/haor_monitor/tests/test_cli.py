"""Command-line, grouping and logging tests (no network)."""

import json
from datetime import date

import pandas as pd

from haor_monitor import __main__ as cli
from haor_monitor import pipeline
from haor_monitor.stac import group_by_day
from tests.test_pipeline import ALL_DAYS, fake_items, fake_load, fake_search


def test_group_by_day_mosaics_tiles_of_one_overpass():
    days = group_by_day(fake_items(ALL_DAYS))
    assert list(days) == ALL_DAYS and all(len(v) == 2 for v in days.values())


def test_cli_run_and_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HAOR_DATA_DIR", str(tmp_path))
    real_run = pipeline.run
    monkeypatch.setattr(cli, "run", lambda s, a, b: real_run(s, a, b, search=fake_search, load=fake_load))
    assert cli.main(["run", "--start", "2024-01-01", "--end", "2024-03-31"]) == 0
    assert cli.main(["report"]) == 0
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report == {"observations": 4, "usable": 4, "first": "2024-01-05", "last": "2024-03-05"}


def test_cli_exit_code_signals_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("HAOR_DATA_DIR", str(tmp_path))

    def broken_load(items, settings):
        raise OSError("catalogue unavailable")

    monkeypatch.setattr(
        cli, "run", lambda s, a, b: pipeline.run(s, a, b, search=fake_search, load=broken_load)
    )
    assert cli.main(["run", "--start", "2024-01-01", "--end", "2024-01-31"]) == 1


def test_run_log_is_json_lines(monkeypatch, tmp_path):
    monkeypatch.setenv("HAOR_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "run", lambda s, a, b: pipeline.run(s, a, b, search=fake_search, load=fake_load))
    cli.main(["run", "--start", "2024-01-01", "--end", "2024-01-31"])
    records = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [r["message"] for r in records] == [
        "run started",
        "day processed",
        "day processed",
        "run finished",
    ]
    assert records[-1]["processed"] == 2 and records[0]["days_new"] == 2
    assert pd.Timestamp(records[0]["time"]).date() == date.today()
