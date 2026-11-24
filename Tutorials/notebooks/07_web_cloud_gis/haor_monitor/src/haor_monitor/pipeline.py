"""The monitoring job: find new observations, process them, append to the table, log the run.

Idempotent: dates already in ``observations.parquet`` are skipped, so the job can run on any
schedule, be retried after a failure, or run twice by accident without duplicating data.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import stac
from .config import Settings
from .process import quality_flags, scene_stats

log = logging.getLogger("haor_monitor")
COLUMNS = [
    "date",
    "scenes",
    "baseline",
    "pixels",
    "clear_fraction",
    "water_fraction",
    "ndvi_land_median",
    "flags",
    "processed_at",
]


class JsonFormatter(logging.Formatter):
    """One JSON object per log line: easy to search and aggregate in any log service."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def setup_logging(path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
    log.handlers[:] = handlers
    log.setLevel(logging.INFO)
    log.propagate = False


def table_path(settings: Settings) -> Path:
    return settings.data_dir / "observations.parquet"


def load_table(settings: Settings) -> pd.DataFrame:
    path = table_path(settings)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=COLUMNS)


def run(settings: Settings, start: date, end: date, *, search=stac.search, load=stac.load_day) -> dict:
    """Process every new daily observation between start and end; return a run summary."""
    run_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    table = load_table(settings)
    done = set(pd.to_datetime(table["date"]).dt.date) if len(table) else set()
    days = stac.group_by_day(search(settings, start, end))
    new = {d: items for d, items in days.items() if d not in done}
    log.info(
        "run started",
        extra={
            "fields": {
                "run_id": run_id,
                "start": str(start),
                "end": str(end),
                "days_found": len(days),
                "days_new": len(new),
            }
        },
    )
    rows, failed = [], []
    for day, items in new.items():
        try:
            bands = load(items, settings)
            stats = scene_stats(bands["green"], bands["red"], bands["nir"], bands["scl"])
            flags = quality_flags(stats, settings.min_clear)
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "scenes": len(items),
                    "baseline": bands["baseline"],
                    **stats,
                    "flags": ";".join(flags),
                    "processed_at": pd.Timestamp.now(tz="UTC"),
                }
            )
            log.info(
                "day processed",
                extra={"fields": {"run_id": run_id, "date": str(day), **stats, "flags": flags}},
            )
        except Exception as exc:  # one bad day must not stop the run
            failed.append(str(day))
            log.error(
                "day failed",
                extra={
                    "fields": {"run_id": run_id, "date": str(day), "error": f"{type(exc).__name__}: {exc}"}
                },
            )
    if rows:
        new_rows = pd.DataFrame(rows, columns=COLUMNS)
        table = new_rows if table.empty else pd.concat([table, new_rows], ignore_index=True)
        table = table.sort_values("date").reset_index(drop=True)
        tmp = table_path(settings).with_suffix(".tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(table_path(settings))  # atomic: readers never see a half-written file
    summary = {
        "run_id": run_id,
        "days_found": len(days),
        "days_new": len(new),
        "processed": len(rows),
        "failed": failed,
        "rows_total": len(table),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log.info("run finished", extra={"fields": summary})
    return summary
