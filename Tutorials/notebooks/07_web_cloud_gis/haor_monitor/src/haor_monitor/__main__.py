"""Command line: ``python -m haor_monitor run --start 2024-01-01 --end 2024-06-30`` or ``... report``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

from .config import Settings
from .pipeline import load_table, run, setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="haor-monitor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="process new observations")
    r.add_argument("--start", type=date.fromisoformat, help="first date (default: 30 days before --end)")
    r.add_argument("--end", type=date.fromisoformat, default=date.today(), help="last date (default: today)")
    sub.add_parser("report", help="print a summary of the observations table")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    setup_logging(settings.data_dir / "runs.jsonl")
    if args.command == "run":
        start = args.start or args.end - timedelta(days=30)
        summary = run(settings, start, args.end)
        return 1 if summary["failed"] else 0  # a non-zero exit code makes the scheduler alert
    table = load_table(settings)
    usable = table[table["flags"] == ""] if len(table) else table
    print(
        json.dumps(
            {
                "observations": len(table),
                "usable": len(usable),
                "first": str(table["date"].min())[:10] if len(table) else None,
                "last": str(table["date"].max())[:10] if len(table) else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
