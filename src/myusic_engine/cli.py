"""Command-line entry points for local pipeline stages."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from myusic_engine.ingest import HistoryIngestionError, prepare_history


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myusic-engine",
        description="Build privacy-safe inputs for the personal music intelligence pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    history_parser = subparsers.add_parser(
        "prepare-history",
        help="Normalize a Spotify Extended Streaming History JSON, directory, or ZIP.",
    )
    history_parser.add_argument("source", type=Path, help="Private export JSON, directory, or ZIP")
    history_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored local directory for cleaned output",
    )
    history_parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid event instead of reporting and skipping it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command and return a process exit code."""

    args = _parser().parse_args(argv)
    if args.command == "prepare-history":
        try:
            result = prepare_history(args.source, args.output_dir, strict=args.strict)
        except (HistoryIngestionError, OSError) as exc:
            print(f"History preparation failed: {exc}", file=sys.stderr)
            return 2
        counts = result.report.media_counts
        print(
            "Prepared "
            f"{result.report.events_written} events "
            f"({counts.get('track', 0)} tracks, "
            f"{counts.get('episode', 0)} episodes, "
            f"{counts.get('unknown', 0)} unknown)."
        )
        print(
            f"Removed {result.report.duplicate_events_removed} duplicates; "
            f"rejected {result.report.records_rejected} invalid records."
        )
        print(f"Wrote privacy-cleaned output to {args.output_dir}")
        return 0
    return 2
