"""Join Spotify history to published MAEST audio embeddings without local music files.

Run from the repository root. --download fetches pinned public research tables only;
the listening history and recording matches remain local.
"""

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from myusic_engine.features.music4all import FILES, SOURCE_VERSION, import_maest_features
from myusic_engine.features.music4all import verify_source_file as verify
from myusic_engine.features.records import write_feature_observations
from myusic_engine.io import atomic_write_text
from myusic_engine.ranking.candidates import CandidateTrack


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, size, digest) in FILES.items():
        path = destination / name
        if path.is_file():
            try:
                verify(path, size, digest)
                continue
            except ValueError:
                pass
        descriptor, temporary_name = tempfile.mkstemp(dir=destination, suffix=".download")
        temporary = Path(temporary_name)
        print(f"Downloading {name} ({size / 1e6:.1f} MB)", flush=True)
        try:
            with (
                os.fdopen(descriptor, "wb") as stream,
                urllib.request.urlopen(url, timeout=60) as response,
            ):
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError("Download exceeds pinned size")
                    stream.write(chunk)
            verify(temporary, size, digest)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", type=Path, default=Path("data/processed/history/user_track_affinity.jsonl")
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/music4all-onion"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/audio/music4all-history")
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.source_dir)
    tracks = []
    with args.history.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            tracks.append(
                CandidateTrack(
                    track_id=row["track_id"],
                    spotify_uri=row.get("track_uri"),
                    track_name=row.get("track_name"),
                    artist_name=row.get("artist_name"),
                    album_name=row.get("album_name"),
                )
            )
    result = import_maest_features(tracks, args.source_dir)
    if not result.observations:
        raise ValueError("No corroborated audio embeddings matched this history")
    report = {
        "schema_version": 1,
        "source_version": SOURCE_VERSION,
        "dataset": "https://zenodo.org/records/15394646",
        "license": "CC-BY-4.0",
        "source_sha256": {name: spec[2] for name, spec in FILES.items()},
        "counts": result.counts,
        "limitations": [
            "Historical catalog; does not establish coverage for current new releases.",
            "30-second research excerpts; upstream audio matches are not independently audited.",
            "Encoder training chronology is not audited; evaluation is a retrospective experiment.",
            "Feature confidence is an excerpt-coverage convention, not match accuracy.",
            "Legacy Spotify audio-feature columns are not imported or used for training.",
        ],
    }
    write_feature_observations(result.observations, args.output_dir / "features.jsonl")
    atomic_write_text(
        args.output_dir / "identity_matches.jsonl",
        "".join(json.dumps(row) + "\n" for row in result.matches),
    )
    atomic_write_text(args.output_dir / "coverage_report.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(result.counts, sort_keys=True))


if __name__ == "__main__":
    main()
