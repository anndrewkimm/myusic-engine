# Myusic Engine

A private, local-first music intelligence pipeline for learning from Spotify listening
behavior and combining it with independently computed or lawfully sourced audio features.

The Spotify Extended Streaming History export is currently pending. Development therefore
starts with synthetic records and contracts; no real listening history or copyrighted audio
is required to build or test the foundation.

> [!IMPORTANT]
> Raw Spotify exports, account data, and audio files are private inputs. They are ignored by
> Git and must never be committed. Only synthetic fixtures, schemas, code, and aggregate
> outputs belong in this repository.

## Current milestone

Phase 0 establishes the local project, privacy boundary, normalized schemas, and testable
interfaces described in the [project brief](docs/project-brief.md). Later phases will add
history ingestion, identity resolution, lawful audio analysis, clustering, and ranking in
that order.

## Development setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

The initial ingestion code intentionally uses the Python standard library. Heavier audio and
machine-learning dependencies will be added as optional groups only when their corresponding
pipeline stages are implemented.

## When the Spotify export arrives

Keep the original JSON files or ZIP under `data/private/`, then run the local privacy-cleaning
step. The command accepts one JSON history file, a directory containing history files, or the
original export ZIP.

```powershell
python -m myusic_engine prepare-history `
  "data/private/spotify-export.zip" `
  --output-dir "data/processed/history" `
  --recommendation-config "configs/recommendation.yaml"
```

The command writes three ignored local files:

- `listening_events.jsonl`: normalized, deduplicated events with tracks, episodes, and unknown
  media explicitly labeled;
- `ingestion_report.json`: record counts, safe validation issues, and counts of sensitive raw
  fields removed (never their values).
- `user_track_affinity.jsonl`: track-level play, completion, skip, repeat, recency, signal
  coverage, and explainable heuristic-score fields.

Use `--strict` to stop on the first invalid record. Without it, malformed records are excluded
and summarized in the report so one unusual row does not discard an otherwise usable export.
Duration-based completion remains null until catalog metadata is available. A local JSON object
mapping track URIs to positive duration milliseconds can be supplied with `--duration-map`;
reason-based completion remains separately coverage-scored when duration is unavailable.

## Repository layout

```text
configs/          Versioned feature and recommendation settings
data/             Instructions only; private and generated data stay ignored
data_contracts/   JSON Schemas for boundaries between pipeline stages
docs/             Project brief and architecture decisions
src/              Installable `myusic_engine` Python package
tests/            Unit tests and clearly synthetic fixtures
```

## Project boundaries

- Listening history is behavioral data, not an audio source.
- Audio features are computed only from owned, licensed, or otherwise permitted inputs.
- Every feature retains its source, version, analyzed coverage, and confidence.
- Similarity, clustering, and personal preference ranking remain separate concerns.
- No private Spotify endpoints, protected-stream extraction, or consumer-client scraping.
