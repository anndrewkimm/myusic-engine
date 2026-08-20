# Myusic Engine

A private, local-first music intelligence pipeline for learning from Spotify listening
behavior and combining it with independently computed or lawfully sourced audio features.

The Spotify Extended Streaming History export is currently pending. Development therefore
uses synthetic records and permitted test signals; no real listening history or copyrighted
Spotify audio is required to build or test the foundation.

> [!IMPORTANT]
> Raw Spotify exports, account data, and audio files are private inputs. They are ignored by
> Git and must never be committed. Only synthetic fixtures, schemas, code, and aggregate
> outputs belong in this repository.

## Current milestone

The data-independent foundation from the [project brief](docs/project-brief.md) is now usable:

- privacy-safe Spotify history ingestion from JSON, directories, or ZIP archives;
- deterministic track behavior aggregation and versioned affinity scoring;
- strict, source-tagged records for numeric, text, and vector audio features;
- a strict manifest for owned, licensed, public-domain, or Creative Commons audio;
- objective DSP descriptors including tempo, key/mode, loudness, bass energy, chroma, and
  spectral/timbre measurements;
- verified cross-platform ONNX inference for 1,280-dimensional Discogs-EffNet embeddings;
- six versioned learned scores for danceable, acoustic, instrumental, happy, aggressive, and
  relaxed audio characteristics;
- exact cosine retrieval with weighted multi-seed queries and provenance-aware filters.

The pending export is not needed for the test suite. Phase 1 is code-complete but still needs
real-export validation. Phase 2 identity resolution is unstarted. Phase 3 now has a tested
foundation, but its exit criterion still requires transformation tests and quality evaluation on
a permitted real music corpus. See [current project status](docs/project-status.md).

## Development setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Install the optional phase-3 stack when extracting real audio:

```powershell
python -m pip install -e ".[dev,phase3]"
```

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

No unpacking or renaming is required: `prepare-history` discovers every recognized history JSON
inside the ZIP and ignores unrelated account files. The detailed handoff and validation checklist
is in [Spotify export handoff](docs/spotify-export-handoff.md).

## Analyze permitted audio

The Spotify export identifies listening behavior but contains no audio. Build a private ignored
`audio_manifest.jsonl` that maps stable IDs (preferably Spotify track URIs after identity review)
to audio that you are allowed to analyze:

```json
{"track_id":"spotify:track:0000000000000000000000","audio_path":"owned/track.wav","rights_basis":"owned"}
```

Download the pinned embedding model after reviewing its CC BY-NC-SA 4.0 license, then run both
objective descriptors, embeddings, and the optional learned score pack:

```powershell
python -m myusic_engine download-embedding-model --accept-noncommercial-license
python -m myusic_engine download-feature-head-models --accept-noncommercial-license
python -m myusic_engine analyze-audio `
  "data/private/audio_manifest.jsonl" `
  --output "data/processed/audio/features.jsonl" `
  --feature-config "configs/features.yaml" `
  --feature-head-model-dir "artifacts/models/feature-heads" `
  --window-output-dir "data/interim/embedding-windows"
```

The aggregate embedding is directly compatible with the existing cosine similarity index.
Window-level vectors remain private and available for later robust or section-aware pooling. Exact
definitions, limitations, and validation requirements are in the
[phase-3 feature/model card](docs/audio-features-and-embeddings.md).
The clean-room mapping to Spotify's published concepts, explicit non-parity boundaries, and
calibration plan are in [Spotify-like feature research](docs/spotify-like-feature-research.md).

## Why this is a data-science project

The goal is not to clone Spotify's proprietary recommender. It is to build an auditable personal
system and demonstrate the full applied-data workflow: privacy-safe ingestion, entity resolution,
feature engineering, representation learning, implicit-feedback labels, leakage-safe evaluation,
retrieval, ranking, ablations, and human review. A strong result is a reproducible improvement over
simple popularity, artist, and behavior baselines—not an unsupported claim of Spotify parity.
The concrete research questions, baselines, temporal split, and portfolio deliverables are laid out
in [data-science value and evaluation](docs/data-science-value-and-evaluation.md).

## Repository layout

```text
configs/          Versioned feature and recommendation settings
data/             Instructions only; private and generated data stay ignored
data_contracts/   JSON Schemas for boundaries between pipeline stages
docs/             Project brief and architecture decisions
src/              Installable `myusic_engine` Python package
tests/            Unit tests and clearly synthetic fixtures
```

The scoring assumptions are documented in [behavioral affinity v1](docs/behavior-affinity.md),
and the retrieval contract is documented in [weighted multi-seed similarity](docs/similarity.md).

## Project boundaries

- Listening history is behavioral data, not an audio source.
- Audio features are computed only from owned, licensed, or otherwise permitted inputs.
- Every feature retains its source, version, analyzed coverage, and confidence.
- Similarity, clustering, and personal preference ranking remain separate concerns.
- No private Spotify endpoints, protected-stream extraction, or consumer-client scraping.
