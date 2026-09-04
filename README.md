# Myusic Engine

A private, local-first music intelligence pipeline for learning from Spotify listening
behavior and combining it with independently computed or lawfully sourced audio features.

Both the compact Spotify Account Data export and the richer Extended Streaming History export have
been validated locally. No copyrighted Spotify audio is required to build or test the foundation.

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
- exact cosine retrieval with weighted multi-seed queries and provenance-aware filters;
- cached, precision-first MusicBrainz mapping and frozen CC0 AcousticBrainz feature conversion,
  each with a live and a fully offline dump-scanning route;
- leakage-safe 90-day taste labels with whole-period train/validation/test splits;
- behavior, descriptor, embedding, and combined logistic ablations on matched audio cohorts;
- standardized K-Means/HDBSCAN taste maps with stability evidence and PCA coordinates;
- explainable candidate ranking, diversity controls, Spotify URI handoff, and feedback logging.

Phase 1 has now been validated against both Spotify export formats. Private EDA confirmed lifetime
coverage, URI-backed track affinities, the expected Extended History playback signals, and a clean
ingestion report. Phase 2 now has an offline account-catalog resolver and an offline canonical-dump
MusicBrainz mapper; live-provider coverage and manual match validation remain. A private
behavior-only chronological model run is complete, and a private descriptor-audio ablation has now
run on real offline-covered history — behavior alone remains the selected model there too.
Embedding ablations and the real taste map remain gated on lawful embedding coverage. Phase 3
has a tested foundation, but its exit criterion still requires quality evaluation on a permitted
real music corpus. See
[current project status](docs/project-status.md).

## Development setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest --cov=myusic_engine --cov-fail-under=75
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

The same gates run on Python 3.11 and 3.14 in GitHub Actions; the current runtime also builds the
wheel so packaging regressions fail before merge.

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
inside the ZIP and ignores unrelated account files. Both Extended Streaming History and the
compact Account Data `StreamingHistory_music_*` / `StreamingHistory_podcast_*` formats are
accepted. Compact Account Data does not include Spotify URIs or the richer playback signals, so
its track identities fall back to deterministic metadata hashes. The detailed handoff and
validation checklist is in [Spotify export handoff](docs/spotify-export-handoff.md).

## Resolve metadata-only track identities

The compact Account Data ZIP includes Spotify URIs for saved and playlist tracks even though its
streaming-history rows do not. Resolve safe exact matches locally and produce explicit fuzzy,
ambiguous, and unmatched queues:

```powershell
python -m myusic_engine resolve-identities `
  "data/processed/history/user_track_affinity.jsonl" `
  "data/private/my_spotify_data.zip" `
  --output-dir "data/interim/identity" `
  --matching-config "configs/identity_resolution.yaml"
```

Exact unique matches receive a stable URI. Fuzzy and ambiguous candidates remain unresolved and
reviewable; the resolver never silently promotes them to ground truth. See
[offline identity resolution](docs/identity-resolution.md) for the policy and output contracts.

## Obtain lawful historical descriptors

The Extended History contains song identities and behavior, not audio samples. With explicit
permission to send title, artist, and album metadata to public lookup services, map the highest
precision MusicBrainz identities and fetch the frozen CC0 AcousticBrainz descriptors:

```powershell
python -m myusic_engine map-musicbrainz `
  "data/processed/history/user_track_affinity.jsonl" `
  --output-dir "data/interim/musicbrainz" `
  --cache-dir "data/raw/provider-cache"

python -m myusic_engine fetch-acousticbrainz `
  "data/interim/musicbrainz/external_identity_matches.jsonl" `
  --output-dir "data/processed/audio/acousticbrainz" `
  --cache-dir "data/raw/provider-cache"
```

The same fallback can be scoped to one imported playlist instead of the full history:

```powershell
python -m myusic_engine map-musicbrainz `
  "data/processed/candidates/my-playlist/candidates.jsonl" `
  --input-kind candidates `
  --output-dir "data/interim/musicbrainz/my-playlist" `
  --cache-dir "data/raw/provider-cache"
```

Only strict exact mapper results receive MBIDs. Fuzzy and unmatched results remain review-only.
The mapper cache is resumable and `--offline` forbids network requests. AcousticBrainz supplies
lawful historical descriptors and learned scores, but it does not replace a deep track embedding.

Mapping can also run entirely offline against an official canonical MusicBrainz dump instead of
calling the live ListenBrainz Labs API. The whole dump is scanned locally and no title/artist/album
metadata ever leaves the machine, so this path does not require the same explicit per-run
disclosure decision as the live mapper:

```powershell
python -m myusic_engine map-musicbrainz `
  "data/processed/history/user_track_affinity.jsonl" `
  --output-dir "data/interim/musicbrainz" `
  --canonical-dump "data/raw/musicbrainz-canonical-dump-20260903-080002.tar.zst"
```

Download the dated `.tar.zst` release and its `.sha256` checksum from
[MusicBrainz's canonical data dumps](https://data.metabrainz.org/pub/musicbrainz/canonical_data/),
published twice monthly; a plain `.csv` or `.tar` is also accepted. This path applies the same
strict exact-match policy as the live mapper and ignores `--cache-dir`/`--offline` since nothing is
cached or fetched over the network.

`fetch-acousticbrainz` has the same kind of offline alternative. AcousticBrainz's own low-level
dataset was frozen in 2022 and published as three derived CSV dumps (lowlevel, rhythm, tonal); a
local scan of any subset of them never sends a recording MBID over the network:

```powershell
python -m myusic_engine fetch-acousticbrainz `
  "data/interim/musicbrainz/history-local-canonical/external_identity_matches.jsonl" `
  --output-dir "data/processed/audio/acousticbrainz" `
  --lowlevel-dump "data/raw/acousticbrainz-lowlevel-features-20220623-lowlevel.tar.zst" `
  --rhythm-dump "data/raw/acousticbrainz-lowlevel-features-20220623-rhythm.tar.zst" `
  --tonal-dump "data/raw/acousticbrainz-lowlevel-features-20220623-tonal.tar.zst"
```

Download the three dated dumps from
[AcousticBrainz's dataset dumps](https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/).
Only the fields present in these CSVs are populated (loudness, dynamic complexity, tempo,
danceability, onset rate, key/mode, and tuning), so the combined low-level descriptor vector is
never emitted from this path, and it never fills in high-level mood/genre predictions, which are
not part of this dataset — both stay honestly absent rather than guessed. It applies the same
"lowest submission offset wins" tie-break as the live client. Any of the three dumps can be
omitted; coverage narrows accordingly instead of failing.

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

## Train and evaluate personal taste

Construct labels from actual later listening outcomes. Predictors are frozen before each target
period, unplayed tracks are omitted rather than called dislikes, preprocessing fits on training
only, and model selection never reads the final test period:

```powershell
python -m myusic_engine build-taste-dataset `
  "data/processed/history/listening_events.jsonl" `
  --output-dir "data/processed/modeling" `
  --modeling-config "configs/modeling.yaml"

# Available immediately from history alone.
python -m myusic_engine train-taste-model `
  "data/processed/modeling/temporal_taste_samples.jsonl" `
  --output-dir "data/processed/models/behavior" `
  --modeling-config "configs/modeling.yaml"

# Repeat after permitted local audio has produced descriptors and deep embeddings.
python -m myusic_engine train-taste-model `
  "data/processed/modeling/temporal_taste_samples.jsonl" `
  --features "data/processed/audio/features.jsonl" `
  --profile local_discogs_effnet `
  --output-dir "data/processed/models/audio" `
  --modeling-config "configs/modeling.yaml"
```

Each fitted model is a content-hashed JSON artifact containing its scaler, coefficients,
point-in-time boundary, and exact feature/profile versions. Held-out predictions and aggregate
ablation metrics stay under ignored private paths. See
[taste modeling and recommendation](docs/taste-modeling-and-recommendation.md).

## Build the taste map and rank candidates

The behavior model can rank immediately without any audio files:

If the candidate songs are already in a playlist captured by the private Spotify Account Data
export, import that playlist directly. This preserves its track metadata and requires neither
manual CSV entry nor OAuth:

```powershell
python -m myusic_engine import-account-playlist `
  "data/private/my_spotify_data.zip" `
  --playlist-name "My playlist" `
  --output-dir "data/processed/candidates/my-playlist"
```

```powershell
python -m myusic_engine rank-candidates `
  "data/processed/candidates/my-playlist/candidates.jsonl" `
  --model "data/processed/models/behavior/selected_model.json" `
  --behavior-snapshots "data/processed/modeling/behavior_snapshots.jsonl" `
  --output-dir "data/processed/recommendations/behavior"
```

When lawful audio representations are available, add acoustic similarity and cluster context:

```powershell
python -m myusic_engine build-taste-map `
  --features "data/processed/audio/features.jsonl" `
  --profile local_discogs_effnet `
  --representation embedding `
  --output-dir "data/processed/taste-map"

python -m myusic_engine rank-candidates `
  "data/private/candidates.csv" `
  --features "data/processed/audio/features.jsonl" `
  --profile local_discogs_effnet `
  --seed "spotify:track:0000000000000000000000=1.0" `
  --model "data/processed/models/audio/selected_model.json" `
  --behavior-snapshots "data/processed/modeling/behavior_snapshots.jsonl" `
  --taste-map-assignments "data/processed/taste-map/taste_map_assignments.jsonl" `
  --output-dir "data/processed/recommendations"
```

Candidate input can be an exactly named account-export playlist, CSV, JSON Lines, or a text file of
Spotify track URIs/URLs. Duplicate playlist tracks are removed in first-seen order and non-track
items are counted in an aggregate import report. Results retain separate acoustic similarity,
predicted preference, novelty, and artist-diversity components.
Behavior-only model results are labeled `preference_ranked`; tracks without either model or audio
coverage are explicitly marked `metadata_only`. Every run ID hashes the exact candidate metadata,
feature observations, behavior snapshots, cluster assignments, model, and configuration. The ordered
`spotify_playlist_uris.txt` remains the safe local handoff. An optional command can turn that file
into a reviewed, deterministic publication plan and—only with an explicit execution flag and a
user-authorized OAuth token—a private Spotify playlist.

Record an outcome without rewriting earlier run artifacts:

```powershell
python -m myusic_engine record-feedback `
  "data/processed/recommendations/feedback.jsonl" `
  "RECOMMENDATION_RUN_ID" `
  "spotify:track:0000000000000000000000" `
  accepted
```

## Publish a private Spotify playlist

First create a dry-run plan. This validates and hashes the exact name, description, order, and
track URIs without reading an OAuth token or making a network request:

```powershell
python -m myusic_engine publish-spotify-playlist `
  "data/processed/recommendations/spotify_playlist_uris.txt" `
  --name "Myusic recommendations" `
  --output-dir "data/processed/recommendations/spotify-output"
```

Review `spotify_playlist_plan.json`. To publish it, obtain a short-lived user OAuth token with
`playlist-modify-private` and `playlist-read-private`, place it in the process environment as
`SPOTIFY_ACCESS_TOKEN`, and rerun the same command with `--execute`. Never put a token in a command
argument, config file, plan, receipt, or Git-tracked file.

```powershell
python -m myusic_engine publish-spotify-playlist `
  "data/processed/recommendations/spotify_playlist_uris.txt" `
  --name "Myusic recommendations" `
  --output-dir "data/processed/recommendations/spotify-output" `
  --execute
```

The publisher uses Spotify's current supported `/me/playlists` and `/playlists/{id}/items`
operations, creates private playlists only, and writes a secret-free resumable receipt after each
confirmed batch. A rerun reconciles the ordered remote prefix before appending; it refuses to
continue if the playlist has drifted. See [Spotify playlist output](docs/spotify-playlist-output.md)
for authorization, failure, and recovery details.

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
