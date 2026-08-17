# Personal Music Intelligence Engine — Project Brief and Build Plan

**Status:** Revised plan; Spotify data export requested / pending
**Scope:** Personal, private research project
**Owner:** Andrew
**Primary goal:** Independently reimplement useful music-audio descriptors, learn a personal taste model, and rank candidate songs by sound similarity and predicted preference

---

## 1. Project Vision

Build a personal music-intelligence pipeline that uses Andrew's lifetime Spotify listening history as **behavioral preference data**, then combines it with independently computed or lawfully obtained **audio features and embeddings**.

The finished system should answer questions such as:

- What distinct sound profiles exist in my listening history?
- Which songs sound most similar to one or several seed songs?
- Which candidates match filters such as heavy bass, a BPM range, melodic similarity, energy, mood, or instrumentalness?
- Of the songs that sound relevant, which ones am I personally most likely to enjoy?
- Can the best-ranked songs be written into a private Spotify playlist using supported API operations?

This project deliberately complements Andrew's existing League of Legends/XGBoost work. It demonstrates:

- music information retrieval and digital signal processing;
- clean-room feature design and validation;
- deep audio embeddings and vector similarity;
- unsupervised clustering;
- implicit-feedback recommendation and learning-to-rank;
- data engineering, experiment tracking, and reproducibility;
- honest handling of data-access, licensing, and measurement limitations.

Lyrics remain out of scope. The focus is sound: rhythm, tempo, bass, melody, harmony, timbre, production, energy, and mood.

---

## 2. What “Reverse Engineering” Means in This Project

The project will use **clean-room behavioral reimplementation**, not extraction of Spotify's private implementation.

We may study the public meaning of legacy descriptors such as tempo, energy, danceability, valence, acousticness, instrumentalness, loudness, speechiness, key, mode, and time signature. We will then design our own independently computed versions from lawful audio inputs.

The project will not claim that these values are Spotify's internal values. Custom fields will be named accordingly—for example:

- `energy_estimate_v1`
- `danceability_estimate_v1`
- `mood_valence_estimate_v1`
- `bass_energy_ratio_v1`

This distinction is important:

| Goal | Included? |
|---|---:|
| Reimplement the useful behavior of audio descriptors | Yes |
| Learn how feature-extraction systems are designed | Yes |
| Compare independent features with appropriately licensed historical reference data | Optional, after provenance review |
| Discover or automate private Spotify endpoints | No |
| Copy Spotify's internal code or protected model | No |
| Capture or decrypt Spotify audio streams/offline files | No |

Exact numerical replication is neither required nor realistically verifiable without a valid reference oracle. Success means that our descriptors are explainable, stable, experimentally validated, and useful for similarity and recommendation.

---

## 3. Core Feasibility Constraint

Spotify's Extended Streaming History is extremely valuable, but it is **not audio**. It contains behavioral and identifying fields such as track URI, artist, title, timestamp, milliseconds played, skip state, start/end reason, platform, shuffle state, and related context.

Therefore:

```text
Spotify history -> what Andrew listened to and how he behaved
Audio waveform   -> what each song actually sounds like
```

BPM, bass, melody, spectrograms, and deep audio embeddings require either:

1. an audio waveform that the project is permitted to analyze; or
2. a permitted database of already-computed audio descriptors.

The history export alone cannot produce those measurements.

---

## 4. Data-Source Strategy

Every track must retain a `feature_source`, `source_version`, `coverage_seconds`, and `feature_confidence`. Features from different sources will never be silently treated as identical measurements.

### Tier A — Spotify Extended Streaming History

**Purpose:** behavioral labels and stable Spotify track URIs.

Expected useful fields include timestamp, track URI, track/artist/album names, milliseconds played, skip indicator, start/end reasons, and shuffle/private/offline context.

Privacy cleaning must remove or ignore IP address, username, user-agent, and other fields unnecessary for modeling.

### Tier B — AcousticBrainz

**Purpose:** immediate precomputed audio descriptors for historical tracks.

AcousticBrainz stopped accepting new submissions in 2022, but its website/API remains available and its existing data is CC0. It contains low-level spectral descriptors and higher-level predictions for millions of recordings.

Proposed match chain:

```text
Spotify URI
  -> supported Spotify metadata lookup
  -> ISRC when available
  -> MusicBrainz recording lookup
  -> MusicBrainz recording ID (MBID)
  -> AcousticBrainz low-level/high-level features
```

Limitations:

- coverage is frozen around mid-2022;
- duplicate releases, remasters, live versions, and recording-ID ambiguity require confidence scoring;
- new releases generally will not be covered;
- AcousticBrainz descriptors were generated by older Essentia versions and should be source-tagged.

### Tier C — Owned or Explicitly Licensed Audio

**Purpose:** highest-fidelity custom feature extraction and embeddings.

Valid examples include purchased downloads, ripped personal CDs where permitted, direct-from-artist files, Creative Commons audio, or another source whose license expressly permits the planned analysis.

Full tracks are preferable to excerpts because they support section-level aggregation and reduce preview-selection bias.

### Tier D — Open Research Audio

**Purpose:** develop and validate the complete audio pipeline before commercial-track coverage is solved.

The Free Music Archive dataset provides Creative Commons-licensed audio, metadata, and precomputed features at multiple dataset sizes. It can be used to:

- validate ingestion and feature extraction;
- test invariance and perturbation experiments;
- compare embeddings and clustering methods;
- build a reproducible public demonstration without exposing personal history or copyrighted Spotify audio.

### Tier E — Preview or Feature Provider, Only After Terms Review

A provider-preview integration is not assumed to be available. Deezer will not be a core dependency: current developer access and authorization have been unstable, and the brief does not yet have authoritative permission establishing that bulk preview analysis is allowed.

Before adding any preview provider, verify all of the following:

- the endpoint is current and documented;
- programmatic access is supported for a new developer;
- the preview may be downloaded or processed for this purpose;
- request volume and caching comply with the provider's terms;
- derived-feature storage is allowed;
- the implementation does not scrape a consumer web client.

If no provider passes this gate, the current-commercial-track feature stage remains unavailable rather than being filled with an undocumented workaround.

### Historical “Spotify Audio Features” CSVs

Old Kaggle/Hugging Face CSVs are **not the primary feature source**. They may be considered only as an optional benchmark after checking original provenance, collection date, redistribution/license status, feature definitions, recording identity, and consistency with Spotify's current terms.

They must not be presented as fresh Spotify data or as ground truth merely because a CSV is downloadable.

---

## 5. Audio-Feature Reimplementation Plan

Each feature family will have a documented definition, extractor version, confidence value, and validation test.

### 5.1 Objective and Interpretable Features

| Feature family | Proposed implementation | Notes |
|---|---|---|
| Tempo/BPM | Beat tracking plus TempoCNN-style prediction | Preserve estimate and confidence; test half/double-tempo errors |
| Beat strength | Onset envelope, beat confidence, periodicity | Useful for danceability and rhythmic stability |
| Key/mode | HPCP/chroma aggregation and key-profile matching | Store key, major/minor mode, and strength |
| Loudness | Integrated loudness and dynamic range | Do not confuse loudness with perceived energy |
| Bass amount | Energy ratio in low-frequency bands, e.g. below 150/250 Hz | Normalize against total spectral energy |
| Bass rhythm | Low-band onset density and beat synchronization | Distinguishes sustained bass from punchy rhythmic bass |
| Brightness | Spectral centroid/rolloff/contrast | Captures dark versus bright production |
| Timbre | MFCCs, spectral contrast, texture statistics | Aggregate mean, variance, and section-level behavior |
| Harmony | Chroma/HPCP, chord-change rate, tonal strength | Useful for harmonic similarity |
| Melody | Predominant-pitch contour, pitch range, interval histogram | Polyphonic melody extraction is difficult; report confidence |
| Vocalness | Pretrained vocal/instrumental classifier | More defensible than copying Spotify instrumentalness |
| Speechiness | Speech/music classifier or vocal temporal characteristics | Keep distinct from ordinary sung vocals |
| Acousticness proxy | Acoustic/electronic tagging plus timbre features | Custom estimate, not Spotify's score |
| Energy proxy | Loudness, onset density, spectral flux, compression, tempo, tags | Validate against human judgments |
| Danceability proxy | Beat stability, tempo suitability, onset regularity, low-band groove | Learned or calibrated composite |
| Mood/valence proxy | Pretrained mood/arousal-valence model | Subjective and culture-dependent; confidence required |
| Time signature | Beat-position and meter analysis | Low confidence on ambiguous or changing meter |

### 5.2 Deep Audio Embeddings

Use a music-oriented pretrained embedding model available through Essentia, such as Discogs-EffNet or MusiCNN, subject to its model license.

Recommended process:

1. Convert audio to the model's expected mono sample rate.
2. Split it into overlapping windows.
3. Extract one embedding per window.
4. Retain window-level embeddings and a robust track-level aggregate.
5. L2-normalize the final vector for cosine similarity.
6. Store model name, version, sample rate, window length, and aggregation method.

Track-level mean pooling is an MVP. A later version can use attention-weighted or section-aware pooling so an intro does not dominate the representation.

### 5.3 Source Separation — Stretch Goal

If licenses and compute allow it, source separation can estimate vocal, drum, bass, and other stems. This could improve bass-only descriptors, melody extraction, drum-groove similarity, and per-stem embeddings. It is not required for the MVP because separation adds compute and artifacts.

---

## 6. Reverse-Engineering Learning Experiments

Instead of guessing whether a feature “looks right,” create controlled tests.

| Test | Expected behavior |
|---|---|
| Metronome at known BPM | Tempo estimate matches within tolerance |
| Time-stretch without pitch shift | Tempo changes; key/timbre remain relatively stable |
| Pitch shift | Key and melody contour shift predictably; tempo remains stable |
| Bass boost/cut | Bass energy changes monotonically |
| Added compression | Dynamic range falls; loudness/energy may rise |
| Added silence | Coverage-aware aggregation prevents large descriptor distortion |
| White noise versus sine wave | Spectral/timbre descriptors separate them clearly |
| Same song encoded at different bitrates | Core similarity remains high |
| Excerpt versus full track | Quantify preview-selection bias |

Create a small human-reviewed sanity set with pairs that are clearly similar, clearly dissimilar, similar in rhythm but not timbre, similar in timbre but not tempo, and similar in bass but not melody.

Run ablations with embeddings only, interpretable features only, behavior only, embeddings plus behavior, and embeddings plus behavior plus filters. This will show which components actually help.

---

## 7. Behavioral Preference Modeling

Listening history supplies implicit feedback, not explicit ratings. A play does not automatically mean “liked.” Aggregate behavior per track using:

- play count and total listening time;
- median completion ratio and completed-play rate;
- early-skip rate;
- repeat-within-session rate;
- recency-weighted plays;
- intentional-start rate versus autoplay/context continuation, when inferable;
- library-save/playlist membership, when supported and available.

Example initial affinity heuristic:

```text
affinity =
    log1p(play_count)
  + 1.5 * completion_rate
  + 1.0 * intentional_start_rate
  + 0.5 * recency_score
  - 2.0 * early_skip_rate
```

This is a starting heuristic. Its weights should later be calibrated using explicit thumbs-up/down feedback.

Positive signals can include repeated full plays, later replays, saved tracks, and confirmed likes. Repeated early skips and explicit rejection can be weak negatives. Unplayed songs are unknown, not automatic negatives.

After representations are stable, train a ranker using audio embeddings, interpretable descriptors, artist/genre familiarity, recency, novelty, and candidate context. Start with logistic regression or XGBoost/LightGBM. Use a chronological split so future listening does not leak into training.

---

## 8. Clustering, Similarity, and Recommendation Are Separate

### 8.1 Taste Clustering

- Scale interpretable numeric features.
- Compare K-Means and HDBSCAN.
- Use PCA as a baseline and UMAP for nonlinear visualization.
- Inspect cluster stability across seeds and feature sets.
- Name clusters only after reviewing representative tracks.
- Evaluate with silhouette, Davies–Bouldin, stability, and human review.

### 8.2 Multi-Seed Audio Similarity

Given seed embeddings `e1 ... en` and optional weights `w1 ... wn`:

```text
query_embedding = normalize(sum(w_i * normalize(e_i)))
```

Retrieve nearest candidates with cosine similarity. Supported filters should include BPM range, bass level, energy, key/mode compatibility, vocal/instrumental preference, mood, release date, excluded artists, novelty, and minimum feature confidence.

### 8.3 Two-Stage Personal Recommendation

**Stage 1 — candidate retrieval:** find acoustically relevant tracks using vector similarity and filters.

**Stage 2 — personalized ranking:** estimate which candidates Andrew is most likely to enjoy.

```text
final_score =
    alpha * audio_similarity
  + beta  * predicted_preference
  + gamma * novelty_bonus
  - delta * artist_repetition_penalty
```

Expose these weights as configuration rather than burying them in code.

---

## 9. New Music Friday and Release Radar

The desired workflow has two separate problems:

1. obtaining a permitted candidate list; and
2. obtaining permitted audio features/embeddings for those candidates.

Current Spotify Development Mode restrictions do not provide a reliable supported way to enumerate Spotify-owned editorial or algorithmic playlist contents. The system must support multiple candidate inputs:

- a manually supplied list of Spotify track URLs/URIs;
- a playlist Andrew owns or collaborates on, when available through the supported API;
- MusicBrainz/ListenBrainz or another permitted catalog/recommendation source;
- a future provider integration that passes the terms-review gate.

Even with a track list, recent songs will usually be absent from AcousticBrainz. Full acoustic ranking of current commercial releases remains dependent on a lawful audio or feature provider. Metadata-only ranking may be offered as a clearly labeled fallback, but it must not pretend to be audio similarity.

---

## 10. Databricks / Lakehouse Design

This project naturally combines raw events, entity resolution, feature pipelines, ML experiments, and vector retrieval.

### Bronze — Raw and Immutable

- `bronze_spotify_history_raw`
- `bronze_spotify_library_raw`
- `bronze_musicbrainz_responses`
- `bronze_acousticbrainz_responses`
- `bronze_candidate_imports`

### Silver — Cleaned and Resolved

- `silver_listening_events`
- `silver_tracks`
- `silver_track_identity_matches`
- `silver_audio_assets`
- `silver_feature_observations`
- `silver_candidate_tracks`

Identity matches retain method, IDs, confidence, ambiguous alternatives, and recording/version details.

### Gold — Modeling and Products

- `gold_user_track_affinity`
- `gold_track_features_current`
- `gold_track_embeddings_current`
- `gold_track_clusters`
- `gold_similarity_neighbors`
- `gold_recommendation_runs`
- `gold_recommendation_feedback`

Use MLflow for feature/model versions, experiments, parameters, metrics, and artifacts. Build a small local version first, then migrate to Databricks once schemas and extractors stabilize.

---

## 11. Repository Structure

```text
personal-music-intelligence/
  README.md
  pyproject.toml
  configs/
    features.yaml
    recommendation.yaml
  data_contracts/
    spotify_history.schema.json
  notebooks/
    01_history_eda.ipynb
    02_identity_matching.ipynb
    03_feature_validation.ipynb
    04_clustering.ipynb
    05_preference_model.ipynb
  src/
    ingest/
    privacy/
    matching/
    audio/
    features/
    embeddings/
    clustering/
    ranking/
    evaluation/
    spotify_output/
  tests/
    fixtures/
    test_privacy_cleaning.py
    test_identity_matching.py
    test_feature_invariance.py
    test_multi_seed_query.py
  reports/
    figures/
    model_cards/
```

Raw personal history and audio must be excluded from Git. Commit schemas, synthetic fixtures, code, aggregate results, and reproducible instructions.

---

## 12. Phased Build Plan

### Phase 0 — While Waiting for the Export

- Create the repository and environment.
- Write a synthetic Spotify-history fixture.
- Define privacy-cleaning and normalized-event schemas.
- Prototype MusicBrainz and AcousticBrainz matching on a manual track list.
- Run Essentia on a few Creative Commons files.
- Define versioned feature names and confidence fields.

**Exit criterion:** one test track flows from input identity to a source-tagged feature record.

### Phase 1 — Personal History Foundation

- Ingest every Extended Streaming History JSON file.
- Remove sensitive/unneeded fields.
- Deduplicate events and normalize track URIs.
- Separate music from podcasts/episodes.
- Create behavioral aggregates and affinity scores.
- Produce coverage and listening-pattern EDA.

**Exit criterion:** one clean track table and one user-track affinity table with validation checks.

### Phase 2 — Identity Resolution and Coverage

- Resolve Spotify URI to supported metadata.
- Match ISRC/title/artist/duration to MusicBrainz.
- Fetch AcousticBrainz features for confident MBIDs.
- Report exact, fuzzy, ambiguous, unmatched, and feature-covered percentages.
- Manually review a stratified match sample.

**Exit criterion:** trustworthy match report and a source-tagged historical feature table.

### Phase 3 — Clean-Room Feature Extractor

- Implement objective features first: BPM, key, loudness, bass energy, spectral/timbre, and chroma.
- Add a pretrained music embedding model.
- Run transformation and invariance tests.
- Add proxy features only after objective features are stable.
- Create a feature/model card documenting definitions and limitations.

**Exit criterion:** reproducible features and embeddings on FMA/owned audio with passing tests.

### Phase 4 — Taste Map and Similarity Engine

- Standardize selected interpretable features.
- Compare K-Means and HDBSCAN.
- Visualize with PCA/UMAP.
- Build cosine nearest-neighbor retrieval.
- Implement weighted multi-seed queries and hard/soft filters.

**Exit criterion:** a seed query returns explainable similar tracks and cluster context.

### Phase 5 — Personal Preference Ranker

- Create chronological train/validation/test splits.
- Train behavior-only, audio-only, and combined baselines.
- Measure ranking quality and conduct ablations.
- Add lightweight explicit feedback.
- Calibrate final recommendation weights.

**Exit criterion:** combined ranking beats simple popularity/nearest-neighbor baselines on held-out personal behavior.

### Phase 6 — Candidate Intake and Spotify Output

- Accept candidate CSV, URLs, URIs, or supported owned-playlist input.
- Resolve candidate identities and feature availability.
- Clearly separate audio-ranked from metadata-only candidates.
- Create a private output playlist through supported Spotify operations.
- Log every recommendation run and its feature/model versions.

**Exit criterion:** reproducible candidate-to-private-playlist workflow without private endpoints or protected-audio extraction.

### Phase 7 — Optional Databricks Productionization

- Move Bronze/Silver/Gold tables to Delta.
- Schedule idempotent ingestion and feature jobs.
- Track experiments and registered models with MLflow.
- Add quality expectations and drift/coverage reporting.
- Build a dashboard for clusters, filters, and recommendation explanations.

---

## 13. Evaluation Plan

### Feature Evaluation

- known BPM/key test clips;
- perturbation/invariance tests;
- confidence calibration;
- excerpt-versus-full-track differences;
- manual failure-case review.

### Identity-Matching Evaluation

- exact and ambiguous match rates;
- manual precision on a stratified sample;
- version/remaster/live mismatch rate.

### Similarity Evaluation

- human pairwise precision@K;
- same-track/different-encoding retrieval;
- artist leakage checks;
- embeddings versus handcrafted features.

### Recommendation Evaluation

- chronological Precision@K, Recall@K, and NDCG@K where labels are meaningful;
- early-skip rate among recommendations;
- catalog/artist diversity;
- novelty and repeated-artist concentration;
- explicit satisfaction feedback.

Silhouette score alone is not evidence that recommendations are good. Clustering, retrieval, and personal ranking must be evaluated separately.

---

## 14. Hard Boundaries

- No extraction, decryption, recording, or reconstruction of Spotify's protected audio stream or offline cache.
- No private Spotify endpoints, copied session tokens, client impersonation, or rate-limit evasion.
- No consumer-web scraping as a substitute for an unavailable developer API.
- No assumption that “personal/private” overrides platform terms or copyright restrictions.
- No redistribution of personal history or copyrighted audio.
- No training on Spotify API content unless the planned use has been reviewed and permitted by the applicable terms.
- No claim that custom outputs are official Spotify audio-feature values.

These constraints do not prevent independent feature engineering, signal processing, model evaluation, or recommendation using permitted inputs.

---

## 15. Decisions Made by This Revision

1. Expand the project from clustering into a complete personal music-intelligence and recommendation system.
2. Treat Spotify history as behavioral data, not an audio source.
3. Make clean-room feature reimplementation a core goal.
4. Make bass and melody features first-class requirements.
5. Include deep embeddings and weighted multi-seed search as core requirements.
6. Separate personal preference prediction from acoustic similarity.
7. Treat AcousticBrainz as a usable frozen source, not a dead API.
8. Make old Spotify-feature CSVs optional benchmarks, not the foundation.
9. Remove Deezer previews as an assumed fallback until access and usage rights are verified.
10. Keep New Music Friday/Release Radar ranking as a target while documenting current-track audio coverage as unresolved.
11. Add Databricks/MLflow productionization after a local proof of concept.

---

## 16. Immediate Next Actions

1. Wait for the Spotify Extended Streaming History export and keep the ZIP private.
2. Before sharing data, run a privacy-cleaning script that removes IP address, username, and user-agent fields.
3. Create a 20–50-track test list spanning Andrew's taste, including similar and intentionally dissimilar songs.
4. Test MusicBrainz-to-AcousticBrainz coverage on that list.
5. Choose several Creative Commons or owned audio files for the first extraction experiments.
6. Implement BPM, key, bass energy, chroma, and embedding extraction before subjective proxies.
7. Do not select final clustering/ranking algorithms until coverage and feature-quality reports exist.

---

## 17. Sources Checked for This Revision

- Spotify, “Understanding your data”: https://support.spotify.com/us/article/understanding-your-data/
- Spotify, November 2024 Web API changes: https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api
- Spotify, February 2026 Development Mode migration guide: https://developer.spotify.com/documentation/web-api/tutorials/february-2026-migration-guide
- Spotify Developer Terms: https://developer.spotify.com/terms
- AcousticBrainz project/API status and CC0 data: https://acousticbrainz.org/
- Essentia pretrained audio models: https://essentia.upf.edu/models.html
- Essentia Python examples: https://essentia.upf.edu/python_examples.html
- MusicBrainz API: https://musicbrainz.org/doc/MusicBrainz_API
- ListenBrainz recommendations: https://listenbrainz.readthedocs.io/en/latest/users/api/recommendation.html
- Free Music Archive dataset: https://github.com/mdeff/fma

**Revision date:** August 14, 2026
