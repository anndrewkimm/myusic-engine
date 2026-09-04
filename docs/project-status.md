# Current project status

This status separates implemented code, private validation already completed, and evidence that
still requires an approved external service or permitted real audio.

| Phase | Status | Evidence still needed |
|---|---|---|
| 0 — data-independent foundation | Complete | None for the synthetic foundation |
| 1 — personal history foundation | Complete — Extended History validated locally | Rerun only when a newer export is imported |
| 2 — identity resolution | Offline resolver plus cached public-provider and offline canonical-dump mappers implemented | Review-sample precision and resulting coverage for each mapper; the live route also needs the user's explicit per-run approval |
| 3 — clean-room audio representation | Foundation implemented | Permitted real-music corpus, pairwise sanity set, and retrieval-quality evaluation |
| 4 — taste map and similarity | Standardized K-Means/HDBSCAN/PCA and retrieval implemented | Real lawful vectors and human retrieval sanity set |
| 5 — personal preference ranker | Behavior baselines validated privately; descriptor-audio lift measured on real offline-covered history (not yet significant); embedding ablations implemented but unrun | Lawful embedding coverage from real permitted audio, and broader/higher-fidelity descriptor coverage |
| 6 — candidate intake/output | Local/account-export playlist intake, explainable ranking, URI handoff, feedback, and guarded private-playlist publication implemented | Real candidate features and an explicitly authorized live OAuth smoke test |

Phase 1 validation read every discovered Extended History shard, rejected no malformed records,
retained Spotify URI identities for all normalized tracks, confirmed the expected rich playback
signals, and removed sensitive network fields from normalized output. Exact counts and listening
patterns remain in ignored local reports and tables.

Phase 2 now supports a precision-first public mapping route through the current ListenBrainz Labs
artist/recording and artist/recording/release endpoints. A public control lookup and its matching
AcousticBrainz low/high-level fetch passed live; only strict exact metadata matches receive
MusicBrainz recording IDs. Provider responses are cached, bounded, retry-safe, and replayable
offline. The private-data run remains paused until the user explicitly approves sending that
limited title/artist/album metadata.

Phase 2 also has an offline mapper that scans an official MusicBrainz canonical data dump instead
of calling a live endpoint, so no title/artist/album metadata leaves the machine. It applies the
same strict-exact-match policy, prefers the documented higher-score row when the dump lists more
than one candidate row for a recording, and reports its own provider name and coverage separately
from the live mapper. This route does not need the live mapper's per-run disclosure approval, but
its matches still need the same manually reviewed stratified sample before use as training labels.

Phase 2 and phase 3 meet at `track_id`, but phase 3 can proceed independently. For a recording that
already has a Spotify track URI, the private audio manifest can use that URI directly. For another
lawful catalog, identity resolution must first decide which stable recording is represented.
Identity confidence is never mixed into audio-feature confidence.

The phase-3 foundation currently provides:

- a rights-declared private audio manifest;
- deterministic mono decoding and resampling;
- objective tempo, key, loudness, bass, spectral, chroma, and MFCC observations;
- a pinned, SHA-256-verified Discogs-EffNet ONNX model;
- six pinned, SHA-256-verified classifier heads for interpretable learned audio scores;
- exact MusiCNN-style 16 kHz log-mel preprocessing;
- retained overlapping window embeddings and learned score trajectories;
- arithmetic-mean pooling followed by L2 normalization;
- direct compatibility with the 1,280-dimensional cosine index;
- synthetic transformation tests plus a real ONNX inference smoke test;
- frozen CC0 AcousticBrainz low/high-level conversion for exact MusicBrainz matches.

AcousticBrainz descriptors are not presented as deep embeddings. A true Discogs-EffNet track
embedding still requires a waveform the user owns or otherwise has permission to analyze.

`fetch-acousticbrainz` can also convert AcousticBrainz's own frozen 2022 low-level dataset
directly from its three local CSV/tar.zst dumps (lowlevel, rhythm, tonal) instead of the live API,
so no recording MBID needs to be disclosed over the network. Coverage from this path is narrower
than the live API by design: only the fields present in the derived CSVs are populated, so the
combined descriptor vector feature is never emitted, and high-level mood/genre predictions are
never available offline, since AcousticBrainz never published a bulk dump of those. Both source
routes tag output with the same `acousticbrainz_cc0` / `frozen-2022-lowlevel-converter-v1`
provenance, because both ultimately read the identical frozen snapshot. The existing
`acousticbrainz_lowlevel` modeling profile requires the combined descriptor vector and so only
ever matches live-sourced coverage; a new `acousticbrainz_lowlevel_offline` profile lists the
seven individual fields both routes can populate, so offline-only coverage can still train and
evaluate a taste model instead of sitting unused.

The phase-5 implementation creates non-overlapping target periods, freezes behavior before each
period, abstains on ambiguous outcomes, and reserves whole later periods for validation and test.
It compares repeat/recency, artist, full behavior, descriptor, embedding, and combined variants;
all audio comparisons use the same covered cohort. Paired bootstrap intervals resample whole
periods, and model selection prefers the simpler variant when a validation lift is uncertain. The
private real-history sweep retained the 90-day period and default regularization, selected the
artist baseline over an uncertain full-behavior lift, and left preference-only novelty disabled by
default after a 101-point validation sweep. Detailed metrics and predictions remain ignored rather
than being committed.

A second private sweep repeated this comparison after the offline AcousticBrainz path populated
real low-level descriptors for roughly half of the exactly matched history. On the resulting
audio-covered cohort, the paired validation comparison did not find the behavior-plus-descriptor
model reliably better than behavior alone — its confidence interval sat at or below zero — so
selection kept the simpler behavior-only model on that cohort; the seven-field descriptor set used
alone performed clearly worse than behavior. This is an honest result, not a defect: seven coarse
interpretable fields are a thin representation next to the 1,280-dimensional embedding phase 3 is
designed to add once real permitted audio is available, and the selection rule is built to prefer
the simpler model exactly when a claimed lift is this uncertain.

The engine deliberately does not emit Spotify-named `danceability`, `energy`, `valence`,
`acousticness`, `speechiness`, or `instrumentalness` values. Learned outputs use custom
`*_score_v1` names and remain uncalibrated until a permitted labeled corpus and held-out benchmark
exist. Unsupported properties remain absent instead of being filled with invented heuristics.

Phase 6 now includes a dry-run-first Spotify publisher aligned to the current `/me/playlists` and
`/playlists/{playlist_id}/items` operations. It accepts OAuth material only from an environment
variable, creates private playlists only, batches at the documented 100-item limit, and stores a
secret-free receipt after each confirmed batch. On resume it reads the remote order and appends
only when that order is an exact prefix of the deterministic plan. No live playlist was created
during repository validation; that final smoke test still requires the user's explicit OAuth
authorization.

Named playlists from a private Spotify Account Data export can now become candidate JSON Lines
directly, with no manual transcription and no OAuth call. The importer selects one exact
case-insensitive playlist name, preserves its exported order and metadata, removes duplicate track
URIs, counts skipped episodes/local items, and records the export snapshot date. A shared playlist
link can therefore be matched to an existing private export snapshot before requesting live API
access.
