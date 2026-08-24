# Current project status

This status distinguishes implemented code from validation on inputs that have not arrived yet.

| Phase | Status | Evidence still needed |
|---|---|---|
| 0 — data-independent foundation | Complete | None for the synthetic foundation |
| 1 — personal history foundation | Compact Account Data validated | Validate the Extended History ZIP for lifetime/URI-rich coverage and perform private EDA |
| 2 — identity resolution | Not started | Metadata provider adapter, exact/fuzzy match policy, confidence report, and manual review sample |
| 3 — clean-room audio representation | Foundation implemented | Permitted real-music corpus, transformation tests, pairwise sanity set, and retrieval-quality evaluation |
| 4 — taste map and similarity | Retrieval primitive implemented early | Standardization, clustering experiments, real vectors, and cluster/retrieval evaluation |
| 5+ | Not started | Requires stable representations and real behavioral labels |

Phase 2 and phase 3 meet at `track_id`, but phase 3 can proceed independently. For a recording that
already has a Spotify track URI, the private audio manifest can use that URI directly. For FMA,
MusicBrainz, Discogs, or another lawful catalog, phase 2 must first decide which stable recording ID
is being represented. Identity confidence must never be mixed into audio-feature confidence.

The phase-3 foundation currently provides:

- a rights-declared private audio manifest;
- deterministic mono decoding and resampling;
- 15–16 objective observations per analyzable track;
- a pinned, SHA-256-verified Discogs-EffNet ONNX model;
- six pinned, SHA-256-verified classifier heads for interpretable learned audio scores;
- exact MusiCNN-style 16 kHz log-mel preprocessing;
- retained overlapping window embeddings and learned score trajectories;
- arithmetic-mean pooling followed by L2 normalization;
- robust onset gating that abstains on stationary tones instead of inventing tempo;
- direct compatibility with the existing 1,280-dimensional cosine index;
- unit and synthetic transformation tests plus a real ONNX inference smoke test.

It deliberately does not emit Spotify-named `danceability`, `energy`, `valence`, `acousticness`,
`speechiness`, or `instrumentalness` values. The learned outputs use custom `*_score_v1` names and
remain uncalibrated until a permitted labeled corpus and held-out benchmark exist. Energy,
speechiness, liveness, time signature, and valence-equivalent outputs remain unimplemented rather
than being filled with unsupported heuristics.
