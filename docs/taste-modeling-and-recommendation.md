# Taste modeling and recommendation

## What the model learns

Spotify Extended Streaming History supplies implicit behavioral outcomes. It does not supply audio
features or audio embeddings. The temporal dataset therefore joins two independent kinds of
evidence by stable `track_id`:

- behavior known strictly before a target period; and
- descriptors or embeddings obtained from a declared lawful source and exact source version.

A complete/long non-skipped playback is a positive signal. A true early skip is a negative signal.
Unknown end states abstain. Multiple known outcomes in one track-period vote; ambiguous fractions
also abstain. Tracks absent from a period never become negative examples.

## Leakage boundary

Periods are fixed, non-overlapping UTC windows. For every track-period sample, all behavior and
artist-familiarity features are computed from events before `period_start`. Only after every sample
in that target period is created does the state update. Entire periods, rather than random rows,
are assigned to train, validation, or test.

The standard scaler and logistic coefficients fit on training rows only. Selection starts with the
best validation NDCG@K inside one deployment cohort, breaking ties with validation average
precision. A more complex variant is retained only when its paired period-bootstrap validation lift
over a simpler candidate is clearly positive. Test metrics are reported after selection and are
never used to switch models.

## Audio ablations

`configs/modeling.yaml` locks every scalar or vector to an exact feature name, source, version, and
dimension. Two initial profiles are included:

- `acousticbrainz_lowlevel`: frozen CC0 low-level descriptors; no deep embedding claim;
- `local_discogs_effnet`: clean-room DSP descriptors plus the 1,280-dimensional Discogs-EffNet
  embedding computed from permitted local audio.

When audio is present, the trainer compares behavior-only, descriptors-only, embedding-only,
audio-combined, and behavior-plus-audio variants. Every fair audio comparison is restricted to the
intersection of tracks having all representations selected by that profile. The broader
`behavior_all` baseline is reported separately.

## Portable artifacts and metrics

Models are JSON, not pickle. Each artifact contains feature order, training-only means/scales,
coefficients, intercept, dataset and model versions, last training-period boundary, feature-group
flags, profile version, and a SHA-256 content ID.

Validation and test reports include ROC AUC, average precision, log loss, Brier score, calibration
error, and per-period Precision@K, Recall@K, and NDCG@K. A one-class held-out split retains valid
calibration/ranking metrics and reports discrimination metrics as null. Model ablations also report
paired mean NDCG lift, period win rate, and a deterministic 95% bootstrap interval that resamples
whole periods rather than pretending individual track rows are independent.

## Taste map and final rank

The taste map standardizes the selected representation, compares K-Means across several seeds and
HDBSCAN across minimum cluster sizes, measures silhouette/Davies-Bouldin and K-Means ARI stability,
and selects only eligible configurations. PCA coordinates are visualization aids, not evidence that
clusters are good.

Final candidate ranking keeps four pieces visible:

1. cosine similarity to a normalized weighted multi-seed embedding;
2. model-predicted preference;
3. novelty from current track history; and
4. a greedy repeated-artist penalty and maximum-per-artist constraint.

The audio-ranked and preference-only novelty bonuses are configured independently. Preference-only
novelty defaults to zero because a 101-point chronological validation sweep did not show a clear
period-bootstrap ranking lift over the selected behavior model alone. This keeps discovery as an
explicit product choice instead of letting a missing audio component silently triple novelty's
relative weight.

Missing components are never silently zero-filled and treated as measurements. Audio-covered,
preference-only, and metadata-only candidates receive distinct tiers. Every run has a deterministic
ID over the exact candidate, feature, behavior-snapshot, cluster, model, and configuration inputs;
exact model/profile versions; per-result explanations; an ordered Spotify URI handoff; and an
append-only explicit-feedback log. A behavior-only model can rank immediately without selecting a
dummy audio profile or supplying an empty feature file.
