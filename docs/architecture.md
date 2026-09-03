# Local architecture

The implementation follows one-way data boundaries so private identifiers cannot leak into
later stages by accident.

```text
private export
    -> privacy cleaning + normalization
    -> listening events
    -> track behavior aggregates
    -> heuristic affinity + point-in-time behavior snapshots
    -> future-period implicit-feedback labels

permitted catalog metadata
    -> identity matches + confidence
    -> source-tagged feature observations

private account-export playlist
    -> named, ordered, deduplicated candidate tracks + import provenance

permitted audio
    -> objective descriptors + embeddings
    -> source-tagged feature observations

features + behavior
    -> training-only preprocessing
    -> behavior/audio ablations on chronological splits
    -> portable taste model

feature representations
    -> standardized K-Means/HDBSCAN experiments + PCA
    -> taste-map assignments

candidates + seeds + current behavior + selected model
    -> acoustic similarity + predicted preference + novelty + diversity
    -> explainable recommendations + Spotify URI handoff + explicit feedback
    -> reviewed publication plan + explicitly authorized private Spotify playlist
```

## Design rules

1. Raw exports are read-only and ignored by version control.
2. Normalized events use an allowlist. Unknown raw keys are not copied forward.
3. Identifiers and measurements are separate records; an identity match is not a feature.
4. Feature provenance is mandatory and includes source, version, analyzed coverage, and
   confidence.
5. Candidate retrieval and personal preference ranking expose separate scores.
6. Network providers sit behind interfaces so synthetic fakes can exercise the pipeline.
7. Pipeline writes are deterministic and safe to repeat.
8. Whole target periods belong to exactly one split; no future-period behavior becomes a feature.
9. Unplayed candidates are unknown, not negative labels.
10. JSON model artifacts are data, never executable pickle payloads.
11. Spotify mutation is dry-run first, private-only, explicit, secret-free on disk, and resumable
    only after reconciling the ordered remote prefix.

## Planned package ownership

| Package | Responsibility |
|---|---|
| `ingest` | Read supported local export formats and normalize records |
| `privacy` | Remove sensitive fields and audit the output boundary |
| `matching` | Resolve catalog identities with explicit match confidence |
| `audio` | Validate and decode permitted audio assets |
| `features` | Produce interpretable, versioned audio descriptors |
| `embeddings` | Produce and aggregate model-specific audio vectors |
| `clustering` | Build and evaluate taste clusters |
| `ranking` | Aggregate behavior, retrieve candidates, and rank results |
| `evaluation` | Test feature behavior and recommendation quality |
| `modeling` | Build temporal labels, lock feature profiles, train portable ablations |
| `spotify_output` | Use supported Spotify operations for private output only |
