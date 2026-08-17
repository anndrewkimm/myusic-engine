# Local architecture

The implementation follows one-way data boundaries so private identifiers cannot leak into
later stages by accident.

```text
private export
    -> privacy cleaning + normalization
    -> listening events
    -> track behavior aggregates
    -> affinity labels

permitted catalog metadata
    -> identity matches + confidence
    -> source-tagged feature observations

permitted audio
    -> objective descriptors + embeddings
    -> source-tagged feature observations

features + behavior
    -> clustering / similarity / preference ranking
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
| `spotify_output` | Use supported Spotify operations for private output only |
