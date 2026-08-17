# Behavioral affinity v1

Spotify history is implicit feedback. The first score is deliberately a transparent heuristic,
not a claim that every play is a like.

```text
affinity =
    play_count_log_weight          * log1p(play_count)
  + completion_rate_weight         * completion_rate
  + intentional_start_rate_weight  * intentional_start_rate
  + recency_score_weight            * recency_score
  + early_skip_rate_weight          * early_skip_rate
```

The committed values live in `configs/recommendation.yaml`, and every output record stores
`scoring_version` plus the individual score components.

## Evidence and missingness

- Duration-based completion uses a capped `ms_played / duration_ms` ratio when a positive
  catalog duration is supplied. It is otherwise left null.
- The completed-play rate falls back to supported end-reason evidence when duration is absent.
- Skip and intentional-start rates use only events with a known signal. Their denominators are
  exposed as coverage rates.
- Missing signals contribute zero to the initial heuristic, but also reduce
  `affinity_confidence`; missingness is never serialized as a negative label.
- Confidence combines weighted signal coverage with a bounded sample-size factor. It measures
  confidence in the behavior summary, not certainty that the listener likes the track.

## Sessions and recency

A new session begins after the configured inactivity gap. A play counts as an in-session repeat
when that track already appeared in the current session. Recency uses exponential half-life
decay relative to the latest event by default, making an unchanged input deterministic. A caller
can supply a later, timezone-aware `as_of` time for a refreshed score.

## Identity caveat

Spotify track URI is the preferred aggregate key. Unavailable tracks without a URI receive a
deterministic metadata hash and `track_identity_source: metadata_hash`. That fallback can merge
same-named recordings and must be resolved before audio features are joined.
