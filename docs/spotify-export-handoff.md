# Spotify export handoff

The eventual handoff is intentionally one local command. Keep the original archive unchanged under
`data/private/`; neither the ZIP nor any derived private table belongs in Git.

```powershell
python -m myusic_engine prepare-history `
  "data/private/spotify-export.zip" `
  --output-dir "data/processed/history" `
  --recommendation-config "configs/recommendation.yaml"
```

No manual extraction is required. The reader walks nested ZIP paths and accepts Extended Streaming
History files named `endsong*.json` or `Streaming_History_Audio_*.json`. It also accepts compact
Account Data files named `StreamingHistory_music_*.json` or
`StreamingHistory_podcast_*.json`, while ignoring unrelated account JSON. It normalizes all
selected arrays, deduplicates across files, separates tracks from episodes, and writes files
atomically.

The compact Account Data format contains only end time, milliseconds played, and track/artist or
podcast/episode names. Its documented `endTime` is interpreted as UTC. Because it has no Spotify
URIs, album names, skip reasons, shuffle state, or other Extended History fields, track aggregation
uses a deterministic metadata-hash identity and reports the unavailable behavioral signals as
missing rather than guessing them.

Spotify's current documentation describes Extended Streaming History as lifetime account activity
with UTC timestamps, milliseconds played, track/artist/album names, Spotify track URIs,
start/end reasons, shuffle, skip, offline, private-session, platform, and sensitive account/network
fields. The parser already accepts those fields. It allowlists useful behavioral values and counts,
but never copies username, IP address, user agent, country, or raw unknown fields into normalized
events. The archive's own `Read Me First - Extended Streaming History` remains the authority if
Spotify changes a field.

## What to inspect after the first run

Open `data/processed/history/ingestion_report.json` and check:

1. Every expected history shard appears in `source_files`.
2. `records_seen` is plausible for the age and activity of the account.
3. `records_rejected / records_seen` is very small; review every rejection code.
4. Track, episode, and unknown counts are plausible.
5. Duplicate removal is nonnegative and not unexpectedly dominant.
6. Sensitive-field counts are present as aggregate evidence that privacy cleaning occurred.
7. `user_track_affinity.jsonl` has stable Spotify URI keys for most music listening.

If Spotify introduces a new filename or field shape, preserve the original ZIP, add a redacted
synthetic fixture reproducing only that shape, update the boundary parser, and rerun. Do not edit the
real export to make it fit the code.

## How history joins audio

```text
Spotify ZIP
  -> normalized listening event (track_uri)
  -> user-track affinity (track_uri)

permitted audio manifest
  -> objective features + embedding (track_id)

track_uri == track_id after exact identity resolution
  -> behavior labels + audio representation
  -> chronological training/evaluation table
```

The ZIP is behavioral evidence, not an audio source. A Spotify URI identifies a recording but does
not authorize downloading, decrypting, recording, or analyzing Spotify's protected stream. Phase 2
will measure which history identities have a confident permitted-audio or lawful external-feature
match; unmatched history remains valid behavioral data rather than being forced into a bad join.
