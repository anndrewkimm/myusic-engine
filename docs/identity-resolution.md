# Offline identity resolution v1

The compact Spotify Account Data history identifies tracks only by title and artist. The same ZIP
also contains URI-bearing metadata in `YourLibrary.json` and `PlaylistN.json`, so the first
identity pass is deliberately local and requires no Spotify API.

```powershell
python -m myusic_engine resolve-identities `
  "data/processed/history/user_track_affinity.jsonl" `
  "data/private/my_spotify_data.zip" `
  --output-dir "data/interim/identity" `
  --matching-config "configs/identity_resolution.yaml"
```

The reader uses saved tracks, banned tracks, and non-local playlist tracks as catalog evidence.
Library membership does not become a preference label; these records are used only to recover a
stable Spotify track URI. Local tracks and podcast episodes are ignored by this music resolver.

## Match policy

Text is normalized with Unicode NFKC normalization, case folding, punctuation-to-space mapping,
and whitespace collapsing. The versioned policy then applies these rules in order:

1. A valid URI already present in history is accepted with confidence `1.0`.
2. An exact title/artist/album match to one URI is accepted with confidence `1.0`.
3. When history has no album, an exact title/artist match to one URI is accepted with the lower
   configured confidence `0.90`.
4. Multiple URIs with the same exact evidence are `ambiguous` and never selected.
5. Fuzzy title comparison is blocked to an exactly normalized artist. A sufficiently strong,
   separated candidate is labeled `fuzzy`, but remains unresolved and requires review.
6. A close fuzzy tie is `ambiguous`; no qualifying candidate is `unmatched`.

This deliberately favors precision over coverage. In particular, remasters, live recordings,
radio edits, and re-releases must not silently collapse into one recording.

## Private outputs

- `identity_matches.jsonl` contains every query, status, resolved URI when safe, and retained
  candidate evidence.
- `identity_resolution_report.json` contains only aggregate identity, play-count-weighted, and
  listening-time-weighted coverage plus method counts.
- `identity_review_sample.jsonl` is a deterministic status-stratified sample for manual quality
  review.

All three paths are under ignored data directories. Fuzzy and ambiguous matches are never written
as resolved IDs. The policy version is retained on every record and in the aggregate report.

## Remaining coverage work

This local catalog can only identify tracks that appear in the account's library or playlists.
Unmatched tracks require a permitted external metadata provider or the eventual Extended Streaming
History URI. A future provider adapter must preserve the same exact/fuzzy/ambiguous boundary,
record provider provenance, and be evaluated against a manually reviewed stratified sample before
its matches are used as training labels.
