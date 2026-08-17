# Local data area

This directory documents where local pipeline inputs and outputs belong. Its data
subdirectories are intentionally ignored by Git.

```text
data/
  private/     Original account exports and other sensitive inputs
  raw/         Immutable, privacy-cleaned source records
  interim/     Resolved identities and intermediate computations
  processed/   Aggregate model-ready tables and local reports
```

Keep the Spotify export ZIP in `data/private/`. Do not rename a real export to match a test
fixture, and never force-add ignored data. Tests use only reserved example values in
`tests/fixtures/`.
