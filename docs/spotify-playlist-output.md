# Spotify private-playlist output

The ranking stage ends with an ordered `spotify_playlist_uris.txt`. That file is useful on its own
and does not require a Spotify developer application. Remote publication is a separate, explicitly
armed boundary because it mutates the user's account.

## Supported API contract

The implementation follows Spotify's current public Web API reference:

- [Create Playlist](https://developer.spotify.com/documentation/web-api/reference/create-playlist)
  with `POST /me/playlists` and `public: false`;
- [Add Items to Playlist](https://developer.spotify.com/documentation/web-api/reference/add-items-to-playlist)
  with `POST /playlists/{playlist_id}/items`, in batches of no more than 100 URIs;
- [Get Playlist Items](https://developer.spotify.com/documentation/web-api/reference/get-playlists-items)
  with `GET /playlists/{playlist_id}/items` to validate and resume ordered progress;
- Spotify's documented
  [OAuth scopes](https://developer.spotify.com/documentation/web-api/concepts/scopes), specifically
  `playlist-modify-private` and `playlist-read-private`.

No legacy private endpoint, consumer-client token, password, client secret, stream, or offline file
is used. Development Mode account and app restrictions still apply.

## Two-step safety boundary

Running `publish-spotify-playlist` without `--execute` performs no network request. It writes
`spotify_playlist_plan.json`, whose SHA-256 plan ID covers:

- schema version;
- playlist name and description;
- the invariant `public: false` setting;
- every canonical Spotify track URI in order.

The execute form recomputes that plan from the same arguments. If a receipt already exists for a
different plan, execution stops before touching Spotify.

An access token is read only from `SPOTIFY_ACCESS_TOKEN` by default. The token is never accepted as
a command-line value and is never written into the plan, receipt, logs, or exception messages. A
different environment-variable name may be selected with `--access-token-env`.

## Checkpoint and resume behavior

After Spotify confirms playlist creation, the command atomically writes
`spotify_playlist_receipt.json`. It then:

1. reads all current remote items;
2. requires them to equal an exact prefix of the planned URI sequence;
3. appends the next batch of at most 100 items;
4. checkpoints the confirmed count and latest snapshot ID;
5. validates the complete remote order before reporting success.

If execution stops after an append, rerun the exact same command. The remote-prefix check discovers
whether Spotify applied the previous batch and prevents duplicate appends. If a person or another
client inserted, removed from the middle, reordered, or replaced items, the command refuses to
guess. Preserve the receipt and inspect the private playlist manually.

The Web API does not provide an idempotency key for playlist creation. If the connection fails
before the initial create response is confirmed, inspect Spotify before retrying so a second empty
playlist is not created. The client intentionally does not retry account mutations automatically.

## Authorization boundary

Building and reviewing a plan does not authorize publication. Set a user-approved, short-lived
OAuth token in the current process environment only when ready, then add `--execute`. Clear the
environment variable after use. A live smoke test is deliberately excluded from automated tests;
tests use an in-memory gateway and synthetic Spotify-shaped IDs.
