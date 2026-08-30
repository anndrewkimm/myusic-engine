"""Supported Spotify output operations.

The current safe handoff is the ordered Spotify URI file emitted by
``ranking.write_recommendations``. Playlist mutation stays separate because it requires a user's
explicit OAuth authorization and a current supported Spotify API operation.
"""
