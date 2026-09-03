"""Track identity resolution."""

from myusic_engine.matching.account_catalog import (
    load_account_catalog,
    load_account_playlist,
    write_account_playlist_report,
)
from myusic_engine.matching.external import (
    ExternalIdentityMatch,
    ExternalIdentityPolicy,
    ExternalIdentityReport,
    ExternalIdentityResult,
    external_review_sample,
    read_external_identity_matches,
    resolve_external_identities,
    write_external_identity_resolution,
)
from myusic_engine.matching.models import (
    AccountPlaylist,
    CatalogLoadResult,
    CatalogTrack,
    IdentityInputError,
    IdentityMatch,
    IdentityResolutionError,
    IdentityResolutionReport,
    IdentityResolutionResult,
    MatchCandidate,
    TrackQuery,
)
from myusic_engine.matching.resolver import (
    IdentityPolicy,
    load_identity_policy,
    normalize_metadata,
    read_track_queries,
    resolve_identities,
    review_sample,
    write_identity_resolution,
)

__all__ = [
    "AccountPlaylist",
    "CatalogLoadResult",
    "CatalogTrack",
    "ExternalIdentityMatch",
    "ExternalIdentityPolicy",
    "ExternalIdentityReport",
    "ExternalIdentityResult",
    "IdentityInputError",
    "IdentityMatch",
    "IdentityPolicy",
    "IdentityResolutionError",
    "IdentityResolutionReport",
    "IdentityResolutionResult",
    "MatchCandidate",
    "TrackQuery",
    "load_account_catalog",
    "load_account_playlist",
    "load_identity_policy",
    "normalize_metadata",
    "external_review_sample",
    "read_external_identity_matches",
    "read_track_queries",
    "resolve_identities",
    "resolve_external_identities",
    "review_sample",
    "write_identity_resolution",
    "write_external_identity_resolution",
    "write_account_playlist_report",
]
