"""Track identity resolution."""

from myusic_engine.matching.account_catalog import load_account_catalog
from myusic_engine.matching.models import (
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
    "CatalogLoadResult",
    "CatalogTrack",
    "IdentityInputError",
    "IdentityMatch",
    "IdentityPolicy",
    "IdentityResolutionError",
    "IdentityResolutionReport",
    "IdentityResolutionResult",
    "MatchCandidate",
    "TrackQuery",
    "load_account_catalog",
    "load_identity_policy",
    "normalize_metadata",
    "read_track_queries",
    "resolve_identities",
    "review_sample",
    "write_identity_resolution",
]
