"""Rate-limited, cached clients for permitted external metadata and features."""

from myusic_engine.providers.acousticbrainz import (
    AcousticBrainzClient,
    AcousticBrainzDocument,
    AcousticBrainzProvider,
)
from myusic_engine.providers.acousticbrainz_bulk import (
    AcousticBrainzBulkError,
    BulkDumpScanReport,
    OfflineAcousticBrainzProvider,
    build_offline_acousticbrainz_provider,
)
from myusic_engine.providers.http import JsonCacheTransport, ProviderError
from myusic_engine.providers.listenbrainz import (
    ListenBrainzMapping,
    ListenBrainzMappingClient,
    MusicBrainzMapper,
)

__all__ = [
    "AcousticBrainzBulkError",
    "AcousticBrainzClient",
    "AcousticBrainzDocument",
    "AcousticBrainzProvider",
    "BulkDumpScanReport",
    "JsonCacheTransport",
    "ListenBrainzMapping",
    "ListenBrainzMappingClient",
    "MusicBrainzMapper",
    "OfflineAcousticBrainzProvider",
    "ProviderError",
    "build_offline_acousticbrainz_provider",
]
