"""Audio embedding extraction and aggregation."""

from myusic_engine.embeddings.discogs_effnet import (
    EMBEDDING_DIMENSIONS,
    FEATURE_NAME,
    FEATURE_SOURCE,
    MODEL_FILENAME,
    MODEL_LICENSE,
    MODEL_SHA256,
    MODEL_URL,
    SOURCE_VERSION,
    DiscogsEffnetOnnxBackend,
    EmbeddingAnalysis,
    PreparedDiscogsEffnetInput,
    download_model,
    file_sha256,
    prepare_discogs_effnet_input,
)
from myusic_engine.embeddings.pooling import (
    EmbeddingExtractionError,
    PooledEmbedding,
    embedding_observation,
    mean_pool_l2_normalize,
)

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "FEATURE_NAME",
    "FEATURE_SOURCE",
    "MODEL_FILENAME",
    "MODEL_LICENSE",
    "MODEL_SHA256",
    "MODEL_URL",
    "SOURCE_VERSION",
    "DiscogsEffnetOnnxBackend",
    "EmbeddingAnalysis",
    "EmbeddingExtractionError",
    "PooledEmbedding",
    "PreparedDiscogsEffnetInput",
    "download_model",
    "embedding_observation",
    "file_sha256",
    "mean_pool_l2_normalize",
    "prepare_discogs_effnet_input",
]
