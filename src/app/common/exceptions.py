class NasaRAGError(Exception):
    """Base exception for all NASA-RAG errors."""


class IngestionError(NasaRAGError):
    """Raised when ingestion pipeline fails."""


class QueryError(NasaRAGError):
    """Raised when query pipeline fails."""


class VectorStoreError(NasaRAGError):
    """Raised on Qdrant operation failure."""


class EmbeddingError(NasaRAGError):
    """Raised when embedding generation fails."""


class VLMError(NasaRAGError):
    """Raised when GPT-4o vision call fails."""


class NotFoundError(NasaRAGError):
    """Raised when a requested resource is not found."""


class BM25IndexError(NasaRAGError):
    """Raised on BM25 index build/load failure."""
