"""
Qdrant singleton client + collection bootstrap.
Three collections: text_chunks, image_store, table_store.
Payload indexes are created for fast filtered lookups.
"""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import VectorStoreError

logger = get_logger("database.qdrant")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant client instance."""
    try:
        kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        client = QdrantClient(**kwargs)
        client.get_collections()  # connectivity probe
        logger.info("Qdrant client connected at %s:%s", settings.qdrant_host, settings.qdrant_port)
        return client
    except Exception as exc:
        raise VectorStoreError(f"Failed to connect to Qdrant: {exc}") from exc


def _vector_config() -> qm.VectorParams:
    return qm.VectorParams(
        size=settings.embedding_dimensions,
        distance=qm.Distance.COSINE,
    )


def _ensure_collection(client: QdrantClient, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(collection_name=name, vectors_config=_vector_config())
        logger.info("Created Qdrant collection: %s", name)
    else:
        logger.debug("Collection already exists: %s", name)


def _ensure_payload_index(client: QdrantClient, collection: str, field: str, schema: qm.PayloadSchemaType) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )
    except Exception:
        pass  # index already exists — Qdrant raises, we swallow


def bootstrap_collections() -> None:
    """Idempotently create all three collections and their payload indexes."""
    client = get_qdrant_client()

    # ── text_chunks ──────────────────────────────────────────────────────────
    _ensure_collection(client, settings.collection_text)
    _ensure_payload_index(client, settings.collection_text, "section_id", qm.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, settings.collection_text, "page_num", qm.PayloadSchemaType.INTEGER)
    _ensure_payload_index(client, settings.collection_text, "content_type", qm.PayloadSchemaType.KEYWORD)

    # ── image_store ───────────────────────────────────────────────────────────
    _ensure_collection(client, settings.collection_images)
    _ensure_payload_index(client, settings.collection_images, "page_num", qm.PayloadSchemaType.INTEGER)
    _ensure_payload_index(client, settings.collection_images, "figure_id", qm.PayloadSchemaType.KEYWORD)

    # ── table_store ───────────────────────────────────────────────────────────
    _ensure_collection(client, settings.collection_tables)
    _ensure_payload_index(client, settings.collection_tables, "section_id", qm.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, settings.collection_tables, "page_num", qm.PayloadSchemaType.INTEGER)

    logger.info("All Qdrant collections bootstrapped.")
