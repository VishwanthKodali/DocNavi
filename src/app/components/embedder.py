"""
Embedder — OpenAI text-embedding-3-small, batched.

Provides both sync and async versions:
  - embed_texts()       sync  — used nowhere now, kept for external callers
  - embed_single()      sync  — used by query_pipeline (LangGraph nodes are sync)
  - embed_texts_async() async — used by ingestion_pipeline (fully async)
  - embed_single_async()async — available for async query paths if needed

Batching: OpenAI allows up to 2048 inputs per call, we use 100 conservatively.
Order: embeddings are sorted by index before returning so they always match input order.
"""
from __future__ import annotations

import asyncio

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import EmbeddingError

logger = get_logger("components.embedder")

BATCH_SIZE = 100
MAX_CHARS = 30000  # ~8000 tokens roughly (avg ~3.75 chars per token)


# ── Truncation helper ───────────────────────────────────────────────────────

def _truncate(text: str) -> str:
    """Truncate text to approximate token limit using character count."""
    if len(text) > MAX_CHARS:
        logger.warning("Truncating text from %d chars to %d", len(text), MAX_CHARS)
        return text[:MAX_CHARS]
    return text


# ── Sync (used by query_pipeline LangGraph nodes) ──────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts synchronously in batches."""
    if not texts:
        return []
    client = OpenAI(api_key=settings.openai_api_key)
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [_truncate(t) for t in texts[start: start + BATCH_SIZE]]
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            batch_emb = [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            all_embeddings.extend(batch_emb)
            logger.debug("Embedded batch %d–%d", start, start + len(batch))
        except Exception as exc:
            raise EmbeddingError(f"Embedding batch {start} failed: {exc}") from exc
    return all_embeddings


def embed_single(text: str) -> list[float]:
    """Embed a single text synchronously."""
    results = embed_texts([text])
    return results[0] if results else []


# ── Async (used by ingestion_pipeline) ─────────────────────────────────────

async def _embed_batch_async(
    client: AsyncOpenAI,
    batch: list[str],
    start: int,
    attempt: int = 0,
) -> list[list[float]]:
    """Embed one batch asynchronously with manual retry."""
    batch = [_truncate(t) for t in batch]
    try:
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        return [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]
    except Exception as exc:
        if attempt < 2:
            wait = 2 ** (attempt + 1)
            logger.debug("Embedding batch %d retry %d in %ds: %s", start, attempt + 1, wait, exc)
            await asyncio.sleep(wait)
            return await _embed_batch_async(client, batch, start, attempt + 1)
        raise EmbeddingError(f"Embedding batch {start} failed after 3 attempts: {exc}") from exc


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts asynchronously.
    Batches are sent concurrently (all at once) — safe because
    OpenAI's embeddings endpoint has generous rate limits.
    """
    if not texts:
        return []

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    batches = [
        texts[start: start + BATCH_SIZE]
        for start in range(0, len(texts), BATCH_SIZE)
    ]

    logger.debug("Embedding %d texts in %d async batches", len(texts), len(batches))

    tasks = [
        _embed_batch_async(client, batch, idx * BATCH_SIZE)
        for idx, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks)

    # Flatten batch results back into a single list (order preserved)
    all_embeddings: list[list[float]] = []
    for batch_result in results:
        all_embeddings.extend(batch_result)

    return all_embeddings


async def embed_single_async(text: str) -> list[float]:
    """Embed a single text asynchronously."""
    results = await embed_texts_async([text])
    return results[0] if results else []