"""
Embedder — wraps OpenAI text-embedding-3-small with batched calls.
Returns List[List[float]] matching input order.
"""
from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import EmbeddingError

logger = get_logger("components.embedder")

BATCH_SIZE = 100  # OpenAI allows up to 2048 but we keep it conservative


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches. Returns embeddings in same order."""
    if not texts:
        return []
    client = OpenAI(api_key=settings.openai_api_key)
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start: batch_start + BATCH_SIZE]
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)
            logger.debug("Embedded batch %d-%d", batch_start, batch_start + len(batch))
        except Exception as exc:
            raise EmbeddingError(f"Embedding batch {batch_start} failed: {exc}") from exc

    return all_embeddings


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for a single text."""
    results = embed_texts([text])
    return results[0] if results else []
