"""
BM25 index — wraps rank_bm25 with build, save, load, and query operations.
The index is built from text_chunks stored in Qdrant (fetched at build time)
and serialised to disk with pickle so it survives server restarts.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import BM25IndexError

logger = get_logger("components.bm25")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for",
    "is", "are", "was", "were", "be", "been", "this", "that",
    "it", "its", "with", "as", "at", "by", "from", "on", "not",
}


@dataclass
class BM25Index:
    bm25: BM25Okapi | None = None
    chunk_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    def is_ready(self) -> bool:
        return self.bm25 is not None and len(self.chunk_ids) > 0

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if not chunk_ids or not texts:
            logger.warning("BM25 build called with empty corpus — skipping.")
            return
        if len(chunk_ids) != len(texts):
            raise BM25IndexError("chunk_ids and texts must have the same length")
        tokenized = [_tokenize(t) for t in texts]
        # Filter out completely empty token lists to avoid BM25Okapi division by zero
        valid = [(cid, toks, txt) for cid, toks, txt in zip(chunk_ids, tokenized, texts) if toks]
        if not valid:
            logger.warning("BM25 build: all tokenized texts are empty — skipping.")
            return
        valid_ids, valid_toks, valid_texts = zip(*valid)
        self.bm25 = BM25Okapi(list(valid_toks), k1=1.5, b=0.75)
        self.chunk_ids = list(valid_ids)
        self.texts = list(valid_texts)
        logger.info("BM25 index built with %d documents", len(self.texts))

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Return top_k results as list of {chunk_id, score, text}."""
        if not self.is_ready():
            logger.warning("BM25 index not ready — returning empty results.")
            return []
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)  # type: ignore[union-attr]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[i],
                "score": float(scores[i]),
                "text": self.texts[i],
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    def save(self, path: str | Path | None = None) -> Path:
        if not self.is_ready():
            logger.warning("BM25 index is empty — skipping save.")
            return Path(path or settings.bm25_index_path)
        save_path = Path(path or settings.bm25_index_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids, "texts": self.texts}, f)
        logger.info("BM25 index saved to %s", save_path)
        return save_path

    def load(self, path: str | Path | None = None) -> None:
        load_path = Path(path or settings.bm25_index_path)
        if not load_path.exists():
            raise BM25IndexError(f"BM25 index file not found: {load_path}")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.texts = data["texts"]
        logger.info("BM25 index loaded from %s (%d docs)", load_path, len(self.chunk_ids))


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# Module-level singleton
_bm25_index: BM25Index = BM25Index()


def get_bm25_index() -> BM25Index:
    return _bm25_index
