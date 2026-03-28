"""
Ingestion pipeline orchestrator — fully async.

All I/O-heavy steps run asynchronously:
  - VLM image description  → async GPT-4o vision (concurrent via semaphore)
  - Text embedding          → async OpenAI embeddings (concurrent batches)
  - Qdrant upserts          → async qdrant-client

CPU-bound steps (PDF parsing, chunking, classification) remain synchronous
and are run in a thread pool executor so they don't block the event loop.

Entry point: run_ingestion(pdf_path) — synchronous wrapper that calls
asyncio.run() / nest_asyncio as appropriate. Called by routers.py background task.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from src.app.components.pdf_parser import parse_pdf, ImageBlock
from src.app.components.content_classifier import classify_page, ClassifiedBlock
from src.app.components.hierarchical_chunker import chunk_classified_blocks, Chunk
from src.app.components.table_parser import (
    extract_tables_from_pages, ParsedTable, build_page_table_map,
)
from src.app.components.vlm_descriptor import describe_images_batch, ImageDescription
from src.app.components.acronym_resolver import get_resolver
from src.app.components.embedder import embed_texts_async
from src.app.components.bm25_index import get_bm25_index
from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import IngestionError

logger = get_logger("components.ingestion")

UPSERT_BATCH = 50


@dataclass
class IngestionResult:
    total_pages: int = 0
    total_chunks: int = 0
    total_images: int = 0
    total_tables: int = 0
    status: str = "pending"
    error: str = ""


# ── Async pipeline ─────────────────────────────────────────────────────────

async def _run_ingestion_async(pdf_path: Path) -> IngestionResult:
    result = IngestionResult()

    # ── Step 1: PDF Parsing (CPU — run in thread pool) ────────────────────
    logger.info("=== INGESTION START: %s ===", pdf_path.name)
    loop = asyncio.get_event_loop()
    pages = await loop.run_in_executor(None, parse_pdf, pdf_path)
    result.total_pages = len(pages)

    # Collect all image blocks
    all_image_blocks: list[ImageBlock] = []
    for page in pages:
        all_image_blocks.extend(page.image_blocks)

    # ── Step 2: Content Classification (CPU — thread pool) ────────────────
    def _classify_all():
        classified: list[ClassifiedBlock] = []
        for page in pages:
            classified.extend(classify_page(page))
        return classified

    all_classified = await loop.run_in_executor(None, _classify_all)

    # ── Step 3: Hierarchical Chunking (CPU — thread pool) ─────────────────
    chunks: list[Chunk] = await loop.run_in_executor(
        None, chunk_classified_blocks, all_classified
    )
    result.total_chunks = len(chunks)

    # Build page → section_id map
    section_map: dict[int, str] = {}
    for c in chunks:
        for p in range(c.page_num, c.page_end + 1):
            if p not in section_map:
                section_map[p] = c.section_id

    # ── Step 4 + Step 5: VLM & Tables run concurrently ───────────────────
    # VLM is the slowest step (~90s async). Tables are fast.
    # Run both in parallel using asyncio.gather.
    logger.info(
        "Starting VLM (%d images) + table extraction (%d pages) concurrently...",
        len(all_image_blocks), result.total_pages,
    )

    async def _extract_tables_async():
        return await loop.run_in_executor(
            None, extract_tables_from_pages, pages, section_map
        )

    image_descriptions_task = asyncio.create_task(
        _run_vlm_async(all_image_blocks, section_map)
    )
    tables_task = asyncio.create_task(_extract_tables_async())

    image_descriptions, tables = await asyncio.gather(
        image_descriptions_task, tables_task
    )
    result.total_images = len(image_descriptions)
    result.total_tables = len(tables)

    # ── Step 6: Build page → ref maps, enrich chunks ──────────────────────
    fig_page_map: dict[int, str] = {}
    for desc in image_descriptions:
        if desc.page_num not in fig_page_map:
            fig_page_map[desc.page_num] = desc.figure_id

    page_table_map: dict[int, str] = build_page_table_map(tables)

    for chunk in chunks:
        if chunk.page_num in fig_page_map and not chunk.image_ref:
            chunk.image_ref = fig_page_map[chunk.page_num]
        if chunk.page_num in page_table_map and not chunk.table_ref:
            chunk.table_ref = page_table_map[chunk.page_num]

    # ── Step 7: Acronym Resolver ──────────────────────────────────────────
    resolver = get_resolver()
    await loop.run_in_executor(
        None, resolver.build_from_chunks, [c.text for c in chunks]
    )

    # ── Steps 8-10: Embed all three collections concurrently ──────────────
    logger.info(
        "Embedding concurrently: %d chunks | %d images | %d tables",
        len(chunks), len(image_descriptions), len(tables),
    )
    await asyncio.gather(
        _embed_and_upsert_chunks(chunks),
        _embed_and_upsert_images(image_descriptions),
        _embed_and_upsert_tables(tables),
    )

    # ── Step 11: BM25 index (CPU — thread pool) ───────────────────────────
    logger.info("Building BM25 index...")
    bm25 = get_bm25_index()
    await loop.run_in_executor(
        None,
        lambda: bm25.build(
            chunk_ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
        ),
    )
    await loop.run_in_executor(None, bm25.save)

    result.status = "complete"
    logger.info(
        "=== INGESTION COMPLETE: %d pages | %d chunks | %d images | %d tables ===",
        result.total_pages, result.total_chunks,
        result.total_images, result.total_tables,
    )
    return result


async def _run_vlm_async(
    image_blocks: list[ImageBlock],
    section_map: dict[int, str],
) -> list[ImageDescription]:
    """Thin async wrapper — describe_images_batch already handles async internally."""
    loop = asyncio.get_event_loop()
    # describe_images_batch uses its own asyncio.run / nest_asyncio logic.
    # Since we ARE already in an async context here, we call the internal
    # async function directly to avoid nested event loop issues.
    from src.app.components.vlm_descriptor import _describe_all_async
    return await _describe_all_async(image_blocks, section_map)


# ── Async Qdrant upsert helpers ────────────────────────────────────────────

def _get_async_qdrant() -> AsyncQdrantClient:
    kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return AsyncQdrantClient(**kwargs)


async def _batch_upsert_async(
    client: AsyncQdrantClient,
    collection: str,
    points: list[qm.PointStruct],
) -> None:
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i: i + UPSERT_BATCH]
        await client.upsert(collection_name=collection, points=batch)
    logger.info("Upserted %d points into '%s'", len(points), collection)


async def _embed_and_upsert_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    embeddings = await embed_texts_async([c.text for c in chunks])
    client = _get_async_qdrant()
    points = [
        qm.PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload={
                "text": chunk.text,
                "page_num": chunk.page_num,
                "page_end": chunk.page_end,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "parent_section_id": chunk.parent_section_id,
                "content_type": "text",
                "cross_refs": chunk.cross_refs,
                "acronyms": chunk.acronyms,
                "image_ref": chunk.image_ref,
                "table_ref": chunk.table_ref,
            },
        )
        for chunk, vector in zip(chunks, embeddings)
    ]
    await _batch_upsert_async(client, settings.collection_text, points)


async def _embed_and_upsert_images(descs: list[ImageDescription]) -> None:
    if not descs:
        return
    embeddings = await embed_texts_async([d.description for d in descs])
    client = _get_async_qdrant()
    points = [
        qm.PointStruct(
            id=desc.figure_id,
            vector=vector,
            payload={
                "text": desc.description,
                "figure_id": desc.figure_id,
                "page_num": desc.page_num,
                "section_id": desc.section_id,
                "description": desc.description,
                "image_name": desc.image_name,
                "image_path": desc.image_path,
                "width": desc.width,
                "height": desc.height,
                "content_type": "image",
            },
        )
        for desc, vector in zip(descs, embeddings)
    ]
    await _batch_upsert_async(client, settings.collection_images, points)


async def _embed_and_upsert_tables(tables: list[ParsedTable]) -> None:
    if not tables:
        return
    embeddings = await embed_texts_async([t.serialised for t in tables])
    client = _get_async_qdrant()
    points = [
        qm.PointStruct(
            id=table.table_id,
            vector=vector,
            payload={
                "text": table.serialised,
                "table_id": table.table_id,
                "section_id": table.section_id,
                "page_num": table.page_num,
                "page_end": table.page_end,
                "headers": table.headers,
                "rows": table.rows,
                "content_type": "table",
            },
        )
        for table, vector in zip(tables, embeddings)
    ]
    await _batch_upsert_async(client, settings.collection_tables, points)


# ── Public sync entry point ────────────────────────────────────────────────

def run_ingestion(pdf_path: str | Path) -> IngestionResult:
    """
    Synchronous entry point called by routers.py background task.
    Runs the async pipeline. Because FastAPI's BackgroundTasks run inside
    uvicorn's event loop, we use nest_asyncio to allow asyncio.run() inside
    an already-running loop.
    """
    pdf_path = Path(pdf_path)
    result = IngestionResult()
    try:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass  # Not inside a running loop — asyncio.run() will work fine

        result = asyncio.run(_run_ingestion_async(pdf_path))

    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        logger.exception("Ingestion failed: %s", exc)
        raise IngestionError(str(exc)) from exc
    finally:
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass
    return result
