"""
Ingestion pipeline orchestrator — Diagram 1.
Coordinates: PDF parse → classify → chunk → VLM → table parse →
metadata enrich → acronym build → embed → Qdrant upsert → BM25 build.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from dataclasses import dataclass

from qdrant_client.http import models as qm

from src.app.components.pdf_parser import parse_pdf, ImageBlock
from src.app.components.content_classifier import classify_page, ClassifiedBlock
from src.app.components.hierarchical_chunker import chunk_classified_blocks, Chunk
from src.app.components.table_parser import extract_tables_from_pages, ParsedTable
from src.app.components.vlm_descriptor import describe_images_batch, ImageDescription
from src.app.components.acronym_resolver import get_resolver
from src.app.components.embedder import embed_texts
from src.app.components.bm25_index import get_bm25_index
from src.app.database.qdrant_client import get_qdrant_client
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


def run_ingestion(pdf_path: str | Path) -> IngestionResult:
    result = IngestionResult()
    pdf_path = Path(pdf_path)

    try:
        # ── 1. PDF Parsing ────────────────────────────────────────────────
        logger.info("=== INGESTION START: %s ===", pdf_path.name)
        pages = parse_pdf(pdf_path)
        result.total_pages = len(pages)

        # ── 2. Content Classification ─────────────────────────────────────
        all_classified: list[ClassifiedBlock] = []
        all_image_blocks: list[ImageBlock] = []
        for page in pages:
            classified = classify_page(page)
            all_classified.extend(classified)
            all_image_blocks.extend(page.image_blocks)

        # ── 3. Hierarchical Chunking ──────────────────────────────────────
        chunks: list[Chunk] = chunk_classified_blocks(all_classified)
        result.total_chunks = len(chunks)

        # Build section_map: page → section_id (from first chunk of that page)
        section_map: dict[int, str] = {}
        for c in chunks:
            for p in range(c.page_num, c.page_end + 1):
                if p not in section_map:
                    section_map[p] = c.section_id

        # ── 4. VLM Image Description ──────────────────────────────────────
        logger.info("Running VLM on %d images...", len(all_image_blocks))
        image_descriptions: list[ImageDescription] = describe_images_batch(all_image_blocks, section_map)
        result.total_images = len(image_descriptions)

        # ── 5. Table Parsing ──────────────────────────────────────────────
        tables: list[ParsedTable] = extract_tables_from_pages(pages, section_map)
        result.total_tables = len(tables)

        # ── 6. Acronym Resolver Build ─────────────────────────────────────
        resolver = get_resolver()
        resolver.build_from_chunks([c.text for c in chunks])

        # Enrich chunks with any newly discovered acronyms
        for chunk in chunks:
            if not chunk.acronyms:
                from src.app.components.hierarchical_chunker import _extract_inline_acronyms
                chunk.acronyms = _extract_inline_acronyms(chunk.text)

        # ── 7. Embed & Upsert text_chunks ─────────────────────────────────
        logger.info("Embedding %d text chunks...", len(chunks))
        _upsert_text_chunks(chunks)

        # ── 8. Embed & Upsert image_store ─────────────────────────────────
        logger.info("Embedding %d image descriptions...", len(image_descriptions))
        _upsert_image_descriptions(image_descriptions)

        # ── 9. Embed & Upsert table_store ─────────────────────────────────
        logger.info("Embedding %d tables...", len(tables))
        _upsert_tables(tables)

        # ── 10. Build & Save BM25 index ───────────────────────────────────
        logger.info("Building BM25 index...")
        bm25 = get_bm25_index()
        bm25.build(
            chunk_ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
        )
        bm25.save()

        result.status = "complete"
        logger.info(
            "=== INGESTION COMPLETE: %d pages | %d chunks | %d images | %d tables ===",
            result.total_pages, result.total_chunks, result.total_images, result.total_tables,
        )

    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        logger.exception("Ingestion failed: %s", exc)
        raise IngestionError(str(exc)) from exc

    return result


def _upsert_text_chunks(chunks: list[Chunk]) -> None:
    client = get_qdrant_client()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    points: list[qm.PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(qm.PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload={
                "text": chunk.text,
                "page_num": chunk.page_num,
                "page_end": chunk.page_end,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "parent_section_id": chunk.parent_section_id,
                "content_type": chunk.content_type,
                "cross_refs": chunk.cross_refs,
                "acronyms": chunk.acronyms,
                "image_ref": chunk.image_ref,
                "table_ref": chunk.table_ref,
            },
        ))

    _batch_upsert(client, settings.collection_text, points)


def _upsert_image_descriptions(descs: list[ImageDescription]) -> None:
    if not descs:
        return
    client = get_qdrant_client()
    texts = [d.description for d in descs]
    embeddings = embed_texts(texts)

    points: list[qm.PointStruct] = []
    for desc, vector in zip(descs, embeddings):
        points.append(qm.PointStruct(
            id=desc.figure_id,
            vector=vector,
            payload={
                "text": desc.description,
                "figure_id": desc.figure_id,
                "page_num": desc.page_num,
                "section_id": desc.section_id,
                "description": desc.description,
                "image_path": desc.image_path,
                "content_type": "image",
            },
        ))

    _batch_upsert(client, settings.collection_images, points)


def _upsert_tables(tables: list[ParsedTable]) -> None:
    if not tables:
        return
    client = get_qdrant_client()
    texts = [t.serialised for t in tables]
    embeddings = embed_texts(texts)

    points: list[qm.PointStruct] = []
    for table, vector in zip(tables, embeddings):
        points.append(qm.PointStruct(
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
        ))

    _batch_upsert(client, settings.collection_tables, points)


def _batch_upsert(client, collection: str, points: list[qm.PointStruct]) -> None:
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i: i + UPSERT_BATCH]
        client.upsert(collection_name=collection, points=batch)
    logger.info("Upserted %d points into '%s'", len(points), collection)
