"""
Table parser — post-processing of TableBlock objects from pdf_parser.

Since pdfplumber already does the heavy lifting of table detection and
row/col extraction, this module focuses on:
  1. Multi-page table stitching (tables that span page boundaries)
  2. Generating ParsedTable objects for Qdrant upsert
  3. Building a page → table_id mapping for cross-reference in chunks
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.app.components.pdf_parser import PageData, TableBlock
from src.app.common.logger import get_logger

logger = get_logger("components.table_parser")


@dataclass
class ParsedTable:
    table_id: str
    section_id: str
    page_num: int
    page_end: int
    headers: list[str]
    rows: list[list[str]]
    raw_text: str       # serialised for embedding

    @property
    def serialised(self) -> str:
        return self.raw_text


def extract_tables_from_pages(
    pages: list[PageData],
    section_map: dict[int, str],
) -> list[ParsedTable]:
    """
    Convert all TableBlock objects from PageData into ParsedTable objects.
    Stitches consecutive same-column-count tables across page boundaries.
    section_map: {page_num → section_id}
    """
    raw: list[ParsedTable] = []

    for page in pages:
        sec_id = section_map.get(page.page_num, "unknown")
        for tb in page.table_blocks:
            if not tb.rows and not tb.raw_text:
                continue
            tid = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"table:{page.page_num}:{tb.block_index}:{tb.raw_text[:40]}",
            ))
            raw.append(ParsedTable(
                table_id=tid,
                section_id=sec_id,
                page_num=page.page_num,
                page_end=page.page_num,
                headers=tb.headers,
                rows=tb.rows,
                raw_text=tb.raw_text,
            ))

    stitched = _stitch_multi_page(raw)
    logger.info("Table parser: %d ParsedTable objects (%d raw blocks)", len(stitched), len(raw))
    return stitched


def _stitch_multi_page(tables: list[ParsedTable]) -> list[ParsedTable]:
    """
    Merge consecutive tables that share the same column count and span
    adjacent pages — common for long NASA handbook tables split across pages.
    """
    if not tables:
        return tables

    merged: list[ParsedTable] = [tables[0]]
    for current in tables[1:]:
        prev = merged[-1]
        consecutive = current.page_num == prev.page_end + 1
        same_cols   = len(current.headers) == len(prev.headers)
        # Don't stitch if current has a different header (new table starting)
        different_header = current.headers != prev.headers and any(h.strip() for h in current.headers)

        if consecutive and same_cols and not different_header:
            prev.rows.extend(current.rows)
            prev.page_end = current.page_num
            prev.raw_text += "\n" + current.raw_text
        else:
            merged.append(current)

    return merged


def build_page_table_map(tables: list[ParsedTable]) -> dict[int, str]:
    """Return {page_num → table_id} for the first table on each page."""
    result: dict[int, str] = {}
    for t in tables:
        for pg in range(t.page_num, t.page_end + 1):
            if pg not in result:
                result[pg] = t.table_id
    return result
