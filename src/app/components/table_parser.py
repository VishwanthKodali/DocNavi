"""
Table parser — extracts row×col data from PDF table blocks.
Stitches multi-page tables by comparing consecutive page fragments.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.app.components.pdf_parser import PageData, TextBlock
from src.app.common.logger import get_logger

logger = get_logger("components.table_parser")

MIN_PIPE_COUNT = 2  # minimum | chars to treat a text block as a table


@dataclass
class ParsedTable:
    table_id: str
    section_id: str
    page_num: int
    headers: list[str]
    rows: list[list[str]]
    raw_text: str
    page_end: int = 0

    @property
    def serialised(self) -> str:
        """Human-readable text for embedding."""
        lines = [" | ".join(self.headers)] if self.headers else []
        for row in self.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)


def extract_tables_from_pages(pages: list[PageData], section_map: dict[int, str]) -> list[ParsedTable]:
    """
    Extract and stitch tables across all pages.
    section_map: {page_num -> section_id}
    """
    raw_tables: list[ParsedTable] = []

    for page in pages:
        sec_id = section_map.get(page.page_num, "unknown")
        for tb in page.text_blocks:
            if _looks_like_table(tb.text):
                pt = _parse_table_text(tb.text, page.page_num, sec_id)
                if pt:
                    raw_tables.append(pt)

    stitched = _stitch_multi_page(raw_tables)
    logger.info("Table parser produced %d tables (%d raw)", len(stitched), len(raw_tables))
    return stitched


def _looks_like_table(text: str) -> bool:
    pipe_count = text.count("|")
    lines = text.strip().split("\n")
    has_multiple_lines = len(lines) >= 2
    return pipe_count >= MIN_PIPE_COUNT and has_multiple_lines


def _parse_table_text(text: str, page_num: int, sec_id: str) -> ParsedTable | None:
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return None

    rows: list[list[str]] = []
    for line in lines:
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                rows.append(cells)

    if not rows:
        return None

    headers = rows[0]
    data_rows = rows[1:]
    # Skip separator rows (e.g. |---|---|)
    data_rows = [r for r in data_rows if not all(set(c) <= {"-", " "} for c in r)]

    tid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"table:{page_num}:{text[:40]}"))
    return ParsedTable(
        table_id=tid,
        section_id=sec_id,
        page_num=page_num,
        page_end=page_num,
        headers=headers,
        rows=data_rows,
        raw_text=text,
    )


def _stitch_multi_page(tables: list[ParsedTable]) -> list[ParsedTable]:
    """Merge consecutive tables that share the same number of columns."""
    if not tables:
        return tables

    merged: list[ParsedTable] = [tables[0]]
    for current in tables[1:]:
        prev = merged[-1]
        # Stitch if consecutive pages and same column count
        same_cols = len(current.headers) == len(prev.headers)
        consecutive = current.page_num == prev.page_end + 1
        if consecutive and same_cols and not current.headers == prev.headers:
            prev.rows.extend(current.rows)
            prev.page_end = current.page_num
            prev.raw_text += "\n" + current.raw_text
        else:
            merged.append(current)

    return merged
