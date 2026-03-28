"""
Hierarchical chunker.
Builds a section tree from classified blocks and emits Chunk objects.

Key fix: the chunker now processes ALL body_text blocks, not just those
after a heading has been seen. The section stack starts pre-seeded with
a "Preamble" section so content before the first heading is captured.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from src.app.components.content_classifier import ClassifiedBlock
from src.app.common.settings import settings
from src.app.common.logger import get_logger

logger = get_logger("components.chunker")

CROSS_REF_RE = re.compile(
    r"[Ss]ection\s+(\d+(?:\.\d+)*)|[Aa]ppendix\s+([A-Z])\b", re.IGNORECASE
)
ACRONYM_INLINE_RE = re.compile(r"([A-Z][A-Za-z\s\-]{3,40}?)\s+\(([A-Z]{2,8})\)")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_num: int
    page_end: int
    section_id: str
    section_title: str
    parent_section_id: str
    content_type: str = "text"
    cross_refs: list[str] = field(default_factory=list)
    acronyms: dict[str, str] = field(default_factory=dict)
    image_ref: str = ""
    table_ref: str = ""


def chunk_classified_blocks(blocks: list[ClassifiedBlock]) -> list[Chunk]:
    """Turn a flat list of classified blocks into hierarchical Chunk objects."""
    chunks: list[Chunk] = []

    # Stack entries: (section_id, section_title, page_num)
    # Pre-seeded so content before first heading is not lost
    section_stack: list[tuple[str, str, int]] = [("0", "Preamble", 1)]

    current_parts: list[str] = []
    current_page_start: int = 1
    current_page_end: int = 1

    def flush(page: int | None = None) -> None:
        nonlocal current_parts, current_page_start, current_page_end
        if not current_parts:
            return
        combined = " ".join(current_parts).strip()
        if not combined or len(combined) < 20:   # skip tiny fragments
            current_parts = []
            return

        sec_id, sec_title, _ = section_stack[-1]
        parent_id = section_stack[-2][0] if len(section_stack) > 1 else "0"
        end_page = page or current_page_end

        for sub_text in _split_by_tokens(combined, settings.chunk_max_tokens):
            cid = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{sec_id}:{current_page_start}:{sub_text[:50]}",
            ))
            chunks.append(Chunk(
                chunk_id=cid,
                text=sub_text,
                page_num=current_page_start,
                page_end=end_page,
                section_id=sec_id,
                section_title=sec_title,
                parent_section_id=parent_id,
                cross_refs=_extract_cross_refs(sub_text),
                acronyms=_extract_inline_acronyms(sub_text),
            ))
        current_parts = []
        current_page_start = end_page

    for cb in blocks:
        src = cb.source
        page = getattr(src, "page_num", current_page_end)
        current_page_end = page

        if cb.is_heading:
            flush(page)
            depth = max(cb.heading_depth, 1)
            heading_text = src.text.strip()  # type: ignore[union-attr]
            sec_id = _extract_section_id(heading_text)

            # Pop stack to correct depth
            while len(section_stack) >= depth:
                section_stack.pop()
            section_stack.append((sec_id, heading_text[:120], page))
            current_page_start = page

        elif cb.kind == "body_text":
            text = src.text.strip()  # type: ignore[union-attr]
            if text:
                current_parts.append(text)
                current_page_end = page

        # image/table blocks are handled separately in the ingestion pipeline

    flush()  # flush final pending parts

    logger.info(
        "Hierarchical chunker produced %d chunks from %d classified blocks",
        len(chunks), len(blocks),
    )
    return chunks


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split text respecting max token budget (word-count proxy: 0.75 words/token)."""
    max_words = max(1, int(max_tokens * 0.75))
    words = text.split()
    if len(words) <= max_words:
        return [text]
    parts: list[str] = []
    while words:
        parts.append(" ".join(words[:max_words]))
        words = words[max_words:]
    return parts


def _extract_section_id(heading_text: str) -> str:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", heading_text)
    if match:
        return match.group(1)
    # Appendix: "Appendix B — Title" → "appendix-B"
    app_match = re.match(r"^\s*[Aa]ppendix\s+([A-Z])", heading_text)
    if app_match:
        return f"appendix-{app_match.group(1)}"
    return heading_text[:30].strip()


def _extract_cross_refs(text: str) -> list[str]:
    refs: list[str] = []
    for m in CROSS_REF_RE.finditer(text):
        ref = m.group(1) or m.group(2)
        if ref:
            refs.append(ref)
    return list(dict.fromkeys(refs))


def _extract_inline_acronyms(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for m in ACRONYM_INLINE_RE.finditer(text):
        full_form = m.group(1).strip()
        abbrev = m.group(2)
        result[abbrev] = full_form.lower()
    return result
