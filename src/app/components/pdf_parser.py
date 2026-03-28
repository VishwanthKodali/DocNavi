"""
PyMuPDF page parser — robust extraction for the NASA handbook.
Uses page.get_text("dict") which is more reliable than "rawdict" for
scanned/tagged PDFs. Falls back to plain text extraction if dict mode
returns no blocks.
"""
from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF

from src.app.common.logger import get_logger
from src.app.common.exceptions import IngestionError

logger = get_logger("components.pdf_parser")


@dataclass
class TextBlock:
    block_type: Literal["text"] = "text"
    text: str = ""
    font_size: float = 12.0
    font_flags: int = 0
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    page_num: int = 0
    block_index: int = 0


@dataclass
class ImageBlock:
    block_type: Literal["image"] = "image"
    image_bytes: bytes = b""
    image_b64: str = ""
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    page_num: int = 0
    block_index: int = 0
    width: int = 0
    height: int = 0


@dataclass
class TableBlock:
    block_type: Literal["table"] = "table"
    raw_text: str = ""
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    page_num: int = 0
    block_index: int = 0


@dataclass
class PageData:
    page_num: int
    text_blocks: list[TextBlock] = field(default_factory=list)
    image_blocks: list[ImageBlock] = field(default_factory=list)
    table_blocks: list[TableBlock] = field(default_factory=list)
    modal_font_size: float = 12.0


def parse_pdf(pdf_path: str | Path) -> list[PageData]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise IngestionError(f"PDF not found: {pdf_path}")

    logger.info("Parsing PDF: %s", pdf_path.name)
    pages: list[PageData] = []

    doc = fitz.open(str(pdf_path))
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            pd = _parse_page(page, page_num)
            pages.append(pd)
            if page_num % 50 == 0:
                logger.info("  Parsed %d / %d pages", page_num, len(doc))
    finally:
        doc.close()

    total_blocks = sum(len(p.text_blocks) for p in pages)
    logger.info("PDF parsing complete. %d pages, %d text blocks.", len(pages), total_blocks)
    return pages


def _parse_page(page: fitz.Page, page_num: int) -> PageData:
    pd = PageData(page_num=page_num)
    font_sizes: list[float] = []

    # ── Primary: structured dict extraction ──────────────────────────────
    try:
        raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = raw.get("blocks", [])
    except Exception:
        blocks = []

    structured_text_found = False

    for block_idx, block in enumerate(blocks):
        btype = block.get("type", -1)

        if btype == 0:  # text block
            lines_text: list[str] = []
            max_font: float = 0.0
            font_flags: int = 0

            for line in block.get("lines", []):
                line_parts: list[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_parts.append(span_text)
                        fs = float(span.get("size", 12.0))
                        if fs > max_font:
                            max_font = fs
                            font_flags = span.get("flags", 0)
                        font_sizes.append(fs)
                if line_parts:
                    lines_text.append(" ".join(line_parts))

            text = "\n".join(lines_text).strip()
            if not text:
                continue

            structured_text_found = True
            if max_font == 0.0:
                max_font = 12.0

            tb = TextBlock(
                text=text,
                font_size=max_font,
                font_flags=font_flags,
                bbox=tuple(block["bbox"]),
                page_num=page_num,
                block_index=block_idx,
            )
            pd.text_blocks.append(tb)

        elif btype == 1:  # image block
            _extract_image_block(page, block, page_num, block_idx, pd)

    # ── Fallback: plain text extraction if structured returned nothing ────
    if not structured_text_found:
        plain = page.get_text("text").strip()
        if plain:
            logger.debug("Page %d: using plain text fallback (%d chars)", page_num, len(plain))
            # Split into pseudo-blocks by double newline
            for idx, para in enumerate(plain.split("\n\n")):
                para = para.strip()
                if len(para) < 10:
                    continue
                pd.text_blocks.append(TextBlock(
                    text=para,
                    font_size=12.0,
                    font_flags=0,
                    bbox=(0, 0, 0, 0),
                    page_num=page_num,
                    block_index=idx,
                ))

    # ── Modal font size ───────────────────────────────────────────────────
    if font_sizes:
        rounded = [round(fs, 1) for fs in font_sizes]
        pd.modal_font_size = Counter(rounded).most_common(1)[0][0]

    return pd


def _extract_image_block(
    page: fitz.Page,
    block: dict,
    page_num: int,
    block_idx: int,
    pd: PageData,
) -> None:
    try:
        clip = fitz.Rect(block["bbox"])
        # Skip tiny blocks (likely decorative lines / bullets)
        if clip.width < 40 or clip.height < 40:
            return
        pix = page.get_pixmap(clip=clip, dpi=120)
        img_bytes = pix.tobytes("png")
        ib = ImageBlock(
            image_bytes=img_bytes,
            image_b64=base64.b64encode(img_bytes).decode(),
            bbox=tuple(block["bbox"]),
            page_num=page_num,
            block_index=block_idx,
            width=int(clip.width),
            height=int(clip.height),
        )
        pd.image_blocks.append(ib)
    except Exception as exc:
        logger.debug("Page %d block %d image extraction failed: %s", page_num, block_idx, exc)
