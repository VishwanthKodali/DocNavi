"""
PyMuPDF + pypdf + pdfplumber page parser.

Why three libraries:
  - fitz (PyMuPDF)  : primary text block extraction with font metadata
  - pypdf           : image byte extraction (page.images gives real JPEG/PNG bytes)
  - pdfplumber      : table extraction (proper PDF table detection, no pipe-counting)

The NASA handbook embeds images as XObject references — they do NOT appear as
type-1 blocks in get_text("dict"). We must use page.get_images() / pypdf.page.images
to find them. Similarly, tables are vector PDF table structures with no pipe chars,
so pdfplumber.extract_tables() is the only reliable approach.
"""
from __future__ import annotations

import base64
import io
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz                     # PyMuPDF — text blocks + font metadata
import pdfplumber               # table extraction
from pypdf import PdfReader     # image byte extraction

from src.app.common.logger import get_logger
from src.app.common.exceptions import IngestionError

logger = get_logger("components.pdf_parser")

# Minimum image dimensions to treat as a real diagram (not a decorative pixel/icon)
MIN_IMAGE_WIDTH  = 60   # points
MIN_IMAGE_HEIGHT = 60   # points
MIN_IMAGE_BYTES  = 1000 # bytes — skip tiny placeholder images


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
    width: float = 0.0
    height: float = 0.0
    name: str = ""


@dataclass
class TableBlock:
    block_type: Literal["table"] = "table"
    raw_text: str = ""          # serialised for embedding
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
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
    """Parse the full PDF. Returns one PageData per page."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise IngestionError(f"PDF not found: {pdf_path}")

    logger.info("Parsing PDF: %s", pdf_path.name)

    # Open with all three libraries
    fitz_doc     = fitz.open(str(pdf_path))
    pypdf_reader = PdfReader(str(pdf_path))
    pages: list[PageData] = []

    with pdfplumber.open(str(pdf_path)) as plumber_doc:
        for page_idx in range(len(fitz_doc)):
            page_num = page_idx + 1
            fitz_page    = fitz_doc[page_idx]
            pypdf_page   = pypdf_reader.pages[page_idx]
            plumber_page = plumber_doc.pages[page_idx]

            pd = _parse_page(fitz_page, pypdf_page, plumber_page, page_num)
            pages.append(pd)

            if page_num % 50 == 0:
                logger.info("  Parsed %d / %d pages", page_num, len(fitz_doc))

    fitz_doc.close()

    total_text   = sum(len(p.text_blocks)  for p in pages)
    total_images = sum(len(p.image_blocks) for p in pages)
    total_tables = sum(len(p.table_blocks) for p in pages)
    logger.info(
        "PDF parsing complete. %d pages | %d text blocks | %d images | %d tables",
        len(pages), total_text, total_images, total_tables,
    )
    return pages


def _parse_page(
    fitz_page: fitz.Page,
    pypdf_page,
    plumber_page,
    page_num: int,
) -> PageData:
    pd = PageData(page_num=page_num)

    # ── 1. Text blocks via fitz (font metadata preserved) ─────────────────
    font_sizes: list[float] = []
    try:
        raw = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block_idx, block in enumerate(raw.get("blocks", [])):
            if block.get("type") != 0:
                continue  # skip non-text blocks here; images handled below
            tb = _extract_text_block(block, page_num, block_idx, font_sizes)
            if tb:
                pd.text_blocks.append(tb)
    except Exception as exc:
        logger.debug("Page %d fitz text extraction failed: %s", page_num, exc)

    # Fallback if fitz gave nothing
    if not pd.text_blocks:
        try:
            plain = fitz_page.get_text("text").strip()
            for idx, para in enumerate(plain.split("\n\n")):
                para = para.strip()
                if len(para) >= 15:
                    pd.text_blocks.append(TextBlock(
                        text=para, font_size=12.0, font_flags=0,
                        bbox=(0, 0, 0, 0), page_num=page_num, block_index=idx,
                    ))
        except Exception:
            pass

    if font_sizes:
        pd.modal_font_size = Counter([round(fs, 1) for fs in font_sizes]).most_common(1)[0][0]

    # ── 2. Images via pypdf (real byte extraction) ─────────────────────────
    try:
        _extract_images(pypdf_page, plumber_page, page_num, pd)
    except Exception as exc:
        logger.debug("Page %d image extraction error: %s", page_num, exc)

    # ── 3. Tables via pdfplumber ───────────────────────────────────────────
    try:
        _extract_tables(plumber_page, page_num, pd)
    except Exception as exc:
        logger.debug("Page %d table extraction error: %s", page_num, exc)

    return pd


def _extract_text_block(
    block: dict,
    page_num: int,
    block_idx: int,
    font_sizes: list[float],
) -> TextBlock | None:
    lines_text: list[str] = []
    max_font: float = 0.0
    font_flags: int = 0

    for line in block.get("lines", []):
        span_parts: list[str] = []
        for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if text:
                span_parts.append(text)
                fs = float(span.get("size", 12.0))
                if fs > max_font:
                    max_font = fs
                    font_flags = span.get("flags", 0)
                font_sizes.append(fs)
        if span_parts:
            lines_text.append(" ".join(span_parts))

    text = "\n".join(lines_text).strip()
    if not text or len(text) < 2:
        return None

    return TextBlock(
        text=text,
        font_size=max_font or 12.0,
        font_flags=font_flags,
        bbox=tuple(block["bbox"]),
        page_num=page_num,
        block_index=block_idx,
    )


def _extract_images(pypdf_page, plumber_page, page_num: int, pd: PageData) -> None:
    """
    Extract real diagram images using pypdf's page.images.
    pypdf gives us the actual JPEG/PNG bytes.
    pdfplumber gives us the bbox (x0, top, x1, bottom) in page coordinates.

    We match images by name (e.g. 'Im138.jpg' ↔ 'Im138') to get bboxes.
    """
    # Build name → bbox map from pdfplumber
    bbox_map: dict[str, tuple[float, float, float, float]] = {}
    size_map: dict[str, tuple[float, float]] = {}
    for img in plumber_page.images:
        raw_name = str(img.get("name", "")).replace(".jpg", "").replace(".png", "")
        w = float(img["x1"]) - float(img["x0"])
        h = float(img["bottom"]) - float(img["top"])
        if w >= MIN_IMAGE_WIDTH and h >= MIN_IMAGE_HEIGHT:
            bbox_map[raw_name] = (
                float(img["x0"]), float(img["top"]),
                float(img["x1"]), float(img["bottom"]),
            )
            size_map[raw_name] = (w, h)

    # Extract bytes from pypdf
    for idx, img_file in enumerate(pypdf_page.images):
        data = img_file.data
        if not data or len(data) < MIN_IMAGE_BYTES:
            continue  # skip empty / tiny images

        # Normalise name for bbox lookup
        raw_name = img_file.name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
        bbox   = bbox_map.get(raw_name, (0.0, 0.0, 0.0, 0.0))
        w, h   = size_map.get(raw_name, (0.0, 0.0))

        # If bbox not found but data is real, still include it
        ib = ImageBlock(
            image_bytes=data,
            image_b64=base64.b64encode(data).decode(),
            bbox=bbox,
            page_num=page_num,
            block_index=idx,
            width=w,
            height=h,
            name=img_file.name,
        )
        pd.image_blocks.append(ib)
        logger.debug("Page %d: image '%s' (%d bytes, %.0fx%.0f)",
                     page_num, img_file.name, len(data), w, h)


def _extract_tables(plumber_page, page_num: int, pd: PageData) -> None:
    """
    Extract tables using pdfplumber.extract_tables().
    Each table → one TableBlock with headers, rows, and serialised text.
    """
    tables = plumber_page.extract_tables()
    for tbl_idx, raw_table in enumerate(tables):
        if not raw_table or len(raw_table) < 2:
            continue

        # Clean cells — replace None with ""
        cleaned: list[list[str]] = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in raw_table
        ]

        # First non-empty row = headers
        headers = cleaned[0]
        data_rows: list[list[str]] = []
        for row in cleaned[1:]:
            # Skip separator-only rows
            if all(c in ("", "-", "—", " ") for c in row):
                continue
            data_rows.append(row)

        # Serialise for embedding — "header1 | header2 \n val1 | val2"
        lines = [" | ".join(h for h in headers if h)]
        for row in data_rows:
            line = " | ".join(c for c in row if c)
            if line.strip():
                lines.append(line)
        serialised = "\n".join(lines)

        if not serialised.strip():
            continue

        # Try to get bbox from pdfplumber table objects
        try:
            plumber_tables = plumber_page.find_tables()
            bbox_tuple = tuple(plumber_tables[tbl_idx].bbox) if tbl_idx < len(plumber_tables) else (0, 0, 0, 0)
        except Exception:
            bbox_tuple = (0, 0, 0, 0)

        pd.table_blocks.append(TableBlock(
            raw_text=serialised,
            headers=headers,
            rows=data_rows,
            bbox=bbox_tuple,
            page_num=page_num,
            block_index=tbl_idx,
        ))
        logger.debug("Page %d: table %d (%d rows, %d cols)",
                     page_num, tbl_idx, len(data_rows), len(headers))
