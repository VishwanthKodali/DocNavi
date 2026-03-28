"""
Content classifier — routes each text block to heading / body_text / table.

NASA handbook has specific patterns:
- Section headings: larger font OR bold flags + match section-number regex
- Tables: pipe chars OR ALL-CAPS short labels in grid-like arrangement
- Body text: everything else

The classifier is intentionally permissive — it is better to have
false-positive body text chunks than to miss content entirely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.app.components.pdf_parser import TextBlock, ImageBlock, TableBlock, PageData
from src.app.common.logger import get_logger

logger = get_logger("components.classifier")

# Matches: "6", "6.3", "6.3.2", "6.3.2.1" followed by a capital letter or space
SECTION_HEADING_RE = re.compile(r"^\s*(\d{1,2}(\.\d{1,2}){0,3})\s+\S")

# Appendix headings: "Appendix A", "APPENDIX B — Title"
APPENDIX_RE = re.compile(r"^\s*[Aa]ppendix\s+[A-Z]", re.IGNORECASE)

# Font size threshold above modal to treat as heading
HEADING_FONT_DELTA = 1.0   # lowered from 1.5 — handbook body is typically 10-11pt

# Bold flag in PDF spans (bit 4 = 16)
BOLD_FLAG = 16

TABLE_RE = re.compile(r"(\|.*\|)|(^\s*[-–]{3,})", re.MULTILINE)
TABLE_KEYWORD_RE = re.compile(r"^\s*(Table|Figure|Exhibit)\s+\d+", re.IGNORECASE)


@dataclass
class ClassifiedBlock:
    kind: Literal["heading", "body_text", "image", "table"]
    source: TextBlock | ImageBlock | TableBlock
    is_heading: bool = False
    heading_depth: int = 0


def classify_page(page: PageData) -> list[ClassifiedBlock]:
    classified: list[ClassifiedBlock] = []

    for tb in page.text_blocks:
        kind, is_heading, depth = _classify_text_block(tb, page.modal_font_size)
        classified.append(ClassifiedBlock(
            kind=kind,
            source=tb,
            is_heading=is_heading,
            heading_depth=depth,
        ))

    for ib in page.image_blocks:
        classified.append(ClassifiedBlock(kind="image", source=ib))

    for tab in page.table_blocks:
        classified.append(ClassifiedBlock(kind="table", source=tab))

    return classified


def _classify_text_block(block: TextBlock, modal_fs: float) -> tuple[str, bool, int]:
    text = block.text.strip()
    if not text:
        return "body_text", False, 0

    # ── Table detection (before heading — some headings say "Table N") ──
    if _is_table(text):
        return "table", False, 0

    # ── Heading detection ─────────────────────────────────────────────
    is_section = bool(SECTION_HEADING_RE.match(text))
    is_appendix = bool(APPENDIX_RE.match(text))
    is_larger = block.font_size >= modal_fs + HEADING_FONT_DELTA
    is_bold = bool(block.font_flags & BOLD_FLAG)
    is_short = len(text) < 120   # headings are rarely very long

    # Accept as heading if: matches section pattern (regardless of font),
    # OR is appendix heading, OR is significantly larger/bold and short.
    if is_section or is_appendix or ((is_larger or is_bold) and is_short and len(text) > 2):
        depth = _heading_depth(text)
        return "heading", True, depth

    return "body_text", False, 0


def _is_table(text: str) -> bool:
    lines = text.split("\n")
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    # At least 2 lines with pipes = likely table
    return pipe_lines >= 2


def _heading_depth(text: str) -> int:
    match = SECTION_HEADING_RE.match(text)
    if match:
        section_num = match.group(1)
        return len(section_num.split("."))
    if APPENDIX_RE.match(text):
        return 1
    # Bold/larger headings without numbering treated as depth 1
    return 1
