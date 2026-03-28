"""
Acronym resolver — builds a dictionary from:
  1. The handbook's glossary appendix (regex-parsed)
  2. Inline parenthetical definitions in body text
  3. A hard-coded seed list of common aerospace acronyms

expand_query() takes a raw query string and returns it with acronyms expanded.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.app.common.logger import get_logger

logger = get_logger("components.acronym_resolver")

INLINE_DEF_RE = re.compile(r"([A-Z][A-Za-z\s\-]+?)\s+\(([A-Z]{2,8})\)")
GLOSSARY_LINE_RE = re.compile(r"^([A-Z]{2,8})\s*[—–-]+\s*(.+)$", re.MULTILINE)

# Seed dictionary for aerospace acronyms that may not appear in glossary
SEED_ACRONYMS: dict[str, str] = {
    "NASA": "National Aeronautics and Space Administration",
    "SE": "systems engineering",
    "TRL": "technology readiness level",
    "KDP": "key decision point",
    "SRR": "system requirements review",
    "PDR": "preliminary design review",
    "CDR": "critical design review",
    "SAR": "system acceptance review",
    "ORR": "operational readiness review",
    "FRR": "flight readiness review",
    "MDR": "mission definition review",
    "SDR": "system definition review",
    "PRR": "production readiness review",
    "DR": "design review",
    "WBS": "work breakdown structure",
    "ICD": "interface control document",
    "MOE": "measures of effectiveness",
    "MOP": "measures of performance",
    "TPM": "technical performance measure",
    "ConOps": "concept of operations",
    "SEMP": "systems engineering management plan",
    "PBS": "product breakdown structure",
    "V&V": "verification and validation",
    "FMEA": "failure modes and effects analysis",
    "FTA": "fault tree analysis",
    "LCC": "life-cycle cost",
    "ROM": "rough order of magnitude",
    "IOC": "initial operational capability",
    "FOC": "full operational capability",
    "ATP": "authority to proceed",
}


class AcronymResolver:
    def __init__(self) -> None:
        self._dict: dict[str, str] = dict(SEED_ACRONYMS)

    def build_from_chunks(self, texts: list[str]) -> None:
        """Scan all text chunks and extract inline acronym definitions."""
        new_found = 0
        for text in texts:
            for m in INLINE_DEF_RE.finditer(text):
                full_form, abbrev = m.group(1).strip(), m.group(2)
                if abbrev not in self._dict:
                    self._dict[abbrev] = full_form.lower()
                    new_found += 1
        logger.info("Acronym resolver: %d from seed, %d newly extracted. Total: %d",
                    len(SEED_ACRONYMS), new_found, len(self._dict))

    def expand_query(self, query: str) -> str:
        """
        Expand all recognized acronyms in the query.
        Returns a string with both original and expanded forms for embedding.
        """
        expanded_parts: list[str] = [query]
        tokens = re.findall(r"[A-Z]{2,8}", query)
        for token in tokens:
            if token in self._dict:
                expanded_parts.append(self._dict[token])
        result = " ".join(dict.fromkeys(expanded_parts))  # deduplicate preserving order
        return result

    def get_dict(self) -> dict[str, str]:
        return dict(self._dict)

    def __len__(self) -> int:
        return len(self._dict)


# Module-level singleton
_resolver: AcronymResolver | None = None


def get_resolver() -> AcronymResolver:
    global _resolver
    if _resolver is None:
        _resolver = AcronymResolver()
    return _resolver
