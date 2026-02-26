"""
Strip XBRL preambles from SEC 10-K filings.

The PleIAs/SEC dataset changed format around 2020-2021.  Post-2020 filings
often start with raw XBRL/iXBRL machine code (CIK numbers, fasb.org URIs,
inline XBRL tags) before the actual narrative text begins.

This module detects contaminated filings and strips everything before the
first real section marker (e.g. "Item 1", "PART I").
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

__all__ = ["clean_filing_text"]

# ── Patterns ────────────────────────────────────────────────────────────────

# Markers that indicate the start of actual 10-K narrative.
# Ordered by specificity — we prefer the most specific match.
_SECTION_PATTERNS = [
    # "Part I" standalone (not inside a URL or XBRL tag)
    re.compile(r"\bPART\s+I\b", re.IGNORECASE),
    # "Item 1" or "Item 1." (the business section)
    re.compile(r"\bItem\s+1[\.\s]", re.IGNORECASE),
    # "UNITED STATES SECURITIES AND EXCHANGE COMMISSION" (cover page)
    re.compile(r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION", re.IGNORECASE),
    # "FORM 10-K" header
    re.compile(r"\bFORM\s+10-K\b", re.IGNORECASE),
]

# Quick heuristic: if the first N chars contain these, the filing is likely
# already clean (no XBRL preamble).
_CLEAN_INDICATORS = re.compile(
    r"(Item\s+1|PART\s+I|UNITED\s+STATES|FORM\s+10-K|ANNUAL\s+REPORT)",
    re.IGNORECASE,
)

# XBRL contamination indicators in the first N chars.
_XBRL_INDICATORS = re.compile(
    r"(fasb\.org|xbrl|false\d{4}FY|\.htm\b|0{6,}\d+\n|--\d{2}-\d{2}\nFY)",
    re.IGNORECASE,
)

_CHECK_WINDOW = 500  # chars to check for clean/contaminated heuristic


def _is_clean(text: str) -> bool:
    """Return True if the filing text appears to start with narrative prose."""
    window = text[:_CHECK_WINDOW]
    return bool(_CLEAN_INDICATORS.search(window)) and not bool(_XBRL_INDICATORS.search(window))


def _find_narrative_start(text: str) -> int:
    """Find byte offset where actual narrative text begins.

    Tries each section pattern in order.  For "Part I" and "Item 1",
    we skip the Table of Contents occurrence (first hit) and look for
    the second hit where the actual section begins — unless the first
    hit is already past the XBRL preamble (> 50% through the document).
    """
    for pattern in _SECTION_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue

        # For short filings or if only one match, take the first
        if len(matches) == 1:
            return matches[0].start()

        # The first match is usually in the Table of Contents.
        # The second is the actual section heading.
        # But only skip the first if it's in the first 20% of the doc
        # (i.e., likely a TOC entry).
        first = matches[0]
        second = matches[1]
        if first.start() < len(text) * 0.2:
            return second.start()
        return first.start()

    # Fallback: find first paragraph of ≥200 chars that looks like English.
    # (no URIs, no XBRL tags, mostly ASCII letters and spaces)
    _english_para = re.compile(r"[A-Z][a-z].{198,}", re.DOTALL)
    m = _english_para.search(text)
    if m:
        return m.start()

    return 0  # give up, return full text


def clean_filing_text(text: str) -> str:
    """Strip XBRL preamble from a 10-K filing text.

    Parameters
    ----------
    text : raw filing text (may or may not have XBRL preamble).

    Returns
    -------
    Cleaned filing text starting from the first real section marker.
    """
    if not text:
        return text

    if _is_clean(text):
        return text

    offset = _find_narrative_start(text)
    if offset > 0:
        cleaned = text[offset:]
        logger.debug(
            "Stripped %d chars of XBRL preamble (%.1f%% of file)",
            offset,
            100 * offset / len(text),
        )
        return cleaned

    return text
