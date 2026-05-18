"""Lyrics text normalization for ASR training and evaluation."""

from __future__ import annotations

import re
import unicodedata

# Characters Whisper and jiwer handle poorly in references.
_PUNCT_TO_SPACE = re.compile(r"[^\w\s']", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_lyrics(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace, keep apostrophes in words."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCT_TO_SPACE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text
