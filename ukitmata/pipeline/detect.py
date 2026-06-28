"""Document-type detection.

A quick VLM text pass reads the visible printed text, which the schema registry then
matches against known form keywords. Thin wrapper kept separate so it can later be
swapped for a faster classifier (e.g. a small text model) without touching the pipeline.
"""

from __future__ import annotations

from PIL import Image

from ukitmata.models.vlm import vision_engine
from ukitmata.schemas import detect_form


def identify_form(image: Image.Image) -> tuple[str | None, str]:
    """Return ``(schema_key_or_None, raw_text)`` for a preprocessed form image."""
    raw_text = vision_engine.read_plain_text(image)
    return detect_form(raw_text), raw_text
