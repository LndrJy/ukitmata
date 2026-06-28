"""Loads form schemas from YAML files in ``schemas/forms/``.

Adding support for a new government form is a *data* change: drop a YAML file in the
forms directory — no Python edits, no redeploy of logic. Each schema declares the
keywords used to auto-detect the form and the list of fields to extract.
"""

from __future__ import annotations

from functools import lru_cache

import yaml

from ukitmata.config import settings

# Minimum keyword hits before we trust a form-type identification.
_DETECTION_MIN_HITS = 2


@lru_cache(maxsize=1)
def load_schemas() -> dict[str, dict]:
    """Load and cache every ``*.yaml`` schema in the forms directory."""
    schemas: dict[str, dict] = {}
    for path in sorted(settings.schemas_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        key = data.get("key") or path.stem.upper()
        data.setdefault("keywords", [])
        data.setdefault("fields", [])
        schemas[key] = data
    return schemas


def list_forms() -> list[str]:
    return list(load_schemas().keys())


def get_schema(key: str) -> dict | None:
    return load_schemas().get(key)


def detect_form(raw_text: str) -> str | None:
    """Return the schema key whose keywords best match ``raw_text``, or None.

    A form is identified when at least ``_DETECTION_MIN_HITS`` of its keywords appear
    in the text. The best-matching form wins if several qualify.
    """
    lowered = raw_text.lower()
    best_key, best_hits = None, 0
    for key, schema in load_schemas().items():
        hits = sum(1 for kw in schema["keywords"] if kw.lower() in lowered)
        if hits >= _DETECTION_MIN_HITS and hits > best_hits:
            best_key, best_hits = key, hits
    return best_key
