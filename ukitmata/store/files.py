"""Raw image archive.

Every processed document's image is copied into the data directory under a content
hash, so the (image, corrected-fields) pairs can later be exported as a training set.
This archive is the asset that makes the self-learning loop possible.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ukitmata.config import settings


def archive_image(src_path: str) -> tuple[str, str]:
    """Copy an image into the archive keyed by content hash.

    Returns ``(doc_hash, stored_path)``. Idempotent — the same image is stored once.
    """
    settings.ensure_dirs()
    src = Path(src_path)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    dest = settings.images_dir / f"{digest}{src.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return digest, str(dest)
