"""Export human corrections into a fine-tuning dataset.

The self-learning loop is a *data pipeline*, not online learning. Only fields a human
corrected (table ``corrections``) are treated as gold labels. Each document becomes one
training example: the form image + the corrected ``Field: ... | Value: ...`` target text,
exactly matching the VLM's output format so the model learns to produce it directly.

Output: a JSONL file under ``data/datasets/`` ready for ``finetune.py``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ukitmata.config import settings
from ukitmata.store import get_db


def build_dataset(min_examples: int = 1) -> Path | None:
    """Write a JSONL training set from corrected documents.

    Returns the dataset path, or None if there are too few corrections yet.
    """
    settings.ensure_dirs()
    db = get_db()

    # Group corrected fields by document so each image yields one target string.
    by_doc: dict[str, dict] = defaultdict(lambda: {"image_path": None, "fields": {}})
    for row in db.corrections():
        entry = by_doc[row["document_id"]]
        entry["image_path"] = row["image_path"]
        entry["fields"][row["field"]] = row["new_value"]

    if len(by_doc) < min_examples:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = settings.datasets_dir / f"corrections_{stamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in by_doc.values():
            target = "\n".join(
                f"Field: {k} | Value: {v}" for k, v in entry["fields"].items()
            )
            fh.write(
                json.dumps(
                    {"image": entry["image_path"], "target": target}, ensure_ascii=False
                )
                + "\n"
            )
    return out_path


if __name__ == "__main__":  # pragma: no cover
    path = build_dataset()
    print(f"Wrote dataset: {path}" if path else "Not enough corrections yet.")
