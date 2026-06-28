"""SQLite-backed store for the human-in-the-loop loop.

Schema (intentionally small and portable — swap the connection for Postgres later):

  documents    one row per processed form image
  extractions  one row per field the pipeline produced (with confidence + status)
  corrections  one row per human edit (the gold labels for retraining)

The ``corrections`` table is the training signal: ``learning/dataset.py`` reads it to
build fine-tuning data, and only corrected fields count as ground truth.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from functools import lru_cache

from ukitmata.config import settings
from ukitmata.pipeline.extract import ExtractionResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_hash    TEXT UNIQUE NOT NULL,
    image_path  TEXT NOT NULL,
    form_key    TEXT,
    form_name   TEXT,
    confidence  REAL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extractions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    field       TEXT NOT NULL,
    value       TEXT,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    extraction_id INTEGER NOT NULL REFERENCES extractions(id),
    document_id   INTEGER NOT NULL REFERENCES documents(id),
    field         TEXT NOT NULL,
    old_value     TEXT,
    new_value     TEXT NOT NULL,
    corrected_by  TEXT,
    created_at    TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | None = None) -> None:
        settings.ensure_dirs()
        self.path = path or str(settings.db_path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Writes ────────────────────────────────────────────────────────────────
    def save_extraction(
        self, doc_hash: str, image_path: str, result: ExtractionResult
    ) -> int:
        """Persist a document and its fields. Returns the document id."""
        cur = self._conn.execute(
            """INSERT INTO documents (doc_hash, image_path, form_key, form_name,
                                      confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_hash) DO UPDATE SET
                   form_key=excluded.form_key,
                   form_name=excluded.form_name,
                   confidence=excluded.confidence""",
            (
                doc_hash,
                image_path,
                result.form_key,
                result.form_name,
                result.confidence,
                _now(),
            ),
        )
        doc_id = cur.lastrowid or self._doc_id(doc_hash)
        for f in result.fields:
            self._conn.execute(
                """INSERT INTO extractions (document_id, field, value, status, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, f["field"], f["value"], f["status"], _now()),
            )
        self._conn.commit()
        return doc_id

    def save_correction(
        self,
        extraction_id: int,
        document_id: int,
        field: str,
        old_value: str,
        new_value: str,
        corrected_by: str = "human",
    ) -> None:
        self._conn.execute(
            """INSERT INTO corrections (extraction_id, document_id, field, old_value,
                                        new_value, corrected_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (extraction_id, document_id, field, old_value, new_value, corrected_by, _now()),
        )
        self._conn.execute(
            "UPDATE extractions SET value = ?, status = 'corrected' WHERE id = ?",
            (new_value, extraction_id),
        )
        self._conn.commit()

    # ── Reads ─────────────────────────────────────────────────────────────────
    def _doc_id(self, doc_hash: str) -> int:
        row = self._conn.execute(
            "SELECT id FROM documents WHERE doc_hash = ?", (doc_hash,)
        ).fetchone()
        return int(row["id"])

    def review_queue(self) -> list[sqlite3.Row]:
        """Fields that need a human: not auto-approved and not yet corrected."""
        return self._conn.execute(
            """SELECT e.*, d.form_name, d.image_path
               FROM extractions e JOIN documents d ON d.id = e.document_id
               WHERE e.status IN ('needs_review', 'low_confidence')
               ORDER BY e.created_at"""
        ).fetchall()

    def corrections(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            """SELECT c.*, d.image_path, d.form_key
               FROM corrections c JOIN documents d ON d.id = c.document_id
               ORDER BY c.created_at"""
        ).fetchall()

    def close(self) -> None:
        self._conn.close()


@lru_cache(maxsize=1)
def get_db() -> Database:
    """Process-wide database singleton."""
    return Database()
