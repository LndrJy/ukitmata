"""FastAPI service — the "local server + API" front-end.

Other projects integrate by POSTing a form image to ``/extract`` and reading back
structured fields. The review/correction endpoints feed the self-learning loop.

Run:  ukitmata-api          (or: uvicorn ukitmata.api.main:app --host 0.0.0.0)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ukitmata.pipeline import process_form
from ukitmata.schemas import list_forms
from ukitmata.store import get_db
from ukitmata.store.files import archive_image

app = FastAPI(title="UkitMata", version="0.1.0")


class CorrectionIn(BaseModel):
    extraction_id: int
    document_id: int
    field: str
    old_value: str = ""
    new_value: str
    corrected_by: str = "api"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "forms": list_forms()}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    """Extract structured fields from an uploaded form image."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = process_form(tmp_path)
        doc_hash, stored_path = archive_image(tmp_path)
        doc_id = get_db().save_extraction(doc_hash, stored_path, result)
    except Exception as exc:  # surface a clean error to API clients
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "document_id": doc_id,
        "form_key": result.form_key,
        "form_name": result.form_name,
        "confidence": result.confidence,
        "fields": result.fields,
    }


@app.get("/review-queue")
def review_queue() -> list[dict]:
    """Fields awaiting human review."""
    return [dict(row) for row in get_db().review_queue()]


@app.post("/correct")
def correct(payload: CorrectionIn) -> dict:
    """Record a human correction (gold label for retraining)."""
    get_db().save_correction(
        payload.extraction_id,
        payload.document_id,
        payload.field,
        payload.old_value,
        payload.new_value,
        payload.corrected_by,
    )
    return {"status": "saved"}


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
