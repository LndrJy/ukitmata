"""Pipeline orchestrator.

Photo -> preprocess -> detect form type -> VLM extract -> LLM structure -> result.
This is the single core both the API and the UI call. It is deliberately free of any
UI/HTTP concerns so it stays reusable as the "backbone for other projects".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ukitmata.config import settings
from ukitmata.models.llm import logic_engine
from ukitmata.models.vlm import vision_engine
from ukitmata.pipeline.detect import identify_form
from ukitmata.pipeline.preprocess import preprocess
from ukitmata.schemas import get_schema


@dataclass
class ExtractionResult:
    form_key: str | None
    form_name: str
    fields: list[dict] = field(default_factory=list)  # [{field, value, status}]
    raw_vlm_text: str = ""
    confidence: float = 0.0

    def status_for(self, value: str) -> str:
        """Per-field routing label driven by the recognition confidence gates."""
        if value.lower() in ("blank", "blank (not detected)"):
            return "blank"
        if self.confidence >= settings.auto_approve_threshold:
            return "auto_approved"
        if self.confidence >= settings.review_threshold:
            return "needs_review"
        return "low_confidence"


def process_form(image_path: str) -> ExtractionResult:
    """Run the full extraction pipeline on one form photo."""
    image = preprocess(image_path)

    form_key, _ = identify_form(image)
    schema = get_schema(form_key) if form_key else None

    if schema:
        vlm = vision_engine.extract_schema(image, schema)
        fields = logic_engine.normalize(vlm.text, schema=schema)
        form_name = schema["name"]
    else:
        vlm = vision_engine.extract_generic(image)
        fields = logic_engine.normalize(vlm.text)
        form_name = "Generic Form"

    result = ExtractionResult(
        form_key=form_key,
        form_name=form_name,
        raw_vlm_text=vlm.text,
        confidence=vlm.confidence,
    )
    result.fields = [
        {"field": f["field"], "value": f["value"], "status": result.status_for(f["value"])}
        for f in fields
    ]
    return result
