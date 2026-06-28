"""Logic engine — Llama 3.1 via Ollama.

Takes the VLM's raw OCR lines and returns a cleaned, deduplicated list of
``{"field": ..., "value": ...}`` dicts. For known forms it also gap-fills any
schema field the VLM missed so downstream consumers get a complete record.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ukitmata.config import settings

_GENERIC_SYSTEM = (
    "You are an automated form data cleaner. "
    "You will receive raw OCR output from a filled form extracted by a Vision model. "
    "Your job is to clean and structure it.\n\n"
    "Rules:\n"
    "1. Output ONLY lines in this format: Field: <label> | Value: <value>\n"
    "2. Fix obvious OCR errors in field labels (e.g. 'Lst Name' -> 'Last Name').\n"
    "3. Fix obvious OCR errors in values (e.g. '0' vs 'O', 'l' vs '1').\n"
    "4. Remove exact duplicate field entries — keep the first occurrence.\n"
    "5. Do NOT hallucinate, guess, or add fields not present in the raw text.\n"
    "6. Do NOT add explanations, headers, or any other text.\n"
    "7. Preserve Tagalog and mixed English-Filipino text exactly as written."
)


def _schema_system(schema: dict) -> str:
    return (
        f"You are an automated form data cleaner for Philippine government forms.\n"
        f"The form being processed is: {schema['name']}.\n\n"
        "You will receive raw OCR output. Your job:\n"
        "1. Output ONLY lines in this format: Field: <label> | Value: <value>\n"
        "2. Fix obvious OCR errors in values (e.g. '0' vs 'O', 'l' vs '1' in TINs).\n"
        "3. For TINs and dates, preserve exact formatting (dashes, slashes).\n"
        "4. Remove duplicate field entries — keep the first occurrence.\n"
        "5. Do NOT hallucinate or guess — if a value is missing, write Value: blank\n"
        "6. Do NOT add explanations, headers, or any other text."
    )


class LogicEngine:
    """Thin wrapper around ChatOllama with form-cleaning prompts."""

    def __init__(self) -> None:
        self._llm: ChatOllama | None = None

    def load(self) -> None:
        if self._llm is None:
            self._llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_host,
                temperature=0.0,
            )

    def normalize(self, raw_vlm_text: str, schema: dict | None = None) -> list[dict]:
        """Clean raw OCR text into structured fields.

        If ``schema`` is given, missing expected fields are appended as
        ``"blank (not detected)"`` so the record is always complete.
        """
        self.load()
        system = SystemMessage(
            content=_schema_system(schema) if schema else _GENERIC_SYSTEM
        )
        user = HumanMessage(content=f"Raw OCR extraction:\n{raw_vlm_text}")
        cleaned = self._llm.invoke([system, user]).content.strip()

        fields = _parse_field_lines(cleaned)

        if schema:
            present = {f["field"].lower() for f in fields}
            for expected in schema["fields"]:
                if expected.lower() not in present:
                    fields.append({"field": expected, "value": "blank (not detected)"})
        return fields


def _parse_field_lines(text: str) -> list[dict]:
    """Parse ``Field: x | Value: y`` lines, dropping duplicates (first wins)."""
    fields: list[dict] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if "|" not in line or not line.lower().startswith("field:"):
            continue
        try:
            field_part, value_part = line.split("|", 1)
        except ValueError:
            continue
        label = field_part.replace("Field:", "").replace("field:", "").strip()
        value = value_part.replace("Value:", "").replace("value:", "").strip()
        if label and label.lower() not in seen:
            seen.add(label.lower())
            fields.append({"field": label, "value": value})
    return fields


logic_engine = LogicEngine()
