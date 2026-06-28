"""Smoke tests that run without GPU/models — exercise the pure-Python core."""

from ukitmata.models.llm import _parse_field_lines
from ukitmata.schemas import detect_form, get_schema, list_forms


def test_schemas_load():
    forms = list_forms()
    assert "BIR_2306" in forms
    assert "BIR_2307" in forms


def test_schema_has_fields():
    schema = get_schema("BIR_2306")
    assert schema is not None
    assert "Payee TIN" in schema["fields"]


def test_detect_form_by_keywords():
    text = "BIR FORM NO 2306 Certificate of Final Tax withheld at source"
    assert detect_form(text) == "BIR_2306"


def test_detect_form_unknown():
    assert detect_form("a random grocery receipt") is None


def test_parse_field_lines_dedupes():
    raw = (
        "Field: Last Name | Value: Dela Cruz\n"
        "Field: Last Name | Value: Ignored Dup\n"
        "Field: First Name | Value: Juan\n"
        "garbage line\n"
    )
    parsed = _parse_field_lines(raw)
    assert parsed == [
        {"field": "Last Name", "value": "Dela Cruz"},
        {"field": "First Name", "value": "Juan"},
    ]
