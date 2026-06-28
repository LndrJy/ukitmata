"""Handwriting second-opinion engine — TrOCR (optional).

Replaces the old EMNIST per-character CNN. The Qwen2-VL vision engine already reads
handwriting natively; this module is a *targeted* cross-check: given a cropped image of
a single handwritten field, it returns TrOCR's transcription so the pipeline can flag
disagreements for human review.

Disabled by default. Enable with ``UKIT_HTR=1`` (and ``pip install ukitmata[htr]``).

Design note: this is line/word-level recognition, not character segmentation — far more
robust than EMNIST on real-world ink. It is intentionally invoked only on low-confidence
regions so it never slows down the common case.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image

from ukitmata.config import settings


@lru_cache(maxsize=1)
def _load():
    """Load TrOCR lazily and only once. Heavy import kept inside the function."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = settings.resolve_device()
    processor = TrOCRProcessor.from_pretrained(settings.htr_model)
    model = VisionEncoderDecoderModel.from_pretrained(settings.htr_model).to(device)
    model.eval()
    return processor, model, device, torch


def transcribe(crop: Image.Image) -> str:
    """Return TrOCR's transcription of a single handwritten field crop."""
    if not settings.htr_enabled:
        return ""
    processor, model, device, torch = _load()
    pixel_values = processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def agrees(vlm_value: str, crop: Image.Image) -> tuple[bool, str]:
    """Compare the VLM's value against TrOCR's reading of the same crop.

    Returns ``(agreement, htr_value)``. Used to decide whether a handwritten field
    should be auto-trusted or routed to human review.
    """
    htr_value = transcribe(crop)
    if not htr_value:
        return True, ""  # HTR disabled or empty — defer to the VLM.
    agreement = vlm_value.strip().lower() == htr_value.strip().lower()
    return agreement, htr_value
