"""Vision engine — Imgscope-OCR-2B (Qwen2-VL).

Reads a form image and returns raw ``Field: ... | Value: ...`` lines plus a
token-level recognition confidence. Loaded once and reused across requests.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ukitmata.config import settings

# Prompt fragments kept as module constants so they are easy to audit and tune.
_GENERIC_PROMPT = (
    "This is a photograph of a filled-out form. "
    "Your task is to extract every single form field and its filled-in value. "
    "Return results in this exact format, one per line:\n"
    "Field: <field label> | Value: <filled value>\n\n"
    "Rules:\n"
    "- Include ALL fields visible on the form.\n"
    "- If a field is blank, write Value: blank\n"
    "- If a checkbox is checked, write Value: checked\n"
    "- If a checkbox is unchecked, write Value: unchecked\n"
    "- If a signature is present, write Value: signed\n"
    "- Preserve exact spelling and capitalization of filled values.\n"
    "- Do not add explanations or headers, only the field-value lines."
)


@dataclass
class VlmResult:
    text: str
    confidence: float


class VisionEngine:
    """Lazy-loaded wrapper around the Qwen2-VL OCR model."""

    def __init__(self) -> None:
        self._processor: AutoProcessor | None = None
        self._model: Qwen2VLForConditionalGeneration | None = None

    # ── Loading ───────────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._model is not None:
            return
        device = settings.resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(settings.vlm_model)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            settings.vlm_model,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self._model.to(device)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── Public extraction API ─────────────────────────────────────────────────
    def extract_generic(self, image: Image.Image) -> VlmResult:
        """Extract all fields from an unknown form."""
        return self._run(image, _GENERIC_PROMPT, settings.vlm_max_new_tokens)

    def extract_schema(self, image: Image.Image, schema: dict) -> VlmResult:
        """Extract using a schema-aware prompt (known government form)."""
        field_list = "\n".join(f"  - {f}" for f in schema["fields"])
        prompt = (
            f"This is a photograph of a Philippine government form: {schema['name']}.\n"
            f"Your task is to extract the filled-in values for EACH of the following "
            f"specific fields.\n\n"
            f"Fields to extract:\n{field_list}\n\n"
            "Return results in this exact format, one per line:\n"
            "Field: <field label> | Value: <filled value>\n\n"
            "Rules:\n"
            "- You MUST output one line for EVERY field listed above, even if blank.\n"
            "- If a field is blank, write Value: blank\n"
            "- For table rows, extract all visible rows using the same Row N pattern.\n"
            "- Preserve exact numbers, TINs, dates, and capitalization as written.\n"
            "- If a signature is present, write Value: signed\n"
            "- Do not include printed form instructions or boilerplate as values.\n"
            "- Do NOT add explanations or headers — only the field-value lines."
        )
        return self._run(image, prompt, settings.vlm_gov_max_new_tokens)

    def read_plain_text(self, image: Image.Image, max_new_tokens: int = 256) -> str:
        """Quick pass that returns raw visible text — used for form-type detection."""
        prompt = (
            "Read all visible printed text in this document image. "
            "Include form numbers, titles, section headers, and any text. "
            "Return plain text only, no formatting."
        )
        return self._run(image, prompt, max_new_tokens).text

    # ── Internal ──────────────────────────────────────────────────────────────
    def _run(self, image: Image.Image, prompt: str, max_new_tokens: int) -> VlmResult:
        if self._model is None:
            self.load()
        device = settings.resolve_device()

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                return_dict_in_generate=True,
                output_scores=True,
            )

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated.sequences)
        ]
        raw = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Mean token confidence as a cheap recognition-quality signal.
        scores = torch.stack(generated.scores, dim=1)
        probs = torch.softmax(scores, dim=-1)
        confidence = float(probs.max(dim=-1).values.mean().item())

        raw = raw.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

        del inputs, generated
        if device == "cuda":
            torch.cuda.empty_cache()

        return VlmResult(text=raw, confidence=confidence)


# Module-level singleton — import this everywhere.
vision_engine = VisionEngine()
