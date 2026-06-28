"""Central configuration.

Every tunable lives here and can be overridden with an environment variable, so the
same code runs unchanged on a developer laptop, a workstation, or a production server.
No more hard-coded ``/content/`` paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # ── Models ────────────────────────────────────────────────────────────────
    vlm_model: str = _env("UKIT_VLM_MODEL", "prithivMLmods/Imgscope-OCR-2B-0527")
    llm_model: str = _env("UKIT_LLM_MODEL", "llama3.1")
    htr_model: str = _env("UKIT_HTR_MODEL", "microsoft/trocr-base-handwritten")
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")

    # ── Inference ─────────────────────────────────────────────────────────────
    device: str = _env("UKIT_DEVICE", "auto")  # auto | cuda | cpu
    max_image_size: int = int(_env("UKIT_MAX_IMAGE_SIZE", "1600"))
    vlm_max_new_tokens: int = int(_env("UKIT_VLM_TOKENS", "1024"))
    vlm_gov_max_new_tokens: int = int(_env("UKIT_VLM_GOV_TOKENS", "2048"))

    # ── Handwriting second-opinion (TrOCR). Off by default — VLM handles HW. ──
    htr_enabled: bool = _env("UKIT_HTR", "0") == "1"

    # ── Human-in-the-loop confidence gates ────────────────────────────────────
    auto_approve_threshold: float = float(_env("UKIT_AUTO_APPROVE", "0.80"))
    review_threshold: float = float(_env("UKIT_REVIEW", "0.50"))

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: Path = Path(_env("UKIT_DATA_DIR", str(PROJECT_ROOT / "data")))

    @cached_property
    def schemas_dir(self) -> Path:
        return PACKAGE_ROOT / "schemas" / "forms"

    @cached_property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @cached_property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @cached_property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    @cached_property
    def db_path(self) -> Path:
        return self.data_dir / "ukitmata.db"

    def ensure_dirs(self) -> None:
        """Create the runtime directory tree if it does not exist."""
        for d in (self.data_dir, self.images_dir, self.datasets_dir, self.adapters_dir):
            d.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Return the concrete torch device string, honouring ``device='auto'``."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


settings = Settings()
