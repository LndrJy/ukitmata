"""Image preprocessing — deskew and resize a form photo for the vision engine."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ukitmata.config import settings


def deskew(image: np.ndarray) -> np.ndarray:
    """Straighten a tilted form. Returns the image unchanged if near-level (<1deg)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 1.0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def preprocess(image_path: str, max_size: int | None = None) -> Image.Image:
    """Load a form photo, deskew it, cap its longest side, and return a PIL image."""
    max_size = max_size or settings.max_image_size
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = deskew(img)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    w, h = pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return pil
