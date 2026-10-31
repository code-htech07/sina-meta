"""Deterministic image quality control (no AI involved).

Metrics: resolution, sharpness (variance of a 3x3 Laplacian on a downscaled grey
image), mean brightness and clipped-pixel fraction. Thresholds come from
``configs/settings.yaml`` and are engineering heuristics, documented as such.
Semantic checks (is it cocoa? is the plant part identifiable?) are delegated to
Muse Spark and reported separately in ``image_assessment``.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from engine.acquisition.images import load_rgb
from engine.config import get_settings
from engine.schemas import EvidenceKind, ImageQualityReport, Provenance, utcnow

ADVICE = {
    "low_resolution": "Reprendre la photo plus près de l'organe, avec un appareil de meilleure résolution.",
    "blurry": "Image floue : stabiliser l'appareil et faire la mise au point sur la zone atteinte.",
    "too_dark": "Image trop sombre : photographier à la lumière du jour, sans contre-jour.",
    "too_bright": "Image surexposée : éviter le soleil direct sur l'organe photographié.",
    "clipped": "Trop de zones saturées (noir/blanc) : changer d'angle ou de lumière.",
}


def laplacian_variance(grey: np.ndarray) -> float:
    g = grey.astype(np.float32)
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var())


def assess_quality(image_id: str, path: str | Path) -> ImageQualityReport:
    cfg = get_settings().settings["quality"]
    img = load_rgb(path)
    w, h = img.size
    small = img.copy()
    small.thumbnail((512, 512))
    grey = np.asarray(small.convert("L"))
    lap_var = laplacian_variance(grey)
    mean_b = float(grey.mean())
    clipped = float(((grey <= 2) | (grey >= 253)).mean())

    issues: list[str] = []
    if min(w, h) < cfg["min_side_px"]:
        issues.append("low_resolution")
    if lap_var < cfg["min_laplacian_variance"]:
        issues.append("blurry")
    if mean_b < cfg["min_mean_brightness"]:
        issues.append("too_dark")
    if mean_b > cfg["max_mean_brightness"]:
        issues.append("too_bright")
    if clipped > cfg["max_clipped_fraction"]:
        issues.append("clipped")
    blocking = {"low_resolution", "blurry", "too_dark", "too_bright"}
    usable = not (set(issues) & blocking)
    return ImageQualityReport(
        image_id=image_id,
        width=w,
        height=h,
        laplacian_variance=round(lap_var, 2),
        mean_brightness=round(mean_b, 2),
        clipped_fraction=round(clipped, 4),
        issues=issues,
        usable=usable,
        retake_advice=[ADVICE[i] for i in issues],
        provenance=Provenance(
            source="Maana Terra image QC (Pillow/NumPy)",
            kind=EvidenceKind.IMAGE_MEASUREMENT,
            retrieved_at=utcnow(),
            notes="Seuils heuristiques définis dans configs/settings.yaml (non issus de la littérature).",
        ),
    )


def to_data_url(img: Image.Image, max_side: int | None = None, quality: int = 88) -> str:
    """Encode an RGB image as a JPEG base64 data URL (format accepted by Meta input_image)."""
    max_side = max_side or get_settings().settings["quality"]["analysis_max_side_px"]
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
