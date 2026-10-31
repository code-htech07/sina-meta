"""Image ingestion: store the original bytes, extract metadata (incl. EXIF date/GPS)."""

from __future__ import annotations

import hashlib
import io
import uuid
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image, ImageOps

from engine.config import get_settings
from engine.schemas import ImageMetadata, PlantPart

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}  # subset of formats accepted by Meta image understanding


class ImageRejected(ValueError):
    pass


def _gps_to_deg(value, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in value)
        deg = d + m / 60 + s / 3600
        return -deg if ref in ("S", "W") else deg
    except Exception:
        return None


def read_exif(img: Image.Image) -> tuple[datetime | None, float | None, float | None]:
    try:
        exif = img.getexif()
    except Exception:
        return None, None, None
    capture: datetime | None = None
    raw_dt = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    if raw_dt:
        try:
            capture = datetime.strptime(str(raw_dt), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            capture = None
    lat = lon = None
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if gps:
        lat = _gps_to_deg(gps.get(2), gps.get(1))
        lon = _gps_to_deg(gps.get(4), gps.get(3))
    return capture, lat, lon


def ingest_image(
    data: bytes,
    *,
    plant_part: PlantPart = "unknown",
    source: str = "user_upload",
    licence: str | None = None,
    attribution: str | None = None,
    images_dir: Path | None = None,
) -> ImageMetadata:
    """Validate and persist an uploaded image. Returns its metadata."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise ImageRejected(f"Fichier image illisible : {exc}") from exc
    if img.format not in ALLOWED_FORMATS:
        raise ImageRejected(f"Format non supporté : {img.format}. Formats acceptés : {sorted(ALLOWED_FORMATS)}")

    capture, lat, lon = read_exif(img)
    oriented = ImageOps.exif_transpose(img)
    sha = hashlib.sha256(data).hexdigest()
    image_id = f"img_{uuid.uuid4().hex[:12]}"
    target_dir = images_dir or get_settings().path("images_dir")
    target_dir.mkdir(parents=True, exist_ok=True)
    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[img.format]
    path = target_dir / f"{image_id}.{ext}"
    path.write_bytes(data)
    return ImageMetadata(
        image_id=image_id,
        path=str(path),
        sha256=sha,
        width=oriented.width,
        height=oriented.height,
        format=img.format,
        capture_date=capture,
        exif_latitude=lat,
        exif_longitude=lon,
        plant_part=plant_part,
        source=source,
        licence=licence,
        attribution=attribution,
    )


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image with EXIF orientation applied, as RGB."""
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")
