"""SAM 3.1 adapter: concept-prompted segmentation through the Meta Model API.

Contract (https://dev.meta.ai/docs/sam/segmenting, /docs/sam/reading-segmentation,
/docs/sam/client-libraries — verified 2026-10-31):

* ``POST /v1/responses``, model ``sam-3.1``, one user message with an ``input_text``
  noun phrase and one ``input_image`` part; ``stream: true`` as in the docs.
* ``metadata.mask_encoding`` = ``"one_bit"`` | ``"lossless"``;
  ``metadata.include_confidence`` = ``"true"`` (string) adds a ``c`` field.
* Output text: one line per image, ``<0f>`` frame header then records
  ``<id><|box;x1;y1;x2;y2;w;h|><|mask;x=0;y=0;data=H,W,<enc>payload|>``; boxes in
  source pixels; empty output = no match.
* Decoding uses Meta's official parser ``meta-sam-parser`` — the base85 mask codec
  is NOT re-implemented here.

SAM localises regions; it never produces a diagnosis. This module never fabricates
a segmentation: on any failure it returns ``status="error"``/``"unavailable"``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from engine.config import get_settings
from engine.meta_ai.client import MetaAPIError, MetaModelClient, MetaNotConfiguredError
from engine.preprocessing.quality import to_data_url
from engine.schemas import SegmentationResult, SegmentedRegion

log = logging.getLogger(__name__)

try:
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format

    _PARSER_AVAILABLE = True
except Exception:  # pragma: no cover
    _PARSER_AVAILABLE = False

OVERLAY_COLOURS = [(230, 57, 70), (69, 123, 157), (244, 162, 97), (42, 157, 143), (131, 56, 236), (255, 190, 11)]


def parse_sam_output(
    text: str, image_id: str, prompt: str, image_size: tuple[int, int]
) -> tuple[list[SegmentedRegion], list[np.ndarray], list[str]]:
    """Parse SAM wire text with the official parser. Returns regions, full-image masks, diagnostics.

    Verified against the live API (2026-10-31): each mask payload is LOCAL to its box
    (``data=H,W`` equals the box size) and the parser exposes its placement in
    ``record.bounds`` (half-open right/bottom). Masks are re-placed on an image-sized canvas.
    """
    if not _PARSER_AVAILABLE:
        raise RuntimeError("meta-sam-parser n'est pas installé (pip install meta-sam-parser)")
    img_w, img_h = image_size
    parser = image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    diagnostics = [f"{d.severity}:{d.code}:{d.message}" for d in result.diagnostics]
    boxes = {r.object_id: r for r in result.records if r.kind == "box"}
    regions: list[SegmentedRegion] = []
    masks: list[np.ndarray] = []
    for rec in result.records:
        if rec.kind != "mask":
            continue
        local = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(rec.mask.height, rec.mask.width)
        left, top = int(round(rec.bounds.left)), int(round(rec.bounds.top))
        full = np.zeros((img_h, img_w), dtype=np.uint8)
        h = max(0, min(local.shape[0], img_h - top))
        w = max(0, min(local.shape[1], img_w - left))
        if h and w and left >= 0 and top >= 0:
            full[top:top + h, left:left + w] = local[:h, :w]
        area = int(full.sum())
        box = boxes.get(rec.object_id)
        bbox = (box.left, box.top, box.right, box.bottom) if box else (rec.bounds.left, rec.bounds.top, rec.bounds.right, rec.bounds.bottom)
        regions.append(
            SegmentedRegion(
                image_id=image_id,
                prompt=prompt,
                object_id=f"{prompt}#{rec.object_id}",
                bbox_xyxy=tuple(float(v) for v in bbox),
                image_width=img_w,
                image_height=img_h,
                mask_area_px=area,
                mask_area_fraction=round(area / float(img_w * img_h), 5),
                box_confidence=getattr(box, "confidence", None) if box else None,
                mask_confidence=rec.confidence,
            )
        )
        masks.append(full)
    return regions, masks, diagnostics


def save_crops_and_overlay(
    image: Image.Image, regions: list[SegmentedRegion], masks: list[np.ndarray], out_dir: Path, stem: str,
    fill_prompts: set[str] | None = None,
) -> None:
    """Overlay: organ regions drawn as outlines only, anomaly regions (``fill_prompts``) filled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.convert("RGBA")
    layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i, (reg, mask) in enumerate(zip(regions, masks)):
        colour = OVERLAY_COLOURS[i % len(OVERLAY_COLOURS)]
        # fill only small anomaly regions; large ones (e.g. "discolored area" matching a whole pod) get outlines
        fill = (fill_prompts is None or reg.prompt in fill_prompts) and (reg.mask_area_fraction or 0) <= 0.10
        if fill and mask.shape == (image.height, image.width):
            m = Image.fromarray((mask * 140).astype(np.uint8), mode="L")
            layer.paste(Image.new("RGBA", overlay.size, colour + (255,)), (0, 0), m)
        x1, y1, x2, y2 = reg.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=colour + (255,), width=3 if fill else 2)
        draw.text((x1 + 4, y1 + 4), reg.prompt, fill=colour + (255,))
        pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        box = (max(0, x1 - pad_x), max(0, y1 - pad_y), min(image.width, x2 + pad_x), min(image.height, y2 + pad_y))
        if box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
            crop_path = out_dir / f"{stem}_roi{i}.jpg"
            image.crop(box).save(crop_path, quality=90)
            reg.crop_path = str(crop_path)
    overlay_path = out_dir / f"{stem}_sam_overlay.png"
    Image.alpha_composite(overlay, layer).convert("RGB").save(overlay_path)
    for reg in regions:
        reg.overlay_path = str(overlay_path)


class SamSegmenter:
    def __init__(self, client: MetaModelClient | None = None, model: str | None = None) -> None:
        s = get_settings()
        self.cfg = s.models["sam"]
        self.client = client or MetaModelClient()
        self.model = model or s.sam_model
        self.enabled = s.sam_enabled

    def build_payload(self, prompt: str, data_url: str) -> dict:
        metadata = {"mask_encoding": self.cfg.get("mask_encoding", "one_bit")}
        if self.cfg.get("include_confidence"):
            metadata["include_confidence"] = "true"
        return {
            "model": self.model,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ]}],
            "metadata": metadata,
        }

    def segment(
        self, image: Image.Image, image_id: str, prompts: list[str] | None = None, artifacts_dir: Path | None = None
    ) -> SegmentationResult:
        if not self.enabled:
            return SegmentationResult(status="disabled", model=self.model)
        if not self.client.configured:
            return SegmentationResult(status="unavailable", model=self.model, error="MODEL_API_KEY absente")
        if not _PARSER_AVAILABLE:
            return SegmentationResult(status="unavailable", model=self.model, error="meta-sam-parser non installé")
        prompts = prompts or (self.cfg.get("organ_prompts", []) + self.cfg.get("anomaly_prompts", []))
        min_conf = float(self.cfg.get("min_region_confidence", 0.0))
        # The image sent to SAM is the downscaled analysis image; boxes are in its pixel space.
        work = image.copy()
        work.thumbnail((get_settings().settings["quality"]["analysis_max_side_px"],) * 2)
        data_url = to_data_url(work, max_side=max(work.size))
        t0 = time.perf_counter()
        regions: list[SegmentedRegion] = []
        masks: list[np.ndarray] = []
        diagnostics: list[str] = []
        errors: list[str] = []
        for prompt in prompts:
            try:
                res = self.client.stream_text(self.build_payload(prompt, data_url))
                r, m, d = parse_sam_output(res.text, image_id, prompt, work.size)
                for reg, mask in zip(r, m):
                    if reg.mask_confidence is None or reg.mask_confidence >= min_conf:
                        regions.append(reg)
                        masks.append(mask)
                diagnostics += [f"[{prompt}] {x}" for x in d]
            except MetaNotConfiguredError as exc:
                return SegmentationResult(status="unavailable", model=self.model, error=str(exc))
            except (MetaAPIError, Exception) as exc:  # one failing prompt must not kill the pipeline
                log.warning("SAM prompt %r failed: %s", prompt, exc)
                errors.append(f"[{prompt}] {exc}")
        latency = (time.perf_counter() - t0) * 1000
        if errors and not regions and len(errors) == len(prompts):
            return SegmentationResult(status="error", model=self.model, error="; ".join(errors)[:2000], latency_ms=latency)
        if regions:
            out_dir = artifacts_dir or get_settings().path("artifacts_dir")
            save_crops_and_overlay(work, regions, masks, out_dir, image_id,
                                   fill_prompts=set(self.cfg.get("anomaly_prompts", [])) or None)
        return SegmentationResult(
            status="ok" if regions else "no_match",
            model=self.model,
            regions=regions,
            diagnostics=diagnostics + errors,
            latency_ms=latency,
        )
