"""End-to-end analysis pipeline.

    images -> quality gate -> [SAM 3.1 localisation] -> context sources (NASA POWER,
    Sentinel-2, knowledge base) -> structured fusion context -> Muse Spark ->
    validation -> deterministic contradiction rules -> caution level -> report

Every step is timed and isolated: a failing source degrades the report (and is
reported as missing), it never aborts the pipeline. If Muse Spark is unavailable
the report is ``failed`` with an explicit reason — no visual analysis is invented.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

from engine.acquisition.images import load_rgb
from engine.config import get_settings
from engine.diagnosis.caution import caution_level
from engine.diagnosis.validator import validate_output
from engine.fusion.context import build_context
from engine.fusion.contradictions import detect_contradictions, weather_consistency
from engine.meta_ai.client import MetaAPIError, MetaNotConfiguredError
from engine.meta_ai.muse_spark import MuseSparkAnalyzer
from engine.meta_ai.sam import SamSegmenter
from engine.preprocessing.quality import assess_quality, to_data_url
from engine.schemas import (
    AnalysisReport,
    ExperimentConfig,
    ImageMetadata,
    ModelUsage,
    Parcel,
    Provenance,
    SatelliteContext,
    SegmentationResult,
    StepTiming,
    WeatherContext,
)
from engine.sources.agronomy.knowledge_base import KnowledgeBase, get_knowledge_base
from engine.sources.satellite.sentinel2 import get_satellite_context
from engine.sources.weather.nasa_power import get_weather_context
from engine.storage.db import new_id

log = logging.getLogger(__name__)


@dataclass
class AnalysisInput:
    images: list[ImageMetadata]
    observation_date: date
    plant_part: str = "unknown"
    farmer_symptoms: str | None = None
    farmer_notes: str | None = None
    parcel: Parcel | None = None
    observation_id: str | None = None
    # Explicit location for standalone runs (evaluation) when there is no parcel
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class ContextCache:
    """Caches context sources per (lat, lon, date) so ablations reuse identical data."""
    weather: dict[tuple, WeatherContext] = field(default_factory=dict)
    satellite: dict[tuple, SatelliteContext] = field(default_factory=dict)
    segmentation: dict[str, SegmentationResult] = field(default_factory=dict)


class AnalysisPipeline:
    def __init__(
        self,
        muse: MuseSparkAnalyzer | None = None,
        sam: SamSegmenter | None = None,
        kb: KnowledgeBase | None = None,
        weather_fn: Callable[..., WeatherContext] = get_weather_context,
        satellite_fn: Callable[..., SatelliteContext] = get_satellite_context,
        cache: ContextCache | None = None,
    ) -> None:
        self.muse = muse or MuseSparkAnalyzer()
        self.sam = sam or SamSegmenter(client=self.muse.client)
        self.kb = kb or get_knowledge_base()
        self.weather_fn = weather_fn
        self.satellite_fn = satellite_fn
        self.cache = cache or ContextCache()

    # ------------------------------------------------------------------ utils
    @contextmanager
    def _step(self, timings: list[StepTiming], name: str) -> Iterator[dict[str, Any]]:
        info: dict[str, Any] = {"status": "ok", "detail": None}
        t0 = time.perf_counter()
        try:
            yield info
        except Exception as exc:
            info["status"], info["detail"] = "error", str(exc)[:500]
            log.exception("Step %s failed", name)
        finally:
            timings.append(StepTiming(step=name, status=info["status"], latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                                      detail=info["detail"]))

    @staticmethod
    def _location(inp: AnalysisInput) -> tuple[float, float] | None:
        if inp.parcel:
            return inp.parcel.latitude, inp.parcel.longitude
        if inp.latitude is not None and inp.longitude is not None:
            return inp.latitude, inp.longitude
        for img in inp.images:
            if img.exif_latitude is not None and img.exif_longitude is not None:
                return img.exif_latitude, img.exif_longitude
        return None

    # -------------------------------------------------------------------- run
    def run(self, inp: AnalysisInput, experiment: ExperimentConfig | None = None) -> AnalysisReport:
        experiment = experiment or ExperimentConfig()
        settings = get_settings()
        timings: list[StepTiming] = []
        report = AnalysisReport(
            analysis_id=new_id("ana"),
            parcel_id=inp.parcel.id if inp.parcel else None,
            observation_id=inp.observation_id,
            experiment=experiment,
            status="failed",
            farmer_input={"plant_part": inp.plant_part, "symptoms": inp.farmer_symptoms, "notes": inp.farmer_notes}
            if experiment.use_farmer_input else None,
        )

        # 1. Quality gate -------------------------------------------------------
        with self._step(timings, "image_quality") as st:
            report.image_quality = [assess_quality(img.image_id, img.path) for img in inp.images]
            if report.image_quality:
                report.provenance.append(report.image_quality[0].provenance)
        usable = [img for img, q in zip(inp.images, report.image_quality) if q.usable]
        if not usable:
            report.status = "needs_new_photo"
            report.failure_reason = "Aucune image exploitable : " + "; ".join(
                a for q in report.image_quality for a in q.retake_advice) if report.image_quality else "Aucune image."
            report.caution_level, report.caution_rationale = "insufficient_data", [report.failure_reason]
            report.timings = timings
            return report

        loc = self._location(inp)
        if loc is None:
            report.missing_information.append("Localisation inconnue : météo et satellite non récupérables.")

        # 2. Segmentation (SAM 3.1) ---------------------------------------------
        seg: SegmentationResult | None = None
        if experiment.use_segmentation:
            with self._step(timings, "sam_segmentation") as st:
                key = usable[0].sha256
                if key not in self.cache.segmentation:
                    self.cache.segmentation[key] = self.sam.segment(load_rgb(usable[0].path), usable[0].image_id)
                seg = self.cache.segmentation[key]
                st["status"] = "ok" if seg.status in ("ok", "no_match") else "skipped"
                st["detail"] = f"{seg.status}: {len(seg.regions)} région(s)" + (f" — {seg.error}" if seg.error else "")
            report.segmentation = seg

        # 3. Context sources ------------------------------------------------------
        weather: WeatherContext | None = None
        satellite: SatelliteContext | None = None
        if experiment.use_weather and loc:
            with self._step(timings, "weather_nasa_power") as st:
                key = (round(loc[0], 4), round(loc[1], 4), inp.observation_date)
                if key not in self.cache.weather:
                    self.cache.weather[key] = self.weather_fn(loc[0], loc[1], inp.observation_date)
                weather = self.cache.weather[key]
                st["detail"] = weather.status
                if weather.status == "unavailable":
                    st["status"] = "error"
                    st["detail"] = weather.error
            report.weather_context = weather
            report.provenance.append(weather.provenance)
        if experiment.use_satellite and loc:
            with self._step(timings, "satellite_sentinel2") as st:
                key = (round(loc[0], 4), round(loc[1], 4), inp.observation_date)
                if key not in self.cache.satellite:
                    poly = inp.parcel.polygon_geojson if inp.parcel else None
                    self.cache.satellite[key] = self.satellite_fn(loc[0], loc[1], inp.observation_date, poly)
                satellite = self.cache.satellite[key]
                st["detail"] = f"{satellite.status}, {satellite.valid_scene_count} scène(s) valide(s)"
                if satellite.status == "unavailable":
                    st["status"] = "error"
            report.satellite_context = satellite
            report.provenance.append(satellite.provenance)

        # 4. Fusion context -------------------------------------------------------
        context, missing = build_context(
            parcel=inp.parcel, observation_date=inp.observation_date, declared_plant_part=inp.plant_part,
            farmer_symptoms=inp.farmer_symptoms, farmer_notes=inp.farmer_notes, quality=report.image_quality,
            weather=weather, satellite=satellite, segmentation=seg, kb=self.kb, experiment=experiment,
        )
        report.missing_information += missing
        if experiment.use_knowledge_base:
            report.provenance.append(Provenance(source="Maana Terra cocoa knowledge base", kind="literature",
                                                notes="data/knowledge_base/cocoa/*.yaml"))

        # 5. Muse Spark -----------------------------------------------------------
        images = [to_data_url(load_rgb(img.path)) for img in usable]
        has_rois = False
        if seg and seg.status == "ok":
            anomaly = set(settings.models["sam"].get("anomaly_prompts", []))
            # anomaly regions first (most confident), then organ regions (largest)
            rois = sorted([r for r in seg.regions if r.crop_path],
                          key=lambda r: (r.prompt not in anomaly, -(r.mask_confidence or 0), -(r.mask_area_fraction or 0)))
            for r in rois[: settings.models["sam"].get("max_regions_to_muse", 4)]:
                images.append(to_data_url(load_rgb(r.crop_path)))
                has_rois = True
        muse_result = None
        with self._step(timings, "muse_spark_analysis") as st:
            try:
                muse_result = self.muse.analyze(images, context, has_rois=has_rois)
                st["detail"] = "; ".join(muse_result.notes) or None
            except MetaNotConfiguredError as exc:
                st["status"], st["detail"] = "skipped", str(exc)
                report.failure_reason = str(exc)
            except MetaAPIError as exc:
                st["status"], st["detail"] = "error", str(exc)[:500]
                report.failure_reason = f"Muse Spark indisponible : {str(exc)[:300]}"
        if muse_result is None:
            report.status = "failed"
            report.failure_reason = report.failure_reason or "Analyse Muse Spark non réalisée."
            report.caution_level, report.caution_rationale = "insufficient_data", [report.failure_reason]
            report.timings = timings
            return report

        out = muse_result.output
        report.model_usage.append(muse_result.usage)
        n_sam = len(settings.models["sam"].get("organ_prompts", [])) + len(settings.models["sam"].get("anomaly_prompts", []))
        if seg and seg.status in ("ok", "no_match"):
            report.model_usage.append(ModelUsage(
                model=seg.model or "sam", images=n_sam,
                estimated_cost_usd=round(n_sam
                                         * settings.models["pricing"].get(seg.model or "", {}).get("per_1k_images", 0) / 1000, 6)))

        # 6. Validation + rules + caution ---------------------------------------
        available: set[str] = {"image"}
        if weather and weather.status != "unavailable":
            available.add("weather")
        if satellite and satellite.status in ("ok", "partial"):
            available.add("satellite")
        if experiment.use_knowledge_base:
            available.add("knowledge_base")
        if seg and seg.status == "ok":
            available.add("segmentation")
        # the declared plant part is always in the context (user_declared), so citing "farmer" for it is legitimate
        if (experiment.use_farmer_input and (inp.farmer_symptoms or inp.farmer_notes)) or inp.plant_part not in ("unknown", None):
            available.add("farmer")
        with self._step(timings, "validation_fusion"):
            primary, alternative, recs, flags = validate_output(out, self.kb, experiment, available)
            for h in primary + alternative:
                h.weather_consistency = weather_consistency(h.problem_id, weather, self.kb) if weather else "unknown"
            contradictions = detect_contradictions(out, weather, satellite if satellite and satellite.status in ("ok", "partial") else None, self.kb)
            level, why = caution_level(out, report.image_quality, primary, contradictions, flags, weather, satellite,
                                       self.kb, experiment.use_weather, experiment.use_satellite)

        report.status = "completed" if out.image_assessment.image_usable and out.overall_assessment != "cannot_assess" else "needs_new_photo"
        report.image_assessment = out.image_assessment
        report.overall_assessment = out.overall_assessment
        report.visual_observations = out.visual_observations
        report.primary_hypotheses, report.alternative_hypotheses = primary, alternative
        report.contradictions = contradictions
        report.recommendations = recs
        report.validation_flags = flags
        report.caution_level, report.caution_rationale = level, why
        report.missing_information = list(dict.fromkeys(report.missing_information + out.missing_information))
        report.recommended_checks = out.recommended_field_checks
        report.reasoning_summary = out.reasoning_summary
        if experiment.use_knowledge_base:
            ids = list(dict.fromkeys([h.problem_id for h in primary + alternative if h.kb_known]
                                     + [r.kb_reference_id for r in recs if r.kb_reference_id]))
            report.knowledge_references = self.kb.reference_summary(ids)
        report.timings = timings
        return report


def image_metadata_from_path(path: str | Path, plant_part: str = "unknown", source: str = "dataset",
                             licence: str | None = None, attribution: str | None = None) -> ImageMetadata:
    """Build ImageMetadata for an existing file without copying it (evaluation use)."""
    import hashlib

    from PIL import Image

    from engine.acquisition.images import read_exif

    p = Path(path)
    data = p.read_bytes()
    with Image.open(p) as img:
        capture, lat, lon = read_exif(img)
        w, h, fmt = img.width, img.height, img.format
    return ImageMetadata(image_id="img_" + hashlib.sha256(data).hexdigest()[:12], path=str(p),
                         sha256=hashlib.sha256(data).hexdigest(), width=w, height=h, format=fmt,
                         capture_date=capture, exif_latitude=lat, exif_longitude=lon, plant_part=plant_part,
                         source=source, licence=licence, attribution=attribution)
