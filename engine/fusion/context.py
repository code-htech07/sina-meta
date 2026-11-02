"""Build the structured, provenance-tagged context passed to Muse Spark.

Each section declares its ``provenance_kind`` so the model (and the jury) can
distinguish: modelled estimates (NASA POWER), remote-sensing observations
(Sentinel-2), computed indicators, user declarations, literature, and MISSING data.

A source disabled by the experiment configuration is reported as
``not_provided_in_this_experiment``; a source that failed is reported as
``unavailable`` with the error. Missing values are never filled in.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from engine.schemas import (
    ExperimentConfig,
    ImageQualityReport,
    Parcel,
    SatelliteContext,
    SegmentationResult,
    WeatherContext,
)
from engine.sources.agronomy.knowledge_base import KnowledgeBase

NOT_PROVIDED = {"status": "not_provided_in_this_experiment", "provenance_kind": "missing"}


def normalization_checks(
    observation_date: date,
    weather: WeatherContext | None,
    satellite: SatelliteContext | None,
    today: date | None = None,
) -> list[str]:
    """Consistency checks on dates/staleness before fusion (6.2). Returns human-readable notes."""
    today = today or date.today()
    notes: list[str] = []
    if observation_date > today:
        notes.append(f"Date d'observation {observation_date} postérieure à aujourd'hui ({today}).")
    if weather and weather.indicators and weather.indicators.last_valid_day:
        lag = (observation_date - weather.indicators.last_valid_day).days
        if lag > 3:
            notes.append(f"Météo : dernier jour valide {lag} jours avant l'observation (latence NASA POWER).")
    if satellite and satellite.scenes:
        usable = [s for s in satellite.scenes if s.usable]
        if usable:
            age = (observation_date - max(s.acquired.date() for s in usable)).days
            if age > 30:
                notes.append(f"Satellite : dernière scène exploitable {age} jours avant l'observation.")
    return notes


def _weather_section(weather: WeatherContext | None) -> dict[str, Any]:
    if weather is None:
        return dict(NOT_PROVIDED)
    if weather.status == "unavailable":
        return {"status": "unavailable", "provenance_kind": "missing", "error": weather.error}
    return {
        "status": weather.status,
        "provenance_kind": "modelled_estimate (indicators: computed)",
        "source": weather.provenance.source,
        "period": f"{weather.provenance.period_start} -> {weather.provenance.period_end}",
        "units": {"rain": "mm", "temperature": "°C", "relative_humidity": "%"},
        "indicators": weather.indicators.model_dump(mode="json", exclude_none=True) if weather.indicators else None,
        "limitations": weather.limitations,
    }


def _satellite_section(sat: SatelliteContext | None) -> dict[str, Any]:
    if sat is None:
        return dict(NOT_PROVIDED)
    if sat.status in ("unavailable", "no_valid_scene"):
        return {"status": sat.status, "provenance_kind": "missing", "error": sat.error,
                "scenes_checked": len(sat.scenes)}
    return {
        "status": sat.status,
        "provenance_kind": "remote_sensing_observation (indices: computed)",
        "source": sat.provenance.source,
        "spatial_resolution": sat.provenance.spatial_resolution,
        "valid_scenes": [
            {"date": s.acquired.date().isoformat(), "ndvi_median": s.ndvi_median, "ndmi_median": s.ndmi_median,
             "valid_pixel_fraction": s.valid_pixel_fraction}
            for s in sat.scenes if s.usable
        ],
        "scenes_rejected_cloud_or_quality": sum(1 for s in sat.scenes if not s.usable),
        "ndvi_change_first_to_last": sat.ndvi_change,
        "ndmi_change_first_to_last": sat.ndmi_change,
        "limitations": sat.limitations,
    }


def _segmentation_section(seg: SegmentationResult | None) -> dict[str, Any]:
    if seg is None:
        return dict(NOT_PROVIDED)
    if seg.status != "ok":
        return {"status": seg.status, "provenance_kind": "missing" if seg.status != "no_match" else "image_measurement",
                "error": seg.error}
    return {
        "status": "ok",
        "provenance_kind": "image_measurement (SAM 3.1 localisation, no diagnostic value)",
        "model": seg.model,
        "regions": [
            {"prompt": r.prompt, "object_id": r.object_id, "bbox_xyxy_px": [round(v) for v in r.bbox_xyxy],
             "image_size_px": [r.image_width, r.image_height], "mask_area_fraction": r.mask_area_fraction,
             "model_score": r.mask_confidence}
            for r in seg.regions
        ],
    }


def _classifier_section(clf: Any) -> dict[str, Any]:
    if clf is None:
        return dict(NOT_PROVIDED)
    if clf.status != "ok":
        return {"status": clf.status, "provenance_kind": "missing", "error": clf.error}
    return {
        "status": "ok",
        "provenance_kind": "trained_model_prediction (DINOv2 + logistic regression trained on 1,285 labelled "
                           "CocoaSwolSet images from Côte d'Ivoire; probability calibrated on held-out data)",
        "task": "healthy vs swollen-shoot-like (CSSVD) appearance ONLY; it knows no other disease",
        "probability_cssvd_like": clf.probability_cssvd_like,
        "decision": clf.decision,
        "decision_reason": clf.reason,
        "in_training_domain": clf.in_domain,
        "trusted_for_this_plant_part": clf.trusted_for_part,
        "held_out_test_performance": clf.test_metrics,
        "use": "Treat as strong but fallible evidence about CSSVD-like vs healthy. Cite it with source_type "
               "'classifier'. It cannot rule in or out other problems (black pod, mirids, ...).",
    }


def build_context(
    *,
    parcel: Parcel | None,
    observation_date: date,
    declared_plant_part: str,
    farmer_symptoms: str | None,
    farmer_notes: str | None,
    quality: list[ImageQualityReport],
    weather: WeatherContext | None,
    satellite: SatelliteContext | None,
    segmentation: SegmentationResult | None,
    kb: KnowledgeBase | None,
    experiment: ExperimentConfig,
    classifier: Any = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return (context_for_llm, missing_information)."""
    missing: list[str] = []
    ctx: dict[str, Any] = {"experiment": experiment.name, "crop": "cocoa (Theobroma cacao)"}

    if parcel is not None:
        ctx["parcel"] = {
            "provenance_kind": "user_declared",
            "latitude": parcel.latitude,
            "longitude": parcel.longitude,
            "planting_year": parcel.planting_year,
            "variety": parcel.variety,
            "area_ha": parcel.area_ha,
            "notes": parcel.notes if experiment.use_farmer_input else None,
        }
        if parcel.planting_year is None:
            missing.append("Année de plantation non renseignée.")
    else:
        missing.append("Aucune parcelle associée (localisation inconnue).")

    ctx["observation"] = {"provenance_kind": "user_declared", "date": observation_date.isoformat(),
                          "declared_plant_part": declared_plant_part}
    if experiment.use_farmer_input:
        ctx["observation"]["farmer_reported_symptoms"] = farmer_symptoms or None
        ctx["observation"]["farmer_notes"] = farmer_notes or None
        if not farmer_symptoms:
            missing.append("Aucun symptôme décrit par l'agriculteur.")

    ctx["image_quality"] = {
        "provenance_kind": "image_measurement",
        "images": [{"image_id": q.image_id, "usable": q.usable, "issues": q.issues,
                    "size_px": [q.width, q.height]} for q in quality],
    }
    ctx["segmentation"] = _segmentation_section(segmentation if experiment.use_segmentation else None)
    ctx["specialist_classifier"] = _classifier_section(classifier if experiment.use_classifier else None)
    ctx["weather"] = _weather_section(weather if experiment.use_weather else None)
    ctx["satellite"] = _satellite_section(satellite if experiment.use_satellite else None)

    if experiment.use_knowledge_base and kb is not None:
        ctx["knowledge_base"] = {
            "provenance_kind": "literature",
            "instruction": "Use these ids as problem_id / reference_id. Do not cite anything else as literature.",
            "entries": kb.context_block(farmer_symptoms if experiment.use_farmer_input else None, declared_plant_part),
        }
    else:
        ctx["knowledge_base"] = dict(NOT_PROVIDED)

    if experiment.use_weather and (weather is None or weather.status == "unavailable"):
        missing.append("Données météorologiques indisponibles.")
    if experiment.use_satellite and (satellite is None or satellite.status in ("unavailable", "no_valid_scene")):
        missing.append("Aucune observation satellitaire exploitable (nuages ou service indisponible).")
    notes = normalization_checks(observation_date, weather if experiment.use_weather else None,
                                 satellite if experiment.use_satellite else None)
    if notes:
        ctx["data_consistency_notes"] = notes
    ctx["missing_information"] = missing
    return ctx, missing
