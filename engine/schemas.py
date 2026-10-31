"""Pydantic data contracts shared by the engine, the API and the UI.

Design rules encoded here:
* every contextual value carries a :class:`Provenance` with an explicit
  :class:`EvidenceKind` (measured / modelled / computed / declared / literature /
  AI-generated / missing);
* hypotheses have a *status* that can never be "confirmed" — a photograph can
  only make a problem *compatible* with what is seen;
* the LLM's ``support_level`` is a qualitative, NON-calibrated label;
* the ``caution_level`` is computed deterministically by the engine, not the LLM.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
class EvidenceKind(str, Enum):
    REMOTE_SENSING_OBSERVATION = "remote_sensing_observation"  # Sentinel-2 reflectances
    MODELLED_ESTIMATE = "modelled_estimate"                    # NASA POWER (gridded, not on-parcel)
    COMPUTED = "computed"                                      # indicators/indices derived by Maana Terra
    USER_DECLARED = "user_declared"                            # farmer input
    IMAGE_MEASUREMENT = "image_measurement"                    # local image QC metrics, SAM masks
    LITERATURE = "literature"                                  # knowledge base fiches
    AI_GENERATED = "ai_generated"                              # Muse Spark observations / hypotheses
    MISSING = "missing"


class Provenance(BaseModel):
    source: str
    kind: EvidenceKind
    url: str | None = None
    retrieved_at: datetime | None = None
    period_start: date | None = None
    period_end: date | None = None
    latitude: float | None = None
    longitude: float | None = None
    spatial_resolution: str | None = None
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Parcels and observations
# --------------------------------------------------------------------------- #
PlantPart = Literal["pod", "leaf", "young_leaf", "stem", "branch", "trunk", "whole_tree", "cherelle", "flower", "root", "other", "unknown"]


class ParcelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    crop: str = "cocoa"
    polygon_geojson: dict[str, Any] | None = None
    planting_year: int | None = Field(default=None, ge=1900, le=2100)
    variety: str | None = None
    area_ha: float | None = Field(default=None, gt=0)
    notes: str | None = None


class Parcel(ParcelCreate):
    id: str
    created_at: datetime


class ImageMetadata(BaseModel):
    image_id: str
    path: str
    sha256: str
    width: int
    height: int
    format: str | None = None
    capture_date: datetime | None = None   # from EXIF if present
    exif_latitude: float | None = None
    exif_longitude: float | None = None
    crop: str = "cocoa"
    plant_part: PlantPart = "unknown"
    source: str = "user_upload"
    licence: str | None = None
    attribution: str | None = None


class ObservationCreate(BaseModel):
    parcel_id: str
    observation_date: date
    plant_part: PlantPart = "unknown"
    farmer_symptoms: str | None = None
    farmer_notes: str | None = None


class Observation(ObservationCreate):
    id: str
    created_at: datetime
    images: list[ImageMetadata] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Image quality
# --------------------------------------------------------------------------- #
class ImageQualityReport(BaseModel):
    image_id: str
    width: int
    height: int
    laplacian_variance: float
    mean_brightness: float
    clipped_fraction: float
    issues: list[str] = Field(default_factory=list)  # machine codes, e.g. "blurry"
    usable: bool
    retake_advice: list[str] = Field(default_factory=list)
    provenance: Provenance


# --------------------------------------------------------------------------- #
# Segmentation (SAM 3.1)
# --------------------------------------------------------------------------- #
class SegmentedRegion(BaseModel):
    image_id: str
    prompt: str
    object_id: str
    bbox_xyxy: tuple[float, float, float, float]
    image_width: int
    image_height: int
    mask_area_px: int | None = None
    mask_area_fraction: float | None = None
    box_confidence: float | None = None   # model-reported score, not a calibrated probability
    mask_confidence: float | None = None
    crop_path: str | None = None
    overlay_path: str | None = None


class SegmentationResult(BaseModel):
    status: Literal["ok", "no_match", "disabled", "unavailable", "error"]
    model: str | None = None
    regions: list[SegmentedRegion] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float | None = None


# --------------------------------------------------------------------------- #
# Weather (NASA POWER)
# --------------------------------------------------------------------------- #
class DailyWeather(BaseModel):
    day: date
    precipitation_mm: float | None = None
    t2m_c: float | None = None
    t2m_max_c: float | None = None
    t2m_min_c: float | None = None
    rh2m_pct: float | None = None


class WeatherIndicators(BaseModel):
    rain_7d_mm: float | None = None
    rain_14d_mm: float | None = None
    rain_30d_mm: float | None = None
    rain_60d_mm: float | None = None
    rainy_days_30d: int | None = None            # days with >= 1 mm
    longest_dry_spell_30d: int | None = None     # consecutive days < 1 mm
    current_dry_spell_days: int | None = None
    mean_t2m_30d_c: float | None = None
    max_t2m_30d_c: float | None = None
    min_t2m_30d_c: float | None = None
    mean_rh2m_30d_pct: float | None = None
    days_rh_ge_90_30d: int | None = None
    climatology_rain_30d_mm: float | None = None  # long-term normal for the same window
    rain_30d_ratio_vs_normal: float | None = None
    climatology_t2m_c: float | None = None
    t2m_anomaly_30d_c: float | None = None
    valid_days: int = 0
    missing_days: int = 0
    last_valid_day: date | None = None


class WeatherContext(BaseModel):
    status: Literal["ok", "partial", "unavailable"]
    indicators: WeatherIndicators | None = None
    daily: list[DailyWeather] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance
    limitations: list[str] = Field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------- #
# Satellite (Sentinel-2 L2A)
# --------------------------------------------------------------------------- #
class SatelliteScene(BaseModel):
    scene_id: str
    acquired: datetime
    scene_cloud_cover: float | None = None
    valid_pixel_fraction: float
    ndvi_median: float | None = None
    ndmi_median: float | None = None
    n_pixels: int
    usable: bool
    note: str | None = None


class SatelliteContext(BaseModel):
    status: Literal["ok", "partial", "unavailable", "no_valid_scene"]
    scenes: list[SatelliteScene] = Field(default_factory=list)
    latest_ndvi: float | None = None
    latest_ndmi: float | None = None
    ndvi_change: float | None = None        # latest - earliest valid scene
    ndmi_change: float | None = None
    valid_scene_count: int = 0
    provenance: Provenance
    limitations: list[str] = Field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
class KBReference(BaseModel):
    citation: str
    url: str | None = None
    accessed: date | None = None
    reliability: Literal["peer_reviewed", "institutional_extension", "institutional_factsheet", "grey_literature"]
    licence: str | None = None


class KBEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    alternative_names: list[str] = Field(default_factory=list)
    category: Literal["disease", "pest", "abiotic", "physiological"]
    causal_agents: list[str] = Field(default_factory=list)
    description: str
    characteristic_symptoms: list[str]
    plant_parts: list[str]
    favourable_conditions: list[str] = Field(default_factory=list)
    differentiation: list[str] = Field(default_factory=list)
    verification_needed: list[str] = Field(default_factory=list)
    management: list[str] = Field(default_factory=list)
    weather_signal: Literal["wet_favourable", "dry_favourable", "none"] = "none"
    keywords: list[str] = Field(default_factory=list)
    references: list[KBReference]
    documentary_reliability: Literal["high", "medium", "low"]
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Muse Spark structured output (what the LLM must return)
# --------------------------------------------------------------------------- #
class MuseImageAssessment(BaseModel):
    is_cocoa: Literal["yes", "no", "uncertain"]
    plant_part_detected: str
    image_usable: bool
    usability_issues: list[str] = Field(default_factory=list)


class MuseVisualObservation(BaseModel):
    description: str
    plant_part: str
    location_in_image: str
    extent: Literal["localized", "multiple_spots", "widespread", "unclear"]


class MuseEvidence(BaseModel):
    statement: str
    source_type: Literal["image", "weather", "satellite", "knowledge_base", "farmer", "segmentation"]
    reference_id: str | None = None   # KB entry id when source_type == knowledge_base


class MuseHypothesis(BaseModel):
    problem_id: str                      # KB entry id or "unknown"
    name: str
    status: Literal["compatible", "plausible", "weakly_supported", "unlikely"]
    support_level: Literal["low", "medium", "high"]
    supporting_evidence: list[MuseEvidence] = Field(default_factory=list)
    contradicting_evidence: list[MuseEvidence] = Field(default_factory=list)
    needed_for_confirmation: list[str] = Field(default_factory=list)


class MuseRecommendation(BaseModel):
    text: str
    kb_reference_id: str | None = None


class MuseAnalysisOutput(BaseModel):
    image_assessment: MuseImageAssessment
    overall_assessment: Literal["no_visible_problem", "abnormality_detected", "cannot_assess"]
    visual_observations: list[MuseVisualObservation] = Field(default_factory=list)
    hypotheses: list[MuseHypothesis] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommended_field_checks: list[str] = Field(default_factory=list)
    recommendations: list[MuseRecommendation] = Field(default_factory=list)
    reasoning_summary: str = ""


# --------------------------------------------------------------------------- #
# Final report
# --------------------------------------------------------------------------- #
class Contradiction(BaseModel):
    description: str
    sources_involved: list[str]
    origin: Literal["rule", "model"]
    rule_id: str | None = None
    related_problem_id: str | None = None


class ValidationFlag(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class Recommendation(BaseModel):
    text: str
    kb_reference_id: str | None = None
    documented: bool              # True only if it maps to a KB fiche's management section
    sources: list[str] = Field(default_factory=list)


class Hypothesis(MuseHypothesis):
    rank: Literal["primary", "alternative"]
    kb_known: bool
    weather_consistency: Literal["consistent", "inconsistent", "neutral", "unknown"] = "unknown"


class StepTiming(BaseModel):
    step: str
    status: Literal["ok", "skipped", "error"]
    latency_ms: float
    detail: str | None = None


class ModelUsage(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    images: int = 0
    estimated_cost_usd: float | None = None
    response_id: str | None = None


class ExperimentConfig(BaseModel):
    name: str = "full"
    use_weather: bool = True
    use_satellite: bool = True
    use_knowledge_base: bool = True
    use_segmentation: bool = True
    use_farmer_input: bool = True


class AnalysisReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    analysis_id: str
    parcel_id: str | None
    observation_id: str | None
    created_at: datetime = Field(default_factory=utcnow)
    experiment: ExperimentConfig
    status: Literal["completed", "needs_new_photo", "failed"]
    failure_reason: str | None = None

    image_quality: list[ImageQualityReport] = Field(default_factory=list)
    image_assessment: MuseImageAssessment | None = None
    overall_assessment: Literal["no_visible_problem", "abnormality_detected", "cannot_assess"] = "cannot_assess"
    visual_observations: list[MuseVisualObservation] = Field(default_factory=list)
    segmentation: SegmentationResult | None = None

    primary_hypotheses: list[Hypothesis] = Field(default_factory=list)
    alternative_hypotheses: list[Hypothesis] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)

    weather_context: WeatherContext | None = None
    satellite_context: SatelliteContext | None = None
    knowledge_references: list[dict[str, Any]] = Field(default_factory=list)  # {id, name, references[]}
    farmer_input: dict[str, Any] | None = None

    missing_information: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)

    caution_level: Literal["standard", "elevated", "high", "insufficient_data"] = "insufficient_data"
    caution_rationale: list[str] = Field(default_factory=list)
    validation_flags: list[ValidationFlag] = Field(default_factory=list)
    reasoning_summary: str = ""

    provenance: list[Provenance] = Field(default_factory=list)
    timings: list[StepTiming] = Field(default_factory=list)
    model_usage: list[ModelUsage] = Field(default_factory=list)
    disclaimer: str = (
        "Analyse d'aide à la décision. Une photographie ne permet pas de confirmer un diagnostic : "
        "les hypothèses doivent être vérifiées sur le terrain et, si nécessaire, par un laboratoire "
        "ou un conseiller agricole (ANADER, CNRA). Les scores de soutien ne sont pas des probabilités calibrées."
    )

    @field_validator("primary_hypotheses", "alternative_hypotheses")
    @classmethod
    def _no_confirmed(cls, v: list[Hypothesis]) -> list[Hypothesis]:
        # Literal type already forbids "confirmed"; keep an explicit guard for clarity.
        for h in v:
            assert h.status != "confirmed", "a photo-based hypothesis can never be 'confirmed'"
        return v
