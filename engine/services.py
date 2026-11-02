"""Application service layer: the single entry point used by the FastAPI routes,
the Streamlit UI and the notebooks. No API call is made outside the adapters."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any

import httpx

from engine.acquisition.images import ingest_image
from engine.config import describe_secret_status, get_settings
from engine.meta_ai.client import MetaModelClient
from engine.orchestration.pipeline import AnalysisInput, AnalysisPipeline
from engine.schemas import (
    AnalysisReport,
    ExperimentConfig,
    Observation,
    ObservationCreate,
    Parcel,
    ParcelCreate,
)
from engine.sources.agronomy.knowledge_base import get_knowledge_base
from engine.storage.db import Database


class MaanaTerraService:
    def __init__(self, db: Database | None = None, pipeline: AnalysisPipeline | None = None) -> None:
        self.db = db or Database()
        self._pipeline = pipeline

    @property
    def pipeline(self) -> AnalysisPipeline:
        if self._pipeline is None:
            self._pipeline = AnalysisPipeline()
        return self._pipeline

    # ----------------------------------------------------------------- parcels
    def create_parcel(self, data: ParcelCreate) -> Parcel:
        return self.db.create_parcel(data)

    def list_parcels(self) -> list[Parcel]:
        return self.db.list_parcels()

    def get_parcel(self, parcel_id: str) -> Parcel | None:
        return self.db.get_parcel(parcel_id)

    # ------------------------------------------------------------ observations
    def create_observation(self, data: ObservationCreate, files: list[bytes]) -> Observation:
        if not files:
            raise ValueError("Au moins une photographie est requise.")
        images = [ingest_image(b, plant_part=data.plant_part) for b in files]
        return self.db.create_observation(data, images)

    def list_observations(self, parcel_id: str | None = None) -> list[Observation]:
        return self.db.list_observations(parcel_id)

    # ---------------------------------------------------------------- analyses
    def analyze_observation(self, observation_id: str, experiment: ExperimentConfig | None = None) -> AnalysisReport:
        obs = self.db.get_observation(observation_id)
        if obs is None:
            raise ValueError(f"Observation inconnue : {observation_id}")
        parcel = self.db.get_parcel(obs.parcel_id)
        report = self.pipeline.run(
            AnalysisInput(images=obs.images, observation_date=obs.observation_date, plant_part=obs.plant_part,
                          farmer_symptoms=obs.farmer_symptoms, farmer_notes=obs.farmer_notes, parcel=parcel,
                          observation_id=obs.id),
            experiment,
        )
        self.db.save_analysis(report)
        return report

    def get_analysis(self, analysis_id: str) -> AnalysisReport | None:
        return self.db.get_analysis(analysis_id)

    def list_analyses(self, parcel_id: str | None = None, limit: int = 50) -> list[AnalysisReport]:
        return self.db.list_analyses(parcel_id, limit)

    # ------------------------------------------------------------------ status
    def dashboard(self) -> dict[str, Any]:
        return {"counts": self.db.counts(), "recent": [
            {"analysis_id": a.analysis_id, "parcel_id": a.parcel_id, "created_at": a.created_at,
             "status": a.status, "overall": a.overall_assessment, "caution": a.caution_level,
             "top_hypothesis": a.primary_hypotheses[0].name if a.primary_hypotheses else None}
            for a in self.db.list_analyses(limit=10)]}


def check_sources(timeout: float = 15.0, check_meta_models: bool = True) -> dict[str, Any]:
    """Live, lightweight health checks of every external dependency (no fake status)."""
    s = get_settings()
    out: dict[str, Any] = {"checked_on": date.today().isoformat()}

    def probe(name: str, fn) -> None:
        try:
            out[name] = {"ok": True, **(fn() or {})}
        except Exception as exc:
            out[name] = {"ok": False, "error": str(exc)[:300]}

    def nasa():
        cfg = s.data_sources["nasa_power"]
        r = httpx.get(cfg["daily_url"], params={"parameters": "T2M", "community": "AG", "latitude": 6.0,
                                                "longitude": -5.0, "start": "20250101", "end": "20250102",
                                                "format": "JSON"}, timeout=timeout)
        r.raise_for_status()
        return {"http": r.status_code}

    def stac():
        r = httpx.get(s.data_sources["sentinel2"]["stac_url"].rstrip("/") + "/collections/"
                      + s.data_sources["sentinel2"]["collection"], timeout=timeout)
        r.raise_for_status()
        return {"http": r.status_code}

    def kb():
        k = get_knowledge_base()
        return {"entries": len(k.ids()), "ids": k.ids()}

    def meta():
        status = describe_secret_status()
        info: dict[str, Any] = {"auth_mode": status["auth_mode"], "muse_model": s.muse_model,
                                "sam_model": s.sam_model, "sam_enabled": s.sam_enabled}
        if status["auth_mode"] == "none":
            raise RuntimeError("Aucune clé Meta (ni variable, ni secret réseau)")
        if check_meta_models:
            ids = MetaModelClient(timeout_s=timeout, max_retries=0).list_models()
            info["muse_listed"] = s.muse_model in ids
            info["sam_listed"] = s.sam_model in ids
        return info

    probe("nasa_power", nasa)
    probe("sentinel2_stac", stac)
    probe("knowledge_base", kb)
    probe("meta_api", meta)
    return out


@lru_cache(maxsize=1)
def get_service() -> MaanaTerraService:
    return MaanaTerraService()
