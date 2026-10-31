"""Maana Terra REST API (FastAPI). Thin layer over engine.services — no model calls here.

Run: uvicorn api.main:app --reload
"""

from __future__ import annotations

from datetime import date

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from engine.acquisition.images import ImageRejected
from engine.config import describe_secret_status, get_settings
from engine.logging_utils import setup_logging
from engine.schemas import AnalysisReport, ExperimentConfig, Observation, ObservationCreate, Parcel, ParcelCreate
from engine.services import MaanaTerraService, check_sources, get_service
from engine.sources.agronomy.knowledge_base import get_knowledge_base

setup_logging()
app = FastAPI(
    title="Maana Terra API",
    version="0.1.0",
    description="Intelligence agricole multimodale pour le cacao (Muse Spark, SAM 3.1, NASA POWER, Sentinel-2).",
)


def service() -> MaanaTerraService:
    return get_service()


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "muse_model": s.muse_model, "sam_model": s.sam_model, "sam_enabled": s.sam_enabled,
            "credentials": describe_secret_status()}


@app.get("/status/sources")
def sources_status() -> dict:
    return check_sources()


@app.get("/knowledge-base")
def knowledge_base() -> list[dict]:
    return [e.model_dump(mode="json") for e in get_knowledge_base().entries.values()]


@app.post("/parcels", response_model=Parcel, status_code=201)
def create_parcel(data: ParcelCreate, svc: MaanaTerraService = Depends(service)) -> Parcel:
    return svc.create_parcel(data)


@app.get("/parcels", response_model=list[Parcel])
def list_parcels(svc: MaanaTerraService = Depends(service)) -> list[Parcel]:
    return svc.list_parcels()


@app.get("/parcels/{parcel_id}", response_model=Parcel)
def get_parcel(parcel_id: str, svc: MaanaTerraService = Depends(service)) -> Parcel:
    p = svc.get_parcel(parcel_id)
    if not p:
        raise HTTPException(404, "Parcelle inconnue")
    return p


@app.post("/observations", response_model=Observation, status_code=201)
async def create_observation(
    parcel_id: str = Form(...),
    observation_date: date = Form(...),
    plant_part: str = Form("unknown"),
    farmer_symptoms: str | None = Form(None),
    farmer_notes: str | None = Form(None),
    images: list[UploadFile] = File(...),
    svc: MaanaTerraService = Depends(service),
) -> Observation:
    try:
        data = ObservationCreate(parcel_id=parcel_id, observation_date=observation_date, plant_part=plant_part,
                                 farmer_symptoms=farmer_symptoms, farmer_notes=farmer_notes)
        return svc.create_observation(data, [await f.read() for f in images])
    except ImageRejected as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/parcels/{parcel_id}/observations", response_model=list[Observation])
def list_observations(parcel_id: str, svc: MaanaTerraService = Depends(service)) -> list[Observation]:
    return svc.list_observations(parcel_id)


@app.post("/observations/{observation_id}/analyze", response_model=AnalysisReport)
def analyze(observation_id: str, experiment: ExperimentConfig | None = None,
            svc: MaanaTerraService = Depends(service)) -> AnalysisReport:
    try:
        return svc.analyze_observation(observation_id, experiment)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/analyses", response_model=list[AnalysisReport])
def list_analyses(parcel_id: str | None = None, limit: int = 50, svc: MaanaTerraService = Depends(service)):
    return svc.list_analyses(parcel_id, limit)


@app.get("/analyses/{analysis_id}", response_model=AnalysisReport)
def get_analysis(analysis_id: str, svc: MaanaTerraService = Depends(service)) -> AnalysisReport:
    a = svc.get_analysis(analysis_id)
    if not a:
        raise HTTPException(404, "Analyse inconnue")
    return a


@app.get("/schema/report")
def report_schema() -> dict:
    """JSON Schema of the analysis report (contract for integrators)."""
    return AnalysisReport.model_json_schema()


@app.get("/images/{image_id}")
def get_image(image_id: str, svc: MaanaTerraService = Depends(service)):
    for obs in svc.list_observations():
        for img in obs.images:
            if img.image_id == image_id:
                return FileResponse(img.path)
    raise HTTPException(404, "Image inconnue")
