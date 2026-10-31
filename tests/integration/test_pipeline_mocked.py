"""Full pipeline with the Meta API replaced by a contract-shaped MockTransport and
context sources replaced by stubs. Verifies orchestration, fusion, validation and
persistence — not model quality."""

import json
from datetime import date

import httpx

from engine.meta_ai.client import MetaModelClient
from engine.meta_ai.muse_spark import MuseSparkAnalyzer
from engine.meta_ai.sam import SamSegmenter
from engine.orchestration.pipeline import AnalysisInput, AnalysisPipeline, image_metadata_from_path
from engine.schemas import (
    EvidenceKind,
    ExperimentConfig,
    ObservationCreate,
    ParcelCreate,
    Provenance,
    SatelliteContext,
    WeatherContext,
    WeatherIndicators,
)
from engine.services import MaanaTerraService
from engine.storage.db import Database


def _stub_weather(lat, lon, d):
    return WeatherContext(status="ok", indicators=WeatherIndicators(rain_30d_mm=20, rain_30d_ratio_vs_normal=0.2,
                                                                    longest_dry_spell_30d=18, valid_days=60),
                          provenance=Provenance(source="stub", kind=EvidenceKind.MODELLED_ESTIMATE))


def _stub_sat(lat, lon, d, poly=None):
    return SatelliteContext(status="no_valid_scene", provenance=Provenance(source="stub", kind=EvidenceKind.REMOTE_SENSING_OBSERVATION),
                            error="clouds")


def _pipeline(muse_fixture, calls):
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"id": "r1", "status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(muse_fixture)}]}],
            "usage": {"input_tokens": 3000, "output_tokens": 800}})
    client = MetaModelClient(api_key="test-key-xxxxxxxxxxxx", transport=httpx.MockTransport(handler))
    sam = SamSegmenter(client=MetaModelClient(api_key=""))  # no key => SAM unavailable, pipeline must continue
    return AnalysisPipeline(muse=MuseSparkAnalyzer(client=client), sam=sam, weather_fn=_stub_weather, satellite_fn=_stub_sat)


def test_pipeline_full_context(muse_fixture, sample_pod):
    calls = []
    pipe = _pipeline(muse_fixture, calls)
    img = image_metadata_from_path(sample_pod, plant_part="pod")
    rep = pipe.run(AnalysisInput(images=[img], observation_date=date(2026, 10, 1), plant_part="pod",
                                 latitude=5.78, longitude=-6.6), ExperimentConfig())
    assert rep.status == "completed"
    assert rep.segmentation.status == "unavailable"
    assert rep.satellite_context.status == "no_valid_scene"
    # weather was available, so the weather evidence stays; satellite evidence (not available) is removed
    assert any(f.code == "hallucinated_source" and "satellite" in f.message for f in rep.validation_flags)
    # dry weather + wet-favoured disease => deterministic contradiction and high caution
    assert any(c.rule_id == "R1_wet_disease_dry_weather" for c in rep.contradictions)
    assert rep.caution_level == "high"
    assert rep.primary_hypotheses[0].weather_consistency == "inconsistent"
    assert rep.knowledge_references and rep.model_usage[0].input_tokens == 3000
    ctx = json.loads(calls[0]["input"][0]["content"][1]["text"].split("CONTEXT:\n", 1)[1])
    assert ctx["weather"]["provenance_kind"].startswith("modelled_estimate")
    assert ctx["satellite"]["status"] == "no_valid_scene"
    names = [t.step for t in rep.timings]
    assert names[0] == "image_quality" and "muse_spark_analysis" in names


def test_pipeline_e0_sends_no_context(muse_fixture, sample_pod):
    calls = []
    pipe = _pipeline(muse_fixture, calls)
    img = image_metadata_from_path(sample_pod, plant_part="pod")
    exp = ExperimentConfig(name="E0", use_weather=False, use_satellite=False, use_knowledge_base=False,
                           use_segmentation=False, use_farmer_input=False)
    rep = pipe.run(AnalysisInput(images=[img], observation_date=date(2026, 10, 1), latitude=5.78, longitude=-6.6), exp)
    ctx = json.loads(calls[0]["input"][0]["content"][1]["text"].split("CONTEXT:\n", 1)[1])
    assert ctx["weather"]["status"] == ctx["knowledge_base"]["status"] == "not_provided_in_this_experiment"
    assert rep.weather_context is None and rep.knowledge_references == []
    # recommendations cannot be "documented" without the knowledge base
    assert not any(r.documented for r in rep.recommendations)


def test_pipeline_without_key_fails_explicitly(sample_pod, tmp_path):
    svc = MaanaTerraService(db=Database(tmp_path / "x.db"),
                            pipeline=AnalysisPipeline(muse=MuseSparkAnalyzer(client=MetaModelClient(api_key="")),
                                                      weather_fn=_stub_weather, satellite_fn=_stub_sat))
    p = svc.create_parcel(ParcelCreate(name="P", latitude=5.78, longitude=-6.6))
    obs = svc.create_observation(ObservationCreate(parcel_id=p.id, observation_date=date(2026, 10, 1), plant_part="pod"),
                                 [sample_pod.read_bytes()])
    rep = svc.analyze_observation(obs.id)
    assert rep.status == "failed" and "MODEL_API_KEY" in rep.failure_reason
    assert rep.weather_context is not None  # real context sources still reported
    assert rep.primary_hypotheses == [] and rep.visual_observations == []  # nothing invented
    assert svc.get_analysis(rep.analysis_id).status == "failed"


def test_unusable_image_requests_new_photo(tmp_path, muse_fixture):
    from PIL import Image

    p = tmp_path / "tiny.png"
    Image.new("RGB", (100, 100), (5, 5, 5)).save(p)
    calls = []
    rep = _pipeline(muse_fixture, calls).run(AnalysisInput(images=[image_metadata_from_path(p)], observation_date=date(2026, 1, 1)))
    assert rep.status == "needs_new_photo" and calls == []  # no paid API call on unusable image
