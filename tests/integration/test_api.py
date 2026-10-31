from datetime import date

from fastapi.testclient import TestClient

from api.main import app, service
from engine.meta_ai.client import MetaModelClient
from engine.meta_ai.muse_spark import MuseSparkAnalyzer
from engine.orchestration.pipeline import AnalysisPipeline
from engine.services import MaanaTerraService
from engine.storage.db import Database
from tests.integration.test_pipeline_mocked import _stub_sat, _stub_weather


def test_api_flow(tmp_path, sample_pod):
    svc = MaanaTerraService(db=Database(tmp_path / "api.db"),
                            pipeline=AnalysisPipeline(muse=MuseSparkAnalyzer(client=MetaModelClient(api_key="")),
                                                      weather_fn=_stub_weather, satellite_fn=_stub_sat))
    app.dependency_overrides[service] = lambda: svc
    try:
        c = TestClient(app)
        assert c.get("/health").json()["credentials"]["meta_api_key_present"] is False
        assert len(c.get("/knowledge-base").json()) == 6
        r = c.post("/parcels", json={"name": "Soubré A", "latitude": 5.785, "longitude": -6.606})
        assert r.status_code == 201
        pid = r.json()["id"]
        assert c.post("/parcels", json={"name": "bad", "latitude": 99, "longitude": 0}).status_code == 422
        r = c.post("/observations", data={"parcel_id": pid, "observation_date": date(2026, 10, 1).isoformat(),
                                          "plant_part": "pod", "farmer_symptoms": "taches brunes"},
                   files=[("images", ("pod.jpg", sample_pod.read_bytes(), "image/jpeg"))])
        assert r.status_code == 201, r.text
        oid = r.json()["id"]
        bad = c.post("/observations", data={"parcel_id": pid, "observation_date": "2026-10-01"},
                     files=[("images", ("x.jpg", b"garbage", "image/jpeg"))])
        assert bad.status_code == 422
        rep = c.post(f"/observations/{oid}/analyze").json()
        assert rep["status"] == "failed" and "MODEL_API_KEY" in rep["failure_reason"]
        assert c.get(f"/analyses/{rep['analysis_id']}").status_code == 200
        assert len(c.get(f"/parcels/{pid}/observations").json()) == 1
    finally:
        app.dependency_overrides.clear()
