from datetime import date

import pytest

from engine.evaluation.experiments import date_from_filename
from engine.evaluation.metrics import classification_metrics, confusion
from engine.schemas import AnalysisReport, ExperimentConfig, ObservationCreate, ParcelCreate
from engine.storage.db import Database


def test_database_roundtrip(tmp_path, sample_pod):
    from engine.acquisition.images import ingest_image

    db = Database(tmp_path / "t.db")
    p = db.create_parcel(ParcelCreate(name="Parcelle A", latitude=5.78, longitude=-6.6, planting_year=2015))
    assert db.get_parcel(p.id).name == "Parcelle A"
    img = ingest_image(sample_pod.read_bytes(), plant_part="pod", images_dir=tmp_path / "img")
    obs = db.create_observation(ObservationCreate(parcel_id=p.id, observation_date=date(2026, 10, 1), plant_part="pod",
                                                  farmer_symptoms="taches brunes"), [img])
    got = db.get_observation(obs.id)
    assert got.images[0].sha256 == img.sha256
    rep = AnalysisReport(analysis_id="ana_1", parcel_id=p.id, observation_id=obs.id, experiment=ExperimentConfig(),
                         status="failed", failure_reason="test")
    db.save_analysis(rep)
    assert db.get_analysis("ana_1").failure_reason == "test"
    assert db.counts() == {"parcels": 1, "observations": 1, "images": 1, "analyses": 1}
    with pytest.raises(ValueError):
        db.create_observation(ObservationCreate(parcel_id="nope", observation_date=date(2026, 1, 1)), [])


def test_parcel_coordinates_validated():
    with pytest.raises(Exception):
        ParcelCreate(name="x", latitude=120, longitude=0)


def test_metrics_known_values():
    y_true = ["healthy", "healthy", "diseased", "diseased", "diseased", "healthy"]
    y_pred = ["healthy", "diseased", "diseased", "healthy", "abstain", "error"]
    m = classification_metrics(y_true, y_pred)
    assert m["n_evaluated"] == 5 and m["n_errors"] == 1 and m["n_answered"] == 4
    assert m["abstention_rate"] == 0.2
    assert m["accuracy"] == 0.5
    assert m["balanced_accuracy"] == 0.5
    assert m["per_class"]["diseased"]["precision"] == 0.5
    assert m["critical_error_rate"] == round(1 / 3, 4)
    assert confusion(y_true[:5], y_pred[:5])["diseased"]["abstain"] == 1


def test_metrics_all_abstained():
    m = classification_metrics(["healthy"], ["abstain"])
    assert m["abstention_rate"] == 1.0 and "accuracy" not in m


def test_date_from_filename():
    assert date_from_filename("20250527_095257.jpg") == date(2025, 5, 27)
    assert date_from_filename("IMG-20250527-WA0223.jpg") == date(2025, 5, 27)
    assert date_from_filename("WhatsApp Image 2025-05-28 at 10.25.54 (2).jpeg") == date(2025, 5, 28)
    assert date_from_filename("GTWX0674.JPG") is None
