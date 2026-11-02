"""Specialist classifier: numpy inference on a synthetic artefact + consensus rule."""

import json

import numpy as np
import pytest

from engine.schemas import ClassifierResult
from engine.vision.classifier import CocoaClassifier
from engine.vision.service import consensus


@pytest.fixture
def clf(tmp_path, monkeypatch):
    import engine.vision.classifier as mod

    dim = 8
    ref = np.eye(dim, dtype=np.float16)[:4]
    monkeypatch.setattr(mod, "ARTEFACT", tmp_path / "a.json")
    np.save(tmp_path / "ref.npy", ref)
    art = {"version": "test", "coef": [5.0] + [0.0] * (dim - 1), "intercept": 0.0, "platt": [1.0, 0.0],
           "abstain_band": [0.4, 0.6], "use_part_features": False, "trusted_parts": ["Leaves-Young"],
           "ood": {"min_similarity": 0.5, "reference_file": "ref.npy"}, "test_metrics_summary": {"accuracy": 0.9}}
    return CocoaClassifier(art)


def test_predict_decisions(clf):
    pos = np.zeros(8, dtype=np.float32); pos[0] = 1.0          # similar to ref[0], score 5 -> p≈0.99
    out = clf.predict(pos, "young_leaf")
    assert out.decision == "cssvd_like" and out.probability_cssvd_like > 0.95 and out.in_domain
    neg = np.zeros(8, dtype=np.float32); neg[1] = 1.0           # score 0 -> p=0.5 -> abstain band
    assert clf.predict(neg, "leaf").decision == "abstain"


def test_out_of_domain_abstains(clf):
    far = np.zeros(8, dtype=np.float32); far[7] = 1.0           # orthogonal to all references
    out = clf.predict(far, "leaf")
    assert not out.in_domain and out.decision == "abstain" and "hors domaine" in out.reason


@pytest.mark.parametrize("clf_dec,muse,final,agree", [
    ("cssvd_like", "abnormality_detected", "anomaly_likely", True),
    ("healthy", "no_visible_problem", "no_anomaly_likely", True),
    ("healthy", "abnormality_detected", "uncertain_needs_check", False),
    ("abstain", "abnormality_detected", "uncertain_needs_check", False),
])
def test_consensus_rule(clf_dec, muse, final, agree):
    c = ClassifierResult(status="ok", decision=clf_dec, probability_cssvd_like=0.5)
    d = consensus(c, muse)
    assert d.final == final and d.agreement is agree


def test_consensus_without_classifier():
    assert consensus(None, "no_visible_problem").final == "uncertain_needs_check"
