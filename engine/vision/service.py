"""Glue between the trained classifier and the analysis pipeline (never raises)."""

from __future__ import annotations

import logging
import time

from engine.acquisition.images import load_rgb
from engine.schemas import ClassifierResult, ConsensusDecision

log = logging.getLogger(__name__)


def run_classifier(image_path: str, plant_part: str | None) -> ClassifierResult:
    t0 = time.perf_counter()
    try:
        from engine.vision.classifier import PLANT_PART_TO_DATASET, get_classifier
        from engine.vision.embedder import get_embedder

        clf = get_classifier()
        if clf is None:
            return ClassifierResult(status="unavailable", error="Artefact du classifieur absent (scripts/train_classifier.py).")
        emb = get_embedder(clf.a["backbone"]["name"]).embed(load_rgb(image_path))
        out = clf.predict(emb, plant_part)
        part = PLANT_PART_TO_DATASET.get(plant_part or "")
        trusted = part in clf.a.get("trusted_parts", []) if part else False
        decision, reason = out.decision, out.reason
        if decision != "abstain" and not trusted:
            decision = "abstain"
            reason = ("Organe non précisé ou pour lequel le classifieur n'est pas jugé fiable "
                      f"(fiabilité mesurée à l'entraînement) : {plant_part or 'inconnu'}.")
        return ClassifierResult(
            status="ok", model_version=out.model_version, probability_cssvd_like=out.probability_cssvd_like,
            decision=decision, reason=reason, in_domain=out.in_domain, trusted_for_part=trusted,
            nearest_train_similarity=out.nearest_train_similarity, test_metrics=out.test_metrics,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    except ImportError as exc:
        return ClassifierResult(status="unavailable", error=f"Dépendance manquante : {exc}")
    except Exception as exc:  # model download / runtime failure must not break the pipeline
        log.warning("Classifier failed: %s", exc)
        return ClassifierResult(status="error", error=str(exc)[:300])


def consensus(clf: ClassifierResult | None, muse_overall: str) -> ConsensusDecision:
    """Rule H2 (pre-declared, evaluated in experiments/reports/classifier/hybrid_*.json)."""
    view = clf.decision if clf and clf.status == "ok" and clf.decision else "unavailable"
    if muse_overall == "cannot_assess":
        return ConsensusDecision(classifier_view=view, muse_view=muse_overall, final="uncertain_needs_check",
                                 agreement=False, explanation="Muse Spark ne peut pas évaluer l'image.")
    muse_anom = muse_overall == "abnormality_detected"
    if view in ("abstain", "unavailable"):
        return ConsensusDecision(
            classifier_view=view, muse_view=muse_overall, final="uncertain_needs_check", agreement=False,
            explanation="Le classifieur spécialisé ne se prononce pas : la seule lecture de Muse Spark n'est pas "
                        "suffisamment fiable (≈ 67 % de bonnes réponses mesurées) pour conclure.")
    clf_anom = view == "cssvd_like"
    if clf_anom == muse_anom:
        return ConsensusDecision(
            classifier_view=view, muse_view=muse_overall, agreement=True,
            final="anomaly_likely" if clf_anom else "no_anomaly_likely",
            explanation="Le classifieur spécialisé et Muse Spark concordent (règle de consensus H2 : "
                        "≈ 91 % de bonnes réponses sur les cas concordants de l'évaluation, 39/43).")
    return ConsensusDecision(
        classifier_view=view, muse_view=muse_overall, final="uncertain_needs_check", agreement=False,
        explanation="Le classifieur spécialisé et Muse Spark divergent : vérification terrain nécessaire.")
