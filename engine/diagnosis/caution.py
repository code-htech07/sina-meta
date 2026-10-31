"""Deterministic caution level (computed by the engine, never by the LLM)."""

from __future__ import annotations

from engine.schemas import (
    Contradiction,
    Hypothesis,
    ImageQualityReport,
    MuseAnalysisOutput,
    SatelliteContext,
    ValidationFlag,
    WeatherContext,
)
from engine.sources.agronomy.knowledge_base import KnowledgeBase


def caution_level(
    output: MuseAnalysisOutput | None,
    quality: list[ImageQualityReport],
    primary: list[Hypothesis],
    contradictions: list[Contradiction],
    flags: list[ValidationFlag],
    weather: WeatherContext | None,
    satellite: SatelliteContext | None,
    kb: KnowledgeBase,
    weather_requested: bool,
    satellite_requested: bool,
) -> tuple[str, list[str]]:
    why: list[str] = []
    if output is None or not any(q.usable for q in quality):
        return "insufficient_data", ["Aucune image exploitable ou analyse visuelle indisponible."]
    if output.overall_assessment == "cannot_assess" or not output.image_assessment.image_usable:
        return "insufficient_data", ["Le modèle indique que l'image ne permet pas d'évaluer l'état de la plante."]

    level = 0  # 0 standard, 1 elevated, 2 high
    if [c for c in contradictions if c.origin == "rule"]:
        level = 2
        why.append("Contradiction(s) détectée(s) entre les sources.")
    if any(f.severity == "error" for f in flags):
        level = 2
        why.append("La validation a retiré des éléments citant une source non fournie.")
    if primary:
        top = primary[0]
        entry = kb.get(top.problem_id)
        if entry is None:
            level = 2
            why.append("L'hypothèse principale ne correspond à aucune fiche documentée.")
        elif entry.documentary_reliability == "low":
            level = max(level, 2)
            why.append(f"Fiche « {entry.name} » de fiabilité documentaire faible.")
        if top.support_level == "low":
            level = max(level, 1)
            why.append("Soutien faible pour l'hypothèse principale.")
        if len(primary) > 1 and primary[0].status == primary[1].status:
            level = max(level, 1)
            why.append("Plusieurs hypothèses actives de même statut : non départagées.")
    if any(q.issues for q in quality):
        level = max(level, 1)
        why.append("Défauts de qualité d'image signalés.")
    if weather_requested and (weather is None or weather.status == "unavailable"):
        level = max(level, 1)
        why.append("Contexte météorologique indisponible.")
    if satellite_requested and (satellite is None or satellite.status in ("unavailable", "no_valid_scene")):
        level = max(level, 1)
        why.append("Contexte satellitaire indisponible.")
    if not why:
        why.append("Sources disponibles et cohérentes ; la confirmation terrain reste nécessaire.")
    return ["standard", "elevated", "high"][level], why
