"""Deterministic cross-source consistency rules.

These rules run AFTER Muse Spark, independently of it, so that contradictions are
surfaced even if the LLM ignores them (an "ignored contradiction" metric compares
rule hits with the contradictions the model reported). Thresholds are engineering
heuristics from ``configs/settings.yaml`` — not literature thresholds — and are
labelled as such in every message.
"""

from __future__ import annotations

from engine.config import get_settings
from engine.schemas import Contradiction, MuseAnalysisOutput, SatelliteContext, WeatherContext
from engine.sources.agronomy.knowledge_base import KnowledgeBase

ACTIVE = ("compatible", "plausible")


def weather_regime(weather: WeatherContext | None) -> str:
    """Classify the last 30 days as 'wet', 'dry', 'normal' or 'unknown' (heuristic)."""
    if weather is None or weather.indicators is None or weather.indicators.valid_days == 0:
        return "unknown"
    cfg = get_settings().settings["fusion"]
    ind = weather.indicators
    if ind.rain_30d_ratio_vs_normal is not None:
        if ind.rain_30d_ratio_vs_normal <= cfg["dry_rain_ratio_vs_normal"]:
            return "dry"
        if ind.rain_30d_ratio_vs_normal >= cfg["wet_rain_ratio_vs_normal"]:
            return "wet"
    if ind.longest_dry_spell_30d is not None and ind.longest_dry_spell_30d >= cfg["dry_spell_days_alert"]:
        return "dry"
    if ind.rain_30d_ratio_vs_normal is None:
        return "unknown"
    return "normal"


def weather_consistency(problem_id: str, weather: WeatherContext | None, kb: KnowledgeBase) -> str:
    entry = kb.get(problem_id)
    regime = weather_regime(weather)
    if entry is None or regime == "unknown":
        return "unknown"
    if entry.weather_signal == "none":
        return "neutral"
    if entry.weather_signal == "wet_favourable":
        return {"wet": "consistent", "dry": "inconsistent"}.get(regime, "neutral")
    if entry.weather_signal == "dry_favourable":
        return {"dry": "consistent", "wet": "inconsistent"}.get(regime, "neutral")
    return "neutral"


def detect_contradictions(
    output: MuseAnalysisOutput,
    weather: WeatherContext | None,
    satellite: SatelliteContext | None,
    kb: KnowledgeBase,
    declared_crop: str = "cocoa",
) -> list[Contradiction]:
    cfg = get_settings().settings["fusion"]
    found: list[Contradiction] = []
    regime = weather_regime(weather)
    ind = weather.indicators if weather else None

    for h in output.hypotheses:
        if h.status not in ACTIVE:
            continue
        entry = kb.get(h.problem_id)
        if entry is None:
            continue
        if entry.weather_signal == "wet_favourable" and regime == "dry":
            found.append(Contradiction(
                rule_id="R1_wet_disease_dry_weather",
                origin="rule",
                related_problem_id=h.problem_id,
                sources_involved=["image", "weather", "knowledge_base"],
                description=(
                    f"L'hypothèse « {entry.name} » est documentée comme favorisée par des conditions humides, "
                    f"mais les 30 derniers jours (NASA POWER) sont classés secs selon les seuils heuristiques du système "
                    f"(cumul {ind.rain_30d_mm} mm, ratio à la normale {ind.rain_30d_ratio_vs_normal}, "
                    f"plus longue période sèche {ind.longest_dry_spell_30d} j)."
                ),
            ))
        if entry.weather_signal == "dry_favourable" and regime == "wet":
            found.append(Contradiction(
                rule_id="R2_dry_problem_wet_weather",
                origin="rule",
                related_problem_id=h.problem_id,
                sources_involved=["image", "weather", "knowledge_base"],
                description=(
                    f"L'hypothèse « {entry.name} » suppose un déficit hydrique, mais la pluviométrie des 30 derniers "
                    f"jours est supérieure à la normale (ratio {ind.rain_30d_ratio_vs_normal}, NASA POWER, seuil heuristique)."
                ),
            ))

    if satellite and satellite.ndvi_change is not None and satellite.ndvi_change <= -cfg["ndvi_drop_alert"]:
        if output.overall_assessment == "no_visible_problem":
            found.append(Contradiction(
                rule_id="R3_satellite_decline_vs_healthy_photo",
                origin="rule",
                sources_involved=["image", "satellite"],
                description=(
                    f"Baisse du NDVI de {satellite.ndvi_change:+.2f} à l'échelle de la parcelle (Sentinel-2) alors que la photo "
                    "ne montre pas d'anomalie : la photo peut ne pas être représentative, ou la baisse a une autre cause "
                    "(taille, récolte, saison, ombrage, nuages résiduels)."
                ),
            ))
        else:
            active = [h for h in output.hypotheses if h.status in ACTIVE]
            if active:
                found.append(Contradiction(
                    rule_id="R4_satellite_decline_ambiguous",
                    origin="rule",
                    sources_involved=["satellite"],
                    description=(
                        f"Baisse du NDVI de {satellite.ndvi_change:+.2f} (Sentinel-2) : signal non spécifique, compatible "
                        "avec une maladie, un stress abiotique ou une intervention culturale ; il ne permet pas de "
                        "départager les hypothèses."
                    ),
                ))

    if output.image_assessment.is_cocoa == "no" and declared_crop == "cocoa":
        found.append(Contradiction(
            rule_id="R5_not_cocoa",
            origin="rule",
            sources_involved=["image", "farmer"],
            description="La parcelle est déclarée en cacao, mais l'image ne semble pas montrer un cacaoyer.",
        ))

    for c in output.contradictions:
        found.append(Contradiction(description=c, sources_involved=[], origin="model"))
    return found
