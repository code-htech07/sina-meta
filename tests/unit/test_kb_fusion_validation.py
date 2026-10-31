from datetime import date, datetime, timezone

from engine.diagnosis.caution import caution_level
from engine.diagnosis.validator import CATEGORICAL, validate_output
from engine.fusion.context import build_context
from engine.fusion.contradictions import detect_contradictions, weather_consistency, weather_regime
from engine.schemas import (
    EvidenceKind,
    ExperimentConfig,
    MuseAnalysisOutput,
    Provenance,
    SatelliteContext,
    WeatherContext,
    WeatherIndicators,
)
from engine.sources.agronomy.knowledge_base import KnowledgeBase, get_knowledge_base


def _weather(ratio: float | None, dry_spell: int = 0) -> WeatherContext:
    return WeatherContext(status="ok", indicators=WeatherIndicators(rain_30d_mm=50, rain_30d_ratio_vs_normal=ratio,
                                                                    longest_dry_spell_30d=dry_spell, valid_days=60),
                          provenance=Provenance(source="test", kind=EvidenceKind.MODELLED_ESTIMATE))


# ------------------------------------------------------------------ knowledge base
def test_every_fiche_is_valid_and_sourced():
    kb = KnowledgeBase.load()
    assert set(kb.ids()) == {"black_pod", "cssvd", "mirids", "anthracnose", "cherelle_wilt", "water_stress"}
    for e in kb.entries.values():
        assert e.references, e.id
        for r in e.references:
            assert r.url and r.accessed, (e.id, r.citation)
        assert e.characteristic_symptoms and e.plant_parts


def test_kb_rank_uses_symptoms_and_part():
    kb = get_knowledge_base()
    ranked = kb.rank("gonflement des rameaux et nervures rouges sur jeune feuille", "young_leaf")
    assert ranked[0][0].id == "cssvd"
    ranked = kb.rank("cabosse noire pourriture avec moisissure blanche", "pod")
    assert ranked[0][0].id == "black_pod"


# ------------------------------------------------------------------------- fusion
def test_context_marks_disabled_sources_not_provided():
    ctx, missing = build_context(parcel=None, observation_date=date(2026, 10, 1), declared_plant_part="pod",
                                 farmer_symptoms=None, farmer_notes=None, quality=[], weather=_weather(1.0),
                                 satellite=None, segmentation=None, kb=get_knowledge_base(),
                                 experiment=ExperimentConfig(name="E0", use_weather=False, use_satellite=False,
                                                             use_knowledge_base=False, use_segmentation=False))
    for k in ("weather", "satellite", "knowledge_base", "segmentation"):
        assert ctx[k]["status"] == "not_provided_in_this_experiment"
    assert any("parcelle" in m.lower() for m in missing)


def test_weather_regime_and_consistency():
    kb = get_knowledge_base()
    assert weather_regime(_weather(0.3)) == "dry"
    assert weather_regime(_weather(1.5)) == "wet"
    assert weather_regime(_weather(1.0)) == "normal"
    assert weather_regime(_weather(None, dry_spell=20)) == "dry"
    assert weather_regime(None) == "unknown"
    assert weather_consistency("black_pod", _weather(0.3), kb) == "inconsistent"
    assert weather_consistency("black_pod", _weather(1.5), kb) == "consistent"
    assert weather_consistency("cssvd", _weather(1.5), kb) == "neutral"
    assert weather_consistency("water_stress", _weather(1.5), kb) == "inconsistent"


def test_rule_detects_wet_disease_in_dry_weather(muse_fixture):
    out = MuseAnalysisOutput.model_validate(muse_fixture)
    contra = detect_contradictions(out, _weather(0.3), None, get_knowledge_base())
    assert any(c.rule_id == "R1_wet_disease_dry_weather" and c.related_problem_id == "black_pod" for c in contra)
    assert not detect_contradictions(out, _weather(1.5), None, get_knowledge_base())


def test_rule_satellite_decline_with_healthy_photo(muse_fixture):
    muse_fixture["overall_assessment"] = "no_visible_problem"
    muse_fixture["hypotheses"] = []
    out = MuseAnalysisOutput.model_validate(muse_fixture)
    sat = SatelliteContext(status="ok", ndvi_change=-0.2, provenance=Provenance(source="t", kind=EvidenceKind.REMOTE_SENSING_OBSERVATION))
    contra = detect_contradictions(out, None, sat, get_knowledge_base())
    assert [c.rule_id for c in contra] == ["R3_satellite_decline_vs_healthy_photo"]


# --------------------------------------------------------------------- validation
def test_validator_removes_hallucinated_sources_and_flags(muse_fixture):
    out = MuseAnalysisOutput.model_validate(muse_fixture)
    exp = ExperimentConfig(name="E4")
    primary, alternative, recs, flags = validate_output(out, get_knowledge_base(), exp, {"image", "knowledge_base"})
    codes = [f.code for f in flags]
    # weather evidence was cited but weather was not available -> removed
    bp = next(h for h in primary if h.problem_id == "black_pod")
    assert all(e.source_type != "weather" for e in bp.supporting_evidence)
    assert "hallucinated_source" in codes
    assert "unknown_problem_id" in codes            # made_up_disease
    assert "undocumented_recommendation" in codes   # "produit miracle"
    assert [r.documented for r in recs] == [True, False]
    assert recs[0].sources  # documented recommendation carries its citations
    assert [h.problem_id for h in primary] == ["black_pod", "mirids"]
    assert alternative[0].problem_id == "made_up_disease" and not alternative[0].kb_known


def test_categorical_language_regex():
    assert CATEGORICAL.search("Le diagnostic est confirmé par la photo")
    assert CATEGORICAL.search("C'est certainement la pourriture brune")
    assert not CATEGORICAL.search("Une analyse de laboratoire est nécessaire pour confirmer")


def test_caution_level_high_on_contradiction(muse_fixture):
    from engine.schemas import ImageQualityReport

    kb = get_knowledge_base()
    out = MuseAnalysisOutput.model_validate(muse_fixture)
    q = [ImageQualityReport(image_id="i", width=1000, height=1000, laplacian_variance=300, mean_brightness=120,
                            clipped_fraction=0, issues=[], usable=True,
                            provenance=Provenance(source="t", kind=EvidenceKind.IMAGE_MEASUREMENT))]
    primary, _, _, flags = validate_output(out, kb, ExperimentConfig(), {"image", "knowledge_base", "weather"})
    contra = detect_contradictions(out, _weather(0.3), None, kb)
    level, why = caution_level(out, q, primary, contra, flags, _weather(0.3), None, kb, True, False)
    assert level == "high" and why
    out.overall_assessment = "cannot_assess"
    assert caution_level(out, q, primary, [], [], None, None, kb, False, False)[0] == "insufficient_data"
