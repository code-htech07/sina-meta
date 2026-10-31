"""Post-LLM validation: traceability, prudence and source-consistency checks.

The validator never invents content. It can only: flag, downgrade labels
(e.g. recommendation ``documented=False``), and drop evidence items that cite a
source that was not provided to the model (hallucinated source).
"""

from __future__ import annotations

import re
from typing import Iterable

from engine.schemas import (
    ExperimentConfig,
    Hypothesis,
    MuseAnalysisOutput,
    MuseEvidence,
    Recommendation,
    ValidationFlag,
)
from engine.sources.agronomy.knowledge_base import KnowledgeBase

# Categorical/over-confident wording (FR + EN). "confirmer" (infinitive, e.g. "pour confirmer")
# is legitimate; past participles / adverbs of certainty are not.
CATEGORICAL = re.compile(
    r"\b(est confirmée?s?|diagnostic confirmé|confirmée? par (la|l')? ?(photo|image)|avec certitude|"
    r"certainement|sans aucun doute|définitivement|indiscutablement|"
    r"is confirmed|confirmed diagnosis|definitely|certainly|without (a|any) doubt)\b",
    re.IGNORECASE,
)

SOURCE_FLAGS = {
    "weather": "use_weather",
    "satellite": "use_satellite",
    "knowledge_base": "use_knowledge_base",
    "segmentation": "use_segmentation",
    "farmer": "use_farmer_input",
}
SUPPORT_ORDER = {"high": 0, "medium": 1, "low": 2}
STATUS_ORDER = {"compatible": 0, "plausible": 1, "weakly_supported": 2, "unlikely": 3}


def _texts(output: MuseAnalysisOutput) -> Iterable[str]:
    yield output.reasoning_summary
    for o in output.visual_observations:
        yield o.description
    for h in output.hypotheses:
        yield h.name
        for e in h.supporting_evidence + h.contradicting_evidence:
            yield e.statement
    for r in output.recommendations:
        yield r.text


def validate_output(
    output: MuseAnalysisOutput,
    kb: KnowledgeBase,
    experiment: ExperimentConfig,
    available_sources: set[str],
) -> tuple[list[Hypothesis], list[Hypothesis], list[Recommendation], list[ValidationFlag]]:
    """Return (primary, alternative, recommendations, flags).

    ``available_sources`` = sources that were actually present (provided AND not failed).
    """
    flags: list[ValidationFlag] = []

    def evidence_ok(ev: MuseEvidence, hyp_id: str) -> bool:
        if ev.source_type in SOURCE_FLAGS and ev.source_type not in available_sources:
            flags.append(ValidationFlag(
                code="hallucinated_source", severity="error",
                message=f"[{hyp_id}] élément cité depuis la source « {ev.source_type} » non fournie : retiré ({ev.statement[:120]})."))
            return False
        if ev.source_type == "knowledge_base" and (not ev.reference_id or kb.get(ev.reference_id) is None):
            flags.append(ValidationFlag(
                code="unsourced_claim", severity="warning",
                message=f"[{hyp_id}] affirmation attribuée à la littérature sans identifiant de fiche valide : {ev.statement[:120]}"))
        return True

    hyps: list[Hypothesis] = []
    for h in output.hypotheses:
        known = kb.get(h.problem_id) is not None
        if not known and h.problem_id != "unknown":
            flags.append(ValidationFlag(code="unknown_problem_id", severity="warning",
                                        message=f"Identifiant « {h.problem_id} » absent de la base de connaissances."))
        if h.support_level == "high" and h.status in ("weakly_supported", "unlikely"):
            flags.append(ValidationFlag(code="inconsistent_support", severity="warning",
                                        message=f"[{h.problem_id}] niveau de soutien « high » incohérent avec le statut « {h.status} »."))
        sup = [e for e in h.supporting_evidence if evidence_ok(e, h.problem_id)]
        con = [e for e in h.contradicting_evidence if evidence_ok(e, h.problem_id)]
        if not sup:
            flags.append(ValidationFlag(code="hypothesis_without_support", severity="warning",
                                        message=f"[{h.problem_id}] aucune preuve à l'appui après validation."))
        rank = "primary" if h.status in ("compatible", "plausible") else "alternative"
        hyps.append(Hypothesis(**{**h.model_dump(), "supporting_evidence": sup, "contradicting_evidence": con},
                               rank=rank, kb_known=known))

    hyps.sort(key=lambda x: (STATUS_ORDER[x.status], SUPPORT_ORDER[x.support_level]))
    primary = [h for h in hyps if h.rank == "primary"]
    alternative = [h for h in hyps if h.rank == "alternative"]

    if output.overall_assessment == "no_visible_problem" and primary:
        flags.append(ValidationFlag(code="assessment_inconsistent", severity="warning",
                                    message="Évaluation globale « aucun problème visible » mais hypothèses actives présentes."))

    recs: list[Recommendation] = []
    for r in output.recommendations:
        entry = kb.get(r.kb_reference_id) if r.kb_reference_id else None
        documented = entry is not None and experiment.use_knowledge_base
        if not documented:
            flags.append(ValidationFlag(code="undocumented_recommendation", severity="warning",
                                        message=f"Recommandation non rattachée à une fiche documentée : {r.text[:120]}"))
        recs.append(Recommendation(
            text=r.text, kb_reference_id=r.kb_reference_id if entry else None, documented=documented,
            sources=[ref.citation for ref in entry.references] if entry else []))

    for text in _texts(output):
        m = CATEGORICAL.search(text or "")
        if m:
            flags.append(ValidationFlag(code="categorical_language", severity="warning",
                                        message=f"Formulation trop catégorique détectée (« {m.group(0)} ») : {text[:140]}"))
    return primary, alternative, recs, flags
