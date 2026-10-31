"""Rendering of an AnalysisReport in Streamlit. Facts and hypotheses are visually separated."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from engine.schemas import AnalysisReport

STATUS_FR = {"compatible": "compatible", "plausible": "plausible", "weakly_supported": "faiblement étayée", "unlikely": "peu probable"}
SUPPORT_FR = {"low": "faible", "medium": "moyen", "high": "élevé"}
CAUTION = {
    "standard": ("🟢", "Prudence standard"),
    "elevated": ("🟠", "Prudence renforcée"),
    "high": ("🔴", "Prudence élevée"),
    "insufficient_data": ("⚪", "Données insuffisantes"),
}
OVERALL_FR = {"no_visible_problem": "Aucune anomalie visible sur cette photo",
              "abnormality_detected": "Anomalie visible détectée", "cannot_assess": "Évaluation impossible"}
WEATHER_CONS = {"consistent": "✅ cohérente avec la météo", "inconsistent": "⚠️ incohérente avec la météo",
                "neutral": "➖ météo non discriminante", "unknown": "❔ météo non évaluée"}


def _images(report: AnalysisReport, image_paths: dict[str, str]) -> None:
    cols = st.columns(2)
    with cols[0]:
        for q in report.image_quality:
            p = image_paths.get(q.image_id)
            if p and Path(p).exists():
                st.image(p, caption=f"Photo {q.image_id}", width="stretch")
    with cols[1]:
        seg = report.segmentation
        if seg and seg.status == "ok" and seg.regions and seg.regions[0].overlay_path and Path(seg.regions[0].overlay_path).exists():
            st.image(seg.regions[0].overlay_path, caption=f"Zones localisées par {seg.model} (localisation, pas diagnostic)",
                     width="stretch")
        elif seg:
            st.info(f"Segmentation SAM : {seg.status}" + (f" — {seg.error}" if seg.error else ""))


def render_report(report: AnalysisReport, image_paths: dict[str, str] | None = None) -> None:
    icon, label = CAUTION[report.caution_level]
    st.subheader(f"{icon} {label} — {OVERALL_FR.get(report.overall_assessment, '')}")
    st.caption(f"Analyse {report.analysis_id} · {report.created_at:%Y-%m-%d %H:%M} UTC · expérience « {report.experiment.name} »")
    if report.status == "failed":
        st.error(f"Analyse non réalisée : {report.failure_reason}")
    elif report.status == "needs_new_photo":
        st.warning("Nouvelle photographie nécessaire. " + (report.failure_reason or ""))
    for r in report.caution_rationale:
        st.markdown(f"- {r}")

    _images(report, image_paths or {})

    tab_facts, tab_hyp, tab_ctx, tab_refs, tab_trace = st.tabs(
        ["🔎 Faits observés", "🧪 Hypothèses", "🌦️🛰️ Contexte", "📚 Références & recommandations", "🧾 Traçabilité"])

    with tab_facts:
        st.markdown("**Contrôle qualité de l'image** (mesure locale)")
        st.dataframe(pd.DataFrame([{"image": q.image_id, "taille": f"{q.width}x{q.height}", "netteté (var. Laplacien)": q.laplacian_variance,
                                    "luminosité": q.mean_brightness, "exploitable": q.usable, "défauts": ", ".join(q.issues)}
                                   for q in report.image_quality]), hide_index=True)
        if report.image_assessment:
            a = report.image_assessment
            st.markdown(f"**Évaluation de l'image par Muse Spark** — cacao : `{a.is_cocoa}` · organe : {a.plant_part_detected} · "
                        f"exploitable : {a.image_usable}")
        st.markdown("**Observations visuelles** *(description de l'image par Muse Spark — pas un diagnostic)*")
        for o in report.visual_observations:
            st.markdown(f"- {o.description} — *{o.plant_part}, {o.location_in_image}, étendue : {o.extent}*")
        if report.farmer_input:
            st.markdown("**Déclaration de l'agriculteur**")
            st.json(report.farmer_input, expanded=False)

    with tab_hyp:
        st.info("Hypothèses générées par IA à partir de l'image et du contexte. Aucune n'est confirmée : "
                "les niveaux de soutien sont qualitatifs et NON calibrés.")
        for title, hyps in (("Hypothèses principales", report.primary_hypotheses), ("Hypothèses alternatives", report.alternative_hypotheses)):
            if not hyps:
                continue
            st.markdown(f"#### {title}")
            for h in hyps:
                with st.container(border=True):
                    kb = "" if h.kb_known else " · ⚠️ hors base documentée"
                    st.markdown(f"**{h.name}** (`{h.problem_id}`) — statut : *{STATUS_FR[h.status]}* · soutien : "
                                f"*{SUPPORT_FR[h.support_level]}* · {WEATHER_CONS[h.weather_consistency]}{kb}")
                    c1, c2 = st.columns(2)
                    c1.markdown("**Éléments favorables**\n" + "\n".join(f"- [{e.source_type}] {e.statement}" for e in h.supporting_evidence))
                    c2.markdown("**Éléments contradictoires**\n" + ("\n".join(f"- [{e.source_type}] {e.statement}" for e in h.contradicting_evidence) or "- —"))
                    if h.needed_for_confirmation:
                        st.markdown("**Nécessaire pour confirmer :** " + " ; ".join(h.needed_for_confirmation))
        if report.contradictions:
            st.markdown("#### ⚠️ Contradictions entre sources")
            for c in report.contradictions:
                origin = "règle déterministe" if c.origin == "rule" else "signalée par le modèle"
                st.warning(f"{c.description}  \n*({origin}{' ' + c.rule_id if c.rule_id else ''})*")
        if report.missing_information:
            st.markdown("#### Informations manquantes")
            st.markdown("\n".join(f"- {m}" for m in report.missing_information))
        if report.recommended_checks:
            st.markdown("#### Vérifications terrain recommandées")
            st.markdown("\n".join(f"- {m}" for m in report.recommended_checks))
        if report.reasoning_summary:
            with st.expander("Résumé du raisonnement (Muse Spark)"):
                st.write(report.reasoning_summary)

    with tab_ctx:
        w = report.weather_context
        st.markdown("#### Météo — NASA POWER *(estimation maillée, pas une mesure sur la parcelle)*")
        if w is None:
            st.caption("Non utilisée dans cette analyse.")
        elif w.status == "unavailable":
            st.error(f"Indisponible : {w.error}")
        else:
            ind = w.indicators
            m = st.columns(4)
            m[0].metric("Pluie 30 j (mm)", ind.rain_30d_mm, help=f"Normale climatologique : {ind.climatology_rain_30d_mm} mm")
            m[1].metric("Ratio / normale", ind.rain_30d_ratio_vs_normal)
            m[2].metric("Plus longue période sèche (j)", ind.longest_dry_spell_30d)
            m[3].metric("HR moyenne 30 j (%)", ind.mean_rh2m_30d_pct)
            df = pd.DataFrame([d.model_dump() for d in w.daily]).set_index("day")
            st.bar_chart(df["precipitation_mm"], height=180)
            st.line_chart(df[["t2m_c", "rh2m_pct"]], height=180)
            for lim in w.limitations:
                st.caption("• " + lim)
        s = report.satellite_context
        st.markdown("#### Satellite — Copernicus Sentinel-2 L2A *(contexte de parcelle, pas un diagnostic)*")
        if s is None:
            st.caption("Non utilisé dans cette analyse.")
        else:
            st.markdown(f"Statut : **{s.status}** · scènes valides : {s.valid_scene_count} · NDVI récent : {s.latest_ndvi} · "
                        f"NDMI récent : {s.latest_ndmi} · variation NDVI : {s.ndvi_change}")
            if s.error:
                st.caption(f"Détail : {s.error}")
            if s.scenes:
                st.dataframe(pd.DataFrame([{"scène": x.scene_id, "date": x.acquired.date(), "nuages scène %": x.scene_cloud_cover,
                                            "pixels valides": x.valid_pixel_fraction, "NDVI": x.ndvi_median, "NDMI": x.ndmi_median,
                                            "note": x.note} for x in s.scenes]), hide_index=True)
            for lim in s.limitations:
                st.caption("• " + lim)

    with tab_refs:
        st.markdown("#### Recommandations")
        for r in report.recommendations:
            badge = "📗 documentée" if r.documented else "⚠️ non documentée"
            st.markdown(f"- {r.text} — *{badge}*" + (f" (fiche `{r.kb_reference_id}`)" if r.kb_reference_id else ""))
        st.markdown("#### Références agronomiques utilisées")
        for k in report.knowledge_references:
            st.markdown(f"**{k['name']}** — fiabilité documentaire : {k['documentary_reliability']}")
            for ref in k["references"]:
                st.markdown(f"- {ref['citation']} — [{ref['url']}]({ref['url']}) (consulté le {ref['accessed']})")
        st.caption(report.disclaimer)

    with tab_trace:
        if report.validation_flags:
            st.markdown("#### Contrôles de validation")
            st.dataframe(pd.DataFrame([f.model_dump() for f in report.validation_flags]), hide_index=True)
        st.markdown("#### Étapes")
        st.dataframe(pd.DataFrame([t.model_dump() for t in report.timings]), hide_index=True)
        if report.model_usage:
            st.markdown("#### Utilisation des modèles")
            st.dataframe(pd.DataFrame([u.model_dump() for u in report.model_usage]), hide_index=True)
        st.markdown("#### Provenance")
        st.dataframe(pd.DataFrame([p.model_dump(mode="json") for p in report.provenance]), hide_index=True)
        st.download_button("Télécharger le rapport JSON", report.model_dump_json(indent=2),
                           file_name=f"{report.analysis_id}.json", mime="application/json")
