import sys  # noqa: I001
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # frontend/ on path

from common import PART_LABELS, page_header, service  # noqa: I001

from datetime import date
import streamlit as st

from frontend.components.report_view import render_report
from engine.acquisition.images import ImageRejected
from engine.config import ROOT
from engine.schemas import ExperimentConfig, ObservationCreate

page_header("Nouvelle observation")
svc = service()
parcels = svc.list_parcels()
if not parcels:
    st.warning("Créez d'abord une parcelle (page Parcelles).")
    st.stop()

samples = sorted((ROOT / "data" / "samples" / "cocoaswolset").glob("*.jpg"))

with st.form("obs"):
    parcel = st.selectbox("Parcelle", parcels, format_func=lambda p: f"{p.name} ({p.latitude:.4f}, {p.longitude:.4f})")
    c1, c2 = st.columns(2)
    part = c1.selectbox("Partie de la plante photographiée", list(PART_LABELS), format_func=PART_LABELS.get, index=0)
    obs_date = c2.date_input("Date de l'observation", value=date.today(), max_value=date.today())
    uploads = st.file_uploader("Photographie(s) (JPEG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    sample = st.selectbox("…ou image de démonstration (CocoaSwolSet, CC BY 4.0)", [None] + samples,
                          format_func=lambda p: "—" if p is None else p.name)
    symptoms = st.text_area("Symptômes observés par l'agriculteur")
    notes = st.text_area("Notes complémentaires (traitements récents, nombre d'arbres touchés…)")
    with st.expander("Options d'analyse (sources utilisées)"):
        e = st.columns(5)
        use_w = e[0].checkbox("Météo", True)
        use_s = e[1].checkbox("Satellite", True)
        use_k = e[2].checkbox("Base agronomique", True)
        use_seg = e[3].checkbox("SAM 3.1", True)
        use_f = e[4].checkbox("Déclarations", True)
    submitted = st.form_submit_button("Enregistrer et analyser", type="primary")

if submitted:
    files = [u.getvalue() for u in uploads or []]
    if sample is not None:
        files.append(Path(sample).read_bytes())
    if not files:
        st.error("Ajoutez au moins une photographie.")
        st.stop()
    try:
        obs = svc.create_observation(ObservationCreate(parcel_id=parcel.id, observation_date=obs_date, plant_part=part,
                                                       farmer_symptoms=symptoms or None, farmer_notes=notes or None), files)
    except ImageRejected as exc:
        st.error(str(exc))
        st.stop()
    exp = ExperimentConfig(name="ui", use_weather=use_w, use_satellite=use_s, use_knowledge_base=use_k,
                           use_segmentation=use_seg, use_farmer_input=use_f)
    with st.status("Analyse en cours…", expanded=True) as status:
        st.write("Contrôle qualité · segmentation SAM · météo NASA POWER · Sentinel-2 · fusion · Muse Spark · validation")
        report = svc.analyze_observation(obs.id, exp)
        status.update(label=f"Analyse terminée ({report.status})", state="complete" if report.status != "failed" else "error")
    st.session_state["last_analysis_id"] = report.analysis_id
    render_report(report, {img.image_id: img.path for img in obs.images})
