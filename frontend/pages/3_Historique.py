import sys  # noqa: I001
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # frontend/ on path

from common import page_header, service  # noqa: I001

import pandas as pd
import streamlit as st

from frontend.components.report_view import render_report

page_header("Historique et rapports")
svc = service()
parcels = svc.list_parcels()
if not parcels:
    st.info("Aucune parcelle.")
    st.stop()

parcel = st.selectbox("Parcelle", parcels, format_func=lambda p: p.name)
analyses = svc.list_analyses(parcel.id, limit=200)
observations = {o.id: o for o in svc.list_observations(parcel.id)}

if not analyses:
    st.info("Aucune analyse pour cette parcelle.")
    st.stop()

rows = [{"date obs.": observations[a.observation_id].observation_date if a.observation_id in observations else None,
         "analyse": a.analysis_id, "statut": a.status, "évaluation": a.overall_assessment, "prudence": a.caution_level,
         "hypothèse principale": a.primary_hypotheses[0].name if a.primary_hypotheses else "—",
         "NDVI récent": a.satellite_context.latest_ndvi if a.satellite_context else None,
         "pluie 30 j (mm)": a.weather_context.indicators.rain_30d_mm if a.weather_context and a.weather_context.indicators else None}
        for a in analyses]
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, width="stretch")
if df["NDVI récent"].notna().sum() >= 2:
    st.line_chart(df.dropna(subset=["NDVI récent"]).set_index("date obs.")["NDVI récent"], height=180)

default = st.session_state.get("last_analysis_id")
ids = [a.analysis_id for a in analyses]
chosen = st.selectbox("Afficher le rapport", ids, index=ids.index(default) if default in ids else 0)
rep = next(a for a in analyses if a.analysis_id == chosen)
obs = observations.get(rep.observation_id)
render_report(rep, {img.image_id: img.path for img in obs.images} if obs else {})
