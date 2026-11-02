"""Maana Terra — tableau de bord. Run: streamlit run frontend/app.py"""

from common import page_header, service, sources_status  # noqa: I001

import pandas as pd
import streamlit as st

page_header("Tableau de bord")
st.caption("Intelligence agricole multimodale pour le cacao — Muse Spark · SAM 3.1 · NASA POWER · Sentinel-2 · base agronomique sourcée")

svc = service()
dash = svc.dashboard()
c = dash["counts"]
cols = st.columns(4)
cols[0].metric("Parcelles", c["parcels"])
cols[1].metric("Observations", c["observations"])
cols[2].metric("Photos", c["images"])
cols[3].metric("Analyses", c["analyses"])

st.subheader("État des sources et services (vérification en direct)")
status = sources_status()
labels = {"meta_api": "Meta Model API (Muse Spark / SAM 3.1)", "nasa_power": "NASA POWER (météo)",
          "sentinel2_stac": "Sentinel-2 L2A (Earth Search STAC)", "knowledge_base": "Base de connaissances cacao"}
rows = []
for key, label in labels.items():
    v = status.get(key, {})
    detail = v.get("error") or ", ".join(f"{k}={val}" for k, val in v.items() if k not in ("ok", "ids"))
    rows.append({"Service": label, "État": "✅ disponible" if v.get("ok") else "❌ indisponible", "Détail": detail})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if st.button("Revérifier"):
    sources_status.clear()
    st.rerun()

st.subheader("Dernières analyses")
if dash["recent"]:
    st.dataframe(pd.DataFrame(dash["recent"]), hide_index=True, width="stretch")
else:
    st.info("Aucune analyse enregistrée. Créez une parcelle puis une observation.")
