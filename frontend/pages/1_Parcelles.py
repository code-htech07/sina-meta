import sys  # noqa: I001
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # frontend/ on path

from common import page_header, service  # noqa: I001

import pandas as pd
import streamlit as st

from engine.schemas import ParcelCreate

page_header("Parcelles")
svc = service()

with st.form("new_parcel", clear_on_submit=True):
    st.markdown("### Enregistrer une parcelle")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Nom de la parcelle *")
    lat = c2.number_input("Latitude (WGS84) *", value=5.785, format="%.5f", min_value=-90.0, max_value=90.0)
    lon = c3.number_input("Longitude (WGS84) *", value=-6.606, format="%.5f", min_value=-180.0, max_value=180.0)
    c4, c5, c6 = st.columns(3)
    crop = c4.selectbox("Culture", ["cocoa"], format_func=lambda x: "Cacao")
    year = c5.number_input("Année de plantation (0 = inconnue)", value=0, min_value=0, max_value=2100, step=1)
    area = c6.number_input("Surface (ha, 0 = inconnue)", value=0.0, min_value=0.0, step=0.1)
    variety = st.text_input("Variété / matériel végétal")
    notes = st.text_area("Informations agronomiques (ombrage, traitements, historique…)")
    if st.form_submit_button("Enregistrer", type="primary"):
        if not name.strip():
            st.error("Le nom est obligatoire.")
        else:
            p = svc.create_parcel(ParcelCreate(name=name.strip(), latitude=lat, longitude=lon, crop=crop,
                                               planting_year=int(year) or None, area_ha=area or None,
                                               variety=variety or None, notes=notes or None))
            st.success(f"Parcelle « {p.name} » enregistrée ({p.id}).")

parcels = svc.list_parcels()
st.markdown("### Parcelles enregistrées")
if not parcels:
    st.info("Aucune parcelle.")
else:
    df = pd.DataFrame([p.model_dump(exclude={"polygon_geojson"}) for p in parcels])
    st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]], zoom=6)
    st.dataframe(df, hide_index=True, width="stretch")
