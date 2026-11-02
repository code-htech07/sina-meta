"""Shared Streamlit helpers (path setup, cached service, source status)."""

from __future__ import annotations

import hmac
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from engine.logging_utils import setup_logging  # noqa: E402
from engine.services import MaanaTerraService, check_sources  # noqa: E402

setup_logging("WARNING")

PART_LABELS = {
    "pod": "Cabosse", "cherelle": "Chérelle (jeune fruit)", "leaf": "Feuille adulte", "young_leaf": "Jeune feuille (flush)",
    "stem": "Tige / rameau", "branch": "Branche", "trunk": "Tronc", "whole_tree": "Arbre entier", "flower": "Fleur",
    "root": "Racine", "other": "Autre", "unknown": "Non précisé",
}


def secret(name: str, default: str | None = None) -> str | None:
    """Env var first (Streamlit Cloud exposes root-level secrets as env vars), then st.secrets."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets[name])
    except Exception:
        return default


def resolve_path(p: str | None) -> Path | None:
    """Paths may be absolute (local runs) or relative to the repo root (bundled demo)."""
    if not p:
        return None
    path = Path(p)
    if path.exists():
        return path
    alt = ROOT / p
    return alt if alt.exists() else None


DEMO_BUNDLE = ROOT / "data" / "demo" / "demo_bundle.json"


def _load_demo(svc: MaanaTerraService) -> None:
    """Pre-load a REAL recorded analysis (see scripts/export_demo.py) when the database is empty."""
    if svc.db.counts()["parcels"] or not DEMO_BUNDLE.exists():
        return
    from engine.schemas import AnalysisReport, ImageMetadata, Observation, Parcel

    bundle = json.loads(DEMO_BUNDLE.read_text(encoding="utf-8"))
    parcel = Parcel.model_validate(bundle["parcel"])
    obs = Observation.model_validate(bundle["observation"])
    with svc.db.connect() as conn:
        conn.execute("INSERT INTO parcels VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (parcel.id, parcel.name, parcel.latitude, parcel.longitude, parcel.crop, None,
                      parcel.planting_year, parcel.variety, parcel.area_ha, parcel.notes, parcel.created_at.isoformat()))
        conn.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                     (obs.id, obs.parcel_id, obs.observation_date.isoformat(), obs.plant_part, obs.farmer_symptoms,
                      obs.farmer_notes, obs.created_at.isoformat()))
        for img in obs.images:
            conn.execute("INSERT INTO images VALUES (?,?,?)", (img.image_id, obs.id, ImageMetadata.model_validate(img).model_dump_json()))
    svc.db.save_analysis(AnalysisReport.model_validate(bundle["analysis"]))


@st.cache_resource
def service() -> MaanaTerraService:
    svc = MaanaTerraService()
    try:
        _load_demo(svc)
    except Exception as exc:  # demo is optional; never block the app
        st.warning(f"Démo préenregistrée non chargée : {exc}")
    return svc


def require_access() -> None:
    """Password gate, active when the APP_PASSWORD secret is set (mandatory for a public deployment)."""
    expected = secret("APP_PASSWORD")
    if not expected or st.session_state.get("auth_ok"):
        return
    pwd = st.text_input("Mot de passe d'accès", type="password")
    if pwd and hmac.compare_digest(pwd, expected):
        st.session_state["auth_ok"] = True
        st.rerun()
    if pwd:
        st.error("Mot de passe incorrect.")
    st.stop()


def analyses_remaining_today(svc: MaanaTerraService) -> int:
    """Daily cap on paid analyses (MAANA_MAX_ANALYSES_PER_DAY, default 20)."""
    cap = int(secret("MAANA_MAX_ANALYSES_PER_DAY", "20") or 20)
    today = datetime.now(timezone.utc).date()
    done = sum(1 for a in svc.list_analyses(limit=500) if a.created_at.date() == today and a.model_usage)
    return max(0, cap - done)


@st.cache_data(ttl=300, show_spinner="Vérification des sources…")
def sources_status() -> dict:
    return check_sources()


def page_header(title: str) -> None:
    st.set_page_config(page_title=f"Maana Terra — {title}", page_icon="🌱", layout="wide")
    st.title(f"🌱 Maana Terra · {title}")
    require_access()
