"""Shared Streamlit helpers (path setup, cached service, source status)."""

from __future__ import annotations

import sys
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


@st.cache_resource
def service() -> MaanaTerraService:
    return MaanaTerraService()


@st.cache_data(ttl=300, show_spinner="Vérification des sources…")
def sources_status() -> dict:
    return check_sources()


def page_header(title: str) -> None:
    st.set_page_config(page_title=f"Maana Terra — {title}", page_icon="🌱", layout="wide")
    st.title(f"🌱 Maana Terra · {title}")
