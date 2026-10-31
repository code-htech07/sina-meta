"""Configuration and secret loading.

Settings come from ``configs/*.yaml``; a small set of environment variables can
override them. The Meta API key is ONLY ever read from the environment (or a
Kaggle secret) and is never logged, returned, or persisted.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"

log = logging.getLogger(__name__)

try:  # optional: load a local .env (ignored by git)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    settings: dict[str, Any]
    models: dict[str, Any]
    data_sources: dict[str, Any]

    # ---- convenience accessors -------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a storage path from settings.yaml relative to the repo root."""
        override = {"db_path": os.environ.get("MAANA_DB_PATH")}.get(key)
        raw = override or self.settings["storage"][key]
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    @property
    def muse_model(self) -> str:
        return os.environ.get("MAANA_MUSE_MODEL") or self.models["muse_spark"]["model"]

    @property
    def sam_model(self) -> str:
        return os.environ.get("MAANA_SAM_MODEL") or self.models["sam"]["model"]

    @property
    def sam_enabled(self) -> bool:
        env = _env_bool("MAANA_SAM_ENABLED")
        return bool(self.models["sam"].get("enabled", True)) if env is None else env

    @property
    def report_language(self) -> str:
        return self.settings.get("app", {}).get("report_language", "fr")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        settings=_load_yaml("settings.yaml"),
        models=_load_yaml("models.yaml"),
        data_sources=_load_yaml("data_sources.yaml"),
    )


def _kaggle_secret(name: str) -> str | None:
    """Read a Kaggle notebook secret if running on Kaggle (Add-ons > Secrets)."""
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return None
    try:  # pragma: no cover - only available on Kaggle
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None


def get_meta_api_key() -> str | None:
    """Return the Meta API key from env / Kaggle secrets, or None. Never log it."""
    names = get_settings().models["meta_api"].get("api_key_env", ["MODEL_API_KEY"])
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for name in names:
        value = _kaggle_secret(name)
        if value:
            os.environ[name] = value  # cache for this process only
            return value.strip()
    return None


def describe_secret_status() -> dict[str, Any]:
    """Non-sensitive description of credential availability (safe to display)."""
    key = get_meta_api_key()
    return {
        "meta_api_key_present": bool(key),
        "meta_api_key_length": len(key) if key else 0,
        "sources_checked": get_settings().models["meta_api"].get("api_key_env", []) + ["kaggle_secrets"],
    }
