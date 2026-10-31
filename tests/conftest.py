from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = ROOT / "data" / "samples" / "cocoaswolset"


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Unit tests never use a real key nor the real DB, unless explicitly marked."""
    if not os.environ.get("MAANA_LIVE_TESTS"):
        monkeypatch.delenv("MODEL_API_KEY", raising=False)
        monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("MAANA_DB_PATH", str(tmp_path / "test.db"))
    yield


@pytest.fixture
def muse_fixture() -> dict:
    data = json.loads((FIXTURES / "muse_output_handwritten.json").read_text(encoding="utf-8"))
    data.pop("_comment", None)
    return data


@pytest.fixture
def sample_pod() -> Path:
    return SAMPLES / "cocoa_pod_Diseased.jpg"


@pytest.fixture
def sample_leaf() -> Path:
    return SAMPLES / "Leaves-Young_Diseased.jpg"


def make_daily(n: int, rain: float | None, end: date, rh: float = 85.0, t: float = 26.0):
    from engine.schemas import DailyWeather

    return [DailyWeather(day=end - timedelta(days=n - 1 - i), precipitation_mm=rain, t2m_c=t, t2m_max_c=t + 5,
                         t2m_min_c=t - 5, rh2m_pct=rh) for i in range(n)]
