"""Live tests against REAL public services (no key needed). Run: pytest -m network"""

from datetime import date, timedelta

import pytest

from engine.sources.satellite.sentinel2 import get_satellite_context
from engine.sources.weather.nasa_power import get_weather_context

pytestmark = pytest.mark.network

SOUBRE = (5.785, -6.606)  # cocoa belt, south-west Côte d'Ivoire


def test_nasa_power_live():
    ctx = get_weather_context(*SOUBRE, date.today() - timedelta(days=10))
    assert ctx.status in ("ok", "partial"), ctx.error
    assert ctx.indicators.valid_days > 30
    assert ctx.indicators.climatology_rain_30d_mm is not None


def test_sentinel2_live():
    ctx = get_satellite_context(*SOUBRE, date.today() - timedelta(days=5))
    assert ctx.status in ("ok", "partial", "no_valid_scene"), ctx.error
    assert ctx.scenes, "STAC search returned no scene at all"
    for sc in ctx.scenes:
        if sc.ndvi_median is not None:
            assert -1 <= sc.ndvi_median <= 1
