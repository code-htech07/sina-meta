from datetime import date

import httpx

from engine.sources.weather.indicators import compute_indicators, longest_run
from engine.sources.weather.nasa_power import NasaPowerClient, climatological_window, get_weather_context
from tests.conftest import make_daily


def test_longest_run():
    assert longest_run([True, True, False, True, True, True]) == 3
    assert longest_run([]) == 0


def test_indicators_wet_series():
    end = date(2026, 10, 30)
    ind = compute_indicators(make_daily(60, 10.0, end, rh=95), clim_rain_30d=150.0, clim_t2m=26.0)
    assert ind.rain_30d_mm == 300.0 and ind.rain_7d_mm == 70.0
    assert ind.rainy_days_30d == 30 and ind.longest_dry_spell_30d == 0
    assert ind.rain_30d_ratio_vs_normal == 2.0
    assert ind.days_rh_ge_90_30d == 30
    assert ind.t2m_anomaly_30d_c == 0.0


def test_missing_days_are_not_counted_as_dry():
    end = date(2026, 10, 30)
    daily = make_daily(60, 5.0, end)
    for d in daily[-5:]:
        d.precipitation_mm = None  # NASA POWER latency
    ind = compute_indicators(daily)
    assert ind.missing_days == 5
    assert ind.last_valid_day == date(2026, 10, 25)
    assert ind.current_dry_spell_days == 0  # windows anchored on last valid day
    assert ind.rain_30d_ratio_vs_normal is None  # no climatology => no invented normal


def test_all_missing():
    ind = compute_indicators(make_daily(10, None, date(2026, 1, 10)))
    assert ind.valid_days == 0 and ind.rain_30d_mm is None


def test_climatological_window():
    clim = {"PRECTOTCORR": {k: 2.0 for k in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]},
            "T2M": {k: 25.0 for k in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]}}
    rain, t = climatological_window(clim, date(2026, 3, 15), 30)
    assert rain == 60.0 and t == 25.0


def _power_transport(fail_climatology=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if "climatology" in request.url.path:
            if fail_climatology:
                return httpx.Response(503)
            months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            return httpx.Response(200, json={"properties": {"parameter": {
                "PRECTOTCORR": {m: 4.0 for m in months} | {"ANN": 4.0}, "T2M": {m: 26.0 for m in months}}}})
        start, end = request.url.params["start"], request.url.params["end"]
        days = [f"202610{d:02d}" for d in range(int(start[-2:]), int(end[-2:]) + 1)] if start[:6] == "202610" else []
        series = {p: {d: (-999.0 if d >= "20261029" else 3.0) for d in days} for p in
                  ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M"]}
        return httpx.Response(200, json={"header": {"fill_value": -999.0}, "properties": {"parameter": series}})
    return httpx.MockTransport(handler)


def test_weather_context_contract_parsing(monkeypatch):
    monkeypatch.setitem(__import__("engine.config", fromlist=["x"]).get_settings().settings["weather"], "lookback_days", 30)
    ctx = get_weather_context(6.0, -5.0, date(2026, 10, 30), client=NasaPowerClient(transport=_power_transport()))
    assert ctx.status == "partial"  # 29-30 Oct are fill values
    assert ctx.indicators.missing_days == 2
    assert ctx.indicators.climatology_rain_30d_mm == 120.0
    assert ctx.provenance.kind.value == "modelled_estimate"
    assert any("maillées" in x for x in ctx.limitations)


def test_weather_context_unavailable_on_error():
    def boom(request):
        return httpx.Response(500)
    ctx = get_weather_context(6.0, -5.0, date(2026, 10, 30), client=NasaPowerClient(transport=httpx.MockTransport(boom)))
    assert ctx.status == "unavailable" and ctx.indicators is None and ctx.error
