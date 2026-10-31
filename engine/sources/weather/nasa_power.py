"""NASA POWER connector (daily point data + long-term monthly climatology).

API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/ (no key needed).
* daily:        GET /api/temporal/daily/point?parameters=...&community=AG&latitude=..&longitude=..&start=YYYYMMDD&end=YYYYMMDD&format=JSON
* climatology:  GET /api/temporal/climatology/point?parameters=...&community=AG&latitude=..&longitude=..&format=JSON
Missing values are encoded with the fill value (-999). Recent days are often
missing because of processing latency: they are reported as missing, never filled.

NASA POWER values are gridded estimates (MERRA-2 / satellite-derived, ~0.5° x 0.625°
for meteorology), NOT on-parcel measurements.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta
from typing import Any

import httpx

from engine.config import get_settings
from engine.schemas import DailyWeather, EvidenceKind, Provenance, WeatherContext, utcnow
from engine.sources.weather.indicators import compute_indicators

log = logging.getLogger(__name__)

PARAM_MAP = {
    "PRECTOTCORR": "precipitation_mm",
    "T2M": "t2m_c",
    "T2M_MAX": "t2m_max_c",
    "T2M_MIN": "t2m_min_c",
    "RH2M": "rh2m_pct",
}
UNITS = {
    "precipitation_mm": "mm/day",
    "t2m_c": "°C",
    "t2m_max_c": "°C",
    "t2m_min_c": "°C",
    "rh2m_pct": "%",
}
MONTH_KEYS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

LIMITATIONS = [
    "Données NASA POWER : estimations maillées (résolution ~0,5° x 0,625° pour la météo), pas des mesures sur la parcelle.",
    "Les derniers jours peuvent être absents (latence de traitement) ; ils sont signalés comme manquants.",
    "La pluviométrie locale sous couvert agroforestier peut différer fortement de la valeur de maille.",
]


class NasaPowerClient:
    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = get_settings().data_sources["nasa_power"]
        self._transport = transport

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.cfg.get("timeout_s", 60), transport=self._transport) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    def fetch_daily(self, lat: float, lon: float, start: date, end: date) -> tuple[list[DailyWeather], dict]:
        params = {
            "parameters": ",".join(self.cfg["parameters"]),
            "community": self.cfg["community"],
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
        }
        data = self._get(self.cfg["daily_url"], params)
        fill = float((data.get("header") or {}).get("fill_value", self.cfg.get("fill_value", -999.0)))
        series = data["properties"]["parameter"]
        days: dict[str, dict[str, float | None]] = {}
        for param, field in PARAM_MAP.items():
            for key, value in (series.get(param) or {}).items():
                v = None if value is None or float(value) == fill else float(value)
                days.setdefault(key, {})[field] = v
        out = [DailyWeather(day=date(int(k[:4]), int(k[4:6]), int(k[6:8])), **vals) for k, vals in sorted(days.items())]
        return out, data.get("header") or {}

    def fetch_climatology(self, lat: float, lon: float) -> dict[str, dict[str, float]]:
        params = {
            "parameters": "PRECTOTCORR,T2M",
            "community": self.cfg["community"],
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "format": "JSON",
        }
        data = self._get(self.cfg["climatology_url"], params)
        return data["properties"]["parameter"]


def climatological_window(clim: dict[str, dict[str, float]], end: date, days: int = 30) -> tuple[float | None, float | None]:
    """Expected rainfall total (mm) and mean T2M over the ``days`` before ``end`` from monthly normals.

    POWER climatology PRECTOTCORR is a mean daily rate (mm/day) per month; the expected
    total is the sum over the window of each day's monthly mean rate.
    """
    rain = clim.get("PRECTOTCORR") or {}
    temp = clim.get("T2M") or {}
    total = 0.0
    temps: list[float] = []
    for i in range(days):
        d = end - timedelta(days=i)
        key = MONTH_KEYS[d.month - 1]
        if key not in rain:
            return None, None
        total += float(rain[key])
        if key in temp:
            temps.append(float(temp[key]))
    return round(total, 1), (round(sum(temps) / len(temps), 2) if temps else None)


def get_weather_context(lat: float, lon: float, observation_date: date, client: NasaPowerClient | None = None) -> WeatherContext:
    s = get_settings()
    lookback = s.settings["weather"]["lookback_days"]
    start = observation_date - timedelta(days=lookback - 1)
    client = client or NasaPowerClient()
    prov = Provenance(
        source="NASA POWER Daily API (community AG)",
        kind=EvidenceKind.MODELLED_ESTIMATE,
        url=s.data_sources["nasa_power"]["docs"],
        retrieved_at=utcnow(),
        period_start=start,
        period_end=observation_date,
        latitude=lat,
        longitude=lon,
        spatial_resolution="~0.5° x 0.625° (météorologie MERRA-2)",
    )
    try:
        daily, _header = client.fetch_daily(lat, lon, start, observation_date)
    except Exception as exc:
        log.warning("NASA POWER daily request failed: %s", exc)
        return WeatherContext(status="unavailable", provenance=prov, limitations=LIMITATIONS, error=str(exc)[:500])

    clim_rain = clim_t = None
    try:
        clim = client.fetch_climatology(lat, lon)
        anchor = max((d.day for d in daily if d.precipitation_mm is not None), default=observation_date)
        clim_rain, clim_t = climatological_window(clim, anchor, 30)
    except Exception as exc:
        log.warning("NASA POWER climatology request failed: %s", exc)
        prov.notes = f"Climatologie indisponible : {str(exc)[:200]}"

    indicators = compute_indicators(daily, clim_rain_30d=clim_rain, clim_t2m=clim_t)
    if indicators.valid_days == 0:
        status = "unavailable"
    elif indicators.missing_days > 0:
        status = "partial"
    else:
        status = "ok"
    limitations = list(LIMITATIONS)
    if indicators.missing_days:
        limitations.append(
            f"{indicators.missing_days} jour(s) sans donnée sur la période ; indicateurs calculés sur les jours valides "
            f"(dernier jour valide : {indicators.last_valid_day})."
        )
    return WeatherContext(
        status=status,
        indicators=indicators,
        daily=daily,
        units=UNITS,
        provenance=prov,
        limitations=limitations,
    )


def days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]
