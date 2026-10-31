"""Weather indicators computed from daily series (pure functions, unit-tested).

Windows are anchored on the LAST VALID DAY (not on the observation date), so that
missing recent days are not silently counted as dry days.
"""

from __future__ import annotations

from datetime import timedelta

from engine.schemas import DailyWeather, WeatherIndicators

RAIN_DAY_MM = 1.0


def _window(daily: list[DailyWeather], end, days: int) -> list[DailyWeather]:
    start = end - timedelta(days=days - 1)
    return [d for d in daily if start <= d.day <= end]


def _sum(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals), 1) if vals else None


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def compute_indicators(
    daily: list[DailyWeather], clim_rain_30d: float | None = None, clim_t2m: float | None = None
) -> WeatherIndicators:
    valid = [d for d in daily if d.precipitation_mm is not None]
    missing = len(daily) - len(valid)
    if not valid:
        return WeatherIndicators(valid_days=0, missing_days=len(daily))
    last = max(d.day for d in valid)
    w7, w14, w30, w60 = (_window(valid, last, n) for n in (7, 14, 30, 60))
    rain30 = _sum([d.precipitation_mm for d in w30])
    dry_flags = [(d.precipitation_mm or 0) < RAIN_DAY_MM for d in sorted(w30, key=lambda x: x.day)]
    current_dry = 0
    for f in reversed(dry_flags):
        if not f:
            break
        current_dry += 1
    mean_t = _mean([d.t2m_c for d in w30])
    rh_vals = [d.rh2m_pct for d in w30 if d.rh2m_pct is not None]
    ratio = None
    if clim_rain_30d and rain30 is not None and clim_rain_30d > 0:
        ratio = round(rain30 / clim_rain_30d, 2)
    return WeatherIndicators(
        rain_7d_mm=_sum([d.precipitation_mm for d in w7]),
        rain_14d_mm=_sum([d.precipitation_mm for d in w14]),
        rain_30d_mm=rain30,
        rain_60d_mm=_sum([d.precipitation_mm for d in w60]),
        rainy_days_30d=sum(1 for d in w30 if (d.precipitation_mm or 0) >= RAIN_DAY_MM),
        longest_dry_spell_30d=longest_run(dry_flags),
        current_dry_spell_days=current_dry,
        mean_t2m_30d_c=mean_t,
        max_t2m_30d_c=max((d.t2m_max_c for d in w30 if d.t2m_max_c is not None), default=None),
        min_t2m_30d_c=min((d.t2m_min_c for d in w30 if d.t2m_min_c is not None), default=None),
        mean_rh2m_30d_pct=_mean(rh_vals),
        days_rh_ge_90_30d=sum(1 for v in rh_vals if v >= 90) if rh_vals else None,
        climatology_rain_30d_mm=clim_rain_30d,
        rain_30d_ratio_vs_normal=ratio,
        climatology_t2m_c=clim_t2m,
        t2m_anomaly_30d_c=round(mean_t - clim_t2m, 2) if (mean_t is not None and clim_t2m is not None) else None,
        valid_days=len(valid),
        missing_days=missing,
        last_valid_day=last,
    )
