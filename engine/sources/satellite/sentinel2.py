"""Copernicus Sentinel-2 L2A connector (Element 84 Earth Search STAC + AWS COGs).

Pipeline per parcel:
1. STAC search (``POST {stac}/search``) for ``sentinel-2-l2a`` items intersecting the
   parcel geometry within the lookback window, scene cloud cover < threshold.
2. For each scene (newest first, up to ``max_scenes``) read only the parcel window from
   the COGs (HTTP range requests via rasterio/GDAL): B04 (red, 10 m), B08 (nir, 10 m),
   B8A (nir08, 20 m), B11 (swir16, 20 m), SCL (20 m).
3. Apply the per-band ``raster:bands`` scale/offset published in the item, mask pixels
   whose SCL class is not in ``scl_valid_classes`` (clouds, shadows, water, ...).
4. NDVI = (B08-B04)/(B08+B04) at 10 m; NDMI = (B8A-B11)/(B8A+B11) at 20 m (Gao 1996 NDWI).

Scientific limit (enforced in prompts and reports): these indices characterise the
canopy/parcel context. They cannot identify a cocoa disease at leaf scale and cannot
alone confirm water stress (shade trees, mixed pixels, phenology).
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import numpy as np

from engine.config import get_settings
from engine.schemas import EvidenceKind, Provenance, SatelliteContext, SatelliteScene, utcnow

log = logging.getLogger(__name__)

LIMITATIONS = [
    "Sentinel-2 (pixels de 10-20 m) caractérise le couvert de la parcelle ; il ne permet pas d'identifier une maladie du cacao à l'échelle d'une feuille ou d'une cabosse.",
    "Les indices (NDVI, NDMI) mélangent cacaoyers, arbres d'ombrage et sol ; une baisse peut avoir de nombreuses causes (récolte, élagage, saison, stress, nuages résiduels).",
    "Un indice hydrique bas ne prouve pas à lui seul un stress hydrique.",
    "Couverture nuageuse fréquente en zone tropicale : les dates utilisables peuvent être rares.",
]


def point_buffer_polygon(lat: float, lon: float, radius_m: float) -> dict[str, Any]:
    """Approximate a circular buffer (WGS84) as a 16-gon GeoJSON polygon."""
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    coords = [
        [lon + dlon * math.cos(2 * math.pi * k / 16), lat + dlat * math.sin(2 * math.pi * k / 16)] for k in range(16)
    ]
    coords.append(coords[0])
    return {"type": "Polygon", "coordinates": [coords]}


def _scale_offset(asset: dict[str, Any], item_properties: dict[str, Any] | None = None) -> tuple[float, float]:
    """Return (scale, offset) to convert DN to surface reflectance.

    Since processing baseline 04.00, ESA L2A DNs carry +1000 (BOA_ADD_OFFSET). Earth Search
    items expose ``raster:bands`` offset=-0.1, but when ``earthsearch:boa_offset_applied`` is
    True the DNs have ALREADY been shifted (verified empirically on tile 30NTN, 2026-10-25:
    applying -0.1 again yields negative red reflectances). In that case only the scale applies.
    """
    bands = asset.get("raster:bands") or [{}]
    scale = float(bands[0].get("scale", 1.0))
    offset = float(bands[0].get("offset", 0.0))
    if (item_properties or {}).get("earthsearch:boa_offset_applied") is True:
        offset = 0.0
    return scale, offset


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a - b) / (a + b)
    nd[~np.isfinite(nd)] = np.nan
    return nd


class Sentinel2Client:
    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        s = get_settings()
        self.cfg = s.data_sources["sentinel2"]
        self.sat_cfg = s.settings["satellite"]
        self._transport = transport

    def search(self, geometry: dict[str, Any], start: date, end: date) -> list[dict[str, Any]]:
        body = {
            "collections": [self.cfg["collection"]],
            "intersects": geometry,
            "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
            "limit": 50,
            "query": {"eo:cloud_cover": {"lt": self.sat_cfg["max_scene_cloud_cover"]}},
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        }
        with httpx.Client(timeout=self.cfg.get("timeout_s", 90), transport=self._transport) as client:
            resp = client.post(self.cfg["stac_url"].rstrip("/") + "/search", json=body)
            resp.raise_for_status()
            feats = resp.json().get("features", [])
        feats.sort(key=lambda f: f["properties"]["datetime"], reverse=True)
        return feats

    @staticmethod
    def _env() -> dict[str, str]:
        return {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "AWS_NO_SIGN_REQUEST": "YES",
                "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1"}

    def reference_grid(self, href: str, geometry: dict[str, Any]) -> dict[str, Any]:
        """Define the 10 m reading grid (bounds, shape, polygon mask) from a 10 m band."""
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.warp import transform_geom
        from rasterio.windows import from_bounds
        from shapely.geometry import shape

        with rasterio.Env(**self._env()), rasterio.open(href) as ds:
            geom = transform_geom("EPSG:4326", ds.crs, geometry)
            minx, miny, maxx, maxy = shape(geom).bounds
            res = ds.res[0]
            window = from_bounds(minx - res, miny - res, maxx + res, maxy + res, ds.transform)
            window = window.round_offsets().round_lengths()
            wt = ds.window_transform(window)
            h, w = int(window.height), int(window.width)
            bounds = (wt.c, wt.f + wt.e * h, wt.c + wt.a * w, wt.f)  # left, bottom, right, top
            outside = geometry_mask([geom], out_shape=(h, w), transform=wt, invert=False, all_touched=True)
        return {"bounds": bounds, "shape": (h, w), "outside": outside}

    def read_on_grid(self, href: str, grid: dict[str, Any]) -> np.ndarray:
        """Read ``href`` over the reference grid (nearest resampling for 20 m bands)."""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds

        with rasterio.Env(**self._env()), rasterio.open(href) as ds:
            window = from_bounds(*grid["bounds"], ds.transform)
            arr = ds.read(1, window=window, out_shape=grid["shape"], boundless=True, fill_value=0,
                          resampling=Resampling.nearest).astype("float32")
        arr[grid["outside"]] = np.nan
        return arr

    def scene_indices(self, item: dict[str, Any], geometry: dict[str, Any]) -> SatelliteScene:
        assets = item["assets"]
        b = self.cfg["bands"]
        valid_classes = list(self.cfg["scl_valid_classes"])
        grid = self.reference_grid(assets[b["red"]]["href"], geometry)

        def band(name: str) -> np.ndarray:
            a = assets[b[name]]
            scale, offset = _scale_offset(a, item["properties"])
            raw = self.read_on_grid(a["href"], grid)
            nodata = raw == 0
            out = raw * scale + offset
            out[nodata] = np.nan
            return out

        scl = self.read_on_grid(assets[b["scl"]]["href"], grid)
        inside = ~grid["outside"]
        scl_ok = np.isin(np.nan_to_num(scl, nan=0).astype(int), valid_classes) & inside
        red, nir = band("red"), band("nir")
        nir08, swir = band("nir08"), band("swir16")
        ndvi = normalized_difference(nir, red)
        ndvi[~scl_ok] = np.nan
        ndmi = normalized_difference(nir08, swir)
        ndmi[~scl_ok] = np.nan

        # Physical sanity guard: surface reflectance must not be markedly negative.
        refl = red[scl_ok]
        negative_share = float((refl < -0.01).mean()) if refl.size else 0.0
        n_inside = int(inside.sum())
        valid_fraction = float((~np.isnan(ndvi) & inside).sum() / n_inside) if n_inside else 0.0
        usable = valid_fraction >= self.sat_cfg["min_valid_pixel_fraction"] and negative_share < 0.05
        note = None
        if negative_share >= 0.05:
            note = "Réflectances négatives détectées (incohérence radiométrique) : scène écartée."
        elif not usable:
            note = "Trop peu de pixels valides (nuages/ombres/eau) sur la parcelle."
        return SatelliteScene(
            scene_id=item["id"],
            acquired=datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")),
            scene_cloud_cover=item["properties"].get("eo:cloud_cover"),
            valid_pixel_fraction=round(valid_fraction, 3),
            ndvi_median=_median(ndvi) if usable else None,
            ndmi_median=_median(ndmi) if usable else None,
            n_pixels=n_inside,
            usable=usable,
            note=note,
        )


def _median(a: np.ndarray) -> float | None:
    vals = a[~np.isnan(a)]
    return round(float(np.median(vals)), 4) if vals.size else None


def get_satellite_context(
    lat: float,
    lon: float,
    observation_date: date,
    polygon_geojson: dict[str, Any] | None = None,
    client: Sentinel2Client | None = None,
) -> SatelliteContext:
    s = get_settings()
    sat = s.settings["satellite"]
    client = client or Sentinel2Client()
    geometry = polygon_geojson or point_buffer_polygon(lat, lon, sat["buffer_m"])
    start = observation_date - timedelta(days=sat["lookback_days"])
    prov = Provenance(
        source="Copernicus Sentinel-2 L2A (ESA) via Element 84 Earth Search / AWS Open Data COGs",
        kind=EvidenceKind.REMOTE_SENSING_OBSERVATION,
        url=s.data_sources["sentinel2"]["docs"],
        retrieved_at=utcnow(),
        period_start=start,
        period_end=observation_date,
        latitude=lat,
        longitude=lon,
        spatial_resolution="10 m (B04, B08, NDVI) / 20 m (B8A, B11, SCL, NDMI)",
        notes="Contient des données Copernicus Sentinel modifiées. Géométrie : "
        + ("polygone fourni" if polygon_geojson else f"cercle de {sat['buffer_m']} m autour du point"),
    )
    try:
        items = client.search(geometry, start, observation_date)
    except Exception as exc:
        log.warning("STAC search failed: %s", exc)
        return SatelliteContext(status="unavailable", provenance=prov, limitations=LIMITATIONS, error=str(exc)[:500])
    if not items:
        return SatelliteContext(status="no_valid_scene", provenance=prov, limitations=LIMITATIONS,
                                error="Aucune scène Sentinel-2 sous le seuil de nébulosité sur la période.")
    scenes: list[SatelliteScene] = []
    errors: list[str] = []
    for item in items[: sat["max_scenes"]]:
        try:
            scenes.append(client.scene_indices(item, geometry))
        except Exception as exc:
            log.warning("Scene %s failed: %s", item.get("id"), exc)
            errors.append(f"{item.get('id')}: {str(exc)[:200]}")
    usable = sorted([sc for sc in scenes if sc.usable and sc.ndvi_median is not None], key=lambda x: x.acquired)
    ctx = SatelliteContext(
        status="ok" if usable and not errors else ("partial" if usable else "no_valid_scene"),
        scenes=sorted(scenes, key=lambda x: x.acquired),
        valid_scene_count=len(usable),
        provenance=prov,
        limitations=list(LIMITATIONS),
        error="; ".join(errors)[:1000] or None,
    )
    if usable:
        ctx.latest_ndvi = usable[-1].ndvi_median
        ctx.latest_ndmi = usable[-1].ndmi_median
        if len(usable) >= 2:
            ctx.ndvi_change = round(usable[-1].ndvi_median - usable[0].ndvi_median, 4)
            if usable[-1].ndmi_median is not None and usable[0].ndmi_median is not None:
                ctx.ndmi_change = round(usable[-1].ndmi_median - usable[0].ndmi_median, 4)
        age = (observation_date - usable[-1].acquired.date()).days
        if age > 30:
            ctx.limitations.append(f"Dernière scène exploitable datant de {age} jours avant l'observation.")
    return ctx
