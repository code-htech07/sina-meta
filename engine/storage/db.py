"""SQLite persistence (stdlib ``sqlite3``).

Kept deliberately thin: a schema, and repository functions that take/return the
Pydantic models from :mod:`engine.schemas`. Analyses are stored as validated
JSON documents, which makes a later migration to PostgreSQL (JSONB) direct.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from engine.config import get_settings
from engine.schemas import (
    AnalysisReport,
    ImageMetadata,
    Observation,
    ObservationCreate,
    Parcel,
    ParcelCreate,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS parcels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    crop TEXT NOT NULL,
    polygon_geojson TEXT,
    planting_year INTEGER,
    variety TEXT,
    area_ha REAL,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    parcel_id TEXT NOT NULL REFERENCES parcels(id) ON DELETE CASCADE,
    observation_date TEXT NOT NULL,
    plant_part TEXT NOT NULL,
    farmer_symptoms TEXT,
    farmer_notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    observation_id TEXT REFERENCES observations(id) ON DELETE CASCADE,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    parcel_id TEXT,
    observation_id TEXT,
    experiment TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    report_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_parcel ON observations(parcel_id);
CREATE INDEX IF NOT EXISTS idx_analyses_parcel ON analyses(parcel_id);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Database:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else get_settings().path("db_path")
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if str(self.path) == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            yield self._memory_conn
            self._memory_conn.commit()
            return
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------- parcels
    def create_parcel(self, data: ParcelCreate) -> Parcel:
        parcel = Parcel(id=new_id("parcel"), created_at=utcnow(), **data.model_dump())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO parcels VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    parcel.id, parcel.name, parcel.latitude, parcel.longitude, parcel.crop,
                    json.dumps(parcel.polygon_geojson) if parcel.polygon_geojson else None,
                    parcel.planting_year, parcel.variety, parcel.area_ha, parcel.notes,
                    parcel.created_at.isoformat(),
                ),
            )
        return parcel

    @staticmethod
    def _row_to_parcel(row: sqlite3.Row) -> Parcel:
        d = dict(row)
        d["polygon_geojson"] = json.loads(d["polygon_geojson"]) if d["polygon_geojson"] else None
        return Parcel(**d)

    def get_parcel(self, parcel_id: str) -> Parcel | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM parcels WHERE id=?", (parcel_id,)).fetchone()
        return self._row_to_parcel(row) if row else None

    def list_parcels(self) -> list[Parcel]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM parcels ORDER BY created_at DESC").fetchall()
        return [self._row_to_parcel(r) for r in rows]

    # ----------------------------------------------------------- observations
    def create_observation(self, data: ObservationCreate, images: list[ImageMetadata]) -> Observation:
        if not self.get_parcel(data.parcel_id):
            raise ValueError(f"Unknown parcel {data.parcel_id}")
        obs = Observation(id=new_id("obs"), created_at=utcnow(), images=images, **data.model_dump())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                (obs.id, obs.parcel_id, obs.observation_date.isoformat(), obs.plant_part,
                 obs.farmer_symptoms, obs.farmer_notes, obs.created_at.isoformat()),
            )
            for img in images:
                conn.execute("INSERT INTO images VALUES (?,?,?)", (img.image_id, obs.id, img.model_dump_json()))
        return obs

    def _images_for(self, conn: sqlite3.Connection, obs_id: str) -> list[ImageMetadata]:
        rows = conn.execute("SELECT metadata_json FROM images WHERE observation_id=?", (obs_id,)).fetchall()
        return [ImageMetadata.model_validate_json(r["metadata_json"]) for r in rows]

    def get_observation(self, obs_id: str) -> Observation | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM observations WHERE id=?", (obs_id,)).fetchone()
            if not row:
                return None
            return Observation(**dict(row), images=self._images_for(conn, obs_id))

    def list_observations(self, parcel_id: str | None = None) -> list[Observation]:
        with self.connect() as conn:
            if parcel_id:
                rows = conn.execute(
                    "SELECT * FROM observations WHERE parcel_id=? ORDER BY observation_date DESC", (parcel_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM observations ORDER BY observation_date DESC").fetchall()
            return [Observation(**dict(r), images=self._images_for(conn, r["id"])) for r in rows]

    # --------------------------------------------------------------- analyses
    def save_analysis(self, report: AnalysisReport) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?)",
                (report.analysis_id, report.parcel_id, report.observation_id, report.experiment.name,
                 report.status, report.created_at.isoformat(), report.model_dump_json()),
            )

    def get_analysis(self, analysis_id: str) -> AnalysisReport | None:
        with self.connect() as conn:
            row = conn.execute("SELECT report_json FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        return AnalysisReport.model_validate_json(row["report_json"]) if row else None

    def list_analyses(self, parcel_id: str | None = None, limit: int = 50) -> list[AnalysisReport]:
        with self.connect() as conn:
            if parcel_id:
                rows = conn.execute(
                    "SELECT report_json FROM analyses WHERE parcel_id=? ORDER BY created_at DESC LIMIT ?",
                    (parcel_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT report_json FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [AnalysisReport.model_validate_json(r["report_json"]) for r in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("parcels", "observations", "images", "analyses")
            }
