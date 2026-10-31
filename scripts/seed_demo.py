#!/usr/bin/env python
"""Create the demo parcel + observation used in docs/DEMO_SCENARIO.md (optionally analyse it).

    python scripts/seed_demo.py            # parcel + observation only
    python scripts/seed_demo.py --analyze  # + full analysis (Muse Spark needs MODEL_API_KEY)

Demo coordinates (5.70, -6.70) lie in the Soubré cocoa belt (south-west Côte d'Ivoire). They were
chosen because Sentinel-2 returned cloud-free vegetated pixels there (NDVI ~0.7 in Oct. 2026);
the actual land cover was NOT verified on the ground. The photo is a CocoaSwolSet image
(CC BY 4.0) that was NOT taken at these coordinates: the pairing is for demonstration only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ROOT  # noqa: E402
from engine.logging_utils import setup_logging  # noqa: E402
from engine.schemas import ObservationCreate, ParcelCreate  # noqa: E402
from engine.services import MaanaTerraService  # noqa: E402

DEMO_PARCEL = ParcelCreate(
    name="Démo — Soubré (coordonnées de démonstration)", latitude=5.70, longitude=-6.70, planting_year=2012,
    notes="Parcelle fictive de démonstration ; photo CocoaSwolSet (CC BY 4.0) non prise à cet endroit.",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--image", type=Path, default=ROOT / "data/samples/cocoaswolset/cocoa_pod_Diseased.jpg")
    ap.add_argument("--date", type=date.fromisoformat, default=date(2026, 10, 28))
    args = ap.parse_args()
    setup_logging("WARNING")
    svc = MaanaTerraService()
    parcel = next((p for p in svc.list_parcels() if p.name == DEMO_PARCEL.name), None) or svc.create_parcel(DEMO_PARCEL)
    obs = svc.create_observation(ObservationCreate(parcel_id=parcel.id, observation_date=args.date, plant_part="pod",
                                                   farmer_symptoms="Taches brunes sur plusieurs cabosses, certaines avec des trous"),
                                 [args.image.read_bytes()])
    print(f"Parcelle {parcel.id} · observation {obs.id}")
    if args.analyze:
        rep = svc.analyze_observation(obs.id)
        print(f"Analyse {rep.analysis_id}: {rep.status} — prudence {rep.caution_level}"
              + (f" — {rep.failure_reason}" if rep.failure_reason else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
