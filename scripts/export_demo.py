#!/usr/bin/env python
"""Export the latest REAL analysis of the demo parcel into data/demo/ (portable, repo-relative paths).

The deployed app pre-loads this bundle when its database is empty, so visitors see a genuine
recorded report without spending API credit. Nothing is generated here: it only copies a run.

    MAANA_DB_PATH=... python scripts/export_demo.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ROOT  # noqa: E402
from engine.services import MaanaTerraService  # noqa: E402

OUT = ROOT / "data" / "demo"


def rel(src: str | None, name: str) -> str | None:
    if not src or not Path(src).exists():
        return None
    dest = OUT / name
    shutil.copyfile(src, dest)
    return str(dest.relative_to(ROOT))


def main() -> int:
    svc = MaanaTerraService()
    reports = [r for r in svc.list_analyses(limit=50) if r.status == "completed"]
    if not reports:
        print("Aucune analyse réelle terminée à exporter.")
        return 1
    rep = reports[0]
    obs = svc.db.get_observation(rep.observation_id)
    parcel = svc.get_parcel(rep.parcel_id)
    OUT.mkdir(parents=True, exist_ok=True)
    for img in obs.images:
        img.path = rel(img.path, f"{img.image_id}{Path(img.path).suffix}") or img.path
    if rep.segmentation:
        overlay = None
        for i, reg in enumerate(rep.segmentation.regions):
            if reg.overlay_path and overlay is None:
                overlay = rel(reg.overlay_path, "sam_overlay.png")
            reg.overlay_path = overlay
            reg.crop_path = rel(reg.crop_path, f"roi{i}.jpg")
    bundle = {
        "_note": "Real analysis recorded with the live Meta Model API; exported by scripts/export_demo.py.",
        "parcel": parcel.model_dump(mode="json"),
        "observation": obs.model_dump(mode="json"),
        "analysis": rep.model_dump(mode="json"),
    }
    (OUT / "demo_bundle.json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Exporté : {rep.analysis_id} ({rep.created_at:%Y-%m-%d %H:%M}) -> data/demo/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
