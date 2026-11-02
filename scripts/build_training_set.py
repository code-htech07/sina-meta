#!/usr/bin/env python
"""Download ALL analysable CocoaSwolSet images (JPEG/PNG, unique SHA-256), verify their hash, store a
downscaled copy (max side 518 px, EXIF orientation applied) and write a manifest.

    python scripts/build_training_set.py [--workers 8]

Output: data/raw/datasets/cocoaswolset_full/{images/*.jpg, manifest.json}  (git-ignored)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ROOT  # noqa: E402
from engine.evaluation.dataset_audit import ANALYSABLE_EXT, build_index, fetch_metadata  # noqa: E402

OUT = ROOT / "data" / "raw" / "datasets" / "cocoaswolset_full"
MAX_SIDE = 518


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    meta, folders = fetch_metadata()
    files = list({f.sha256: f for f in build_index(meta, folders) if f.ext in ANALYSABLE_EXT and f.label}.values())
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=180, follow_redirects=True)

    def one(f):
        dest = OUT / "images" / f"{f.sha256[:16]}.jpg"
        if not dest.exists():
            data = client.get(f.download_url).raise_for_status().content
            if hashlib.sha256(data).hexdigest() != f.sha256:
                return None
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            img.save(dest, quality=92)
        return {"file": str(dest.relative_to(ROOT)), "sha256": f.sha256, "label": f.label, "part": f.part,
                "source_path": f.path, "filename": f.filename, "licence": "CC BY 4.0"}

    rows, failed = [], 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(lambda f: _safe(one, f), files), 1):
            if r:
                rows.append(r)
            else:
                failed += 1
            if i % 100 == 0:
                print(f"{i}/{len(files)} (échecs : {failed})", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK : {len(rows)} images, échecs : {failed}")
    return 0


def _safe(fn, f):
    try:
        return fn(f)
    except Exception as exc:
        print(f"échec {f.filename}: {exc}", flush=True)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
