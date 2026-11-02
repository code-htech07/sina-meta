#!/usr/bin/env python
"""Phase 1 — verify REAL access to Muse Spark and SAM 3.1 with the configured key.

Usage:  python scripts/check_meta_access.py [--image path] [--skip-sam]
Writes: experiments/reports/meta_access_check.json   (never contains the key)

Checks, in order:
 1. key present (length only)            4. Muse Spark image + structured output
 2. GET /v1/models (ids listed?)          5. SAM 3.1 segmentation with a data-URL image
 3. Muse Spark minimal text request
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.acquisition.images import load_rgb  # noqa: E402
from engine.config import ROOT, describe_secret_status, get_settings  # noqa: E402
from engine.logging_utils import redact, setup_logging  # noqa: E402
from engine.meta_ai.client import MetaModelClient  # noqa: E402
from engine.meta_ai.muse_spark import MuseSparkAnalyzer  # noqa: E402
from engine.meta_ai.sam import SamSegmenter  # noqa: E402
from engine.preprocessing.quality import to_data_url  # noqa: E402

DEFAULT_IMAGE = ROOT / "data" / "samples" / "cocoaswolset" / "cocoa_pod_Diseased.jpg"


def step(results: dict, name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        info = fn() or {}
        results[name] = {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000), **info}
    except Exception as exc:
        results[name] = {"ok": False, "latency_ms": round((time.perf_counter() - t0) * 1000),
                         "error_type": type(exc).__name__, "error": redact(str(exc))[:800],
                         "http_status": getattr(exc, "status_code", None)}
    print(f"[{'OK ' if results[name]['ok'] else 'ERR'}] {name}: "
          + json.dumps({k: v for k, v in results[name].items() if k != 'ok'}, ensure_ascii=False)[:400])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--skip-sam", action="store_true")
    args = ap.parse_args()
    setup_logging("WARNING")
    s = get_settings()
    results: dict = {"run_at": datetime.now(timezone.utc).isoformat(), "base_url": s.models["meta_api"]["base_url"],
                     "muse_model": s.muse_model, "sam_model": s.sam_model, "credentials": describe_secret_status()}
    if results["credentials"]["auth_mode"] == "none":
        print("MODEL_API_KEY absente : définir la variable (ou le secret Kaggle) puis relancer.")
        results["blocked"] = "no_api_key"
    else:
        client = MetaModelClient(max_retries=0)

        def models():
            ids = client.list_models()
            return {"n_models": len(ids), "muse_listed": s.muse_model in ids, "sam_listed": s.sam_model in ids,
                    "matching_ids": [i for i in ids if "muse" in i or "sam" in i]}
        step(results, "list_models", models)

        def text():
            r = client.create_response({"model": s.muse_model, "input": "Reply with exactly: OK"})
            return {"output": r.text[:50], "usage": r.usage, "response_id": r.response_id}
        step(results, "muse_text_minimal", text)

        def image():
            r = MuseSparkAnalyzer(client=client).analyze([to_data_url(load_rgb(args.image))], {"experiment": "access_check"})
            o = r.output
            return {"is_cocoa": o.image_assessment.is_cocoa, "overall": o.overall_assessment,
                    "hypotheses": [(h.problem_id, h.status) for h in o.hypotheses], "notes": r.notes,
                    "usage": r.usage.model_dump()}
        step(results, "muse_image_structured", image)

        if not args.skip_sam:
            def sam():
                seg = SamSegmenter(client=client)
                seg.enabled = True
                r = seg.segment(load_rgb(args.image), "access_check", prompts=["cocoa pod"],
                                artifacts_dir=ROOT / "experiments" / "reports" / "sam_check")
                if r.status == "error":
                    raise RuntimeError(r.error)
                return {"status": r.status, "regions": len(r.regions), "diagnostics": r.diagnostics[:5],
                        "areas": [x.mask_area_fraction for x in r.regions]}
            step(results, "sam_segmentation", sam)
    out = ROOT / "experiments" / "reports" / "meta_access_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nRésultats enregistrés : {out.relative_to(ROOT)}")
    if results.get("blocked"):
        return 2
    return 0 if all(v.get("ok", True) for v in results.values() if isinstance(v, dict) and "ok" in v) else 1


if __name__ == "__main__":
    raise SystemExit(main())
