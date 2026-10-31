#!/usr/bin/env python
"""Run the E0–E4 ablation study on a labelled manifest (requires MODEL_API_KEY).

Example (8 committed samples, proxy location = Yamoussoukro / INP-HB, fallback date):
    python scripts/run_ablation.py --manifest data/samples/cocoaswolset/manifest.json \
        --proxy-lat 6.8276 --proxy-lon -5.2767 --fallback-date 2025-05-27

Larger run: first download a stratified sample with scripts/run_dataset_audit.py --download N.
Estimated cost is printed BEFORE running; pass --yes to skip the confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import get_meta_api_key  # noqa: E402
from engine.evaluation.experiments import EXPERIMENTS, load_manifest, run_ablation  # noqa: E402
from engine.logging_utils import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--experiments", default="E0,E1,E2,E3,E4")
    ap.add_argument("--proxy-lat", type=float, required=True, help="PROXY latitude (dataset has no GPS)")
    ap.add_argument("--proxy-lon", type=float, required=True)
    ap.add_argument("--fallback-date", type=date.fromisoformat, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    setup_logging("WARNING")
    if not get_meta_api_key():
        print("MODEL_API_KEY absente : l'ablation exige des appels réels à Muse Spark. Aucun résultat produit.")
        return 2
    exps = [e.strip() for e in args.experiments.split(",") if e.strip() in EXPERIMENTS]
    n = len(load_manifest(args.manifest)[: args.limit or None])
    # Rough upper bound: ~6k input + ~3k output tokens per call at muse-spark-1.3 list price
    est = n * len(exps) * (6000 * 1.25 + 3000 * 4.25) / 1e6
    print(f"{n} images x {len(exps)} expériences = {n * len(exps)} appels Muse Spark (coût estimé ~{est:.2f} USD max).")
    if not args.yes and input("Continuer ? [o/N] ").strip().lower() not in ("o", "y", "oui", "yes"):
        return 1
    summary = run_ablation(args.manifest, exps, args.proxy_lat, args.proxy_lon, args.fallback_date, args.limit)
    print(json.dumps({k: v["classification"] for k, v in summary["experiments"].items()}, indent=1, ensure_ascii=False))
    print(f"Résultats : experiments/ablation/runs/{summary['run_id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
