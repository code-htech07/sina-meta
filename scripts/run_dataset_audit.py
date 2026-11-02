#!/usr/bin/env python
"""Audit CocoaSwolSet through the public Mendeley API; optionally download a stratified sample.

    python scripts/run_dataset_audit.py                    # audit only (no image download)
    python scripts/run_dataset_audit.py --download 10      # + 10 images per (plant part, label)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ROOT  # noqa: E402
from engine.evaluation.dataset_audit import build_index, download, fetch_metadata, run_audit, stratified_sample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=int, default=0, help="images per (part,label) stratum")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = run_audit()
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if args.download:
        meta, folders = fetch_metadata()
        sample = stratified_sample(build_index(meta, folders), args.download, seed=args.seed)
        target = ROOT / "data" / "raw" / "datasets" / f"cocoaswolset_n{args.download}_s{args.seed}"
        man = download(sample, target)
        print(f"{len(man)} images téléchargées, SHA-256 vérifiés : {sum(m['sha256_verified'] for m in man)} -> {target}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
