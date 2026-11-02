#!/usr/bin/env python
"""Evaluate pre-declared hybrid decision rules (classifier + Muse Spark) on the 80 ablation images.

Rules were fixed BEFORE looking at test results (docs/CLASSIFIER.md):
  H1 routing   : classifier decides if (in domain, organ trusted, outside abstention band); else Muse Spark.
  H2 consensus : answer only when classifier (p>=0.5) and Muse Spark agree; otherwise abstain (human check).
Uses recorded Muse predictions (no API cost) and the trained classifier artefact.

    python scripts/eval_hybrid.py --backbone dinov2-small
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ROOT  # noqa: E402
from engine.evaluation.metrics import classification_metrics, mcnemar_exact  # noqa: E402
from engine.vision.classifier import ARTEFACT, CocoaClassifier  # noqa: E402

DATA = ROOT / "data" / "raw" / "datasets" / "cocoaswolset_full"
ABL = ROOT / "experiments" / "reports" / "ablation_80_20261102" / "predictions.csv"
ABL_MAN = ROOT / "data" / "raw" / "datasets" / "cocoaswolset_n10_s42" / "manifest.json"
REV = {"Leaves-Young": "young_leaf", "Leaves-Adults": "leaf", "Trees": "whole_tree", "cocoa pod": "pod"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2-small")
    args = ap.parse_args()
    art = json.loads(ARTEFACT.with_name(f"cocoa_cssvd_classifier_{args.backbone}.json").read_text())
    clf = CocoaClassifier(art)
    z = np.load(DATA / f"embeddings_{args.backbone}.npz", allow_pickle=True)
    emb = dict(zip(z["sha"], z["emb"]))
    name_to = {Path(m["file"]).name: m for m in json.loads(ABL_MAN.read_text())}
    muse = {(r["exp"], r["image"]): r for r in csv.DictReader(ABL.open(encoding="utf-8"))}
    images = sorted({img for _, img in muse})
    out: dict = {"backbone": args.backbone, "n": len(images), "trusted_parts": art["trusted_parts"]}
    rows = []
    for exp in ("E0", "E4"):
        y, pc, h1, h2, mp = [], [], [], [], []
        for img in images:
            m = name_to[img]
            label = "diseased" if m["label"] == "Diseased" else "healthy"
            c = clf.predict(emb[m["sha256"]], REV.get(m["part"]))
            c_bin = "diseased" if c.probability_cssvd_like >= 0.5 else "healthy"
            mu = muse[(exp, img)]["prediction"]
            trusted = m["part"] in art["trusted_parts"]
            route = c.decision != "abstain" and trusted
            h1_pred = ("diseased" if c.decision == "cssvd_like" else "healthy") if route else mu
            h2_pred = c_bin if c_bin == mu else "abstain"
            y.append(label); pc.append(c_bin); h1.append(h1_pred); h2.append(h2_pred); mp.append(mu)
            if exp == "E4":
                rows.append({"image": img, "part": m["part"], "label": label, "p_cssvd": c.probability_cssvd_like,
                             "classifier": c.decision, "in_domain": c.in_domain, "muse_E4": mu, "H1": h1_pred, "H2": h2_pred})
        ok = lambda preds: [p == t for p, t in zip(preds, y)]  # noqa: E731
        out[exp] = {
            "muse_alone": classification_metrics(y, mp),
            "classifier_alone": classification_metrics(y, pc),
            "H1_routing": classification_metrics(y, h1),
            "H2_consensus": classification_metrics(y, h2),
            "mcnemar_H1_vs_muse": mcnemar_exact(ok(h1), ok(mp)),
            "mcnemar_H1_vs_classifier": mcnemar_exact(ok(h1), ok(pc)),
        }
    rep = ROOT / "experiments" / "reports" / "classifier"
    (rep / f"hybrid_{args.backbone}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with (rep / f"hybrid_predictions_{args.backbone}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for exp in ("E0", "E4"):
        for k in ("muse_alone", "classifier_alone", "H1_routing", "H2_consensus"):
            m = out[exp][k]
            print(f"{exp} {k:17s} acc={m.get('accuracy')} bacc={m.get('balanced_accuracy')} "
                  f"abst={m.get('abstention_rate')} crit={m.get('critical_error_rate')}")
        print(f"{exp} McNemar H1 vs Muse {out[exp]['mcnemar_H1_vs_muse']} | H1 vs classifier {out[exp]['mcnemar_H1_vs_classifier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
