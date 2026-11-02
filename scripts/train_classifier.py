#!/usr/bin/env python
"""Train and evaluate the specialist classifier (DINOv2 ONNX embeddings + logistic regression).

    python scripts/train_classifier.py --backbone dinov2-small

Protocol (anti-leakage, documented in docs/CLASSIFIER.md):
1. Near-duplicate groups: perceptual hash (pHash, Hamming <= 8) + union-find. Groups never
   straddle train/test.
2. The 80 images of the Muse Spark ablation (ablation_80_20261102) are forced into the TEST set,
   so the classifier and Muse Spark can be compared on exactly the same images.
3. Remaining groups: ~20% more test groups (stratified by part x label), rest = train.
4. On train only: 5-fold StratifiedGroupKFold to choose C and whether to add part features;
   out-of-fold scores -> Platt calibration, abstention band, OOD threshold.
5. Final model refit on all train; evaluated ONCE on test (incl. bootstrap 95% CIs).
Requires: scikit-learn, imagehash (training only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ROOT  # noqa: E402
from engine.vision.classifier import ARTEFACT, PARTS, part_onehot  # noqa: E402
from engine.vision.embedder import MODELS, Embedder  # noqa: E402

DATA = ROOT / "data" / "raw" / "datasets" / "cocoaswolset_full"
ABLATION_PRED = ROOT / "experiments" / "reports" / "ablation_80_20261102" / "predictions.csv"
ABLATION_MANIFEST = ROOT / "data" / "raw" / "datasets" / "cocoaswolset_n10_s42" / "manifest.json"
OUT_REPORT = ROOT / "experiments" / "reports" / "classifier"


def load_or_embed(rows: list[dict], backbone: str) -> np.ndarray:
    cache = DATA / f"embeddings_{backbone}.npz"
    shas = [r["sha256"] for r in rows]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["sha"]) == shas:
            return z["emb"]
    emb = Embedder(backbone, threads=4)
    out, t0 = [], time.time()
    for i in range(0, len(rows), 16):
        batch = [Image.open(ROOT / r["file"]) for r in rows[i:i + 16]]
        out.append(emb.embed_batch(batch))
        if i % 160 == 0:
            print(f"  embeddings {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    E = np.concatenate(out)
    np.savez(cache, emb=E, sha=np.array(shas))
    return E


def near_duplicate_groups(rows: list[dict], max_dist: int = 8) -> np.ndarray:
    import imagehash

    hashes = np.array([np.array(imagehash.phash(Image.open(ROOT / r["file"])).hash, dtype=bool).ravel() for r in rows])
    n = len(rows)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0] + i + 1:
            parent[find(i)] = find(int(j))
    roots = [find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([uniq[r] for r in roots])


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return round(float(total), 4)


def metrics(y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> dict:
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

    pred = (p >= 0.5).astype(int)
    answered = (p <= lo) | (p >= hi)
    sel = {}
    if answered.any():
        ya, pa = y[answered], (p[answered] >= hi).astype(int)
        sel = {"coverage": round(float(answered.mean()), 4), "selective_accuracy": round(float((ya == pa).mean()), 4),
               "selective_balanced_accuracy": round(float(balanced_accuracy_score(ya, pa)), 4) if len(set(ya)) > 1 else None,
               "selective_false_healthy_rate": round(float(((pa == 0) & (ya == 1)).sum() / max(1, (ya == 1).sum())), 4)}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            boots.append((float((pred[idx] == y[idx]).mean()), float(balanced_accuracy_score(y[idx], pred[idx]))))
    b = np.array(boots)
    return {
        "n": int(len(y)), "positives_cssvd": int(y.sum()),
        "accuracy": round(float((pred == y).mean()), 4),
        "accuracy_ci95": [round(float(np.percentile(b[:, 0], 2.5)), 4), round(float(np.percentile(b[:, 0], 97.5)), 4)],
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "balanced_accuracy_ci95": [round(float(np.percentile(b[:, 1], 2.5)), 4), round(float(np.percentile(b[:, 1], 97.5)), 4)],
        "f1_cssvd": round(float(f1_score(y, pred)), 4),
        "recall_cssvd": round(float(pred[y == 1].mean()), 4), "recall_healthy": round(float(1 - pred[y == 0].mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 4), "ece": ece(p, y),
        "confusion": {"healthy->healthy": int(((pred == 0) & (y == 0)).sum()), "healthy->cssvd": int(((pred == 1) & (y == 0)).sum()),
                      "cssvd->healthy": int(((pred == 0) & (y == 1)).sum()), "cssvd->cssvd": int(((pred == 1) & (y == 1)).sum())},
        "with_abstention": sel,
    }


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2-small", choices=list(MODELS))
    ap.add_argument("--target-selective-accuracy", type=float, default=0.95)
    ap.add_argument("--allow-part-features", action="store_true",
                    help="allow the declared organ as an input feature (off by default: a wrong declaration "
                         "flipped a live prediction from 0.80 to 0.01 on 2026-11-02)")
    ap.add_argument("--min-part-reliability", type=float, default=0.75,
                    help="organs whose out-of-fold balanced accuracy is below this are deferred to Muse Spark")
    args = ap.parse_args()

    rows = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    y = np.array([1 if r["label"] == "Diseased" else 0 for r in rows])
    parts = [r["part"] for r in rows]
    strata = np.array([f"{p}|{lab}" for p, lab in zip(parts, y)])
    print(f"{len(rows)} images, {y.sum()} CSSVD / {len(y) - y.sum()} sains")

    groups = near_duplicate_groups(rows)
    print(f"{groups.max() + 1} groupes de quasi-doublons (pHash <= 8) pour {len(rows)} images")
    E = load_or_embed(rows, args.backbone)

    # ---- split: ablation images (whole groups) forced to test, + ~20% more groups
    abl_shas = {m["sha256"] for m in json.loads(ABLATION_MANIFEST.read_text())} if ABLATION_MANIFEST.exists() else set()
    sha_idx = {r["sha256"]: i for i, r in enumerate(rows)}
    forced_groups = {groups[sha_idx[s]] for s in abl_shas if s in sha_idx}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, extra_test = next(sgkf.split(E, strata, groups))
    test_groups = forced_groups | set(groups[extra_test])
    test = np.array([g in test_groups for g in groups])
    train = ~test
    print(f"train {train.sum()} / test {test.sum()} (dont {len([s for s in abl_shas if s in sha_idx])} images de l'ablation Muse)")

    def feats(idx: np.ndarray, use_part: bool) -> np.ndarray:
        X = E[idx]
        return np.hstack([X, np.stack([part_onehot(parts[i]) for i in np.nonzero(idx)[0]])]) if use_part else X

    # ---- model selection on train only (grouped CV)
    tr_idx = np.nonzero(train)[0]
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(tr_idx, strata[tr_idx], groups[tr_idx]))
    best = None
    for use_part in ((False, True) if args.allow_part_features else (False,)):
        Xtr = feats(train, use_part)
        for C in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            oof = np.zeros(len(tr_idx))
            for a, b in cv:
                m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr[a], y[tr_idx][a])
                oof[b] = m.decision_function(Xtr[b])
            bacc = float((((oof > 0) == y[tr_idx]) * np.where(y[tr_idx] == 1, 0.5 / y[tr_idx].mean(), 0.5 / (1 - y[tr_idx].mean()))).sum() / len(oof))
            print(f"  CV use_part={use_part} C={C}: balanced acc {bacc:.4f}")
            if best is None or bacc > best[0]:
                best = (bacc, use_part, C, oof)
    cv_bacc, use_part, C, oof = best
    print(f"retenu : use_part={use_part}, C={C}, balanced acc CV={cv_bacc:.4f}")

    # ---- calibration, abstention band, OOD threshold (all from train/OOF)
    platt = LogisticRegression(C=1e6, max_iter=5000).fit(oof.reshape(-1, 1), y[tr_idx])
    pa, pb = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    p_oof = 1 / (1 + np.exp(-(pa * oof + pb)))
    lo, hi = 0.5, 0.5
    for w in np.arange(0.0, 0.45, 0.01):
        l, h = 0.5 - w, 0.5 + w
        ans = (p_oof <= l) | (p_oof >= h)
        if ans.mean() < 0.5:
            break
        if ((p_oof[ans] >= h) == y[tr_idx][ans]).mean() >= args.target_selective_accuracy:
            lo, hi = round(float(l), 3), round(float(h), 3)
            break
    else:
        lo, hi = 0.35, 0.65
    from sklearn.metrics import balanced_accuracy_score

    part_reliability = {}
    for part in PARTS:
        mk = np.array([parts[i] == part for i in tr_idx])
        if mk.sum() and len(set(y[tr_idx][mk])) > 1:
            part_reliability[part] = {"n_train": int(mk.sum()),
                                      "oof_balanced_accuracy": round(float(balanced_accuracy_score(y[tr_idx][mk], p_oof[mk] >= 0.5)), 4)}
    trusted_parts = [p for p, v in part_reliability.items() if v["oof_balanced_accuracy"] >= args.min_part_reliability]
    print("fiabilité par organe (OOF, entraînement) :", part_reliability, "-> organes de confiance :", trusted_parts)
    final = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(feats(train, use_part), y[train])
    Etr = E[train]
    # OOD: similarity of held-out-fold images to their training folds -> 1st percentile
    sims = []
    for a, b in cv:
        sims.append((E[tr_idx][b] @ E[tr_idx][a].T).max(axis=1))
    min_sim = float(np.percentile(np.concatenate(sims), 1))

    # ---- single evaluation on test
    s_test = final.decision_function(feats(test, use_part))
    p_test = 1 / (1 + np.exp(-(pa * s_test + pb)))
    m_test = metrics(y[test], p_test, lo, hi)
    per_part = {}
    for part in PARTS:
        mask = np.array([p == part for p in parts])[test]
        if mask.sum() and len(set(y[test][mask])) > 1:
            per_part[part] = {k: v for k, v in metrics(y[test][mask], p_test[mask], lo, hi).items()
                              if k in ("n", "accuracy", "balanced_accuracy", "recall_cssvd", "recall_healthy", "roc_auc")}

    # ---- paired comparison with Muse Spark on the 80 ablation images
    comparison = {}
    if ABLATION_PRED.exists():
        import csv

        from engine.evaluation.metrics import mcnemar_exact

        abl = list(csv.DictReader(ABLATION_PRED.open(encoding="utf-8")))
        name_to_sha = {Path(m["file"]).name: m["sha256"] for m in json.loads(ABLATION_MANIFEST.read_text())}
        test_rows = {rows[i]["sha256"]: k for k, i in enumerate(np.nonzero(test)[0])}
        clf_correct = {}
        for sha, k in test_rows.items():
            clf_correct[sha] = int(p_test[k] >= 0.5) == y[test][k]
        for exp in ("E0", "E4"):
            pairs = [(clf_correct[name_to_sha[r["image"]]], r["prediction"] == r["label"]) for r in abl
                     if r["exp"] == exp and name_to_sha.get(r["image"]) in clf_correct]
            comparison[f"classifier_vs_muse_{exp}"] = {
                "n": len(pairs), "classifier_accuracy": round(sum(a for a, _ in pairs) / len(pairs), 4),
                "muse_accuracy": round(sum(b for _, b in pairs) / len(pairs), 4),
                "mcnemar": mcnemar_exact([a for a, _ in pairs], [b for _, b in pairs])}

    version = f"{args.backbone}-lr-{datetime.now(timezone.utc):%Y%m%d}"
    summary = {k: m_test[k] for k in ("n", "accuracy", "accuracy_ci95", "balanced_accuracy", "balanced_accuracy_ci95",
                                      "roc_auc", "recall_cssvd", "recall_healthy")}
    summary["with_abstention"] = m_test["with_abstention"]
    # OOD reference set: ALL train embeddings (float16 .npy next to the JSON, same set used for the threshold)
    ref_path = ARTEFACT.with_name(f"cocoa_cssvd_reference_{args.backbone}.npy")
    artefact = {
        "version": version, "task": "healthy vs cssvd_like (CocoaSwolSet domain)",
        "backbone": {"name": args.backbone, **MODELS[args.backbone], "preprocessing": "resize-256 bicubic, crop-224, ImageNet norm, flip-TTA, concat(CLS, mean patch), L2"},
        "use_part_features": use_part, "parts": PARTS, "C": C,
        "coef": [round(float(v), 6) for v in final.coef_[0]], "intercept": float(final.intercept_[0]),
        "platt": [pa, pb], "abstain_band": [lo, hi],
        "part_reliability_oof": part_reliability, "trusted_parts": trusted_parts,
        "ood": {"min_similarity": round(min_sim, 4), "reference_file": ref_path.name},
        "training_data": {"dataset": "CocoaSwolSet v2 (CC BY 4.0) doi:10.17632/hvwth9dsfd.2", "n_train": int(train.sum()),
                          "n_test": int(test.sum()), "manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest()},
        "test_metrics_summary": summary,
    }
    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    np.save(ref_path, Etr.astype(np.float16))
    ARTEFACT.with_name(f"cocoa_cssvd_classifier_{args.backbone}.json").write_text(json.dumps(artefact), encoding="utf-8")
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    report = {"version": version, "backbone": args.backbone, "cv_balanced_accuracy": round(cv_bacc, 4),
              "selected": {"use_part_features": use_part, "C": C}, "abstain_band": [lo, hi],
              "ood_min_similarity": round(min_sim, 4), "part_reliability_oof": part_reliability,
              "trusted_parts": trusted_parts, "n_groups": int(groups.max() + 1),
              "split": {"train": int(train.sum()), "test": int(test.sum())}, "test": m_test, "test_per_part": per_part,
              "paired_comparison_with_muse_spark": comparison}
    (OUT_REPORT / f"report_{args.backbone}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
