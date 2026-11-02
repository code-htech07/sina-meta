"""Specialist cocoa classifier: DINOv2 embeddings + calibrated logistic regression.

Task (CocoaSwolSet domain): "healthy" vs "cssvd_like" (swollen-shoot labelled images).
Inference is numpy-only (weights in a JSON file), so the app needs no PyTorch/sklearn.

Safeguards built into the artefact:
* Platt calibration fitted on out-of-fold scores (probabilities measured on unseen groups);
* abstention band [lo, hi] chosen on out-of-fold predictions;
* out-of-distribution guard: if the nearest training embedding is less similar than the
  1st percentile observed on held-out data, the classifier abstains ("hors domaine").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import ROOT

ARTEFACT = ROOT / "data" / "models_trained" / "cocoa_cssvd_classifier.json"
PARTS = ["Leaves-Young", "Leaves-Adults", "Trees", "cocoa pod"]
PLANT_PART_TO_DATASET = {
    "young_leaf": "Leaves-Young", "leaf": "Leaves-Adults", "whole_tree": "Trees", "trunk": "Trees",
    "branch": "Trees", "stem": "Trees", "pod": "cocoa pod", "cherelle": "cocoa pod",
}


def part_onehot(part: str | None) -> np.ndarray:
    v = np.zeros(len(PARTS), dtype=np.float32)
    if part in PARTS:
        v[PARTS.index(part)] = 1.0
    return v


@dataclass
class ClassifierOutput:
    probability_cssvd_like: float | None
    decision: str            # "cssvd_like" | "healthy" | "abstain"
    reason: str
    nearest_train_similarity: float | None
    in_domain: bool
    model_version: str
    test_metrics: dict[str, Any]


class CocoaClassifier:
    def __init__(self, artefact: dict[str, Any]) -> None:
        self.a = artefact
        self.coef = np.array(artefact["coef"], dtype=np.float32)
        self.intercept = float(artefact["intercept"])
        self.platt_a, self.platt_b = artefact["platt"]
        self.lo, self.hi = artefact["abstain_band"]
        self.use_part = artefact["use_part_features"]
        self.ref = np.load(ARTEFACT.parent / artefact["ood"]["reference_file"]).astype(np.float32)
        self.ood_threshold = artefact["ood"]["min_similarity"]

    @classmethod
    def load(cls, path: Path = ARTEFACT) -> "CocoaClassifier":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def features(self, emb: np.ndarray, part: str | None) -> np.ndarray:
        return np.concatenate([emb, part_onehot(part)]) if self.use_part else emb

    def predict(self, emb: np.ndarray, plant_part: str | None) -> ClassifierOutput:
        part = PLANT_PART_TO_DATASET.get(plant_part or "", plant_part)
        sim = float((self.ref @ emb).max())
        in_domain = sim >= self.ood_threshold
        score = float(self.features(emb, part) @ self.coef + self.intercept)
        p = float(1 / (1 + np.exp(-(self.platt_a * score + self.platt_b))))
        if not in_domain:
            decision, reason = "abstain", "Image trop différente des images d'entraînement (hors domaine)."
        elif self.lo < p < self.hi:
            decision, reason = "abstain", f"Probabilité dans la zone d'incertitude ({self.lo:.2f}–{self.hi:.2f})."
        else:
            decision = "cssvd_like" if p >= self.hi else "healthy"
            reason = "Probabilité hors de la zone d'incertitude."
        return ClassifierOutput(round(p, 4), decision, reason, round(sim, 4), in_domain,
                                self.a["version"], self.a["test_metrics_summary"])


@lru_cache(maxsize=1)
def get_classifier() -> CocoaClassifier | None:
    return CocoaClassifier.load() if ARTEFACT.exists() else None
