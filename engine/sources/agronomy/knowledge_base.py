"""Cocoa knowledge base: load validated fiches and select the relevant ones.

Retrieval is deliberately transparent (no embeddings): fiches are ranked by keyword
overlap with farmer-declared symptoms and plant-part compatibility. With only a handful
of fiches, ALL are passed to Muse Spark by default (it then must cite ids); ranking is
used to order them and to report which fiches were judged most relevant.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

from engine.config import ROOT, _load_yaml
from engine.schemas import KBEntry


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


class KnowledgeBase:
    def __init__(self, entries: list[KBEntry]) -> None:
        self.entries = {e.id: e for e in entries}

    @classmethod
    def load(cls, directory: Path | None = None) -> "KnowledgeBase":
        cfg = _load_yaml("diseases.yaml")
        directory = directory or ROOT / cfg["knowledge_base_dir"]
        entries = []
        for path in sorted(directory.glob("*.yaml")):
            with path.open(encoding="utf-8") as fh:
                entries.append(KBEntry.model_validate(yaml.safe_load(fh)))
        return cls(entries)

    def get(self, entry_id: str) -> KBEntry | None:
        return self.entries.get(entry_id)

    def ids(self) -> list[str]:
        return list(self.entries)

    def rank(self, symptoms_text: str | None = None, plant_part: str | None = None) -> list[tuple[KBEntry, float]]:
        tokens = set(re.findall(r"[a-z]+", _norm(symptoms_text or "")))
        scored = []
        for e in self.entries.values():
            score = 0.0
            for kw in e.keywords:
                kw_tokens = set(re.findall(r"[a-z]+", _norm(kw)))
                if kw_tokens and kw_tokens <= tokens:
                    score += 1.0
            if plant_part and plant_part not in ("unknown", "other"):
                if plant_part in e.plant_parts or (plant_part == "leaf" and "young_leaf" in e.plant_parts):
                    score += 0.5
                else:
                    score -= 0.5
            scored.append((e, score))
        return sorted(scored, key=lambda x: (-x[1], x[0].id))

    def context_block(self, symptoms_text: str | None, plant_part: str | None) -> list[dict]:
        """Compact representation of fiches for the LLM context (all fiches, ranked)."""
        out = []
        for e, score in self.rank(symptoms_text, plant_part):
            out.append({
                "id": e.id,
                "name": e.name,
                "category": e.category,
                "relevance_score_keyword": score,
                "characteristic_symptoms": e.characteristic_symptoms,
                "plant_parts": e.plant_parts,
                "favourable_conditions": e.favourable_conditions,
                "differentiation": e.differentiation,
                "verification_needed": e.verification_needed,
                "management": e.management,
                "documentary_reliability": e.documentary_reliability,
                "sources": [r.citation for r in e.references],
            })
        return out

    def reference_summary(self, entry_ids: list[str]) -> list[dict]:
        out = []
        for i in entry_ids:
            e = self.entries.get(i)
            if e:
                out.append({
                    "id": e.id,
                    "name": e.name,
                    "documentary_reliability": e.documentary_reliability,
                    "references": [r.model_dump(mode="json") for r in e.references],
                })
        return out


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.load()
