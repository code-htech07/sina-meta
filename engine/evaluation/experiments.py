"""Ablation study E0–E4 on a labelled image manifest.

| Exp | Context given to Muse Spark                                  |
|-----|--------------------------------------------------------------|
| E0  | image only                                                   |
| E1  | image + weather (NASA POWER)                                 |
| E2  | image + satellite (Sentinel-2)                               |
| E3  | image + weather + satellite                                  |
| E4  | image + weather + satellite + agronomic knowledge base       |

Controls: same images, same prompt version, same model, same cached context per
(location, date) for every experiment; SAM segmentation and farmer input are held
constant (disabled by default, because the dataset has no farmer declarations).

Honesty rules: results are computed ONLY from real runs written to
``experiments/ablation/runs/<run_id>/``. Nothing is written to the summary that was
not measured. When the dataset has no per-image location, a PROXY location must be
passed explicitly and is recorded in the summary (``context_location_is_proxy``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from engine.config import ROOT, get_settings
from engine.evaluation.metrics import (
    classification_metrics,
    cost_latency,
    disease_specific,
    quality_metrics,
    report_to_prediction,
)
from engine.meta_ai.prompts import PROMPT_VERSION
from engine.orchestration.pipeline import AnalysisInput, AnalysisPipeline, ContextCache, image_metadata_from_path
from engine.schemas import AnalysisReport, ExperimentConfig

EXPERIMENTS: dict[str, ExperimentConfig] = {
    "E0": ExperimentConfig(name="E0_image_only", use_weather=False, use_satellite=False, use_knowledge_base=False,
                           use_segmentation=False, use_farmer_input=False),
    "E1": ExperimentConfig(name="E1_image_weather", use_weather=True, use_satellite=False, use_knowledge_base=False,
                           use_segmentation=False, use_farmer_input=False),
    "E2": ExperimentConfig(name="E2_image_satellite", use_weather=False, use_satellite=True, use_knowledge_base=False,
                           use_segmentation=False, use_farmer_input=False),
    "E3": ExperimentConfig(name="E3_image_weather_satellite", use_weather=True, use_satellite=True,
                           use_knowledge_base=False, use_segmentation=False, use_farmer_input=False),
    "E4": ExperimentConfig(name="E4_full_context", use_weather=True, use_satellite=True, use_knowledge_base=True,
                           use_segmentation=False, use_farmer_input=False),
}

PART_MAP = {"Leaves-Young": "young_leaf", "Leaves-Adults": "leaf", "Trees": "whole_tree", "cocoa pod": "pod"}
LABEL_MAP = {"Healthy": "healthy", "Diseased": "diseased"}

_DATE_PATTERNS = [
    re.compile(r"(20\d{2})(\d{2})(\d{2})_\d{6}"),          # 20250527_095257.jpg
    re.compile(r"IMG-(20\d{2})(\d{2})(\d{2})-WA"),          # IMG-20250527-WA0223.jpg
    re.compile(r"(20\d{2})-(\d{2})-(\d{2})"),               # WhatsApp Image 2025-05-28 at ...
]


def date_from_filename(name: str) -> date | None:
    """Best-effort date from camera/WhatsApp filenames (may be a sharing date, not capture date)."""
    for pat in _DATE_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


@dataclass
class Sample:
    path: str
    label: str
    plant_part: str
    obs_date: date | None
    date_source: str


def load_manifest(path: Path) -> list[Sample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        d = date_from_filename(r.get("filename", ""))
        out.append(Sample(path=r.get("sample_file") or r["file"], label=LABEL_MAP.get(r["label"], r["label"]),
                          plant_part=PART_MAP.get(r.get("part") or "", "unknown"), obs_date=d,
                          date_source="filename" if d else "fallback"))
    return out


def run_ablation(
    manifest: Path,
    experiments: list[str],
    proxy_lat: float | None,
    proxy_lon: float | None,
    fallback_date: date,
    limit: int | None = None,
    out_root: Path | None = None,
    pipeline: AnalysisPipeline | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    samples = load_manifest(manifest)[: limit or None]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (out_root or ROOT / "experiments" / "ablation" / "runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = pipeline or AnalysisPipeline(cache=ContextCache())
    s = get_settings()
    summary: dict[str, Any] = {
        "run_id": run_id,
        "manifest": str(manifest),
        "n_samples": len(samples),
        "label_distribution": {lab: sum(1 for x in samples if x.label == lab) for lab in ("healthy", "diseased")},
        "muse_model": s.muse_model,
        "prompt_version": PROMPT_VERSION,
        "context_location": [proxy_lat, proxy_lon],
        "context_location_is_proxy": True,
        "date_sources": {k: sum(1 for x in samples if x.date_source == k) for k in ("filename", "fallback")},
        "experiments": {},
        "limitations": [
            "Le dataset ne fournit pas la localisation des images : le contexte météo/satellite est celui d'une localisation PROXY fournie par l'opérateur.",
            "Les dates sont déduites des noms de fichiers (date de prise ou de partage WhatsApp) quand c'est possible.",
            "Effectif réduit : les écarts entre expériences doivent être interprétés avec prudence (pas de test de significativité à ce stade).",
        ],
    }
    for exp_key in experiments:
        cfg = EXPERIMENTS[exp_key]
        reports: list[AnalysisReport] = []
        y_true: list[str] = []
        with (out_dir / f"{exp_key}.jsonl").open("w", encoding="utf-8") as fh:
            def one(smp: Sample) -> tuple[AnalysisReport, float]:
                img = image_metadata_from_path(smp.path, plant_part=smp.plant_part)
                inp = AnalysisInput(images=[img], observation_date=smp.obs_date or fallback_date,
                                    plant_part=smp.plant_part, latitude=proxy_lat, longitude=proxy_lon)
                t0 = time.perf_counter()
                return pipeline.run(inp, cfg), (time.perf_counter() - t0) * 1000

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:  # results kept in sample order
                for smp, (rep, wall) in zip(samples, ex.map(one, samples)):
                    reports.append(rep)
                    y_true.append(smp.label)
                    fh.write(json.dumps({"sample": smp.path, "label": smp.label, "prediction": report_to_prediction(rep),
                                         "wall_ms": round(wall, 1), "report": rep.model_dump(mode="json")},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
        y_pred = [report_to_prediction(r) for r in reports]
        summary["experiments"][exp_key] = {
            "config": cfg.model_dump(),
            "classification": classification_metrics(y_true, y_pred),
            "cssvd_specific": disease_specific(reports, y_true, "cssvd"),
            "quality": quality_metrics(reports),
            "cost_latency": cost_latency(reports),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# Ablation E0–E4 — run {summary['run_id']}", "",
             f"- Échantillons : {summary['n_samples']} ({summary['label_distribution']})",
             f"- Modèle : {summary['muse_model']} — prompt {summary['prompt_version']}",
             f"- Localisation du contexte (PROXY) : {summary['context_location']}", "",
             "| Exp | n | abstention | accuracy | balanced acc. | macro-F1 | erreurs critiques | CSSVD top-1 | hallucinations/rapport | coût moyen $ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary["experiments"].items():
        c, q, cl, d = v["classification"], v["quality"], v["cost_latency"], v["cssvd_specific"]
        lines.append(f"| {k} | {c.get('n_evaluated')} | {c.get('abstention_rate')} | {c.get('accuracy')} | "
                     f"{c.get('balanced_accuracy')} | {c.get('macro_f1')} | {c.get('critical_error_rate')} | "
                     f"{d.get('top1_hit_rate')} | {q.get('hallucinated_source_items_per_report')} | "
                     f"{cl.get('estimated_cost_usd_per_analysis')} |")
    lines += ["", "## Limites", *[f"- {x}" for x in summary["limitations"]]]
    return "\n".join(lines) + "\n"
