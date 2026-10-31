"""Dataset audit for CocoaSwolSet (Mendeley Data, DOI 10.17632/hvwth9dsfd.2, CC BY 4.0).

Uses the public Mendeley API (no download of the 7.7 GB archive) to audit:
licence, folder/class structure, counts per class and plant part, file formats,
exact duplicates (identical SHA-256 published by Mendeley), RAW/HEIC files that
need conversion, and class imbalance. A stratified sample can then be downloaded.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from engine.config import ROOT, get_settings

PUBLIC_API = "https://data.mendeley.com/public-api/datasets/{id}"
FOLDERS_API = "https://data.mendeley.com/public-api/datasets/{id}/folders/{version}"
DATASET_ID = "hvwth9dsfd"
ANALYSABLE_EXT = {"jpg", "jpeg", "png"}
# Folder-name variants observed in the published dataset (v2): "Diseaded" is a typo of "Diseased".
LABEL_ALIASES = {"Healthy": "Healthy", "Diseased": "Diseased", "Diseaded": "Diseased"}


@dataclass
class DatasetFile:
    filename: str
    path: str            # folder path, e.g. "CocoaSwolSet/Leaves/Young/Diseased"
    label: str | None    # Healthy / Diseased (normalised)
    part: str | None     # Leaves-Young, Leaves-Adults, cocoa pod, Trees ...
    sha256: str
    size: int
    ext: str
    download_url: str
    label_folder: str | None = None  # raw folder name as published


def fetch_metadata(dataset_id: str = DATASET_ID, timeout: float = 60) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = httpx.get(PUBLIC_API.format(id=dataset_id), timeout=timeout).json()
    folders = httpx.get(FOLDERS_API.format(id=dataset_id, version=meta["version"]), timeout=timeout).json()
    return meta, folders


def build_index(meta: dict[str, Any], folders: list[dict[str, Any]]) -> list[DatasetFile]:
    by_id = {f["id"]: f for f in folders}

    def path_of(fid: str | None) -> list[str]:
        parts: list[str] = []
        while fid and fid in by_id:
            parts.append(by_id[fid]["name"])
            fid = by_id[fid].get("parent_id")
        return list(reversed(parts))

    out = []
    for f in meta.get("files", []):
        segs = path_of(f.get("folder_id"))
        raw_label = next((s for s in reversed(segs) if s in LABEL_ALIASES), None)
        label = LABEL_ALIASES.get(raw_label) if raw_label else None
        part_segs = [s for s in segs if s not in LABEL_ALIASES and s != "CocoaSwolSet"]
        cd = f.get("content_details", {})
        out.append(DatasetFile(
            filename=f["filename"], path="/".join(segs), label=label,
            part="-".join(part_segs[-2:]) if part_segs else None,
            label_folder=raw_label,
            sha256=cd.get("sha256_hash", ""), size=int(f.get("size") or cd.get("size") or 0),
            ext=f["filename"].rsplit(".", 1)[-1].lower(), download_url=cd.get("download_url", ""),
        ))
    return out


def audit(index: list[DatasetFile], meta: dict[str, Any]) -> dict[str, Any]:
    by_hash: dict[str, list[DatasetFile]] = defaultdict(list)
    for f in index:
        by_hash[f.sha256].append(f)
    dup_groups = [g for g in by_hash.values() if len(g) > 1 and g[0].sha256]
    cross_label_dups = [g for g in dup_groups if len({x.label for x in g}) > 1]
    stems = Counter((f.path, f.filename.rsplit(".", 1)[0]) for f in index)
    analysable = [f for f in index if f.ext in ANALYSABLE_EXT]
    unique_analysable = {f.sha256: f for f in analysable}.values()
    per_class = Counter(f.label for f in unique_analysable)
    per_part_class = Counter(f"{f.part} | {f.label}" for f in unique_analysable)
    majority = max(per_class.values()) if per_class else 0
    minority = min(per_class.values()) if per_class else 0
    return {
        "dataset": meta.get("name"),
        "doi": (meta.get("doi") or {}).get("id"),
        "version": meta.get("version"),
        "published": meta.get("publish_date"),
        "licence": (meta.get("data_licence") or {}).get("short_name"),
        "institutions": sorted({c.get("institution", {}).get("name") for c in meta.get("contributors", []) if c.get("institution")}),
        "total_files": len(index),
        "total_size_gb": round(sum(f.size for f in index) / 1e9, 2),
        "formats": dict(Counter(f.ext for f in index)),
        "files_same_stem_multiple_formats": sum(1 for v in stems.values() if v > 1),
        "exact_duplicate_groups": len(dup_groups),
        "exact_duplicate_extra_files": sum(len(g) - 1 for g in dup_groups),
        "duplicates_across_labels": len(cross_label_dups),
        "unlabelled_files": sum(1 for f in index if f.label is None),
        "unlabelled_folders": sorted({f.path for f in index if f.label is None}),
        "label_folder_anomalies": {f"{f.path}": f.label_folder for f in index if f.label_folder and f.label_folder != f.label},
        "analysable_unique_images": len(unique_analysable),
        "per_class_unique_analysable": dict(per_class),
        "per_part_and_class_unique_analysable": dict(sorted(per_part_class.items())),
        "imbalance_ratio_majority_minority": round(majority / minority, 2) if minority else None,
        "limitations": [
            "Étiquettes binaires Healthy/Diseased (swollen shoot) : pas d'annotation des autres maladies ni de localisation des symptômes.",
            "Pas de coordonnées GPS ni de date par image dans les métadonnées Mendeley : le contexte météo/satellite ne peut pas être rattaché image par image sans EXIF.",
            "Le diagnostic de référence est visuel (experts agronomes INP-HB) ; la publication ne mentionne pas de confirmation PCR.",
            "Anomalie de nommage : dossier « Diseaded » (Leaves/Adults) interprété comme « Diseased » — à confirmer auprès des auteurs.",
            "Fichiers DNG/HEIC à convertir ; doublons RAW/JPEG d'une même prise (même nom de fichier) à dédoublonner pour éviter les fuites entre jeux.",
        ],
    }


def stratified_sample(index: list[DatasetFile], per_stratum: int, seed: int = 42, max_bytes: int = 4_000_000) -> list[DatasetFile]:
    """Sample up to ``per_stratum`` unique analysable images per (part, label), preferring files < max_bytes."""
    rng = random.Random(seed)
    unique = {f.sha256: f for f in index if f.ext in ANALYSABLE_EXT and f.label}.values()
    strata: dict[tuple, list[DatasetFile]] = defaultdict(list)
    for f in unique:
        strata[(f.part, f.label)].append(f)
    out: list[DatasetFile] = []
    for key in sorted(strata, key=str):
        files = sorted(strata[key], key=lambda x: x.filename)
        small = [f for f in files if f.size <= max_bytes] or files
        rng.shuffle(small)
        out += small[:per_stratum]
    return out


def download(files: list[DatasetFile], target: Path, timeout: float = 120) -> list[dict[str, Any]]:
    """Download files, verify SHA-256 against Mendeley metadata, write a manifest."""
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for f in files:
            dest = target / f"{f.sha256[:12]}_{f.filename}"
            if not dest.exists():
                r = client.get(f.download_url)
                r.raise_for_status()
                dest.write_bytes(r.content)
            ok = hashlib.sha256(dest.read_bytes()).hexdigest() == f.sha256
            manifest.append({"file": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else str(dest),
                             "label": f.label, "part": f.part, "source_path": f.path, "filename": f.filename,
                             "sha256": f.sha256, "sha256_verified": ok, "licence": "CC BY 4.0",
                             "attribution": "CocoaSwolSet, Kopoin N.C. et al., INP-HB / ESATIC, Mendeley Data, doi:10.17632/hvwth9dsfd.2"})
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_audit(out_path: Path | None = None) -> dict[str, Any]:
    meta, folders = fetch_metadata()
    result = audit(build_index(meta, folders), meta)
    out_path = out_path or ROOT / "experiments" / "reports" / "dataset_audit_cocoaswolset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _ = get_settings()
    return result
