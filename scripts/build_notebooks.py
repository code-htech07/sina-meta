#!/usr/bin/env python
"""Generate the Kaggle notebooks in notebooks/ from source cells (keeps them reviewable in git).

All notebooks share the same bootstrap cell: clone the repo, install requirements, and read
MODEL_API_KEY from Kaggle Secrets (Add-ons > Secrets). Internet must be ON in the notebook settings.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

BOOTSTRAP = '''# --- Bootstrap (Kaggle or local) -------------------------------------------------
# Kaggle: Settings > Internet = ON ; Add-ons > Secrets > add "MODEL_API_KEY" (and "GITHUB_TOKEN" if the repo is private)
import os, subprocess, sys
REPO = "code-htech07/sina-meta"
BRANCH = "ccr-c8fab4b3-0jmvi1"  # change to "main" once merged
if os.path.exists("engine"):            # already inside the repo (local run)
    pass
else:
    if not os.path.exists("sina-meta"):
        url = f"https://github.com/{REPO}.git"
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
            url = f"https://{tok}@github.com/{REPO}.git"
        except Exception:
            pass
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, "sina-meta"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # silent: never print a tokenised URL
    os.chdir("sina-meta")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
sys.path.insert(0, os.getcwd())
from engine.config import describe_secret_status
from engine.logging_utils import setup_logging
setup_logging("WARNING")
print(describe_secret_status())  # shows only presence/length of the key, never its value'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


NOTEBOOKS: dict[str, list[dict]] = {
    "01_meta_model_benchmark.ipynb": [
        md("# 01 — Vérification des modèles Meta (Muse Spark 1.3, SAM 3.1)\nTeste l'accès réel : liste des modèles, "
           "requête texte minimale, image + sortie structurée, segmentation SAM. Résultats → `experiments/reports/meta_access_check.json`."),
        code(BOOTSTRAP),
        code("!python scripts/check_meta_access.py"),
        code("import json; print(json.dumps(json.load(open('experiments/reports/meta_access_check.json')), indent=1, ensure_ascii=False))"),
    ],
    "02_dataset_audit.ipynb": [
        md("# 02 — Audit du dataset CocoaSwolSet\nLicence, classes, doublons, formats, déséquilibre — via l'API publique Mendeley (sans télécharger 7,7 Go)."),
        code(BOOTSTRAP),
        code("from engine.evaluation.dataset_audit import run_audit\nimport pandas as pd, json\naudit = run_audit()\n"
             "print(json.dumps({k: v for k, v in audit.items() if not k.startswith('per_')}, indent=1, ensure_ascii=False))\n"
             "pd.Series(audit['per_part_and_class_unique_analysable']).plot.barh(title='Images uniques analysables');"),
        md("## Échantillon stratifié pour l'ablation (optionnel)\n`N` images par (partie de plante, classe), SHA-256 vérifié."),
        code("N = 10\n!python scripts/run_dataset_audit.py --download {N}"),
    ],
    "03_image_analysis.ipynb": [
        md("# 03 — Analyse visuelle : contrôle qualité, SAM 3.1, Muse Spark\nNécessite `MODEL_API_KEY`."),
        code(BOOTSTRAP),
        code("from pathlib import Path\nfrom engine.preprocessing.quality import assess_quality\n"
             "samples = sorted(Path('data/samples/cocoaswolset').glob('*.jpg'))\n"
             "for p in samples: q = assess_quality(p.stem, p); print(p.name, q.usable, q.issues, q.laplacian_variance)"),
        code("from engine.acquisition.images import load_rgb\nfrom engine.meta_ai.sam import SamSegmenter\n"
             "img = load_rgb('data/samples/cocoaswolset/cocoa_pod_Diseased.jpg')\n"
             "seg = SamSegmenter().segment(img, 'nb03')\nprint(seg.status, seg.error, [(r.prompt, r.mask_area_fraction) for r in seg.regions])\n"
             "from IPython.display import Image as I, display\nif seg.regions: display(I(seg.regions[0].overlay_path))"),
        code("from engine.meta_ai.muse_spark import MuseSparkAnalyzer\nfrom engine.preprocessing.quality import to_data_url\n"
             "res = MuseSparkAnalyzer().analyze([to_data_url(img)], {'experiment': 'nb03_image_only'})\n"
             "print(res.output.model_dump_json(indent=1)); print(res.usage)"),
    ],
    "04_weather_integration.ipynb": [
        md("# 04 — Météo NASA POWER\nDonnées journalières + climatologie ; indicateurs (cumuls, périodes sèches, anomalies). "
           "Estimations maillées, pas des mesures sur parcelle."),
        code(BOOTSTRAP),
        code("from datetime import date\nimport pandas as pd\nfrom engine.sources.weather.nasa_power import get_weather_context\n"
             "w = get_weather_context(5.70, -6.70, date.today())\nprint(w.status, w.limitations)\n"
             "print(w.indicators.model_dump_json(indent=1))\n"
             "df = pd.DataFrame([d.model_dump() for d in w.daily]).set_index('day')\n"
             "df[['precipitation_mm']].plot.bar(figsize=(14,3), legend=False, title='Pluie journalière (mm)');"),
    ],
    "05_satellite_integration.ipynb": [
        md("# 05 — Sentinel-2 L2A (Earth Search / AWS COG)\nRecherche STAC, masque SCL, NDVI/NDMI sur la parcelle. "
           "Contexte de parcelle uniquement — pas d'identification de maladie."),
        code(BOOTSTRAP),
        code("from datetime import date\nimport pandas as pd\nfrom engine.sources.satellite.sentinel2 import get_satellite_context\n"
             "s = get_satellite_context(5.70, -6.70, date.today())\nprint(s.status, s.valid_scene_count, s.latest_ndvi, s.ndvi_change)\n"
             "pd.DataFrame([x.model_dump() for x in s.scenes])"),
    ],
    "06_multisource_fusion.ipynb": [
        md("# 06 — Fusion multimodale de bout en bout\nParcelle de démo → observation → pipeline complet → rapport JSON validé."),
        code(BOOTSTRAP),
        code("!python scripts/seed_demo.py --analyze"),
        code("from engine.services import MaanaTerraService\nrep = MaanaTerraService().list_analyses(limit=1)[0]\n"
             "print(rep.status, rep.caution_level, rep.caution_rationale)\n"
             "for h in rep.primary_hypotheses: print(h.problem_id, h.status, h.support_level, h.weather_consistency)\n"
             "for c in rep.contradictions: print('CONTRADICTION', c.origin, c.description)\n"
             "print(rep.model_dump_json(indent=1)[:4000])"),
    ],
    "07_ablation_study.ipynb": [
        md("# 07 — Ablation E0–E4\nMêmes images, même prompt, même modèle ; seul le contexte varie. "
           "Localisation PROXY explicite (le dataset n'a pas de GPS). Les coûts sont estimés avant exécution."),
        code(BOOTSTRAP),
        code("MANIFEST = 'data/samples/cocoaswolset/manifest.json'  # or data/raw/datasets/cocoaswolset_n10_s42/manifest.json\n"
             "!python scripts/run_ablation.py --manifest {MANIFEST} --proxy-lat 6.8276 --proxy-lon -5.2767 --fallback-date 2025-05-27 --yes"),
    ],
    "08_final_evaluation.ipynb": [
        md("# 08 — Évaluation finale\nAgrège le dernier run d'ablation. Aucun chiffre n'est produit sans run réel."),
        code(BOOTSTRAP),
        code("import json, pandas as pd\nfrom pathlib import Path\nruns = sorted(Path('experiments/ablation/runs').glob('*/summary.json'))\n"
             "assert runs, 'Aucun run : exécuter le notebook 07'\nsummary = json.loads(runs[-1].read_text())\n"
             "rows = {k: {**v['classification'], **{'cssvd_top1': v['cssvd_specific'].get('top1_hit_rate')}, **v['quality'], **v['cost_latency']} "
             "for k, v in summary['experiments'].items()}\npd.DataFrame(rows).T"),
        code("print(open(runs[-1].parent / 'REPORT.md').read())"),
    ],
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
