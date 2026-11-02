# Sources de données

| Source | Accès | Licence / conditions | Statut vérifié | Limites |
|---|---|---|---|---|
| **Meta Model API** — Muse Spark 1.3, SAM 3.1 | `https://api.meta.ai/v1/responses`, Bearer `MODEL_API_KEY` | Conditions Meta | ✅ testé en réel (2026-10-31) : modèles listés, Muse Spark et SAM 3.1 fonctionnels | Voir docs/STATUS.md |
| **NASA POWER** Daily + Climatology | `power.larc.nasa.gov/api/temporal/{daily,climatology}/point`, sans clé | Données publiques NASA | ✅ testé en réel (tests `-m network`) | Estimations maillées (~0,5° × 0,625°) ; latence de quelques jours (`-999`) |
| **Copernicus Sentinel-2 L2A** via Element 84 Earth Search + AWS Open Data (`sentinel-cogs`) | STAC `earth-search.aws.element84.com/v1`, COG publics | Licence Copernicus (libre, attribution « Contains modified Copernicus Sentinel data ») | ✅ testé en réel | Nébulosité tropicale ; pixels 10–20 m mélangeant cacaoyers/ombrage/sol ; pas de diagnostic foliaire |
| **CocoaSwolSet** (Mendeley Data, doi:10.17632/hvwth9dsfd.2) | API publique Mendeley | **CC BY 4.0** | ✅ audit réel, 8 images échantillons redistribuées (redimensionnées) | Binaire sain/malade (swollen shoot) ; pas de GPS/date ; annotations visuelles |
| **Base de connaissances** `data/knowledge_base/cocoa/*.yaml` | Fichiers du dépôt | Voir chaque référence | 6 fiches validées par schéma | Fiches anthracnose / chérelles : fiabilité faible (à compléter) |
| **DINOv2-small** (Meta) — export ONNX `onnx-community/dinov2-small` rév. 8b1f705 | Hugging Face, sans clé | Apache-2.0 (modèle d'origine `facebook/dinov2-small`) | ✅ utilisé (embeddings du classifieur) | Licence de l'export ONNX non indiquée séparément |

Note audit : 2 fichiers PNG de CocoaSwolSet sont illisibles (`20250527_110144.png`, `20250529_094740.png`).

## Provenance dans les rapports
Chaque `AnalysisReport.provenance[]` contient : source, type de preuve (`EvidenceKind`), URL, date de récupération,
période, coordonnées, résolution spatiale, notes.

## Références de la base de connaissances (consultées le 2026-10-31)
| Fiche | Références | Fiabilité |
|---|---|---|
| black_pod | Plantwise/CABI factsheet SL014En (CC BY-SA 4.0) ; Vanegtern, Rogers & Nelson 2015, UH-CTAHR PD-108 | haute |
| cssvd | Ameyaw, Domfeh & Gyamera 2023, *Viruses* 16(1):43, doi:10.3390/v16010043 (CC BY 4.0) | haute |
| mirids | Field observations on cacao mirids in the Gold Coast, *Bull. Entomol. Res.* (auteur/date à vérifier) ; Yede et al. 2024, IJAR 12(3), doi:10.21474/IJAR01/18447 | moyenne |
| anthracnose | Takata et al. 2024, *J. Fungi* 10(3):204, doi:10.3390/jof10030204 | faible |
| cherelle_wilt | UH-CTAHR PD-108 ; Takata et al. 2024 | faible |
| water_stress | Carr & Lockwood 2011, *Exp. Agric.* 47(4):653-676, doi:10.1017/S0014479711000421 | moyenne |

**À compléter en priorité** : fiches CNRA/ANADER (Côte d'Ivoire), CABI Compendium (P. megakarya, mirides, anthracnose),
ICCO. Ne pas ajouter de fiche sans référence vérifiable.

## Datasets étudiés et non retenus
- **KaraAgroAI Cocoa** : sources contradictoires (3 classes / 17 703 images selon arXiv 2405.04535 ; 8 classes /
  11 978 images selon Dataset Ninja) ; licence non confirmée lors de l'audit → non utilisé.
- **Jeux CIRAD** : non localisé de jeu d'images de symptômes sous licence claire pendant cette session.
