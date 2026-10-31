# Protocole d'évaluation

## Question
L'ajout de contexte (météo, satellite, connaissances agronomiques) améliore-t-il **mesurablement** la qualité et la
prudence des analyses de Muse Spark, à images identiques ?

## Expériences (`engine/evaluation/experiments.py`)
| Exp | Données transmises à Muse Spark |
|---|---|
| E0 | Image seule |
| E1 | Image + météo (NASA POWER) |
| E2 | Image + satellite (Sentinel-2) |
| E3 | Image + météo + satellite |
| E4 | Image + météo + satellite + base de connaissances |

**Contrôles** : mêmes images, même modèle (`muse-spark-1.3`), même `PROMPT_VERSION`, même `reasoning.effort`,
contexte mis en cache par (lieu, date) donc identique entre expériences ; SAM et déclarations agriculteur désactivés
(constants) ; une source non fournie est marquée `not_provided_in_this_experiment`.

## Vérité terrain et mapping
CocoaSwolSet : `Healthy → healthy`, `Diseased → diseased` (swollen shoot).
Prédiction : `abnormality_detected → diseased`, `no_visible_problem → healthy`,
`cannot_assess`/`needs_new_photo → abstain`, échec technique → `error` (exclu, compté).

## Indicateurs (`engine/evaluation/metrics.py`)
- Classification (sur prédictions non abstenues) : accuracy, balanced accuracy, précision/rappel/F1 par classe, macro-F1,
  matrice de confusion (avec colonne abstention), **taux d'abstention**, **taux d'erreurs critiques** (malade → « aucun problème »).
- Spécifique : CSSVD en hypothèse principale (top-1) chez les malades ; CSSVD proposé à tort chez les sains.
- Qualité / prudence : éléments citant une source non fournie (hallucinations de source), affirmations attribuées à la
  littérature sans fiche valide, langage catégorique, identifiants inconnus, part de recommandations non documentées,
  **contradictions ignorées** (règle déterministe déclenchée mais aucune contradiction rapportée par le modèle),
  distribution des niveaux de prudence.
- Coût et latence : latence pipeline (moyenne, p50, p95), latence Muse, tokens, coût estimé (prix publics `configs/models.yaml`).

## Exécution
```bash
python scripts/run_dataset_audit.py --download 10          # ~80 images stratifiées
python scripts/run_ablation.py --manifest data/raw/datasets/cocoaswolset_n10_s42/manifest.json \
    --proxy-lat 6.8276 --proxy-lon -5.2767 --fallback-date 2025-05-27
```
Sorties : `experiments/ablation/runs/<run_id>/{E0..E4}.jsonl, summary.json, REPORT.md`.

## Règles d'interprétation
- **Aucune conclusion sans run réel** ; aucun chiffre n'est écrit à la main dans la documentation.
- Avec 8 images (échantillon du dépôt), les résultats sont **illustratifs** ; viser ≥ 80 images (10/strate) et rapporter
  les effectifs. Pas de test de significativité implémenté à ce stade (à ajouter : McNemar apparié E0 vs E4).
- Le contexte étant un proxy (dataset sans GPS), E1–E3 mesurent surtout l'effet du contexte sur la **prudence** et le
  **raisonnement**, pas un gain de précision attribuable à une météo réellement liée à l'image.
- Un score de soutien Muse Spark n'est pas une probabilité : pas de courbe de calibration sans étude dédiée.

## Résultats
Aucun run d'ablation n'a encore été exécuté : la clé `MODEL_API_KEY` n'était pas disponible dans l'environnement de
développement (voir `docs/STATUS.md`).
