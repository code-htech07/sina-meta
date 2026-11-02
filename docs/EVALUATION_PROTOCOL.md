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

## Résultats — essai n°1 (run 20261031-155838, 8 images, 40 appels réels, coût réel 0,78 $)
Fichiers : `experiments/reports/ablation_20261031-155838/` (`REPORT.md`, `summary.json`, `predictions.csv`).

| Exp | Contexte | Accuracy (n=8) | Erreurs critiques | CSSVD en hypothèse principale (malades) | Coût moyen |
|---|---|---|---|---|---|
| E0 | image seule | 5/8 | 1/4 | 0/4 | 0,016 $ |
| E1 | + météo | 5/8 | 0/4 | 0/4 | 0,017 $ |
| E2 | + satellite | 4/8 | 1/4 | 0/4 | 0,018 $ |
| E3 | + météo + satellite | 5/8 | 0/4 | 0/4 | 0,020 $ |
| E4 | + base agronomique | 7/8 | 0/4 | 2/4 | 0,027 $ |

Lecture prudente :
- **n = 8, une seule exécution par condition** ; la variabilité entre exécutions identiques a été observée : les écarts
  (5/8 vs 7/8) ne sont **pas** statistiquement établis.
- E0–E3 ne fournissent pas la base de connaissances : les hypothèses y sont toutes `unknown` et les recommandations
  « non documentées » — c'est attendu par construction. Le gain d'E4 sur l'identification (CSSVD 2/4, mirides, pourriture brune)
  vient donc surtout de la base de connaissances.
- Faux positifs fréquents sur images saines en E1–E3 (3/4) : le contexte seul ne réduit pas la tendance à voir une anomalie.
- **Artefact de mesure** : les 30 éléments « source non fournie » (5–8 par expérience) citaient tous `farmer` pour la partie
  de plante déclarée, présente dans le contexte. Corrigé dans le pipeline après ce run ; ce compteur est invalide pour ce run.
- Aucune règle de contradiction déterministe ne s'est déclenchée (contexte proxy avec météo « normale ») : la métrique
  « contradictions ignorées » n'est pas évaluable ici.

## Résultats — évaluation n°2 (80 images, 2 novembre 2026, coût réel 7,77 $)
Rapport complet : `experiments/reports/ablation_80_20261102/REPORT.md`.
Accuracy : E0 0,58 · E1 0,61 · E2 0,67 · E3 0,61 · E4 0,67. McNemar exact vs E0 : p = 0,77 / 0,065 / 0,75 / 0,14 →
**aucune différence significative**. Forte tendance aux faux positifs sur plantes saines. E4 : meilleur macro-F1 (0,67),
meilleur rappel des sains (0,58), seul cas où CSSVD est identifié (17,5 % des malades). Le gain brut d'E2 ne peut pas être
causal (satellite d'un lieu proxy). Conclusion : il faut des photos de terrain géolocalisées pour évaluer la fusion.

## Résultats — E5, architecture hybride (80 images, 2,31 $)
Consensus classifieur + Muse Spark : 91,2 % des cas tranchés (31/34), abstention 57,5 %, 0 fausse alerte sur les sains,
7,5 % de malades manqués. Rapport : `experiments/reports/e5_20261102/REPORT.md` ; protocole du classifieur : `docs/CLASSIFIER.md`.
