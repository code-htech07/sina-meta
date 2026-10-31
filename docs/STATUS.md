# État réel du projet — 2026-10-31 (mis à jour après accès réel à l'API Meta)

> Document de reprise : une nouvelle session doit lire `PROJECT_BRIEF.md`, puis ce fichier, puis `IMPLEMENTATION_PLAN.md`.

## Environnement de développement constaté
- Dépôt initialement **vide** (aucun commit, aucune configuration ni exemple de requête Meta).
- Python 3.13, venv local ; Kaggle non accessible depuis la session (pas d'identifiants).
- Clé Meta fournie comme **secret réseau** de l'environnement cloud (type Bearer, hôte `api.meta.ai`) : injectée par le
  proxy, jamais visible du code (`auth_mode = proxy_injected`).
- Réseau sortant OK vers NASA POWER, Earth Search STAC, COG AWS, Mendeley.

## TERMINÉ (développé et testé)
- Fondations : configuration YAML + surcharges env, gestion des secrets (env / `.env` / secrets Kaggle), logs avec masquage.
- Contrats Pydantic (`engine/schemas.py`) avec provenance typée ; rapport JSON versionné.
- Stockage SQLite + fichiers ; ingestion d'images (EXIF date/GPS) ; contrôle qualité déterministe.
- **NASA POWER** journalier + climatologie, indicateurs (cumuls, périodes sèches, HR, anomalies) — **testé en réel**.
- **Sentinel-2 L2A** (Earth Search + COG) : STAC, masque SCL, NDVI/NDMI, gestion de l'offset radiométrique — **testé en réel**.
- Base de connaissances : 6 fiches sourcées, validées par schéma.
- Fusion : contexte typé par provenance, contrôles de cohérence, 5 règles de contradiction déterministes,
  validateur post-LLM, niveau de prudence déterministe.
- Pipeline de bout en bout avec dégradation gracieuse — **exécuté en réel** (météo + satellite réels ; Muse Spark
  explicitement « skipped » faute de clé, aucun contenu inventé).
- Audit **réel** de CocoaSwolSet (API Mendeley) ; 8 images CC BY 4.0 redistribuées avec attribution.
- Métriques d'évaluation et exécuteur E0–E4 (testés sur valeurs connues).
- API FastAPI (testée), interface Streamlit 5 pages (testée en headless + capture navigateur réelle).
- 8 notebooks Kaggle générés ; notebooks 02/04/05 exécutés avec succès en local.
- Tests : 47 hors-ligne + 2 réseau réels passent ; 4 tests Meta réels prêts (ignorés sans clé).

## VALIDÉ CONTRE L'API META RÉELLE (2026-10-31)
- `GET /v1/models` : `muse-spark-1.3` et `sam-3.1` listés (9 modèles accessibles).
- Muse Spark : texte minimal OK ; image + sortie structurée OK ; analyse complète de la démo : ~27 s, ~0,035 $.
- SAM 3.1 : data URL acceptée ; segmentation OK (~0,0025 $/prompt). Benchmark : `experiments/reports/sam_prompt_benchmark.md`.
- Pipeline complet réel (qualité → SAM → NASA POWER → Sentinel-2 → Muse Spark → validation) : ~90 s, ~0,047 $ par analyse.

### Écarts découverts en réel et corrigés
1. **Clés JSON corrompues** avec `strict: false` (`"hypotheses_foo"`, `"contradictions_contrad"`) → rapport vide silencieux.
   Correctif : schéma strict (`strict: true`) + modèles Pydantic `extra="forbid"`. Deux runs successifs propres ensuite.
2. **Masques SAM locaux à la boîte** (`data=H,W` = taille de la boîte) : replacement via `record.bounds`.
3. **Vocabulaire SAM** : « cocoa pod », « pod », « lesion », « rot » → aucune détection ; retenu : `fruit`, `leaf`,
   `brown spot`, `discolored area`, `hole` (seuil de score 0,5).
4. Le tour de réparation reprend les consignes (langue, règles).

### Observations préliminaires (n = 1 image, non généralisables)
- Variabilité entre exécutions identiques (pourriture brune « plausible » puis « faiblement étayée ») → à mesurer.
- Sur la cabosse étiquetée « swollen shoot » dans CocoaSwolSet, Muse Spark propose pourriture brune / mirides / cause
  inconnue, pas CSSVD : cohérent avec l'aspect de l'image, à confronter à l'étiquette lors de l'ablation.

## PARTIELLEMENT TERMINÉ
- Base de connaissances : fiches anthracnose et chérelles de fiabilité faible ; sources CNRA/ANADER à ajouter.
- Prompts SAM choisis sur 3 images seulement.

## EN ATTENTE D'ACCORD
| Élément | Raison | Estimation |
|---|---|---|
| Ablation E0–E4 | Dépense | 8 images × 5 ≈ 1,2 $ ; 80 images × 5 ≈ 12 $ (~0,03 $ par appel Muse mesuré) |
| Interface accessible depuis un téléphone | Publication externe | Streamlit Community Cloud (gratuit) ou équivalent |

## NON COMMENCÉ
- Analyse temporelle multi-observations (P2), cartographie avancée (P2).
- Test de significativité apparié (McNemar) pour E0 vs E4.
- Conversion HEIC/DNG du dataset ; prise en charge de polygones de parcelle dans l'UI.

## PROCHAINE ACTION
Avec l'accord du porteur : ablation E0–E4 sur les 8 images du dépôt (~1,2 $), puis sur ~80 images (~12 $).
