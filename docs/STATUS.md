# État réel du projet — 2026-10-31

> Document de reprise : une nouvelle session doit lire `PROJECT_BRIEF.md`, puis ce fichier, puis `IMPLEMENTATION_PLAN.md`.

## Environnement de développement constaté
- Dépôt initialement **vide** (aucun commit, aucune configuration ni exemple de requête Meta).
- Python 3.13, venv local ; Kaggle non accessible depuis la session (pas d'identifiants).
- **Aucune clé Meta** (`MODEL_API_KEY`/`META_API_KEY` absentes, pas de `.env`). L'endpoint `https://api.meta.ai/v1` est
  joignable (HTTP 401 `invalid_api_key` sans clé, format d'erreur conforme au client).
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
- Tests : 45 hors-ligne + 2 réseau réels passent ; 4 tests Meta réels prêts (ignorés sans clé).

## PARTIELLEMENT TERMINÉ
- **Adaptateur Muse Spark 1.3** : implémenté selon la documentation officielle (Responses API, `input_image` data URL,
  `json_schema`, `reasoning.effort`), testé contre le contrat via mocks. **Non validé contre l'API réelle.**
- **Adaptateur SAM 3.1** : implémenté (SSE, `metadata.mask_encoding`, `include_confidence`, décodage par le parseur
  officiel `meta-sam-parser`), testé avec des masques encodés par le codec officiel. **Non validé contre l'API réelle** ;
  l'envoi d'image en data URL est à confirmer pour SAM (la doc SAM montre une URL publique).
- Base de connaissances : fiches anthracnose et chérelles de fiabilité faible ; sources ivoiriennes (CNRA/ANADER) à ajouter.

## BLOQUÉ
| Élément | Cause | Déjà tenté | Conséquence | Action requise |
|---|---|---|---|---|
| Vérification réelle Muse Spark / SAM 3.1 | Pas de `MODEL_API_KEY` dans l'environnement | Recherche env, `.env`, secrets Kaggle ; doc officielle consultée ; endpoint sondé | Aucune analyse visuelle réelle produite | Porteur : `cp .env.example .env`, renseigner la clé, lancer `python scripts/check_meta_access.py` (ou notebook 01 sur Kaggle avec le secret `MODEL_API_KEY`) |
| Résultats d'ablation E0–E4 | Dépend de la clé | Exécuteur prêt et testé | Aucun résultat expérimental (aucun n'a été inventé) | Lancer `scripts/run_ablation.py` (coût estimé affiché avant exécution) |
| Exécution sur Kaggle | Pas d'accès au compte Kaggle | Notebooks générés avec bootstrap | Non exécuté sur Kaggle | Porteur : importer les notebooks, Internet ON, secret `MODEL_API_KEY` (+ `GITHUB_TOKEN` si dépôt privé) |

## NON COMMENCÉ
- Analyse temporelle multi-observations (P2), cartographie avancée (P2).
- Test de significativité apparié (McNemar) pour E0 vs E4.
- Conversion HEIC/DNG du dataset ; prise en charge de polygones de parcelle dans l'UI.

## PROCHAINE ACTION
Exécuter `python scripts/check_meta_access.py` avec la clé, corriger l'adaptateur si un écart de contrat apparaît
(identifiant SAM, data URL SAM, `text.format`), puis lancer l'ablation sur ≥ 80 images.
