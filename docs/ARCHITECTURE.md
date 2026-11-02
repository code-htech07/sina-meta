# Architecture

## Vue d'ensemble

```
             ┌────────────── frontend/ (Streamlit) ──────────────┐   ┌── api/ (FastAPI) ──┐   notebooks/ (Kaggle)
             └───────────────────────────┬────────────────────────┘   └─────────┬──────────┘          │
                                         ▼                                      ▼                     ▼
                                   engine/services.py  (façade unique — aucune route/page n'appelle une API externe)
                                         │
                                         ▼
                         engine/orchestration/pipeline.py  (étapes chronométrées, isolées)
   ┌──────────────┬───────────────┬──────────────────┬──────────────────┬────────────────┬──────────────────┐
   ▼              ▼               ▼                  ▼                  ▼                ▼                  ▼
preprocessing  meta_ai/sam    sources/weather   sources/satellite  sources/agronomy  fusion/context   meta_ai/muse_spark
 quality.py    (SAM 3.1)      nasa_power.py     sentinel2.py       knowledge_base    (contexte typé)  (raisonnement)
                                                                                           │                  │
                                                                                           ▼                  ▼
                                                    diagnosis/validator.py ◄── fusion/contradictions.py (règles)
                                                    diagnosis/caution.py (niveau de prudence déterministe)
                                                                     │
                                                                     ▼
                                                     AnalysisReport (Pydantic, JSON Schema) → storage/db.py (SQLite)
```

## Mise à jour : architecture hybride (voir `docs/CLASSIFIER.md`)
Un classifieur spécialisé entraîné (DINOv2-small ONNX + régression logistique calibrée, `engine/vision/`) s'exécute
après le contrôle qualité ; sa sortie est transmise à Muse Spark (`specialist_classifier` dans le contexte) et la décision
finale suit une règle de consensus (`AnalysisReport.consensus`). Désaccord ⇒ contradiction R6 + prudence élevée.

## Flux d'une analyse
1. **Réception** de la/des photo(s) → `acquisition/images.py` (format, SHA-256, EXIF date/GPS, stockage fichier).
2. **Contrôle qualité** déterministe → `preprocessing/quality.py` (résolution, flou, luminosité, saturation).
   Aucune image exploitable ⇒ `needs_new_photo`, **aucun appel payant**.
3. **SAM 3.1** (si activé et accessible) → masques/boîtes par prompt nominal (« cocoa pod », « brown lesion »…),
   recadrages des régions d'intérêt (ROI) + superposition. Échec ⇒ statut `error/unavailable`, jamais de masque inventé.
4. **Contexte** : NASA POWER (journalier + climatologie), Sentinel-2 L2A (STAC + COG, masque SCL, NDVI/NDMI),
   base de connaissances (fiches YAML sourcées). Chaque source échoue isolément et est signalée « manquante ».
5. **Normalisation et fusion** → `fusion/context.py` : contexte JSON où chaque section porte son `provenance_kind`
   (`modelled_estimate`, `remote_sensing_observation`, `computed`, `user_declared`, `literature`, `missing`,
   `not_provided_in_this_experiment`) + contrôles de cohérence (dates, latence, ancienneté des scènes).
6. **Muse Spark** : image(s) + ROI SAM + contexte → sortie JSON contrainte par `json_schema` puis **revalidée** par Pydantic
   (un tour de réparation au plus).
7. **Validation** : retrait des preuves citant une source non fournie (hallucination de source), signalement des
   affirmations non sourcées, identifiants de fiche inconnus, langage catégorique, recommandations non documentées.
8. **Règles de contradiction déterministes** (indépendantes du LLM) : maladie favorisée par l'humidité vs météo sèche,
   stress hydrique vs météo humide, baisse NDVI vs photo saine, signal satellite non spécifique, image non-cacao.
9. **Niveau de prudence** calculé par le moteur (`standard / elevated / high / insufficient_data`) avec justification.
10. **Rapport** `AnalysisReport` (schéma JSON versionné) persisté en SQLite.

## Séparation des responsabilités IA
- `meta_ai/client.py` : transport HTTP unique (auth Bearer, retries, SSE, gestion 4xx, redaction).
- `meta_ai/muse_spark.py` et `meta_ai/sam.py` : adaptateurs **indépendants** ; SAM utilise son propre format de sortie
  (décodé par le paquet officiel `meta-sam-parser`).
- Les prompts sont versionnés dans `meta_ai/prompts.py` (`PROMPT_VERSION`).

## Données et stockage
- SQLite (`data/maana_terra.db`) : `parcels`, `observations`, `images` (métadonnées JSON), `analyses` (rapport JSON complet).
  Les rapports étant des documents JSON validés, la migration vers PostgreSQL/JSONB est directe.
- Fichiers : `data/raw/uploads/` (photos), `data/processed/artifacts/` (ROI et superpositions SAM).

## Expérimentation
`engine/evaluation/` : audit de dataset (API Mendeley), métriques sans dépendance, ablation E0–E4 avec cache de contexte
partagé (mêmes données pour toutes les expériences). Résultats bruts JSONL + `summary.json` + `REPORT.md` par run.
