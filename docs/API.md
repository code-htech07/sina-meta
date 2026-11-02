# API REST (FastAPI)

Lancement : `uvicorn api.main:app --reload` · documentation interactive : `/docs` · schéma du rapport : `GET /schema/report`.

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Modèles configurés, présence de la clé (booléen + longueur uniquement) |
| GET | `/status/sources` | Vérification en direct : Meta API, NASA POWER, STAC Sentinel-2, base de connaissances |
| GET | `/knowledge-base` | Fiches agronomiques (avec références) |
| POST | `/parcels` | Créer une parcelle (JSON `ParcelCreate`) |
| GET | `/parcels`, `/parcels/{id}` | Lister / lire |
| POST | `/observations` | `multipart/form-data` : `parcel_id`, `observation_date`, `plant_part`, `farmer_symptoms`, `farmer_notes`, `images[]` |
| GET | `/parcels/{id}/observations` | Historique des observations |
| POST | `/observations/{id}/analyze` | Lance le pipeline ; corps optionnel `ExperimentConfig` (sources activées) |
| GET | `/analyses?parcel_id=`, `/analyses/{id}` | Rapports `AnalysisReport` |
| GET | `/images/{image_id}` | Fichier image |
| GET | `/schema/report` | JSON Schema de `AnalysisReport` |

## Exemple
```bash
curl -X POST localhost:8000/parcels -H 'content-type: application/json' \
  -d '{"name":"Parcelle A","latitude":5.70,"longitude":-6.70,"planting_year":2012}'
curl -X POST localhost:8000/observations -F parcel_id=parcel_xxx -F observation_date=2026-10-28 \
  -F plant_part=pod -F "farmer_symptoms=taches brunes" -F images=@data/samples/cocoaswolset/cocoa_pod_Diseased.jpg
curl -X POST localhost:8000/observations/obs_xxx/analyze
```

## Contrat de sortie (`AnalysisReport`, `schema_version` 1.0) — champs principaux
`status` (`completed` / `needs_new_photo` / `failed` + `failure_reason`) · `image_quality[]` · `image_assessment` ·
`overall_assessment` · `visual_observations[]` · `segmentation` · `primary_hypotheses[]` / `alternative_hypotheses[]`
(`status` ∈ compatible/plausible/weakly_supported/unlikely — jamais « confirmé » ; `support_level` non calibré ;
`weather_consistency` ; preuves favorables/contradictoires avec `source_type`) · `contradictions[]` (`origin` rule/model) ·
`weather_context` · `satellite_context` · `knowledge_references[]` · `missing_information[]` · `recommended_checks[]` ·
`recommendations[]` (`documented`, `sources`) · `caution_level` + `caution_rationale` · `validation_flags[]` ·
`provenance[]` · `timings[]` · `model_usage[]` · `disclaimer`.
