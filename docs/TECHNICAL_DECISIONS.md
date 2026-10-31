# Décisions techniques (ADR condensés)

Format : décision — alternatives — justification. Date : 2026-10-31 sauf mention.

### TD-01 · Meta Model API via `httpx`, sans SDK OpenAI
- Alternatives : SDK `openai` pointé sur `https://api.meta.ai/v1` (montré dans la doc Meta).
- Choix : client `httpx` maison (`engine/meta_ai/client.py`). Contrat explicite et testable par `MockTransport`,
  une dépendance de moins, contrôle fin des retries/SSE/redaction. `output_text` (helper SDK) n'est pas utilisé :
  le texte est reconstruit depuis `output[].content[]` (`type: output_text`).

### TD-02 · Identifiants de modèles
- `muse-spark-1.3` et `sam-3.1` : listés sur https://dev.meta.ai/docs/models et utilisés dans les guides.
  La fiche modèle SAM affiche `sam-3-1` : divergence documentaire → valeur configurable (`configs/models.yaml`,
  `MAANA_SAM_MODEL`) et vérification par `GET /v1/models` dans `scripts/check_meta_access.py`.

### TD-03 · Sortie structurée Muse Spark
- `text.format = {type: json_schema, strict: false}` (le mode strict exige `additionalProperties:false` sur chaque objet
  et que `required` liste toutes les propriétés, ce que le schéma Pydantic — champs avec valeurs par défaut — ne respecte pas) + **revalidation Pydantic locale** systématique + 1 tour de réparation (`previous_response_id`).
- Si l'API rejette `text` ou `reasoning` (HTTP 400), le paramètre est retiré **et la note apparaît dans le rapport**.

### TD-04 · SAM 3.1 : flux SSE + parseur officiel
- La doc recommande `stream: true` ; le format des masques (base85, codage par plage/arith.) n'est pas spécifié
  publiquement → décodage via `meta-sam-parser` (paquet officiel Meta) plutôt qu'une ré-implémentation.
- Images envoyées en data URL base64 (`input_image.image_url`), format documenté pour `input_image` en général ;
  à confirmer pour SAM par `check_meta_access.py` (alternative documentée : `file_id` via Files API).
- SAM n'est appelé que sur la 1ʳᵉ image exploitable, avec 4 prompts (coût ≈ 4 × 0,0025 $).

### TD-05 · Sentinel-2 via Element 84 Earth Search (AWS Open Data) plutôt que Copernicus Data Space
- Alternatives : CDSE (OAuth requis), Microsoft Planetary Computer (signature d'URL).
- Choix : Earth Search STAC + COG publics, **aucune authentification**, lecture par fenêtres (HTTP range) — compatible Kaggle.
- **Offset radiométrique** : les items ont `raster:bands.offset = -0.1` mais `earthsearch:boa_offset_applied = true`.
  Vérifié empiriquement (tuile 30NTN, 2026-10-25) : appliquer l'offset donne des réflectances rouges négatives
  (médiane −0,033) ⇒ l'offset est déjà appliqué ; seul `scale` est utilisé. Garde-fou : scène rejetée si >5 % de
  réflectances < −0,01.
- Toutes les bandes sont lues sur la grille 10 m de B04 (rééchantillonnage au plus proche pour B8A/B11/SCL).
- Pixels valides : SCL ∈ {4 végétation, 5 non-végétal} ; eau, ombres, nuages, cirrus exclus.
- Indices : NDVI (B08/B04), NDMI (B8A/B11, NDWI de Gao 1996).

### TD-06 · NASA POWER : journalier + climatologie
- Anomalie pluviométrique = cumul 30 j / normale climatologique (somme des moyennes journalières mensuelles POWER).
- Fenêtres ancrées sur le **dernier jour valide** (les jours `-999` récents ne comptent jamais comme jours secs).

### TD-07 · Seuils heuristiques de fusion
- `wet_rain_ratio_vs_normal = 1.2`, `dry_rain_ratio_vs_normal = 0.5`, `dry_spell_days_alert = 14`, `ndvi_drop_alert = 0.1`.
- Ce sont des **heuristiques d'ingénierie**, pas des seuils publiés ; elles sont nommées ainsi dans chaque message.
  À calibrer avec des agronomes (CNRA/ANADER).

### TD-08 · Niveau de prudence déterministe
- Calculé par `diagnosis/caution.py` à partir des contradictions, flags de validation, fiabilité documentaire de la
  fiche, qualité d'image et disponibilité des sources — jamais par le LLM.

### TD-09 · Base de connaissances en YAML validé (pas de RAG vectoriel)
- 6 fiches ; toutes transmises au modèle, classées par mots-clés/organe. Un RAG vectoriel n'apporte rien à cette taille
  et réduirait la traçabilité. Schéma `KBEntry` (`extra=forbid`) : une fiche sans référence est invalide.

### TD-10 · Stockage SQLite stdlib
- Alternatives : SQLAlchemy + Alembic. Choix : `sqlite3` + rapports en JSON ; migration PostgreSQL/JSONB directe.

### TD-11 · GeoPandas non utilisé
- Le master prompt le cite ; il n'est pas nécessaire (géométries simples : cercle tampon ou polygone GeoJSON fourni,
  traitées par Shapely + Rasterio). Dépendance lourde évitée ; à ajouter si gestion multi-parcelles vectorielles.

### TD-12 · Streamlit appelle le moteur directement
- L'UI et l'API partagent `engine/services.py`. Sur Kaggle, lancer deux serveurs est peu pratique ; l'API reste
  disponible pour intégrations (`uvicorn api.main:app`).

### TD-13 · Dataset d'évaluation : CocoaSwolSet
- Licence vérifiée **CC BY 4.0** (API Mendeley), origine Côte d'Ivoire (INP-HB / ESATIC). KaraAgroAI : licence non
  trouvée lors de l'audit (sources contradictoires sur le contenu) → non utilisé tant que la licence n'est pas confirmée.
- Le dossier `Leaves/Adults/Diseaded` (faute de frappe dans le dataset publié) est interprété comme `Diseased` et
  signalé comme anomalie d'audit.

### TD-14 · Licence du code : MIT (à confirmer par le porteur)
- Choix par défaut réversible pour un projet de hackathon ; le porteur peut la changer avant publication.
