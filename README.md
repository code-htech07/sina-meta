# 🌱 Maana Terra

**Intelligence agricole multimodale pour le cacao en Côte d'Ivoire** — Meta AI Developer Hackathon.

Maana Terra rapproche **la photo d'un symptôme**, **la météo** (NASA POWER), **le satellite** (Copernicus Sentinel-2),
**des connaissances agronomiques sourcées** et **l'historique de la parcelle** pour produire une analyse
contextualisée, traçable et prudente : hypothèses *compatibles* (jamais « confirmées »), éléments favorables et
contradictoires, contradictions entre sources, informations manquantes et vérifications terrain.

| Modèle Meta | Rôle |
|---|---|
| **Muse Spark 1.3** | Analyse multimodale et raisonnement (que voit-on ? quelles hypothèses ? qu'est-ce qui se contredit ?) |
| **SAM 3.1** | Segmentation : *où* se trouve la région concernée (masques, boîtes) — sans valeur diagnostique |

## Pourquoi ce n'est pas « un chatbot autour d'une photo »
- **Contexte typé par provenance** : estimation modélisée, observation satellite, valeur calculée, déclaration, littérature, donnée manquante.
- **Règles de contradiction déterministes** indépendantes du LLM (p. ex. maladie favorisée par l'humidité *vs* 30 jours secs).
- **Validation post-LLM** : retrait des preuves citant une source non fournie, détection du langage catégorique,
  recommandations marquées « non documentées » si elles ne viennent pas d'une fiche sourcée.
- **Niveau de prudence calculé par le moteur**, pas par le modèle ; scores du modèle explicitement non calibrés.
- **Ablation E0–E4** pour mesurer ce qu'apporte chaque source.

## Démarrage rapide
```bash
pip install -r requirements.txt
cp .env.example .env                       # renseigner MODEL_API_KEY (jamais commité)
python scripts/check_meta_access.py        # vérifie Muse Spark + SAM 3.1
python scripts/seed_demo.py --analyze      # parcelle + analyse de démonstration
streamlit run frontend/app.py              # interface
```
Kaggle : voir [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) (Internet ON + secret `MODEL_API_KEY`) et `notebooks/01…08`.

## État
Voir **[docs/STATUS.md](docs/STATUS.md)**. En bref : météo et satellite réels, fusion, validation, interface, API,
audit du dataset et tests sont opérationnels. Les appels Muse Spark / SAM 3.1 sont implémentés selon la documentation
officielle mais **n'ont pas encore été validés contre l'API réelle** (clé non disponible dans l'environnement de
développement). Aucun résultat expérimental n'a été produit à ce jour — aucun n'est inventé.

## Structure
```
engine/   acquisition · preprocessing · meta_ai (client, muse_spark, sam, prompts) · sources (weather, satellite, agronomy)
          fusion · diagnosis · orchestration · storage · evaluation · services.py
api/      FastAPI            frontend/  Streamlit (5 pages)        notebooks/  Kaggle 01–08
configs/  settings, models, data_sources, diseases                 data/       base de connaissances, échantillons CC BY
scripts/  check_meta_access, seed_demo, run_dataset_audit, run_ablation, build_notebooks
tests/    unit · integration (mock + réseau réel + Meta réel)      docs/       architecture, décisions, protocoles, démo
```

## Documentation
[Brief](docs/PROJECT_BRIEF.md) · [Architecture](docs/ARCHITECTURE.md) · [Décisions](docs/TECHNICAL_DECISIONS.md) ·
[Sources](docs/DATA_SOURCES.md) · [Datasets](docs/DATASET_PROTOCOL.md) · [Évaluation](docs/EVALUATION_PROTOCOL.md) ·
[API](docs/API.md) · [Déploiement](docs/DEPLOYMENT.md) · [Démo](docs/DEMO_SCENARIO.md) · [Plan](docs/IMPLEMENTATION_PLAN.md)

## Avertissement
Outil d'aide à la décision. Une photographie ne permet pas de confirmer un diagnostic ; toute hypothèse doit être
vérifiée sur le terrain et, si nécessaire, en laboratoire ou auprès d'un conseiller agricole (ANADER, CNRA).

## Licences
Code : MIT (à confirmer par le porteur). Images d'exemple : CocoaSwolSet, CC BY 4.0 (voir `data/samples/cocoaswolset/ATTRIBUTION.md`).
Contient des données Copernicus Sentinel modifiées. Données météo : NASA POWER.
