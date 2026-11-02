# Installation, exécution et déploiement

## 1. Kaggle (cible principale)
1. Créer un notebook Kaggle (ou importer un fichier de `notebooks/`).
2. **Settings → Internet : ON** (nécessite un compte Kaggle vérifié par téléphone).
3. **Add-ons → Secrets** : ajouter `MODEL_API_KEY` (clé Meta) et l'attacher au notebook.
   Si le dépôt GitHub est privé, ajouter aussi `GITHUB_TOKEN` (jeton *fine-grained* en lecture seule sur ce dépôt).
4. Exécuter la première cellule (bootstrap) : clone du dépôt, `pip install -r requirements.txt`, lecture du secret.
   Le moteur lit automatiquement le secret Kaggle via `kaggle_secrets` (jamais affiché).
5. Notebooks : `01` accès Meta → `02` audit → `03` analyse visuelle → `04` météo → `05` satellite → `06` fusion →
   `07` ablation → `08` évaluation finale.

> Kaggle n'est pas un serveur de production : l'interface Streamlit se lance en local (ou sur un hébergeur) ;
> sur Kaggle on exécute le moteur, les expériences et l'évaluation.

## 2. Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # puis renseigner MODEL_API_KEY (fichier ignoré par git)
python scripts/check_meta_access.py        # vérifie Muse Spark + SAM 3.1 avec la clé
python scripts/seed_demo.py --analyze      # crée la parcelle et l'analyse de démonstration
streamlit run frontend/app.py              # interface (http://localhost:8501)
uvicorn api.main:app --reload              # API (http://localhost:8000/docs)
```

## 3. Tests
```bash
pytest -m "not network and not meta"       # 47 tests hors-ligne (aucun appel externe)
pytest -m network                          # NASA POWER + Sentinel-2 réels
MAANA_LIVE_TESTS=1 pytest -m meta          # Meta Model API réelle (clé requise, coût minime)
```

## 4. Variables d'environnement
| Variable | Rôle |
|---|---|
| `MODEL_API_KEY` (ou `META_API_KEY`) | Clé Meta Model API — **jamais** dans le code, les logs, les notebooks partagés ou git |
| `MAANA_MUSE_MODEL`, `MAANA_SAM_MODEL` | Surcharge des identifiants de modèles |
| `MAANA_SAM_ENABLED` | `false` pour désactiver SAM |
| `MAANA_DB_PATH` | Chemin de la base SQLite |
| `MAANA_LOG_LEVEL` | Niveau de log (les jetons Bearer et la clé sont masqués par un filtre) |

## 6. Publication de l'interface (Streamlit Community Cloud — faisable depuis un téléphone)
1. Ouvrir https://share.streamlit.io et se connecter avec GitHub (autoriser l'accès au dépôt `code-htech07/sina-meta`).
2. **Create app** → *Deploy a public app from GitHub* :
   - Repository : `code-htech07/sina-meta` · Branch : `ccr-c8fab4b3-0jmvi1` (ou `main` après fusion)
   - Main file path : `frontend/app.py` · App URL : au choix (ex. `maana-terra`)
3. **Advanced settings** → Python 3.12 ; **Secrets** (format TOML) :
   ```toml
   MODEL_API_KEY = "votre-clé-meta"
   APP_PASSWORD = "un-mot-de-passe-à-choisir"
   MAANA_MAX_ANALYSES_PER_DAY = "20"
   ```
   Les secrets de premier niveau sont exposés comme variables d'environnement ; ils ne sont jamais dans le dépôt.
4. **Deploy** (premier démarrage : quelques minutes, installation de rasterio etc.).

Protections : mot de passe obligatoire dès que `APP_PASSWORD` est défini ; plafond quotidien d'analyses payantes ;
au démarrage sur base vide, l'application charge une **analyse réelle préenregistrée** (`data/demo/demo_bundle.json`,
produite par `scripts/export_demo.py`) — aucune dépense pour la consulter.
Limite : le disque de Streamlit Cloud est éphémère (la base SQLite est réinitialisée à chaque redémarrage).

## 5. Sécurité
- `.env`, `kaggle.json`, `*.key` sont ignorés par git ; `.env.example` ne contient aucune valeur.
- `describe_secret_status()` n'expose que la présence et la longueur de la clé.
- Le bootstrap des notebooks clone en mode silencieux pour ne jamais afficher une URL contenant un jeton.
- Avant chaque commit : `git grep -nI "MODEL_API_KEY=" -- ':!.env.example'` doit être vide.
