# Plan de réalisation

Légende : ✅ fait et testé · 🟡 fait, validation réelle en attente · ⛔ bloqué · ⬜ à faire

## Phase 0 — Audit initial ✅
Dépôt vide constaté ; environnement, réseau et accès vérifiés ; documents de référence créés.

## Phase 1 — Vérification des modèles Meta 🟡
- ✅ Contrats officiels relevés (docs dev.meta.ai : models, image-understanding, structured-output, reasoning,
  responses schemas, sam/segmenting, sam/reading-segmentation, sam/client-libraries, pricing-rate-limits).
- ✅ Adaptateurs indépendants Muse Spark / SAM 3.1 + client HTTP commun.
- ✅ Script `scripts/check_meta_access.py` (models, texte, image structurée, SAM) → `experiments/reports/meta_access_check.json`.
- ⛔ Exécution réelle : clé absente.

## Phase 2 — Fondations ✅
Structure, configs, schémas, SQLite, validations, stockage d'images, tests.

## Phase 3 — Moteur visuel ✅/🟡
Import, contrôle qualité (✅), SAM (🟡), Muse Spark (🟡), structuration et erreurs (✅).

## Phase 4 — Sources contextuelles ✅
NASA POWER (✅ réel) → base de connaissances (✅) → Sentinel-2 (✅ réel).

## Phase 5 — Fusion ✅
Normalisation, contexte typé, contradictions (règles + modèle), validation, prudence, rapport.

## Phase 6 — Évaluation ✅/⛔
Audit dataset (✅ réel), métriques (✅), exécuteur E0–E4 (✅), runs (⛔ clé).

## Phase 7 — Interface ✅
Streamlit : tableau de bord, parcelles (carte), nouvelle observation, historique/rapport, évaluation.

## Phase 8 — Stabilisation 🟡
✅ tests, sécurité (gitignore, redaction, scan), documentation, scénario de démo.
⬜ test de démonstration complet avec Muse Spark réel.

## Tâches restantes priorisées
1. **P0** — Valider Muse Spark avec la clé (`check_meta_access.py`), ajuster si besoin.
2. **P1** — Valider SAM 3.1 (identifiant `sam-3.1` vs `sam-3-1`, data URL) ; ajuster les prompts SAM sur des images réelles.
3. **P1** — Ablation E0–E4 sur ≥ 80 images (`run_dataset_audit.py --download 10` puis `run_ablation.py`).
4. **P1** — Compléter la base (CNRA/ANADER, CABI : P. megakarya, mirides, anthracnose).
5. **P2** — McNemar E0/E4, évolution temporelle par parcelle, polygones de parcelle dans l'UI.
