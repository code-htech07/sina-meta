# Ablation E0–E4 — run 20261031-155838

- Échantillons : 8 ({'healthy': 4, 'diseased': 4})
- Modèle : muse-spark-1.3 — prompt 2026-10-31.1
- Localisation du contexte (PROXY) : [6.8276, -5.2767]

| Exp | n | abstention | accuracy | balanced acc. | macro-F1 | erreurs critiques | CSSVD top-1 | hallucinations/rapport | coût moyen $ |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 8 | 0.0 | 0.625 | 0.625 | 0.619 | 0.25 | 0.0 | 0.75 | 0.01607 |
| E1 | 8 | 0.0 | 0.625 | 0.625 | 0.5636 | 0.0 | 0.0 | 0.75 | 0.01707 |
| E2 | 8 | 0.0 | 0.5 | 0.5 | 0.4667 | 0.25 | 0.0 | 0.625 | 0.01828 |
| E3 | 8 | 0.0 | 0.625 | 0.625 | 0.5636 | 0.0 | 0.0 | 1.0 | 0.01954 |
| E4 | 8 | 0.0 | 0.875 | 0.875 | 0.873 | 0.0 | 0.5 | 0.625 | 0.0271 |

## Limites
- Le dataset ne fournit pas la localisation des images : le contexte météo/satellite est celui d'une localisation PROXY fournie par l'opérateur.
- Les dates sont déduites des noms de fichiers (date de prise ou de partage WhatsApp) quand c'est possible.
- Effectif réduit : les écarts entre expériences doivent être interprétés avec prudence (pas de test de significativité à ce stade).
