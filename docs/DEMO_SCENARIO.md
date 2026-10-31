# Scénario de démonstration (3–5 minutes)

**Préparation** (avant le jury) : `python scripts/check_meta_access.py` doit être vert ;
`python scripts/seed_demo.py` (parcelle de démo) ; `streamlit run frontend/app.py`.
Prévoir un **run de secours** enregistré (`scripts/seed_demo.py --analyze` exécuté juste avant) en cas de réseau lent.

| Temps | Écran | Message clé |
|---|---|---|
| 0:00–0:30 | **Tableau de bord** | « Chaque source est vérifiée en direct : Meta (Muse Spark + SAM 3.1), NASA POWER, Sentinel-2, base agronomique. Aucune statistique fictive. » |
| 0:30–1:00 | **Parcelles** | Montrer la parcelle de démo (Soubré, coordonnées de démonstration) sur la carte. Préciser que la photo provient de CocoaSwolSet (CC BY 4.0) et n'a pas été prise à cet endroit. |
| 1:00–1:40 | **Nouvelle observation** | Sélectionner `cocoa_pod_Diseased.jpg`, organe « Cabosse », symptômes « taches brunes, trous ». Lancer l'analyse : expliquer les étapes (qualité → SAM → météo → satellite → fusion → Muse Spark → validation). |
| 1:40–2:20 | Rapport — en-tête + images | Niveau de prudence **calculé par le moteur** ; à droite, zones localisées par SAM 3.1 : « SAM dit *où*, pas *quoi* ». |
| 2:20–3:10 | Onglet **Hypothèses** | Hypothèses compatibles (jamais « confirmées »), preuves favorables/contradictoires avec leur source, cohérence météo par hypothèse, **contradictions** (règles déterministes vs modèle), informations manquantes, vérifications terrain. Point fort : la cabosse étiquetée « swollen shoot » dans le dataset montre des lésions qui évoquent aussi pourriture brune/mirides → le système doit le dire plutôt que trancher. |
| 3:10–3:50 | Onglet **Contexte** | Pluie journalière NASA POWER (estimation maillée), ratio à la normale, NDVI/NDMI Sentinel-2 avec scènes rejetées pour nuages. Rappeler : le satellite caractérise la parcelle, il ne diagnostique pas une feuille. |
| 3:50–4:20 | Onglets **Références** et **Traçabilité** | Recommandations rattachées à une fiche sourcée (sinon marquées « non documentées »), références avec URL et date, flags de validation, provenance, tokens/coût, JSON téléchargeable conforme au schéma. |
| 4:20–5:00 | **Évaluation** | Audit réel du dataset (licence, doublons, faute « Diseaded », déséquilibre) et protocole E0–E4 ; présenter uniquement les résultats d'un run réel, avec leurs limites (localisation proxy, effectif). |

**Phrase de conclusion** : « Maana Terra ne remplace pas l'agronome : il assemble des preuves traçables, montre leurs
contradictions et dit ce qu'il faut vérifier sur le terrain. »
