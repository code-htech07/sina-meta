# E5 — pipeline complet hybride (classifieur + Muse Spark + consensus), 80 images, 2 novembre 2026

Mêmes 80 images que E0–E4 (10 par strate organe × classe). Configuration E4 + classifieur spécialisé (DINOv2-small,
sans organe en entrée) + règle de consensus H2. 80 appels réels, 0 échec, **coût réel 2,31 $**.

| Système (mêmes 80 images) | Bonnes réponses | Abstention | Malades déclarés sains | Fausses alertes sur sains |
|---|---|---|---|---|
| Muse Spark seul (E0) | 58,2 % | 1 % | 22,5 % | 24/40 |
| Muse Spark + contexte + base (E4) | 67,1 % | 1 % | 22,5 % | 17/40 |
| Muse Spark dans E5 (voit le classifieur) | 73,4 % | 1 % | 12,5 % | — |
| **E5 — décision de consensus** | **91,2 %** des cas tranchés (31/34) | 57,5 % | **7,5 %** (3/40) | **0/40** |

- Muse Spark progresse quand il reçoit la sortie du classifieur (67 → 73 %), mais l'écart n'est pas significatif
  (McNemar p = 0,18). Le swollen shoot est son hypothèse principale pour 14 malades sur 40 (contre 7 en E4).
- Le consensus ne produit **aucune fausse alerte** sur les 40 plantes saines et 3 malades manqués sur 40.
- Prix de cette fiabilité : **57,5 % de cas renvoyés en vérification**, dont les 20 photos d'arbres entiers
  (le classifieur n'est pas jugé fiable sur cet organe) et 16 cabosses sur 20.
- Par organe (tranchés) : feuilles adultes 13/15, jeunes feuilles 14/15, cabosses 4/4, arbres 0 tranché.

Limites : 80 images d'un seul dataset ; images de test du classifieur mais même domaine que l'entraînement ;
étiquettes visuelles ; contexte météo/satellite proxy. Fichiers : `summary.json`, `predictions.csv`.
