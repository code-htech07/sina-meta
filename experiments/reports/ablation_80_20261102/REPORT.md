# Ablation E0–E4 — 80 images CocoaSwolSet (2 novembre 2026)

- 80 images uniques (10 par strate organe × classe ; 40 saines, 40 swollen shoot), SHA-256 vérifiés, graine 42.
- `muse-spark-1.3`, prompt 2026-10-31.1, `reasoning.effort=medium`, SAM et déclarations désactivés (constants).
- Contexte météo/satellite : **localisation PROXY** (6,8276 ; −5,2767) — le dataset n'a pas de GPS.
- 400 appels réels + 4 relances (HTTP 502 de l'API sur 4 analyses E4). **Coût réel total : 7,77 $.**

| Exp | Contexte | Accuracy | Balanced acc. | Macro-F1 | Rappel sains | Rappel malades | Erreurs critiques | CSSVD hypothèse principale | Coût/analyse |
|---|---|---|---|---|---|---|---|---|---|
| E0 | image | 0,582 | 0,585 | 0,569 | 0,40 | 0,77 | 0,225 | 0 % | 0,017 $ |
| E1 | + météo | 0,608 | 0,610 | 0,595 | 0,43 | 0,79 | 0,200 | 0 % | 0,018 $ |
| E2 | + satellite | 0,671 | 0,674 | 0,655 | 0,45 | 0,90 | 0,100 | 0 % | 0,018 $ |
| E3 | + météo + satellite | 0,608 | 0,610 | 0,591 | 0,40 | 0,82 | 0,175 | 0 % | 0,019 $ |
| E4 | + base agronomique | 0,671 | 0,672 | 0,668 | 0,58 | 0,77 | 0,225 | 17,5 % | 0,026 $ |

Accuracy calculée sur les réponses non abstenues (1 abstention par expérience).
Erreurs critiques = plante malade déclarée « aucun problème visible ».

## Tests appariés (McNemar exact, même 80 images)
| Comparaison | Seul E0 correct | Seul l'autre correct | p |
|---|---|---|---|
| E0 vs E1 | 5 | 7 | 0,77 |
| E0 vs E2 | 2 | 9 | 0,065 |
| E0 vs E3 | 4 | 6 | 0,75 |
| E0 vs E4 | 5 | 12 | 0,14 |

**Aucune différence n'est statistiquement significative au seuil de 5 %.**

## Interprétation
1. **Performance de base modeste** (58–67 %) et **forte tendance aux faux positifs** : 42 à 60 % des plantes saines
   sont jugées anormales. Le système est prudent dans le sens « signaler », pas fiable comme filtre « plante saine ».
2. **E2 (satellite) a les meilleurs chiffres bruts mais l'effet ne peut pas être causal** : le contexte satellite est celui
   d'un lieu proxy sans lien avec les photos. L'écart E0–E2 (p = 0,065) relève vraisemblablement de la variabilité du
   modèle ou d'un effet du contexte sur son seuil de décision, pas d'une information utile. C'est un résultat important :
   **un contexte non pertinent peut modifier les réponses**.
3. **E4 (base agronomique)** : meilleur rappel des plantes saines (0,58), meilleur macro-F1, seul cas où CSSVD apparaît
   (17,5 % des malades en hypothèse principale, 0 faux CSSVD sur les sains), et niveaux de prudence plus nuancés
   (43 « standard », 21 « renforcée », 15 « élevée », contre ~69 « élevée » sur 80 en E1–E3). Gain non significatif.
4. **Swollen shoot difficile à reconnaître visuellement** : même avec la fiche documentée, CSSVD n'est proposé en premier
   que pour 7 malades sur 40 — cohérent avec la littérature (symptômes variables, diagnostic par PCR).
5. Aucune règle de contradiction ne s'est déclenchée (météo proxy « normale ») : la détection des contradictions n'est pas
   évaluée par ce protocole ; il faut des images géolocalisées.
6. Les éléments « source non fournie » sont à 0 dans toutes les expériences après la correction du validateur.

## Ce qu'il faudrait pour conclure
- Images **géolocalisées et datées** (photos de terrain réelles) pour que météo et satellite soient pertinents.
- Plusieurs répétitions par condition (variabilité du modèle), puis effectifs plus grands.
- Étiquettes confirmées (PCR) et d'autres maladies que le swollen shoot.

Fichiers : `summary.json` (métriques complètes), `predictions.csv` (400 prédictions).
