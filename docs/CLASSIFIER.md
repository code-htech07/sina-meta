# Classifieur spécialisé (architecture « hybride ») — 2 novembre 2026

## Pourquoi
L'évaluation sur 80 images a montré que Muse Spark seul, utilisé sans entraînement (zero-shot), plafonne à 58–67 %
de bonnes réponses sur la question « sain vs swollen shoot », avec de nombreux faux positifs, et qu'aucun
contexte n'apporte de gain significatif. Nous disposons pourtant de 1 692 photos ivoiriennes **étiquetées par des
agronomes** (CocoaSwolSet, CC BY 4.0). Un modèle **entraîné** sur ces données répond à cette question précise bien mieux
qu'un modèle généraliste ; Muse Spark garde le rôle de raisonnement (autres hypothèses, contexte, recommandations).

## Architecture
```
photo ─┬─► DINOv2-small (Meta, Apache-2.0, ONNX) ─► embedding ─► régression logistique calibrée ─► p(swollen shoot)
       │                                                    └─► garde « hors domaine » (similarité au jeu d'entraînement)
       ├─► SAM 3.1 (localisation) ─┐
       └───────────────────────────┴─► Muse Spark (raisonnement + contexte + base agronomique + sortie du classifieur)
                                                   │
                     décision de consensus (règle H2) : on conclut seulement si classifieur ET Muse Spark concordent ;
                     sinon « incertain — vérification terrain » + contradiction R6 + prudence élevée
```
- Inférence 100 % locale et gratuite (~0,3 s/photo sur CPU, ONNX Runtime, sans PyTorch) ; modèle téléchargé une fois
  depuis Hugging Face (`onnx-community/dinov2-small`, révision figée) ; poids du classifieur : `data/models_trained/`.
- Code : `engine/vision/{embedder,classifier,service}.py` ; entraînement : `scripts/train_classifier.py` ;
  évaluation hybride : `scripts/eval_hybrid.py`.

## Protocole (anti-fuite)
1. 1 692 images uniques (2 PNG corrompus dans le dataset publié, exclus).
2. **Quasi-doublons regroupés** (pHash, distance de Hamming ≤ 8) → 1 546 groupes ; un groupe n'est jamais à la fois en
   entraînement et en test.
3. **Les 80 images de l'évaluation Muse Spark sont forcées en test** (comparaison sur les mêmes images).
4. Test = 407 images ; entraînement = 1 285. Sélection des hyperparamètres (C, ajout de l'organe) par validation croisée
   groupée 5 plis **sur l'entraînement uniquement** ; calibration (Platt), zone d'abstention [0,39 ; 0,61], seuil hors
   domaine et fiabilité par organe calculés sur les prédictions hors-pli de l'entraînement.
5. Évaluation unique sur le test, IC à 95 % par bootstrap.

## Résultats (test, 407 images jamais vues) — modèle retenu
| Mesure | DINOv2-small, **sans l'organe en entrée** (retenu) | avec l'organe en entrée | DINOv2-base (avec organe) |
|---|---|---|---|
| Exactitude | **88,9 %** (IC95 85,8–91,7) | 89,7 % (86,5–92,6) | 90,4 % (87,2–93,1) |
| Exactitude équilibrée | 88,1 % (84,8–91,2) | 88,8 % | 89,7 % |
| AUC | 0,946 | 0,946 | 0,957 |
| Rappel swollen shoot / sains | 81 % / 95,2 % | 81 % / 96,5 % | 84 % / 95,6 % |
| Calibration (ECE) | 0,051 | 0,058 | 0,050 |
| Avec abstention [0,35 ; 0,65] | 91,2 % sur 94,8 % des photos | — | — |

**Pourquoi « sans l'organe »** : lors du test réel du 2 novembre, une jeune feuille déclarée par erreur comme « cabosse »
a fait passer la probabilité de 0,80 à 0,01. Une erreur de saisie ne doit pas inverser une décision : l'organe déclaré
ne sert plus qu'à décider si le classifieur est jugé fiable (arbres exclus : fiabilité 0,68 à l'entraînement).
Coût : −0,8 point, non significatif. Small préféré à base : écart non significatif, modèle 4× plus léger.

Par organe (test) : jeunes feuilles 90,5 %, feuilles adultes 94,1 %, arbres 81,3 %, **cabosses 50,8 % (échec)** —
seulement 70 cabosses à l'entraînement ; la fiabilité estimée à l'entraînement (82 %) ne l'avait pas prédit.
**Les cabosses devraient être exclues** (constat fait sur le test : à valider sur des données nouvelles, sans réajuster
sur ces mêmes données).

## Comparaison avec Muse Spark (les mêmes 80 images, 10 par strate organe × classe)
| Système | Exactitude | Abstention | Malades déclarés sains |
|---|---|---|---|
| Muse Spark seul, image (E0) | 58,2 % | 1 % | 22,5 % |
| Muse Spark + contexte + base (E4) | 67,1 % | 1 % | 22,5 % |
| Classifieur seul | 76,3 % | 0 % | 40 % |
| H1 aiguillage | 77,2 % | 1 % | 35 % |
| **H2 consensus classifieur + E4** | **90,7 %** des cas tranchés (39/43) | 46 % | **7,5 %** |

Classifieur vs Muse E0 : McNemar p = 0,020 (significatif) ; vs E4 : p = 0,24 (non significatif).
Ces 80 images sur-représentent cabosses et arbres (où le classifieur est faible), d'où 76 % contre 89 % sur
l'ensemble du test.

**Choix retenu : règle H2 (consensus).** Elle ne tranche que lorsque les deux systèmes indépendants concordent et
renvoie les autres cas à une vérification humaine : mieux vaut « je ne sais pas » qu'une erreur. Dans le pipeline, H2
est un peu plus stricte que dans l'évaluation hors ligne (classifieur hors zone d'abstention et sur un organe fiable) :
la mesure exacte nécessite l'expérience E5 (≈ 2 $ d'appels Muse Spark).

## Limites
- Le classifieur ne connaît que « sain » vs « aspect swollen shoot » dans le domaine CocoaSwolSet (mêmes photographes,
  mêmes zones) : performance attendue plus faible sur des photos d'autres régions/appareils ; garde « hors domaine »
  imparfaite.
- Étiquettes visuelles non confirmées par PCR.
- Les 8 photos d'exemple du dépôt font partie de l'entraînement : elles ne servent qu'à la démonstration.
- Il ne détecte pas la pourriture brune, les mirides, etc. : ces hypothèses restent du ressort de Muse Spark + base.
