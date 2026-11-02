# Maana Terra — Project Brief

> Source de vérité initiale : le « Master Prompt » du porteur de projet (Meta AI Developer Hackathon).
> Ce document en est la synthèse persistante ; toute session future doit le lire en premier, puis `docs/STATUS.md`.

## Vision
Plateforme d'intelligence agricole **multimodale** pour le **cacao en Côte d'Ivoire** : rapprocher
observations visuelles (photos), météo, satellite, connaissances agronomiques documentées et données de
parcelle pour produire une analyse **contextualisée, explicable et prudente** — pas un chatbot qui décrit une photo.

## Problème
Un même symptôme visuel peut avoir plusieurs causes (maladie, ravageur, stress abiotique, désordre
physiologique). Le système doit exploiter le contexte pour éviter les conclusions simplistes et **signaler
les contradictions** plutôt que forcer un diagnostic.

## Modèles Meta (rôles non interchangeables)
| Modèle | Rôle | Ce qu'il ne fait pas |
|---|---|---|
| **Muse Spark 1.3** (`muse-spark-1.3`) | Analyse multimodale et raisonnement : symptômes visibles, hypothèses, confrontation au contexte, contradictions, informations manquantes | N'est pas une source de vérité ; ses scores ne sont pas calibrés |
| **SAM 3.1** (`sam-3.1`) | Segmentation : *où* se trouve la région (masques, boîtes) | Ne diagnostique rien |

## Périmètre MVP (P0 → P2)
- **P0** : Muse Spark fonctionnel, analyse d'image de cacao, rapport structuré, météo réelle, base agronomique documentée, interface de démo, dépôt reproductible.
- **P1** : satellite réel, fusion multimodale, SAM 3.1, historique des parcelles, ablation E0–E4.
- **P2** : analyse temporelle, cartographie avancée, optimisations.

## Règles non négociables (résumé)
Ne jamais inventer de source, de résultat, de segmentation ; jamais « confirmer » une maladie depuis une photo ;
jamais présenter un indice satellitaire comme preuve de maladie ni une sortie LLM comme mesure calibrée ;
jamais exposer la clé API ; jamais masquer une limite.

## Contraintes
Pas de GPU local ; calculs lourds via API ; **exécution cible : Kaggle Notebooks** (internet ON, secrets Kaggle) ;
GitHub = source centrale du code.
