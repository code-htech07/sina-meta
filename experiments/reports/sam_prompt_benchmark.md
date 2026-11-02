# Benchmark des prompts SAM 3.1 (API réelle, 2026-10-31)

3 images CocoaSwolSet (redimensionnées ≤ 1600 px, envoyées en data URL), 13 prompts, `mask_encoding=one_bit`,
`include_confidence=true`. Valeurs : (fraction de l'image couverte par le masque, score du modèle).

| Prompt | cocoa_pod_Diseased | Leaves-Young_Diseased | Leaves-Adults_Healthy |
|---|---|---|---|
| fruit | 3 régions (0,236/0,89 ; 0,045/0,88 ; 0,021/0,84) | — | — |
| pod | — | — | — |
| leaf | — | 7 régions (feuille principale 0,241/0,98) | 8 régions (0,376/0,95) |
| plant leaf | — | 6 | 6 |
| brown spot | 4 petites régions (0,002–0,006 ; 0,52–0,80) | — | — |
| dark spot | 5 (≤0,006 ; 0,51–0,64) | — | — |
| lesion | — | — | 1 (0,002/0,53) |
| stain | 2 (0,69 ; 0,60) | — | — |
| hole | 2 (0,71 ; 0,51) | — | — |
| rot | — | — | — |
| discolored area | 3 (recouvre les fruits ; 0,53–0,65) | 1 (feuille entière, 0,69) | — |
| yellow area | — | — | — |
| tree trunk | 1 (0,254/0,90) | — | 3 (≤0,57) |

Constats :
- « cocoa pod », « pod », « rot » : aucune détection sur une cabosse évidente → vocabulaire générique préférable.
- Les prompts d'organe sont fiables (« fruit », « leaf », scores 0,84–0,98) et spécifiques (aucun « fruit » sur les feuilles).
- Les prompts d'anomalie détectent de petites zones sur la cabosse malade et rien sur la feuille saine, mais ne
  détectent pas les symptômes diffus de la jeune feuille malade (swollen shoot) : SAM localise des taches, pas une chlorose diffuse.
- Les masques renvoyés sont **locaux à la boîte** (`data=H,W` = taille de la boîte) ; le parseur officiel fournit `bounds`.

Choix retenu (`configs/models.yaml`) : organes `fruit`, `leaf` ; anomalies `brown spot`, `discolored area`, `hole` ;
seuil de score 0,5. Coût : 5 appels × 0,0025 $ = 0,0125 $ par image. Effectif très faible (3 images) : choix provisoire.
