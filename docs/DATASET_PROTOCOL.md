# Protocole d'utilisation des datasets

## Checklist avant usage (appliquée à CocoaSwolSet)
| Critère | CocoaSwolSet v2 | Méthode |
|---|---|---|
| Origine | INP-HB / ESATIC, projet Deep Farm (Erasmus+), Côte d'Ivoire | Métadonnées Mendeley |
| Licence | CC BY 4.0 (attribution requise, modifications à signaler) | `data_licence` API Mendeley |
| Classes | Healthy / Diseased (swollen shoot) × Leaves-Young, Leaves-Adults, Trees, cocoa pod | Arborescence des dossiers |
| Volume | 2 094 fichiers, 7,73 Go ; 1 694 images uniques analysables (JPEG/PNG) | `scripts/run_dataset_audit.py` |
| Doublons | 10 groupes de doublons exacts (11 fichiers en trop, SHA-256) ; 159 prises en double format (RAW+JPEG) ; aucun doublon inter-classes | SHA-256 publiés par Mendeley |
| Déséquilibre | 968 sains / 726 malades (ratio 1,33) ; très déséquilibré par organe (cabosses malades : 28) | Comptage par strate |
| Annotations | Diagnostic visuel par agronomes ; pas de confirmation PCR mentionnée ; pas de localisation des symptômes | Description du dataset |
| Anomalies | Dossier `Leaves/Adults/Diseaded` (faute de frappe) ; nombreuses images transitées par WhatsApp (EXIF supprimé, recompression) | Audit + noms de fichiers |
| Représentativité | Pas de GPS ni de date par image ; zone de collecte non précisée ; dataset « constamment mis à jour » (version figée : v2) | — |

Résultat machine : `experiments/reports/dataset_audit_cocoaswolset.json`.

## Règles d'échantillonnage
- Dédoublonnage par SHA-256 ; formats retenus : JPEG/PNG (DNG/HEIC exclus faute de conversion validée).
- Échantillon **stratifié** par (organe, classe), graine fixe (`--seed 42`), fichiers ≤ 4 Mo de préférence.
- Le manifeste conserve : fichier, étiquette, organe, chemin source, SHA-256 vérifié, licence, attribution.
- Les images redistribuées dans `data/samples/` sont redimensionnées (≤ 1280 px) : modification déclarée dans le manifeste.

## Limites à rappeler dans toute communication de résultats
1. Tâche binaire « swollen shoot vs sain » : ne mesure pas la capacité à distinguer pourriture brune, mirides, etc.
2. Pas de vérité terrain contextuelle : le contexte météo/satellite des expériences est une **localisation proxy**.
3. Étiquettes visuelles non confirmées en laboratoire : les erreurs d'annotation sont possibles.
4. Une image de cabosse « Diseased » peut montrer des lésions évoquant d'autres causes (cf. `cocoa_pod_Diseased.jpg`).
