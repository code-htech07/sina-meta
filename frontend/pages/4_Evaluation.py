import sys  # noqa: I001
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # frontend/ on path

from common import page_header  # noqa: I001

import json

import pandas as pd
import streamlit as st

from engine.config import ROOT

page_header("Évaluation scientifique")
st.markdown("Résultats **mesurés** uniquement : audit du dataset et runs d'ablation E0–E4 présents dans `experiments/`. "
            "Aucun chiffre n'est affiché s'il n'a pas été produit par un run réel.")

audit_path = ROOT / "experiments" / "reports" / "dataset_audit_cocoaswolset.json"
st.subheader("Audit du dataset CocoaSwolSet")
if audit_path.exists():
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    c = st.columns(4)
    c[0].metric("Fichiers", audit["total_files"])
    c[1].metric("Images uniques analysables", audit["analysable_unique_images"])
    c[2].metric("Doublons exacts", audit["exact_duplicate_extra_files"])
    c[3].metric("Ratio déséquilibre", audit["imbalance_ratio_majority_minority"])
    st.bar_chart(pd.Series(audit["per_part_and_class_unique_analysable"]), height=220)
    st.markdown(f"Licence : **{audit['licence']}** · DOI {audit['doi']}")
    for lim in audit["limitations"]:
        st.caption("• " + lim)
else:
    st.info("Audit non exécuté : `python scripts/run_dataset_audit.py`.")

st.subheader("Ablation E0–E4")
runs = sorted(list((ROOT / "experiments" / "reports").glob("ablation*/summary.json"))
              + list((ROOT / "experiments" / "ablation" / "runs").glob("*/summary.json")), key=lambda p: p.parent.name, reverse=True)
if not runs:
    st.warning("Aucun run d'ablation disponible. Il nécessite la clé Meta : "
               "`python scripts/run_ablation.py --manifest ... --proxy-lat ... --proxy-lon ... --fallback-date ...`")
    st.stop()
run = st.selectbox("Run", runs, format_func=lambda p: p.parent.name)
summary = json.loads(run.read_text(encoding="utf-8"))
rows = []
for k, v in summary["experiments"].items():
    cl, q, cost, d = v["classification"], v["quality"], v["cost_latency"], v["cssvd_specific"]
    rows.append({"exp": k, "n": cl.get("n_evaluated"), "abstention": cl.get("abstention_rate"), "accuracy": cl.get("accuracy"),
                 "balanced_acc": cl.get("balanced_accuracy"), "macro_F1": cl.get("macro_f1"),
                 "erreurs critiques": cl.get("critical_error_rate"), "CSSVD top-1": d.get("top1_hit_rate"),
                 "hallucinations/rapport": q.get("hallucinated_source_items_per_report"),
                 "contradictions ignorées": q.get("contradictions_ignored_by_model_rate"),
                 "latence moy. (ms)": cost.get("pipeline_latency_ms_mean"), "coût/analyse ($)": cost.get("estimated_cost_usd_per_analysis")})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if summary.get("paired_tests_mcnemar_exact"):
    st.markdown("**Tests appariés (McNemar exact, vs E0)** — p < 0,05 requis pour conclure")
    st.dataframe(pd.DataFrame(summary["paired_tests_mcnemar_exact"]).T, width="stretch")
report_md = run.parent / "REPORT.md"
if report_md.exists():
    with st.expander("Rapport et interprétation"):
        st.markdown(report_md.read_text(encoding="utf-8"))
st.json({k: summary[k] for k in ("n_samples", "label_distribution", "muse_model", "prompt_version", "context_location",
                                  "context_location_is_proxy", "limitations")}, expanded=False)
