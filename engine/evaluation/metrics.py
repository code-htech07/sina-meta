"""Evaluation metrics (dependency-free, unit-tested).

Prediction space for the binary CocoaSwolSet task: ``healthy`` / ``diseased`` /
``abstain`` (the system declined: unusable image or cannot_assess). Classification
metrics are computed on non-abstained predictions and reported alongside coverage.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable

from engine.schemas import AnalysisReport

LABELS = ("healthy", "diseased")


def report_to_prediction(report: AnalysisReport) -> str:
    if report.status == "failed":
        return "error"
    if report.status == "needs_new_photo" or report.overall_assessment == "cannot_assess":
        return "abstain"
    if report.consensus is not None:
        return {"anomaly_likely": "diseased", "no_anomaly_likely": "healthy"}.get(report.consensus.final, "abstain")
    return "diseased" if report.overall_assessment == "abnormality_detected" else "healthy"


def confusion(y_true: list[str], y_pred: list[str], labels: Iterable[str] = LABELS) -> dict[str, dict[str, int]]:
    cols = list(labels) + ["abstain"]
    m = {t: {p: 0 for p in cols} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in m and p in m[t]:
            m[t][p] += 1
    return m


def classification_metrics(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = LABELS) -> dict[str, Any]:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "error"]
    n_total = len(pairs)
    answered = [(t, p) for t, p in pairs if p in labels]
    out: dict[str, Any] = {
        "n_evaluated": n_total,
        "n_errors": sum(1 for p in y_pred if p == "error"),
        "n_answered": len(answered),
        "abstention_rate": round(1 - len(answered) / n_total, 4) if n_total else None,
        "confusion_matrix": confusion([t for t, _ in pairs], [p for _, p in pairs], labels),
    }
    if not answered:
        return out
    acc = sum(t == p for t, p in answered) / len(answered)
    per_class = {}
    recalls = []
    for c in labels:
        tp = sum(1 for t, p in answered if t == c and p == c)
        fp = sum(1 for t, p in answered if t != c and p == c)
        fn = sum(1 for t, p in answered if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(1 for t, _ in answered if t == c)
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": support}
        if support:
            recalls.append(rec)
    diseased_total = sum(1 for t, _ in pairs if t == "diseased")
    missed = sum(1 for t, p in pairs if t == "diseased" and p == "healthy")
    out.update({
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "per_class": per_class,
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(per_class), 4),
        # critical error = a diseased plant reported as showing no visible problem
        "critical_error_rate": round(missed / diseased_total, 4) if diseased_total else None,
    })
    return out


def quality_metrics(reports: list[AnalysisReport]) -> dict[str, Any]:
    done = [r for r in reports if r.status != "failed"]
    flags = Counter(f.code for r in done for f in r.validation_flags)
    n = len(done) or 1
    rule_contra = [r for r in done if any(c.origin == "rule" for c in r.contradictions)]
    ignored = [r for r in rule_contra if not any(c.origin == "model" for c in r.contradictions)]
    recs = [rec for r in done for rec in r.recommendations]
    return {
        "n_reports": len(done),
        "hallucinated_source_items_per_report": round(flags["hallucinated_source"] / n, 4),
        "unsourced_claims_per_report": round(flags["unsourced_claim"] / n, 4),
        "categorical_language_rate": round(sum(1 for r in done if any(f.code == "categorical_language" for f in r.validation_flags)) / n, 4),
        "unknown_problem_id_per_report": round(flags["unknown_problem_id"] / n, 4),
        "undocumented_recommendation_share": round(sum(1 for x in recs if not x.documented) / len(recs), 4) if recs else None,
        "reports_with_rule_contradiction": len(rule_contra),
        "contradictions_ignored_by_model_rate": round(len(ignored) / len(rule_contra), 4) if rule_contra else None,
        "caution_levels": dict(Counter(r.caution_level for r in done)),
        "flag_counts": dict(flags),
    }


def cost_latency(reports: list[AnalysisReport]) -> dict[str, Any]:
    lat = [sum(t.latency_ms for t in r.timings) for r in reports if r.timings]
    muse_lat = [t.latency_ms for r in reports for t in r.timings if t.step == "muse_spark_analysis" and t.status == "ok"]
    costs = [sum(u.estimated_cost_usd or 0 for u in r.model_usage) for r in reports]
    tokens_in = sum(u.input_tokens for r in reports for u in r.model_usage)
    tokens_out = sum(u.output_tokens for r in reports for u in r.model_usage)

    def pct(v: list[float], q: float) -> float | None:
        if not v:
            return None
        s = sorted(v)
        return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 1)

    return {
        "pipeline_latency_ms_mean": round(statistics.mean(lat), 1) if lat else None,
        "pipeline_latency_ms_p50": pct(lat, 0.5),
        "pipeline_latency_ms_p95": pct(lat, 0.95),
        "muse_latency_ms_mean": round(statistics.mean(muse_lat), 1) if muse_lat else None,
        "estimated_cost_usd_total": round(sum(costs), 4),
        "estimated_cost_usd_per_analysis": round(sum(costs) / len(costs), 5) if costs else None,
        "input_tokens_total": tokens_in,
        "output_tokens_total": tokens_out,
    }


def disease_specific(reports: list[AnalysisReport], y_true: list[str], target_problem: str = "cssvd") -> dict[str, Any]:
    """Among truly diseased samples: is the target problem the top primary hypothesis / among primaries?"""
    pairs = [(r, t) for r, t in zip(reports, y_true) if t == "diseased" and r.status != "failed"]
    if not pairs:
        return {"n_diseased": 0}
    top1 = sum(1 for r, _ in pairs if r.primary_hypotheses and r.primary_hypotheses[0].problem_id == target_problem)
    anyp = sum(1 for r, _ in pairs if any(h.problem_id == target_problem for h in r.primary_hypotheses))
    healthy_false = sum(1 for r, t in zip(reports, y_true) if t == "healthy" and r.status != "failed"
                        and any(h.problem_id == target_problem for h in r.primary_hypotheses))
    n_healthy = sum(1 for r, t in zip(reports, y_true) if t == "healthy" and r.status != "failed")
    return {
        "target_problem": target_problem,
        "n_diseased": len(pairs),
        "top1_hit_rate": round(top1 / len(pairs), 4),
        "in_primary_hypotheses_rate": round(anyp / len(pairs), 4),
        "false_target_rate_on_healthy": round(healthy_false / n_healthy, 4) if n_healthy else None,
    }


def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> dict[str, Any]:
    """Exact (binomial) two-sided McNemar test on paired correctness of two systems."""
    from math import comb

    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if not x and y)
    n = b + c
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "discordant": n, "p_value": round(p, 4)}
