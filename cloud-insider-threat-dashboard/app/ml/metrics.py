"""
Shared evaluation metrics + threshold tuning.

Every experiment in the evaluation plan (architecture.md section 5) reports
the same four selection metrics -- recall, precision, F1, PR-AUC -- plus
accuracy for reference only. Accuracy is *not* used for any selection
decision: under heavy class imbalance a model that flags nobody scores
>95% accuracy.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
)

METRIC_NAMES = ["recall", "precision", "f1", "pr_auc", "accuracy"]


def pr_auc(y_true, score) -> float:
    """Average precision (area under the precision-recall curve). NaN if no positives."""
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, np.asarray(score, dtype=float)))


def tune_threshold(y_true, score, min_recall: float = 0.9) -> float:
    """
    Picks a decision threshold on a *validation* set, prioritizing recall
    (a missed insider costs more than a false alert): among all thresholds
    that reach at least `min_recall`, take the one with the highest
    precision (ties -> the higher threshold, i.e. fewer alerts).

    Works on any real-valued score (LSTM probability, SVM margin, anomaly
    score). Prediction rule is `score >= threshold`. The lowest threshold
    always reaches recall 1.0, so a valid choice always exists.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        raise ValueError("Cannot tune a threshold: the validation set has no positive samples.")
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]   # align with `thresholds`
    ok = np.flatnonzero(recall >= min_recall)
    best = ok[np.lexsort((thresholds[ok], precision[ok]))[-1]]
    return float(thresholds[best])


@dataclass
class ThresholdPolicy:
    """
    How the alert threshold is chosen on the VALIDATION set.

      budget     flag the top `budget` fraction of validation items (windows or
                 users). Depends on the score distribution of the many normal
                 items, not on the handful of positives, so it transfers to new
                 data far better than a recall target. Think "analyst capacity".
      precision  highest-recall threshold whose validation precision >= min_precision
                 (falls back to the best-F1 threshold if none reaches it)
      fbeta      threshold maximizing F-beta (beta < 1 favours precision)
      recall     legacy rule: best precision subject to recall >= min_recall
    """
    mode: str = "budget"
    budget: float = 0.01
    min_precision: float = 0.2
    min_recall: float = 0.9
    beta: float = 1.0

    @classmethod
    def coerce(cls, p) -> "ThresholdPolicy":
        if p is None:
            return cls()
        return p if isinstance(p, cls) else cls(**p)


def choose_threshold(y_true, score, policy: Optional[ThresholdPolicy] = None) -> float:
    """Pick a decision threshold on validation data according to `policy`. Prediction rule: score >= threshold."""
    pol = ThresholdPolicy.coerce(policy)
    s = np.asarray(score, dtype=float)
    y = None if y_true is None else np.asarray(y_true).astype(int)
    if pol.mode == "recall":
        return tune_threshold(y, s, pol.min_recall)
    if pol.mode == "budget":
        if not 0 < pol.budget < 1:
            raise ValueError("budget must be in (0, 1)")
        n_flag = max(1, int(round(pol.budget * len(s))))
        return float(np.sort(s)[::-1][n_flag - 1])               # the n_flag-th highest score is the cut-off
    if y is None or y.sum() == 0:
        raise ValueError("Cannot tune a threshold: the validation set has no positive samples.")
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]
    beta2 = pol.beta ** 2
    f = (1 + beta2) * precision * recall / np.maximum(beta2 * precision + recall, 1e-12)
    if pol.mode == "precision":
        ok = np.flatnonzero(precision >= pol.min_precision)
        if len(ok):
            return float(thresholds[ok[np.argmax(recall[ok])]])
        return float(thresholds[int(np.argmax(f))])
    if pol.mode == "fbeta":
        return float(thresholds[int(np.argmax(f))])
    raise ValueError(f"unknown threshold mode '{pol.mode}'")


def precision_recall_at_k(y_true, score, ks: Iterable[int]) -> Dict[str, float]:
    """Threshold-free view: if an analyst reviews the K highest-scored items, how many are real?"""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    out = {}
    for k in ks:
        k = int(min(k, len(y)))
        hit = int(y[order[:k]].sum())
        out[f"precision@{k}"] = round(hit / max(k, 1), 4)
        out[f"recall@{k}"] = round(hit / max(int(y.sum()), 1), 4)
    return out


def metrics_at_budget(y_true, score, frac: float) -> Dict[str, float]:
    """Flag the top `frac` of THIS set by score -- no threshold carried over from validation, so no drift."""
    y, sc = np.asarray(y_true).astype(int), np.asarray(score, dtype=float)
    k = max(1, int(round(frac * len(y))))
    tp = int(y[np.argsort(-sc, kind="stable")[:k]].sum())
    return {"recall": round(tp / max(int(y.sum()), 1), 4), "precision": round(tp / k, 4), "n_flagged": k, "true_flagged": tp}


def compute_metrics(y_true, score, threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    return {
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "pr_auc": round(pr_auc(y, s), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_flagged": int(pred.sum()),
    }


def aggregate(runs: Iterable[Dict[str, float]], keys: List[str] = None) -> Dict[str, Dict[str, float]]:
    """mean +/- std across seeds (NaN-safe). Std is the sample std when n > 1."""
    runs = list(runs)
    keys = keys or METRIC_NAMES
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if r.get(k) is not None], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = {"mean": None, "std": None, "n": 0}
        else:
            out[k] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
                "n": int(len(vals)),
            }
    return out


def to_jsonable(obj):
    """Recursively converts numpy scalars/arrays and NaN/inf to JSON-safe values
    (Postgres JSON columns reject NaN). Keys starting with '_' are dropped --
    they carry in-memory-only arrays."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj