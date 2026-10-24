"""
Layer-2 detector: gradient boosting (HistGradientBoostingClassifier). This is THE production Layer-2 model.

Why: on CERT r4.2 (7-fold user-level out-of-fold, 1000 users / 70 insiders) HGB on last / max / mean summaries of the
T-window history beat the LSTM, the BiLSTM-attention and every blend of them (user PR-AUC 0.976 vs 0.937 / 0.939 /
0.939 stack; window PR-AUC 0.855 vs 0.631). The stacked ensemble was significantly WORSE than HGB alone, so the
sequence models are kept only as an ablation (app/ml/ensemble.py, notebook section 7c).

Layer 1 (the sampler) is unchanged: the winning spec is applied to the TRAIN split only, then HGB is fitted on it.

  train_production_hgb   fit on one user-level split, tune window + user thresholds on VALIDATION, report TEST, save artifact
  HGBDetector            load an artifact and score windows / users
  run_oof_hgb            k-fold user-level out-of-fold scores for EVERY user (main evidence)
  budget_table           pooled precision / recall / F1 when the top X% of users of each fold are flagged
  matched_recall         precision / F1 at the smallest budget that reaches a target recall (paper comparison)
  importance_table       permutation importance per behavioural feature (explainability)
"""
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from app.ml.metrics import ThresholdPolicy, choose_threshold, compute_metrics, pr_auc, to_jsonable
from app.ml.sampling import kwargs_for, resample
from app.ml.sequences import flatten
from app.ml.user_level import DEFAULT_AGG, aggregate_user_scores, evaluate_user_level

# Reference numbers reported by the selected paper (Asha, Shanmugapriya & Padmavathi 2023): NM-2 + One-Class SVM.
PAPER = {"recall": 1.0, "f1": 0.7872, "accuracy": 0.8246}
PAPER["precision_implied"] = round(PAPER["f1"] * PAPER["recall"] / (2 * PAPER["recall"] - PAPER["f1"]), 3)   # ~0.649


def _summary(X: np.ndarray) -> np.ndarray:
    from app.ml.ensemble import summarize_history
    return summarize_history(X)


def summary_columns(feature_columns: Sequence[str]) -> list:
    """Column names of summarize_history(): last-window, max and mean of every behavioural feature."""
    return [f"{c}__{s}" for s in ("last", "max", "mean") for c in feature_columns]


# ------------------------------------------------------------------ production model (one split)

def train_production_hgb(prep, sampler: str, seed: int, window_policy=None, user_policy=None,
                         user_agg: Optional[dict] = None, artifact_dir: Optional[str] = None,
                         val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Fits HGB on the winning sampler's output for one user-level split. Both thresholds (window + user) are tuned on
    VALIDATION and frozen before TEST is touched. Returns (artifact, metrics, split_data).
    """
    from app.ml import ensemble
    from app.ml.train import DEFAULT_USER_POLICY, DEFAULT_WINDOW_POLICY, make_split
    T, F = prep.seq_len, len(prep.seq.feature_columns)
    wpol = ThresholdPolicy.coerce(window_policy) if window_policy is not None else DEFAULT_WINDOW_POLICY
    upol = ThresholdPolicy.coerce(user_policy) if user_policy is not None else DEFAULT_USER_POLICY
    agg = dict(user_agg or DEFAULT_AGG)
    sd = make_split(prep, seed, val_frac, test_frac)
    rr = resample(sampler, flatten(sd.X_train), sd.y_train, seed=seed, **kwargs_for(sampler, None))
    preds = ensemble.fit_members(sd, rr, ["hgb"], None, seed, 1, T, F, keep_models=True)
    clf, pv, pt = preds["hgb"]["models"][0], preds["hgb"]["val"], preds["hgb"]["test"]
    thr = choose_threshold(sd.y_val, pv, wpol)                           # tuned on validation, then frozen
    methods = {"max": {"method": "max"}, "topk": {"method": "topk", "k": 3}, "consecutive": {"method": "consecutive", "n": 3}}
    methods[agg["method"]] = agg
    user_res = evaluate_user_level(prep, sd.idx_val, sd.idx_test, pv, pt, upol, methods)
    chosen = user_res[agg["method"]]
    metrics = {
        "window_policy": wpol.__dict__, "user_policy": upol.__dict__, "user": user_res, "user_selected": agg["method"],
        "val": compute_metrics(sd.y_val, pv, thr), "test": compute_metrics(sd.y_test, pt, thr),
        "threshold": round(float(thr), 6), "train_size_after_sampling": int(len(rr.y)), "sampler_info": rr.info,
        "split": {"train_users": int(len(sd.split.train)), "val_users": int(len(sd.split.val)), "test_users": int(len(sd.split.test))},
    }
    artifact = {
        "format": "hgb-v1", "model": clf, "scaler": sd.scaler, "sampler": sampler, "seed": seed,
        "seq_len": T, "window_days": prep.window_days, "feature_columns": list(prep.seq.feature_columns),
        "window_threshold": float(thr), "user_threshold": chosen["threshold"], "user_agg": chosen["agg"],
        "metrics": to_jsonable(metrics),
    }
    if artifact_dir:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(artifact_dir) / "hgb.pkl", "wb") as fh:
            pickle.dump(artifact, fh)
    return artifact, metrics, sd


class HGBDetector:
    """Scores windows / users with a saved (or in-memory) hgb-v1 artifact. Inputs are SCALED (T, F) sequences."""

    def __init__(self, artifact: dict):
        if artifact.get("format") != "hgb-v1":
            raise ValueError("not an hgb-v1 artifact")
        self.a = artifact

    @classmethod
    def load(cls, path: str) -> "HGBDetector":
        p = Path(path)
        with open(p / "hgb.pkl" if p.is_dir() else p, "rb") as fh:
            return cls(pickle.load(fh))

    def score_windows(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.a["model"].predict_proba(_summary(X_scaled))[:, 1]

    def score_users(self, user_ids, window_starts, X_scaled: np.ndarray) -> pd.DataFrame:
        """One row per user, sorted by score: user_score + flagged (stored aggregation and user threshold)."""
        agg = {k: v for k, v in self.a["user_agg"].items() if k in ("method", "k", "n")}
        s = aggregate_user_scores(user_ids, window_starts, self.score_windows(X_scaled),
                                  window_days=self.a["window_days"], **agg)
        out = s.sort_values(ascending=False).to_frame("user_score")
        out["flagged"] = out["user_score"] >= self.a["user_threshold"]
        return out


# ------------------------------------------------------------------ out-of-fold evaluation (main evidence)

def run_oof_hgb(prep, sampler: str = "none", k: int = 7, user_agg: Optional[dict] = None,
                progress: Callable[[str], None] = print):
    """
    Same user k-fold as the ensemble ablation (fold i = test, i+1 = validation, rest = train): every user is scored only
    by a model that never trained on them. Returns (users, windows); users has method == "hgb".
    """
    from app.ml import ensemble
    users, windows, _ = ensemble.run_oof_ensemble(prep, sampler, k=k, n_inits=1, members=["hgb"],
                                                  user_agg=user_agg, progress=progress)
    return users, windows


def _pooled(g: pd.DataFrame, top: pd.Series) -> dict:
    n_ins = int(g["is_insider"].sum())
    tp, nf = int((top & g["is_insider"]).sum()), int(top.sum())
    prec, rec = tp / max(nf, 1), tp / max(n_ins, 1)
    return {"flagged": nf, "caught": tp, "insiders": n_ins, "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}


def budget_table(users: pd.DataFrame, method: str = "hgb", budgets: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30)) -> pd.DataFrame:
    """Flag the top X% of users WITHIN EACH FOLD, pool the counts. No validation threshold has to carry over."""
    g = users[users["method"] == method]
    rows = {}
    for b in budgets:
        top = g["rank"] <= np.maximum(1, np.round(b * g["n_fold"]))
        rows[f"top {b:.0%}"] = _pooled(g, top)
    return pd.DataFrame(rows).T


def matched_recall(users: pd.DataFrame, method: str = "hgb", target: float = 1.0) -> dict:
    """Precision / F1 at the smallest per-fold budget (1% steps) whose pooled recall reaches `target`."""
    g = users[users["method"] == method]
    for b in np.arange(0.01, 1.0001, 0.01):
        top = g["rank"] <= np.maximum(1, np.round(b * g["n_fold"]))
        r = _pooled(g, top)
        if r["recall"] >= target - 1e-9:
            return {"budget": round(float(b), 2), **r}
    return {"budget": 1.0, **_pooled(g, g["rank"] > 0)}


def paper_comparison(users: pd.DataFrame, method: str = "hgb", budget: float = 0.10) -> pd.DataFrame:
    """Paper row next to ours at (a) a fixed analyst budget and (b) matched (100%) recall. Accuracy is NOT compared
    (user-level accuracy is dominated by the majority class and is not measured the same way as the paper's)."""
    g = users[users["method"] == method]
    fixed = _pooled(g, g["rank"] <= np.maximum(1, np.round(budget * g["n_fold"])))
    matched = matched_recall(users, method, 1.0)
    rows = {
        "paper (NM-2 + OCSVM)": {"flagged": np.nan, "recall": PAPER["recall"], "precision": PAPER["precision_implied"], "f1": PAPER["f1"]},
        f"{method} @ top {budget:.0%} of users": {k: fixed[k] for k in ("flagged", "recall", "precision", "f1")},
        f"{method} @ matched recall ({matched['budget']:.0%} budget)": {k: matched[k] for k in ("flagged", "recall", "precision", "f1")},
    }
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ explainability

def importance_table(artifact: dict, X_scaled: np.ndarray, y: np.ndarray, n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """
    Permutation importance (drop in window PR-AUC when one column is shuffled) on held-out windows, summed over the
    last / max / mean summaries of each behavioural feature. Global explanation for the alert-detail page.
    """
    from sklearn.inspection import permutation_importance
    Xs = _summary(X_scaled)
    r = permutation_importance(artifact["model"], Xs, y, scoring="average_precision", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    cols = summary_columns(artifact["feature_columns"])
    df = pd.DataFrame({"column": cols, "importance": r.importances_mean})
    df["feature"] = df["column"].str.rsplit("__", n=1).str[0]
    out = df.groupby("feature")["importance"].sum().sort_values(ascending=False).round(4)
    return out.to_frame("PR-AUC drop when shuffled")
