"""
User-level scoring.

The LSTM scores every (user, window). Analysts act on USERS, and taking the max
window probability per user is the noisiest possible aggregate: a user with 250
windows gets 250 chances for one random spike. Three aggregators:

  max          legacy: highest window probability
  topk         mean of the user's k highest window probabilities
  consecutive  highest value v such that the user has n back-to-back windows
               ALL with probability >= v. A user is flagged at threshold t iff
               n consecutive windows are >= t (a single spike never flags).
               Windows with no activity (gaps in the grid) break a run.

All three are continuous scores, so the same threshold policies and PR-AUC
apply to them.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from app.ml.metrics import ThresholdPolicy, choose_threshold, pr_auc, precision_recall_at_k

AGG_METHODS = ("max", "topk", "consecutive")
DEFAULT_AGG = {"method": "topk", "k": 3, "n": 3}


def _consecutive_min(user: np.ndarray, widx: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    """Arrays sorted by (user, widx). out[t] = min(p[t..t+n-1]) if those windows are contiguous, else -inf."""
    N = len(p)
    out = np.full(N, -np.inf)
    if n <= 1:
        return p.astype(float).copy()
    if N < n:
        return out
    m = N - n + 1
    ok = np.ones(m, dtype=bool)
    val = p[:m].astype(float).copy()
    for i in range(1, n):
        ok &= (user[i:m + i] == user[:m]) & (widx[i:m + i] - widx[:m] == i)
        val = np.minimum(val, p[i:m + i])
    out[:m][ok] = val[ok]
    return out


def aggregate_user_scores(user_ids, window_starts, probs, method: str = "topk", k: int = 3, n: int = 3,
                          window_days: Optional[int] = None) -> pd.Series:
    """Returns a Series indexed by user_id: one score per user."""
    df = pd.DataFrame({"user": np.asarray(user_ids), "ws": pd.to_datetime(np.asarray(window_starts)),
                       "p": np.asarray(probs, dtype=float)})
    if df.empty:
        return pd.Series(dtype=float)
    if method == "max":
        return df.groupby("user")["p"].max()
    if method == "topk":
        return df.groupby("user")["p"].apply(lambda s: float(np.sort(s.to_numpy())[::-1][:max(1, k)].mean()))
    if method == "consecutive":
        df = df.sort_values(["user", "ws"]).reset_index(drop=True)
        step = pd.Timedelta(days=window_days) if window_days else (df["ws"].diff()[df["ws"].diff() > pd.Timedelta(0)].min())
        widx = ((df["ws"] - df["ws"].min()) / step).round().astype("int64").to_numpy()
        run = _consecutive_min(df["user"].to_numpy(), widx, df["p"].to_numpy(), n)
        df["run"] = run
        score = df.groupby("user")["run"].max()
        return score.where(np.isfinite(score), 0.0)         # no n contiguous windows -> cannot show a sustained pattern
    raise ValueError(f"unknown aggregation '{method}'; choose from {AGG_METHODS}")


def user_truth(users: Sequence, unique_users: np.ndarray, user_is_insider: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(unique_users, np.asarray(users))
    return user_is_insider[pos].astype(int)


def user_metrics(truth, score, threshold: float, ks=(5, 10, 15, 20, 30)) -> Dict[str, float]:
    t, s = np.asarray(truth).astype(int), np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    tp = int((pred & t).sum())
    n_flag, n_pos = int(pred.sum()), int(t.sum())
    prec = tp / n_flag if n_flag else 0.0
    rec = tp / n_pos if n_pos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"recall": round(rec, 4), "precision": round(prec, 4), "f1": round(f1, 4), "pr_auc": round(pr_auc(t, s), 4),
            "n_users": int(len(t)), "n_insiders": n_pos, "n_flagged": n_flag, "true_flagged": tp,
            **precision_recall_at_k(t, s, [k for k in ks if k <= len(t)])}


def evaluate_user_level(prep, idx_val: np.ndarray, idx_test: np.ndarray, p_val: np.ndarray, p_test: np.ndarray,
                        policy=None, methods: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """
    For each aggregation: score val/test users, tune the user threshold on VALIDATION users only,
    report val/test user-level metrics. Returns {method_name: {"threshold", "val", "test", "agg"}}.
    """
    pol = ThresholdPolicy.coerce(policy) if policy is not None else ThresholdPolicy(mode="budget", budget=0.10)
    uid = prep.features_df["user_id"].to_numpy()
    ws = prep.seq.window_starts
    methods = methods or {"max": {"method": "max"}, "topk": {"method": "topk", "k": 3},
                          "consecutive": {"method": "consecutive", "n": 3}}
    out = {}
    for name, cfg in methods.items():
        kw = {k: v for k, v in cfg.items() if k in ("method", "k", "n")}
        sv = aggregate_user_scores(uid[idx_val], ws[idx_val], p_val, window_days=prep.window_days, **kw)
        st = aggregate_user_scores(uid[idx_test], ws[idx_test], p_test, window_days=prep.window_days, **kw)
        tv = user_truth(sv.index.to_numpy(), prep.unique_users, prep.user_is_insider)
        tt = user_truth(st.index.to_numpy(), prep.unique_users, prep.user_is_insider)
        thr = choose_threshold(tv, sv.to_numpy(), pol)
        out[name] = {"threshold": round(float(thr), 6), "agg": kw,
                     "val": user_metrics(tv, sv.to_numpy(), thr), "test": user_metrics(tt, st.to_numpy(), thr)}
    return out


def flag_users(scored: pd.DataFrame, agg: dict, threshold: float, window_days: Optional[int] = None,
               user_col: str = "user", score_col: str = "score") -> pd.DataFrame:
    """Apply a stored aggregation + user threshold to scored windows (output of LSTMDetector.score)."""
    kw = {k: v for k, v in agg.items() if k in ("method", "k", "n")}
    s = aggregate_user_scores(scored[user_col].to_numpy(), scored["window_start"].to_numpy(),
                              scored[score_col].to_numpy(), window_days=window_days, **kw)
    out = s.sort_values(ascending=False).to_frame("user_score")
    out["flagged"] = out["user_score"] >= threshold
    return out
