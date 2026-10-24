"""
Temporal component of the final system (architecture.md section on temporal modelling).

The selected paper's stated gap: every window is scored independently, so slow escalation is invisible. Two ways of
closing it are built here and compared on the SAME 7-fold user-level out-of-fold setup as section 7b:

  1. Trend features inside HGB   summarize_trend(): last / max / mean (as before) PLUS, for every behavioural feature,
                                 the linear slope over the T-window history, last window minus the mean of the earlier
                                 ones, and the length of the trailing run of rising windows.
  2. CUSUM as a ranking score    cusum_peak_by_user(): one-sided CUSUM over each user's out-of-fold window scores
                                 (logit scale), against an EXPANDING per-user baseline (prior windows only, no
                                 lookahead). The user's score is the PEAK of the CUSUM statistic, so it ranks users
                                 and works with the alert-budget metrics -- no fixed alarm threshold h has to be
                                 carried over from synthetic data (that was the old failure: h=8 flagged every user
                                 on real CERT scores). The slack k is chosen per fold on VALIDATION users only.
                                 An equivalent alarm threshold h is frozen from the validation users' peaks.

Compared methods (all out-of-fold; every user is scored only by models that never trained on them):

  hgb                 baseline (identical to section 7b)
  hgb_trend           HGB with the trend features
  cusum               CUSUM peak over the baseline HGB window scores
  hgb+cusum           mean within-fold percentile rank of hgb (top-k user score) and cusum peak
  hgb_trend+cusum     same, using hgb_trend

Output frames have the same columns as ensemble.run_oof_ensemble, so ensemble.summarize_oof / paired_bootstrap work.
"""
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from app.ml.metrics import pr_auc
from app.ml.sampling import kwargs_for, resample
from app.ml.sequences import flatten, kfold_user_splits, unflatten
from app.ml.user_level import DEFAULT_AGG, aggregate_user_scores, user_truth

K_GRID_DEFAULT = (0.25, 0.5, 1.0, 2.0)
MIN_BASELINE = 3          # windows of own history needed before a user's deviations accumulate
LOGIT_CLIP = 1e-6
STD_FLOOR = 0.5           # logit units; stops a near-constant user's tiny std from exploding z


# ------------------------------------------------------------------ 1. trend features

def summarize_trend(X: np.ndarray) -> np.ndarray:
    """(N, T, F) -> (N, 6F): last / max / mean (as summarize_history) + slope + last-minus-earlier-mean + rising run."""
    X = np.nan_to_num(X).astype(np.float64)
    N, T, F = X.shape
    base = np.concatenate([X[:, -1], X.max(1), X.mean(1)], axis=1)
    tc = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    slope = (X * tc[None, :, None]).sum(1) / (tc ** 2).sum()
    delta = X[:, -1] - X[:, :-1].mean(1) if T > 1 else np.zeros((N, F))
    rises = X[:, 1:] > X[:, :-1] + 1e-9                       # (N, T-1, F)
    run = np.zeros((N, F))
    alive = np.ones((N, F), dtype=bool)
    for j in range(T - 2, -1, -1):                            # trailing run of consecutive rises, newest first
        alive &= rises[:, j]
        run += alive
    return np.concatenate([base, slope, delta, run], axis=1).astype(np.float32)


def trend_columns(feature_columns: Sequence[str]) -> list:
    return [f"{c}__{s}" for s in ("last", "max", "mean", "slope", "delta", "rise_run") for c in feature_columns]


def _fit_hgb_trend(Xr: np.ndarray, yr: np.ndarray, pos_weight: Optional[float], seed: int):
    """Same hyper-parameters and class weighting as ensemble._fit_hgb; only the summary differs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    n_pos = max(int((yr == 1).sum()), 1)
    pw = min(50.0, float(pos_weight)) if pos_weight else min(50.0, float(np.sqrt((yr == 0).sum() / n_pos)))
    w = np.where(yr == 1, pw, 1.0)
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=seed) \
        .fit(summarize_trend(Xr), yr, sample_weight=w)


# ------------------------------------------------------------------ 2. CUSUM peak score

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def cusum_series(x: np.ndarray, k: float) -> np.ndarray:
    """
    One-sided CUSUM of one user's chronologically ordered scores (logit scale).
    z[t] = (x[t] - mean(x[:t])) / max(std(x[:t]), STD_FLOOR)       (prior windows only; z = 0 until MIN_BASELINE)
    c[t] = max(0, c[t-1] + z[t] - k)
    """
    n = len(x)
    out = np.zeros(n)
    if n <= MIN_BASELINE:
        return out
    cs, cs2 = np.cumsum(x), np.cumsum(x * x)
    c = 0.0
    for t in range(MIN_BASELINE, n):
        m = cs[t - 1] / t
        var = max(cs2[t - 1] / t - m * m, 0.0)
        z = (x[t] - m) / max(np.sqrt(var), STD_FLOOR)
        c = max(0.0, c + z - k)
        out[t] = c
    return out


def cusum_peak_by_user(user_ids, window_starts, probs, k: float) -> pd.Series:
    """One score per user: the peak of that user's CUSUM statistic over the given window scores."""
    df = pd.DataFrame({"user": np.asarray(user_ids), "ws": pd.to_datetime(np.asarray(window_starts)),
                       "x": _logit(probs)}).sort_values(["user", "ws"], kind="stable")
    u, x = df["user"].to_numpy(), df["x"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, u[1:] != u[:-1], True])
    idx, peaks = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx.append(u[a])
        peaks.append(cusum_series(x[a:b], k).max())
    return pd.Series(peaks, index=pd.Index(idx, name="user"), dtype=float).sort_index()


# ------------------------------------------------------------------ out-of-fold comparison

def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def _rows(st: pd.Series, truth: np.ndarray, fold: int, method: str) -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": st.index.to_numpy(), "fold": fold, "method": method, "user_score": st.to_numpy(),
        "rank": st.rank(ascending=False, method="first").to_numpy(), "n_fold": len(st), "is_insider": truth.astype(bool)})


def run_oof_temporal(prep, sampler: str = "none", k: int = 7, user_agg: Optional[dict] = None,
                     k_grid: Sequence[float] = K_GRID_DEFAULT, make_split=None,
                     progress: Callable[[str], None] = print):
    """
    Returns (users, windows, window_scores, cusum_info):
      users          one row per (user, method) -- see module docstring for the methods
      windows        one row per (fold, hgb / hgb_trend): test window PR-AUC
      window_scores  every window of every user scored out-of-fold: user_id, window_start, fold, y, p_hgb, p_hgb_trend
      cusum_info     one row per (fold, base model): chosen slack k, frozen alarm threshold h (top-10% of validation
                     users' peaks), and the alarm's pooled result on the fold's test users
    """
    if make_split is None:
        from app.ml.train import make_split
    from app.ml import ensemble
    agg = user_agg or DEFAULT_AGG
    kw = {a: v for a, v in agg.items() if a in ("method", "k", "n")}
    T, F = prep.seq_len, len(prep.seq.feature_columns)
    uid, ws = prep.features_df["user_id"].to_numpy(), prep.seq.window_starts
    urows, wrows, srows, irows = [], [], [], []

    for i, sp in enumerate(kfold_user_splits(prep.unique_users, prep.user_is_insider, k)):
        progress(f"[oof-temporal:{sampler}] fold {i + 1}/{k}")
        sd = make_split(prep, i, split=sp)
        rr = resample(sampler, flatten(sd.X_train), sd.y_train, seed=i, **kwargs_for(sampler, None))
        base = ensemble.fit_members(sd, rr, ["hgb"], None, i, 1, T, F)["hgb"]            # identical to section 7b
        clf = _fit_hgb_trend(unflatten(rr.X, T, F), rr.y, rr.pos_weight, i)
        P = {"hgb": (base["val"], base["test"]),
             "hgb_trend": (clf.predict_proba(summarize_trend(sd.X_val))[:, 1],
                           clf.predict_proba(summarize_trend(sd.X_test))[:, 1])}

        srows.append(pd.DataFrame({"user_id": uid[sd.idx_test], "window_start": ws[sd.idx_test], "fold": i,
                                   "y": sd.y_test, "p_hgb": P["hgb"][1], "p_hgb_trend": P["hgb_trend"][1]}))
        score_val, score_test = {}, {}
        for name, (pv, pt) in P.items():
            sv = aggregate_user_scores(uid[sd.idx_val], ws[sd.idx_val], pv, window_days=prep.window_days, **kw)
            st = aggregate_user_scores(uid[sd.idx_test], ws[sd.idx_test], pt, window_days=prep.window_days, **kw)
            tv = user_truth(sv.index.to_numpy(), prep.unique_users, prep.user_is_insider)
            tt = user_truth(st.index.to_numpy(), prep.unique_users, prep.user_is_insider)
            urows.append(_rows(st, tt, i, name))
            wrows.append({"fold": i, "method": name, "window_pr_auc": pr_auc(sd.y_test, pt)})

            # CUSUM on this model's window scores: slack k chosen on VALIDATION users only
            best_k, best_ap = None, -1.0
            for kk in k_grid:
                ap = pr_auc(tv, cusum_peak_by_user(uid[sd.idx_val], ws[sd.idx_val], pv, kk).reindex(sv.index).to_numpy())
                if ap > best_ap:
                    best_k, best_ap = kk, ap
            cv = cusum_peak_by_user(uid[sd.idx_val], ws[sd.idx_val], pv, best_k).reindex(sv.index)
            ct = cusum_peak_by_user(uid[sd.idx_test], ws[sd.idx_test], pt, best_k).reindex(st.index)
            h = float(np.quantile(cv.to_numpy(), 0.90))                                  # frozen alarm threshold
            alarm = ct.to_numpy() > h
            irows.append({"fold": i, "base": name, "k": best_k, "h_frozen": round(h, 3), "val_user_pr_auc": round(best_ap, 3),
                          "flagged": int(alarm.sum()), "caught": int((alarm & (tt == 1)).sum()), "insiders": int(tt.sum())})
            score_val[name], score_test[name] = sv, st
            if name == "hgb":
                urows.append(_rows(ct, tt, i, "cusum"))
            urows.append(_rows((0.5 * (_pct(st) + _pct(ct))).rename(None), tt, i, f"{name}+cusum"))

    users = pd.concat(urows, ignore_index=True)
    return (users, pd.DataFrame(wrows), pd.concat(srows, ignore_index=True), pd.DataFrame(irows))
