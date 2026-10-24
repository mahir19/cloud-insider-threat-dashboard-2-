"""
Layer-2 ensemble (architecture.md section 6).

Members       lstm  |  bilstm_attn  |  hgb   (gradient boosting on last / max / mean over the T-window history)
Combination   stack      logistic regression on the members' logits, fitted on VALIDATION users' window scores
                         (users no member trained on). Validation scores are produced cross-fitted by user, so
                         the user threshold tuned on them is not in-sample for the stacker.
              rank_avg   mean percentile rank (fallback comparison)
Evaluation    run_oof_ensemble: the same k-fold user split as run_oof (fold i test, i+1 val, rest train); each user is
              scored only by models that never trained on them. Alert-budget metrics, ablation, paired bootstrap.

Layer 1 (the sampler) is unchanged: the winning spec is applied to the TRAIN split only, for every member.
"""
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.ml.metrics import ThresholdPolicy, choose_threshold, compute_metrics, pr_auc, to_jsonable
from app.ml.sampling import kwargs_for, resample
from app.ml.sequences import flatten, kfold_user_splits, unflatten
from app.ml.user_level import DEFAULT_AGG, aggregate_user_scores, evaluate_user_level, user_truth

MEMBERS_DEFAULT = ("lstm", "bilstm_attn", "hgb")
NN_MEMBERS = ("lstm", "bilstm_attn")
BLENDS = ("rank_avg", "stack")


# ------------------------------------------------------------------ members

def summarize_history(X: np.ndarray) -> np.ndarray:
    """(N, T, F) -> (N, 3F): last window + max + mean over the history (same summary as the 7c sanity check)."""
    X = np.nan_to_num(X)
    return np.concatenate([X[:, -1], X.max(1), X.mean(1)], axis=1)


def _fit_hgb(Xr: np.ndarray, yr: np.ndarray, pos_weight: Optional[float], seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n_pos = max(int((yr == 1).sum()), 1)
    pw = min(50.0, float(pos_weight)) if pos_weight else min(50.0, float(np.sqrt((yr == 0).sum() / n_pos)))
    w = np.where(yr == 1, pw, 1.0)
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, random_state=seed) \
        .fit(summarize_history(Xr), yr, sample_weight=w)


def fit_members(sd, rr, members: Sequence[str], cfg, seed: int, n_inits: int, T: int, F: int, keep_models: bool = False) -> dict:
    """
    Trains every member on the resampled TRAIN split `rr` and predicts the untouched val/test windows.
    Returns {member: {"val": p, "test": p, "models": [...]}}  (models only when keep_models).
    """
    Xr = unflatten(rr.X, T, F)
    out = {}
    for name in members:
        if name == "hgb":
            clf = _fit_hgb(Xr, rr.y, rr.pos_weight, seed)
            pv = clf.predict_proba(summarize_history(sd.X_val))[:, 1]
            pt = clf.predict_proba(summarize_history(sd.X_test))[:, 1]
            out[name] = {"val": pv, "test": pt, "models": [clf] if keep_models else []}
        elif name in NN_MEMBERS:
            from app.ml.lstm_detector import predict_proba, train_lstm      # torch is only needed for the neural members
            pv, pt, models = 0.0, 0.0, []
            for j in range(n_inits):
                m, _ = train_lstm(Xr, rr.y, cfg.scaled_for(len(rr.y)), seed=seed * 100 + j, pos_weight=rr.pos_weight,
                                  X_val=sd.X_val, y_val=sd.y_val, arch=name)
                pv = pv + predict_proba(m, sd.X_val) / n_inits
                pt = pt + predict_proba(m, sd.X_test) / n_inits
                if keep_models:
                    models.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
            out[name] = {"val": pv, "test": pt, "models": models}
        else:
            raise ValueError(f"unknown ensemble member '{name}'; choose from {MEMBERS_DEFAULT}")
    return out


# ------------------------------------------------------------------ blending

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _rank01(p: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(p) / len(p)


def _stack(member_val: Dict[str, np.ndarray], member_test: Dict[str, np.ndarray], y_val: np.ndarray,
           val_users: np.ndarray, cv_groups: int = 3):
    """
    Logistic-regression stacker on member logits. Returns (val_scores, test_scores, coefs).
    val_scores are cross-fitted by user (GroupKFold) so a threshold tuned on them is not in-sample;
    test_scores come from the stacker fitted on all validation windows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    names = list(member_val)
    Zv = np.column_stack([_logit(member_val[n]) for n in names])
    Zt = np.column_stack([_logit(member_test[n]) for n in names])
    y = np.asarray(y_val).astype(int)
    mk = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    if len(np.unique(y)) < 2:                                   # cannot fit: plain mean logit
        return Zv.mean(1), Zt.mean(1), {n: 1.0 / len(names) for n in names}
    final = mk().fit(Zv, y)
    sv = np.full(len(y), np.nan)
    try:
        for tr, te in GroupKFold(n_splits=min(cv_groups, len(np.unique(val_users)))).split(Zv, y, groups=val_users):
            if len(np.unique(y[tr])) < 2:
                raise ValueError("a cross-fit fold has no positives")
            sv[te] = mk().fit(Zv[tr], y[tr]).decision_function(Zv[te])
    except ValueError:
        sv = final.decision_function(Zv)                        # fall back to in-sample validation scores
    return sv, final.decision_function(Zt), {n: float(round(c, 4)) for n, c in zip(names, final.coef_[0])}


def blend_all(preds: dict, y_val: np.ndarray, val_users: np.ndarray, blends: Sequence[str] = BLENDS):
    """{method: (val_scores, test_scores)} for each member, rank_avg and stack, plus the stacker coefficients."""
    out = {n: (p["val"], p["test"]) for n, p in preds.items()}
    coefs = {}
    if len(preds) > 1:
        if "rank_avg" in blends:
            out["rank_avg"] = (np.mean([_rank01(p["val"]) for p in preds.values()], axis=0),
                               np.mean([_rank01(p["test"]) for p in preds.values()], axis=0))
        if "stack" in blends:
            sv, st, coefs = _stack({n: p["val"] for n, p in preds.items()}, {n: p["test"] for n, p in preds.items()},
                                   y_val, val_users)
            out["stack"] = (sv, st)
    return out, coefs


# ------------------------------------------------------------------ out-of-fold evaluation

def run_oof_ensemble(prep, sampler: str = "none", k: int = 7, cfg=None, n_inits: int = 2,
                     members: Sequence[str] = MEMBERS_DEFAULT, user_agg: Optional[dict] = None,
                     progress: Callable[[str], None] = print, make_split=None):
    """
    Returns (users, windows, coefs):
      users    one row per (user, method): user_id, fold, method, user_score, rank (1 = highest in its fold), n_fold, is_insider
      windows  one row per (fold, method): test window PR-AUC
      coefs    one row per fold: the stacker's member weights
    """
    if make_split is None:
        from app.ml.train import make_split
    if cfg is None and any(m in NN_MEMBERS for m in members):          # HGB-only runs need no torch
        from app.ml.lstm_detector import LSTMConfig
        cfg = LSTMConfig(hidden_size=64, num_layers=1, dropout=0.2, epochs=25, patience=5)
    agg = user_agg or DEFAULT_AGG
    kw = {a: v for a, v in agg.items() if a in ("method", "k", "n")}
    T, F = prep.seq_len, len(prep.seq.feature_columns)
    uid, ws = prep.features_df["user_id"].to_numpy(), prep.seq.window_starts
    urows, wrows, crows = [], [], []
    for i, sp in enumerate(kfold_user_splits(prep.unique_users, prep.user_is_insider, k)):
        progress(f"[oof-ens:{sampler}] fold {i + 1}/{k}")
        sd = make_split(prep, i, split=sp)
        rr = resample(sampler, flatten(sd.X_train), sd.y_train, seed=i, **kwargs_for(sampler, None))
        preds = fit_members(sd, rr, members, cfg, i, n_inits, T, F)
        scores, coefs = blend_all(preds, sd.y_val, uid[sd.idx_val])
        if coefs:
            crows.append({"fold": i, **coefs})
        for method, (_, pt) in scores.items():
            st = aggregate_user_scores(uid[sd.idx_test], ws[sd.idx_test], pt, window_days=prep.window_days, **kw)
            tt = user_truth(st.index.to_numpy(), prep.unique_users, prep.user_is_insider)
            urows.append(pd.DataFrame({
                "user_id": st.index.to_numpy(), "fold": i, "method": method, "user_score": st.to_numpy(),
                "rank": st.rank(ascending=False, method="first").to_numpy(), "n_fold": len(st), "is_insider": tt.astype(bool)}))
            wrows.append({"fold": i, "method": method, "window_pr_auc": pr_auc(sd.y_test, pt)})
    return (pd.concat(urows, ignore_index=True), pd.DataFrame(wrows), pd.DataFrame(crows))


def summarize_oof(users: pd.DataFrame, windows: Optional[pd.DataFrame] = None,
                  budgets: Sequence[float] = (0.05, 0.10, 0.15, 0.30)) -> pd.DataFrame:
    """
    One row per method. precision@X / recall@X flag the top X% of users WITHIN EACH FOLD and pool the counts, so no
    validation threshold has to carry over. user PR-AUC is computed per fold, then mean +/- std over folds.
    """
    from sklearn.metrics import average_precision_score
    rows = {}
    for method, g in users.groupby("method", sort=False):
        n_ins = int(g["is_insider"].sum())
        aps = [average_precision_score(f["is_insider"], f["user_score"]) for _, f in g.groupby("fold") if f["is_insider"].any()]
        row = {"user PR-AUC": f"{np.mean(aps):.3f} ± {np.std(aps):.3f}"}
        if windows is not None and len(windows):
            w = windows[windows["method"] == method]["window_pr_auc"].dropna()
            row["window PR-AUC"] = f"{w.mean():.3f} ± {w.std(ddof=0):.3f}"
        for b in budgets:
            top = g["rank"] <= np.maximum(1, np.round(b * g["n_fold"]))
            tp, nf = int((top & g["is_insider"]).sum()), int(top.sum())
            row[f"precision@{b:.0%}"] = round(tp / max(nf, 1), 3)
            row[f"recall@{b:.0%}"] = round(tp / max(n_ins, 1), 3)
        rows[method] = row
    return pd.DataFrame(rows).T


def best_method(users: pd.DataFrame, candidates: Sequence[str]) -> str:
    """The candidate with the highest mean per-fold user PR-AUC (used to pick 'best single model' for the bootstrap)."""
    from sklearn.metrics import average_precision_score
    def score(m):
        g = users[users["method"] == m]
        return np.mean([average_precision_score(f["is_insider"], f["user_score"]) for _, f in g.groupby("fold") if f["is_insider"].any()])
    return max([c for c in candidates if c in set(users["method"])], key=score)


def paired_bootstrap(users: pd.DataFrame, a: str, b: str, metric: str = "recall", budget: float = 0.10,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """
    Paired bootstrap over users (resampled within each fold, same users for both methods): difference a - b.
    metric = "recall" (at `budget`, flagging the top X% of each fold) or "pr_auc" (mean of per-fold user PR-AUC).
    """
    from sklearn.metrics import average_precision_score
    ua = users[users["method"] == a].set_index("user_id")
    ub = users[users["method"] == b].set_index("user_id").loc[ua.index]
    rng = np.random.RandomState(seed)
    folds = [np.flatnonzero(ua["fold"].to_numpy() == f) for f in np.unique(ua["fold"])]
    y = ua["is_insider"].to_numpy()
    sa, sb = ua["user_score"].to_numpy(), ub["user_score"].to_numpy()
    fa = (ua["rank"] <= np.maximum(1, np.round(budget * ua["n_fold"]))).to_numpy()
    fb = (ub["rank"] <= np.maximum(1, np.round(budget * ub["n_fold"]))).to_numpy()
    diffs = []
    for _ in range(n_boot):
        idx = [f[rng.randint(0, len(f), len(f))] for f in folds]
        if metric == "recall":
            ii = np.concatenate(idx)
            n_ins = max(int(y[ii].sum()), 1)
            diffs.append((fa[ii] & y[ii]).sum() / n_ins - (fb[ii] & y[ii]).sum() / n_ins)
        else:
            d = [average_precision_score(y[ix], sa[ix]) - average_precision_score(y[ix], sb[ix]) for ix in idx if y[ix].any()]
            diffs.append(np.mean(d))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"a": a, "b": b, "metric": metric if metric != "recall" else f"recall@{budget:.0%}",
            "mean_diff": round(float(diffs.mean()), 4), "ci95": (round(float(lo), 4), round(float(hi), 4)),
            "p_diff_le_0": round(float((diffs <= 0).mean()), 3)}


# ------------------------------------------------------------------ production ensemble (single held-out split)

def train_production_ensemble(prep, sampler: str, seed: int, members: Sequence[str] = MEMBERS_DEFAULT, cfg=None,
                              n_inits: int = 2, window_policy=None, user_policy=None, user_agg: Optional[dict] = None,
                              artifact_dir: Optional[str] = None, make_split=None):
    """
    Fits every member on the winning sampler's output for one user-level split, blends them, tunes the window and user
    thresholds on VALIDATION and reports TEST. Returns (artifact_dict, table, per_method_results).
    The saved ensemble.pkl holds weights + stacker + scaler + thresholds; an API-side loader still has to be written.
    """
    from app.ml.lstm_detector import LSTMConfig
    from app.ml.train import DEFAULT_USER_POLICY, DEFAULT_WINDOW_POLICY
    from app.ml.train import make_split as _default_split
    make_split = make_split or _default_split
    cfg = cfg or LSTMConfig.production()
    T, F = prep.seq_len, len(prep.seq.feature_columns)
    wpol = ThresholdPolicy.coerce(window_policy) if window_policy is not None else DEFAULT_WINDOW_POLICY
    upol = ThresholdPolicy.coerce(user_policy) if user_policy is not None else DEFAULT_USER_POLICY
    agg = dict(user_agg or DEFAULT_AGG)
    sd = make_split(prep, seed)
    rr = resample(sampler, flatten(sd.X_train), sd.y_train, seed=seed, **kwargs_for(sampler, None))
    preds = fit_members(sd, rr, members, cfg, seed, n_inits, T, F, keep_models=True)
    uid = prep.features_df["user_id"].to_numpy()
    scores, coefs = blend_all(preds, sd.y_val, uid[sd.idx_val])
    results, rows = {}, {}
    for method, (pv, pt) in scores.items():
        thr = choose_threshold(sd.y_val, pv, wpol)
        ur = evaluate_user_level(prep, sd.idx_val, sd.idx_test, pv, pt, upol, {"sel": agg})["sel"]
        t = ur["test"]
        results[method] = {"window_threshold": thr, "window_test": compute_metrics(sd.y_test, pt, thr), "user": ur}
        rows[method] = {"window PR-AUC (test)": round(pr_auc(sd.y_test, pt), 3), "user PR-AUC (test)": t["pr_auc"],
                        "user precision": t["precision"], "user recall": t["recall"], "flagged": t["n_flagged"],
                        "precision@10": t.get("precision@10"), "recall@15": t.get("recall@15"), "recall@30": t.get("recall@30")}
    artifact = {"format": "ensemble-v1", "sampler": sampler, "members": list(preds), "stack_coefs": coefs,
                "member_models": {n: p["models"] for n, p in preds.items()}, "scaler": sd.scaler,
                "seq_len": T, "window_days": prep.window_days, "feature_columns": list(prep.seq.feature_columns),
                "user_agg": agg, "thresholds": {m: {"window": r["window_threshold"], "user": r["user"]["threshold"]} for m, r in results.items()},
                "seed": seed, "metrics": to_jsonable(results)}
    if artifact_dir:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(artifact_dir) / "ensemble.pkl", "wb") as fh:
            pickle.dump(artifact, fh)
    return artifact, pd.DataFrame(rows).T, results
