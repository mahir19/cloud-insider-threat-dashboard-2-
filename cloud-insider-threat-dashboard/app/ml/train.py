"""
Offline training job (architecture.md sections 3.5, 3.6, 4.2).

    python -m app.ml.train --mode train    --answers seed/synthetic_cert/answers.csv --window-days 2
    python -m app.ml.train --mode compare  ...     # Layer 1 only (sampler comparison)
    python -m app.ml.train --mode evaluate ...     # full evaluation table (section 5)

Training NEVER happens inside an API request: the API launches this module as
a separate process (see app/jobs.py) and the job reports back through its
DetectionRun row (status, metrics, config).

Pipeline
--------
  events -> 12 features/user/window -> T x 12 sequences -> USER-LEVEL split
  Layer 1  every candidate sampler (NM-2 / DeepSMOTE / none / class_weight)
           is applied to the TRAIN split only, a small LSTM is trained with
           identical seeds, and candidates are ranked on the VALIDATION split
           (test is only reported, never used to choose). Repeated over
           several seeds -> mean +/- std.
  Layer 2  the production LSTM is trained once on the winning sampler's
           output; the alert threshold is tuned on validation (recall
           first) and frozen; test metrics are reported; the artifact
           (weights, scaler, threshold, sampler) is saved.
"""
import argparse
import logging
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app.ml.metrics import ThresholdPolicy, aggregate, choose_threshold, compute_metrics, metrics_at_budget, to_jsonable
from app.ml.sampling import SAMPLER_CANDIDATES, SWEEP_CANDIDATES, base_of, kwargs_for, resample
from app.ml.user_level import DEFAULT_AGG, evaluate_user_level
from app.ml.sequences import (
    FeatureScaler, SequenceData, UserSplit, build_sequences, flatten, insider_flags,
    label_samples, load_labels, unflatten, user_level_split, kfold_user_splits,
)

log = logging.getLogger("train")

DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_WINDOW_POLICY = ThresholdPolicy(mode="budget", budget=0.01)   # flag the top 1% of windows
DEFAULT_USER_POLICY = ThresholdPolicy(mode="budget", budget=0.10)     # flag the top 10% of users (~prevalence x1.5)
USER_KEYS = ["recall", "precision", "f1", "pr_auc", "n_flagged", "true_flagged",
             "precision@5", "precision@10", "precision@15", "precision@20", "recall@10", "recall@15", "recall@20", "recall@30"]


class SplitError(ValueError):
    """A user-level split that can't support training/evaluation (e.g. no positives in val)."""


# ------------------------------------------------------------------ data prep

@dataclass
class PreparedData:
    features_df: pd.DataFrame
    seq: SequenceData
    y: np.ndarray                 # per-sample label (N,)
    unique_users: np.ndarray
    user_is_insider: np.ndarray   # per unique user
    dataset: Optional[str]
    window_days: int
    seq_len: int


@dataclass
class SplitData:
    split: UserSplit
    scaler: FeatureScaler
    X_train: np.ndarray           # scaled (N, T, F)
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray         # positions into PreparedData.features_df
    idx_val: np.ndarray
    idx_test: np.ndarray


def prepare_data(features_df: pd.DataFrame, labels: pd.DataFrame, user_code_by_id: Dict[int, str],
                 window_days: int, seq_len: int, dataset: Optional[str] = None,
                 feature_columns: Optional[List[str]] = None) -> PreparedData:
    if features_df.empty:
        raise ValueError("No feature windows: ingest events first (POST /pipeline/ingest).")
    features_df = features_df.reset_index(drop=True)
    seq = build_sequences(features_df, seq_len, feature_columns)
    y = label_samples(features_df, labels, user_code_by_id)
    users = np.unique(features_df["user_id"].to_numpy())
    flags = insider_flags(users, labels, user_code_by_id)
    if flags.sum() == 0:
        raise ValueError("None of the labeled insiders appear in the ingested data for this dataset.")
    return PreparedData(features_df, seq, y, users, flags, dataset, window_days, seq_len)


def prepare_from_db(db, answers_csv: str, dataset: str = "cert", window_days: int = 2, seq_len: int = 7,
                    labels_dataset: Optional[str] = None) -> PreparedData:
    from app.ml.features import build_feature_windows
    from app.models import User
    features_df = build_feature_windows(db, window_days=window_days, dataset=dataset)
    labels = load_labels(answers_csv, dataset=labels_dataset)
    # Non-CERT users are stored as "<dataset>:<id>" to avoid colliding with CERT ids;
    # labels files use the bare id, so strip the prefix when matching.
    prefix = f"{dataset}:" if dataset != "cert" else ""
    user_code_by_id = {u.id: (u.user_code[len(prefix):] if prefix and u.user_code.startswith(prefix) else u.user_code)
                       for u in db.query(User).all()}
    return prepare_data(features_df, labels, user_code_by_id, window_days, seq_len, dataset)


def prepare_from_features(features_df: pd.DataFrame, user_codes: Dict[int, str], answers_csv: str,
                          dataset: str = "cert", window_days: int = 2, seq_len: int = 7,
                          labels_dataset: Optional[str] = None,
                          feature_columns: Optional[List[str]] = None) -> PreparedData:
    """DB-free entry point: features from app/ml/offline_features.py (streamed from CSVs, or a cached file)."""
    labels = load_labels(answers_csv, dataset=labels_dataset)
    return prepare_data(features_df, labels, user_codes, window_days, seq_len, dataset, feature_columns)


def make_split(prep: PreparedData, seed: int, val_frac: float = 0.15, test_frac: float = 0.15,
               split: Optional[UserSplit] = None) -> SplitData:
    if split is None:
        split = user_level_split(prep.unique_users, prep.user_is_insider, val_frac, test_frac, seed)
    uid = prep.features_df["user_id"].to_numpy()
    m_tr, m_va, m_te = split.masks(uid)
    # scaler is fit on TRAIN-split users only (one row per user-window)
    scaler = FeatureScaler().fit(prep.features_df[m_tr], prep.seq.feature_columns)
    Xs = scaler.transform(prep.seq.X)
    sd = SplitData(
        split, scaler,
        Xs[m_tr], prep.y[m_tr], Xs[m_va], prep.y[m_va], Xs[m_te], prep.y[m_te],
        np.flatnonzero(m_tr), np.flatnonzero(m_va), np.flatnonzero(m_te),
    )
    for name, yy in (("train", sd.y_train), ("val", sd.y_val), ("test", sd.y_test)):
        if yy.sum() == 0:
            raise SplitError(f"seed {seed}: the {name} split has no positive samples "
                             "(insider malicious period falls outside its windows?)")
    return sd


# ------------------------------------------------------------------ Layer 1

EVALUATORS = ("hgb", "lstm")     # model that scores each sampler candidate in Layer 1


def _fit_eval_candidate(sd: SplitData, name: str, cfg, seed: int, window_policy, sampler_kwargs: dict,
                        keep_predictions: bool, seq_len: int, n_feat: int, prep: PreparedData = None,
                        user_policy=None, ae_cache: Optional[dict] = None, evaluator: str = "hgb") -> dict:
    kw = kwargs_for(name, sampler_kwargs)
    if base_of(name) == "DeepSMOTE" and ae_cache is not None:
        kw["cache"] = ae_cache                      # autoencoder is trained once per split, reused across ratios
    rr = resample(name, flatten(sd.X_train), sd.y_train, seed=seed, **kw)
    if evaluator == "hgb":
        # same model + same class-weighting rule as production Layer 2 (ensemble._fit_hgb), so the winner is the
        # best sampler for the model that actually ships
        from app.ml import ensemble
        pr = ensemble.fit_members(sd, rr, ["hgb"], None, seed, 1, seq_len, n_feat)["hgb"]
        p_val, p_test = pr["val"], pr["test"]
    elif evaluator == "lstm":
        from app.ml.lstm_detector import predict_proba, train_lstm
        Xr = unflatten(rr.X, seq_len, n_feat)
        model, _ = train_lstm(Xr, rr.y, cfg.scaled_for(len(rr.y)), seed=seed, pos_weight=rr.pos_weight,
                              X_val=sd.X_val, y_val=sd.y_val)
        p_val, p_test = predict_proba(model, sd.X_val), predict_proba(model, sd.X_test)
    else:
        raise ValueError(f"unknown evaluator '{evaluator}'; choose from {EVALUATORS}")
    thr = choose_threshold(sd.y_val, p_val, window_policy)
    rec = {
        "seed": seed, "threshold": round(thr, 6),
        "val": compute_metrics(sd.y_val, p_val, thr),
        "test": compute_metrics(sd.y_test, p_test, thr),
        "train_size_after_sampling": int(len(rr.y)),
        "train_positives_after_sampling": int(rr.y.sum()),
        "sampler_info": rr.info,
    }
    if window_policy.mode == "budget":
        rec["test_at_budget"] = metrics_at_budget(sd.y_test, p_test, window_policy.budget)
    if prep is not None:
        rec["user"] = evaluate_user_level(prep, sd.idx_val, sd.idx_test, p_val, p_test, user_policy)
    if keep_predictions:
        rec["_pred"] = {"val_prob": p_val, "test_prob": p_test, "threshold": thr}
    return rec


def run_sampler_comparison(
    prep: PreparedData, seeds=DEFAULT_SEEDS, candidates=SAMPLER_CANDIDATES, eval_cfg=None,
    selection_metric: str = "pr_auc", min_recall: float = 0.9, sampler_kwargs: Optional[dict] = None,
    keep_predictions: bool = False, val_frac: float = 0.15, test_frac: float = 0.15,
    progress: Callable[[str], None] = log.info, window_policy=None, user_policy=None,
    kfold: Optional[int] = None, evaluator: str = "hgb",
) -> dict:
    """
    Layer 1. For each seed: one user-level split shared by every candidate
    (identical seeds and splits), the sampler touches only the train split,
    the `evaluator` model is trained ("hgb" = production Layer-2 model, default;
    "lstm" = the original small LSTM proxy, needs PyTorch), and the candidate is
    scored on the untouched validation and test sets (original class ratio).

    Selection uses mean VALIDATION `selection_metric` (default PR-AUC, which
    is threshold-independent and robust under heavy imbalance). Test metrics
    are reported for every candidate but never influence the choice.
    A candidate is excluded from winning if it failed on any usable seed, or
    (DeepSMOTE) failed to converge on more than half of them -- and that is
    recorded as a finding instead of being forced.
    """
    if evaluator not in EVALUATORS:
        raise ValueError(f"unknown evaluator '{evaluator}'; choose from {EVALUATORS}")
    if evaluator == "lstm":
        from app.ml.lstm_detector import LSTMConfig
        eval_cfg = eval_cfg or LSTMConfig.evaluator()
    sampler_kwargs = sampler_kwargs or {}
    # `min_recall` is only used by the legacy "recall" threshold mode
    window_policy = ThresholdPolicy.coerce(window_policy) if window_policy is not None else DEFAULT_WINDOW_POLICY
    if window_policy.mode == "recall":
        window_policy = ThresholdPolicy(mode="recall", min_recall=min_recall)
    user_policy = ThresholdPolicy.coerce(user_policy) if user_policy is not None else DEFAULT_USER_POLICY
    T, F = prep.seq_len, len(prep.seq.feature_columns)

    splits, skipped = {}, []
    if kfold:      # every user is a test user exactly once; the "seed" label becomes the fold index
        plan = dict(enumerate(kfold_user_splits(prep.unique_users, prep.user_is_insider, kfold)))
    else:
        plan = {s: None for s in seeds}
    for s, sp in plan.items():
        try:
            splits[s] = make_split(prep, s, val_frac, test_frac, split=sp)
        except ValueError as e:
            skipped.append({"seed": s, "reason": str(e)})
            progress(f"skipping seed {s}: {e}")
    if not splits:
        raise SplitError("No usable seeds: " + "; ".join(x["reason"] for x in skipped))

    results = {}
    ae_caches = {s: {} for s in splits}
    for name in candidates:
        per_seed, errors = [], []
        for s, sd in splits.items():
            progress(f"[layer1] {name} seed={s}")
            try:
                per_seed.append(_fit_eval_candidate(sd, name, eval_cfg, s, window_policy, sampler_kwargs,
                                                    keep_predictions, T, F, prep, user_policy, ae_caches[s], evaluator))
            except Exception as e:      # a sampler failing is a finding, not a crash
                errors.append({"seed": s, "error": f"{type(e).__name__}: {e}"})
                progress(f"[layer1] {name} seed={s} FAILED: {e}")
        unstable = sum(1 for r in per_seed if r["sampler_info"].get("converged") is False)
        entry = {
            "n_ok_seeds": len(per_seed), "errors": errors, "unstable_seeds": unstable, "per_seed": per_seed,
            "summary": {
                "val": aggregate([r["val"] for r in per_seed]),
                "test": aggregate([r["test"] for r in per_seed]),
                "test_at_budget": aggregate([r["test_at_budget"] for r in per_seed if "test_at_budget" in r], ["recall", "precision"]),
                "user": {m: {"val": aggregate([r["user"][m]["val"] for r in per_seed], USER_KEYS),
                             "test": aggregate([r["user"][m]["test"] for r in per_seed], USER_KEYS)}
                         for m in per_seed[0].get("user", {})},
            } if per_seed else None,
        }
        if base_of(name) == "DeepSMOTE" and per_seed:
            entry["convergence_note"] = (
                f"autoencoder converged in {len(per_seed) - unstable}/{len(per_seed)} seeds "
                f"(mean reconstruction ratio {np.mean([r['sampler_info'].get('recon_ratio', np.nan) for r in per_seed]):.2f})"
            )
        results[name] = entry

    excluded, eligible = {}, {}
    for i, (name, e) in enumerate(results.items()):
        if e["n_ok_seeds"] == 0:
            excluded[name] = "failed on every seed: " + (e["errors"][0]["error"] if e["errors"] else "unknown")
        elif e["errors"]:
            excluded[name] = f"failed on {len(e['errors'])} of {len(splits)} seeds"
        elif e["unstable_seeds"] * 2 > e["n_ok_seeds"]:
            excluded[name] = f"did not train reliably (autoencoder unconverged on {e['unstable_seeds']}/{e['n_ok_seeds']} seeds)"
        else:
            sel = e["summary"]["val"][selection_metric]["mean"]
            f1 = e["summary"]["val"]["f1"]["mean"] or 0.0
            if sel is not None:
                eligible[name] = (sel, f1, -i)
    winner = max(eligible, key=eligible.get) if eligible else None
    return {
        "seeds_used": list(splits), "seeds_skipped": skipped, "selection_metric": selection_metric,
        "selection_split": "validation", "min_recall": min_recall,
        "window_policy": window_policy.__dict__, "user_policy": user_policy.__dict__,
        "evaluator": evaluator, "eval_config": asdict(eval_cfg) if eval_cfg is not None else None, "candidates": results,
        "excluded": excluded, "winner": winner,
        "winner_reason": (f"highest mean validation {selection_metric} = {eligible[winner][0]:.4f}" if winner else "no eligible candidate"),
    }   # NB: may hold private `_pred` arrays (keep_predictions=True); run to_jsonable() before persisting


# ------------------------------------------------------------------ Layer 2

def train_production(prep: PreparedData, sampler: str, seed: int, cfg=None, min_recall: float = 0.9,
                     sampler_kwargs: Optional[dict] = None, artifact_dir: Optional[Path] = None,
                     val_frac: float = 0.15, test_frac: float = 0.15,
                     window_policy=None, user_policy=None, user_agg: Optional[dict] = None):
    """Trains the production LSTM once on the chosen sampler's output; tunes + freezes the threshold on val."""
    from app.ml.lstm_detector import LSTMArtifact, LSTMConfig, predict_proba, train_lstm
    cfg = cfg or LSTMConfig.production()
    T, F = prep.seq_len, len(prep.seq.feature_columns)
    sd = make_split(prep, seed, val_frac, test_frac)
    rr = resample(sampler, flatten(sd.X_train), sd.y_train, seed=seed, **kwargs_for(sampler, sampler_kwargs))
    model, history = train_lstm(unflatten(rr.X, T, F), rr.y, cfg.scaled_for(len(rr.y)), seed=seed, pos_weight=rr.pos_weight,
                                X_val=sd.X_val, y_val=sd.y_val)
    p_val, p_test = predict_proba(model, sd.X_val), predict_proba(model, sd.X_test)
    window_policy = ThresholdPolicy.coerce(window_policy) if window_policy is not None else DEFAULT_WINDOW_POLICY
    if window_policy.mode == "recall":
        window_policy = ThresholdPolicy(mode="recall", min_recall=min_recall)
    user_policy = ThresholdPolicy.coerce(user_policy) if user_policy is not None else DEFAULT_USER_POLICY
    thr = choose_threshold(sd.y_val, p_val, window_policy)     # tuned on validation, then frozen
    user_agg = dict(user_agg or DEFAULT_AGG)
    methods = {"max": {"method": "max"}, "topk": {"method": "topk", "k": 3}, "consecutive": {"method": "consecutive", "n": 3}}
    methods[user_agg["method"]] = user_agg
    user_res = evaluate_user_level(prep, sd.idx_val, sd.idx_test, p_val, p_test, user_policy, methods)
    chosen = user_res[user_agg["method"]]
    metrics = {
        "window_policy": window_policy.__dict__, "user_policy": user_policy.__dict__,
        "user": user_res, "user_selected": user_agg["method"],
        "val": compute_metrics(sd.y_val, p_val, thr), "test": compute_metrics(sd.y_test, p_test, thr),
        "threshold": round(thr, 6), "epochs_run": len(history),
        "train_size_after_sampling": int(len(rr.y)), "sampler_info": rr.info,
        "split": {"train_users": int(len(sd.split.train)), "val_users": int(len(sd.split.val)),
                  "test_users": int(len(sd.split.test))},
    }
    artifact = LSTMArtifact(
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=asdict(cfg),
        scaler=sd.scaler, threshold=thr, sampler=sampler, seq_len=T, window_days=prep.window_days,
        feature_columns=list(prep.seq.feature_columns), dataset=prep.dataset, seed=seed,
        metrics=to_jsonable(metrics), user_threshold=chosen["threshold"], user_agg=chosen["agg"],
    )
    if artifact_dir is not None:
        artifact.save(artifact_dir)
    return artifact, metrics, history


# ------------------------------------------------------------------ out-of-fold scores for EVERY user

def run_oof(prep: PreparedData, sampler: str = "none", k: int = 7, cfg=None, n_inits: int = 3,
            user_agg: Optional[dict] = None, user_policy=None, progress: Callable[[str], None] = print) -> pd.DataFrame:
    """
    k-fold over users. Each user is scored by models that never trained on them (fold i = test, fold i+1 = val).
    Several initialisations per fold are averaged. The user threshold is tuned on that fold's validation users and
    applied to its test users. Returns one row per user: user_id, fold, user_score, threshold, flagged, is_insider.
    """
    from app.ml.lstm_detector import LSTMConfig, predict_proba, train_lstm
    from app.ml.user_level import aggregate_user_scores, user_truth
    cfg = cfg or LSTMConfig.production()
    agg = user_agg or DEFAULT_AGG
    kw = {a: v for a, v in agg.items() if a in ("method", "k", "n")}
    pol = ThresholdPolicy.coerce(user_policy) if user_policy is not None else DEFAULT_USER_POLICY
    T, F = prep.seq_len, len(prep.seq.feature_columns)
    uid, ws = prep.features_df["user_id"].to_numpy(), prep.seq.window_starts
    out = []
    for i, sp in enumerate(kfold_user_splits(prep.unique_users, prep.user_is_insider, k)):
        progress(f"[oof] fold {i + 1}/{k}")
        sd = make_split(prep, i, split=sp)
        rr = resample(sampler, flatten(sd.X_train), sd.y_train, seed=i, **kwargs_for(sampler, None))
        Xr = unflatten(rr.X, T, F)
        pv = pt = 0.0
        for j in range(n_inits):
            model, _ = train_lstm(Xr, rr.y, cfg.scaled_for(len(rr.y)), seed=100 * i + j, pos_weight=rr.pos_weight,
                                  X_val=sd.X_val, y_val=sd.y_val)
            pv = pv + predict_proba(model, sd.X_val) / n_inits
            pt = pt + predict_proba(model, sd.X_test) / n_inits
        sv = aggregate_user_scores(uid[sd.idx_val], ws[sd.idx_val], pv, window_days=prep.window_days, **kw)
        st = aggregate_user_scores(uid[sd.idx_test], ws[sd.idx_test], pt, window_days=prep.window_days, **kw)
        tv = user_truth(sv.index.to_numpy(), prep.unique_users, prep.user_is_insider)
        thr = choose_threshold(tv, sv.to_numpy(), pol)
        tt = user_truth(st.index.to_numpy(), prep.unique_users, prep.user_is_insider)
        n_top = max(1, int(round((pol.budget if pol.mode == "budget" else 0.10) * len(st))))
        top = st.rank(ascending=False, method="first").to_numpy() <= n_top      # threshold-free: top X% of this fold
        out.append(pd.DataFrame({"user_id": st.index.to_numpy(), "fold": i, "user_score": st.to_numpy(),
                                 "threshold": thr, "flagged": st.to_numpy() >= thr, "flagged_at_budget": top,
                                 "is_insider": tt.astype(bool)}))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ run bookkeeping
# Two recorders with the same interface: the Postgres/API path stores results on a DetectionRun row;
# the DB-free (Kaggle) path writes them to JSON files in --out-dir.

class _DbRecorder:
    def __init__(self, db, run):
        self.db, self.run = db, run

    def update(self, **fields):
        for k, v in fields.items():
            setattr(self.run, k, v)
        self.db.commit()

    def rollback(self):
        self.db.rollback()


class _FileRecorder:
    def __init__(self, out_dir: Path, kind: str):
        self.path, self.state = Path(out_dir) / f"result_{kind}.json", {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, **fields):
        import json
        self.state.update({k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in fields.items()})
        self.path.write_text(json.dumps(to_jsonable(self.state), indent=2))

    def rollback(self):
        pass


def _acquire_data(args, out_dir: Path, use_db: bool, db=None):
    """Returns (PreparedData, extra_config). Source: cached features file > streamed CSVs > database."""
    import pickle
    from app.ml.offline_features import (
        build_features_from_csvs, find_dataset_dir, list_users, load_features, save_features, select_users,
    )
    extra: dict = {}
    if use_db:
        return prepare_from_db(db, args.answers, args.dataset, args.window_days, args.seq_len, args.labels_dataset), extra

    cache = Path(args.features) if args.features else out_dir / f"features_w{args.window_days}.pkl"
    labels = load_labels(args.answers, dataset=args.labels_dataset)
    insiders = set(labels.loc[labels["malicious"] == 1, "user"])
    if cache.exists() and not args.rebuild_features:
        log.info("loading cached features from %s", cache)
        features_df, user_codes, meta = load_features(cache)
        if meta.get("window_days") not in (None, args.window_days):
            raise ValueError(f"cached features use window_days={meta['window_days']}, requested {args.window_days} "
                             "(delete the cache or pass --rebuild-features)")
    elif args.csv_dir:
        directory = find_dataset_dir(args.csv_dir)
        log.info("streaming CERT CSVs from %s (this is the slow, one-time pass)", directory)
        all_users = list_users(directory)
        keep = select_users(all_users, insiders, args.max_benign_users) if args.max_benign_users else None
        features_df, user_codes = build_features_from_csvs(
            directory, args.window_days, args.internal_domain, keep, chunksize=args.chunksize, progress=log.info)
        meta = {"window_days": args.window_days, "internal_domain": args.internal_domain,
                "n_users_source": len(all_users), "max_benign_users": args.max_benign_users}
        save_features(cache, features_df, user_codes, meta)
        log.info("cached features to %s", cache)
    else:
        raise FileNotFoundError(f"No cached features at {cache} and no --csv-dir given to build them.")

    # optional post-hoc benign-user sampling (also applies to a cache built on more users)
    codes_present = set(user_codes.values())
    if args.max_benign_users:
        keep = select_users(codes_present, insiders, args.max_benign_users)
        keep_ids = {i for i, c in user_codes.items() if c in keep}
        features_df = features_df[features_df["user_id"].isin(keep_ids)].reset_index(drop=True)
        user_codes = {i: c for i, c in user_codes.items() if i in keep_ids}
    n_source = meta.get("n_users_source") or len(codes_present)
    n_kept = len(set(user_codes.values()))
    extra = {"features_cache": str(cache), "users_in_source": n_source, "users_used": n_kept, "internal_domain": meta.get("internal_domain")}
    if n_kept < n_source:
        extra["sampling_note"] = (f"benign users subsampled ({n_kept} of {n_source} users kept, all insiders retained): "
                                  "insider prevalence in val/test is higher than in the full population, so precision and "
                                  "PR-AUC are optimistic relative to the real population")
    return prepare_from_features(features_df, user_codes, args.answers, args.dataset, args.window_days,
                                 args.seq_len, args.labels_dataset), extra


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline LSTM training / sampler comparison / evaluation job")
    ap.add_argument("--mode", choices=["compare", "train", "evaluate"], default="train")
    ap.add_argument("--answers", required=True, help="ground-truth CSV (user,malicious[,start,end] or CERT insiders.csv)")
    ap.add_argument("--dataset", default="cert", help="cert (CERT r4.2 only)")
    ap.add_argument("--labels-dataset", default=None, help="filter for CERT insiders.csv 'dataset' column, e.g. r4.2")
    ap.add_argument("--window-days", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--candidates", nargs="+", default=None, help="sampler specs, e.g. none NM-2@0.05 DeepSMOTE@0.05+cw")
    ap.add_argument("--selection-metric", default="pr_auc", choices=["pr_auc", "f1", "recall", "precision"])
    ap.add_argument("--min-recall", type=float, default=0.9, help="only used with --threshold-mode recall")
    ap.add_argument("--threshold-mode", default="budget", choices=["budget", "precision", "fbeta", "recall"])
    ap.add_argument("--window-budget", type=float, default=0.01, help="budget mode: fraction of windows to flag")
    ap.add_argument("--user-budget", type=float, default=0.10, help="fraction of users to flag")
    ap.add_argument("--user-agg", default="topk", choices=["max", "topk", "consecutive"])
    ap.add_argument("--evaluator", default="hgb", choices=list(EVALUATORS), help="Layer 1: model that scores each sampler candidate")
    ap.add_argument("--sweep", action="store_true", help="use the partial-ratio / hybrid candidate sweep")
    # --- data source (default: the Postgres database, as launched by the API)
    ap.add_argument("--csv-dir", default=None, help="DB-FREE: stream the raw CERT CSVs from here (e.g. /kaggle/input/<dataset>)")
    ap.add_argument("--features", default=None, help="DB-FREE: path of the cached features file (built from --csv-dir if missing)")
    ap.add_argument("--rebuild-features", action="store_true", help="ignore an existing features cache")
    ap.add_argument("--max-benign-users", type=int, default=None, help="keep all insiders + this many random benign users (memory/time)")
    ap.add_argument("--internal-domain", default="@company.com", help='e-mail domain counted as internal (real CERT: "@dtaa.com")')
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="rows per CSV chunk when streaming")
    ap.add_argument("--out-dir", default="insider_out", help="DB-free mode: where features, model and result JSON are written")
    # --- DB mode bookkeeping
    ap.add_argument("--run-id", type=int, default=None, help="existing DetectionRun to update (set by the API)")
    ap.add_argument("--artifact-dir", default=None)
    args = ap.parse_args(argv)
    args.candidates = args.candidates or list(SWEEP_CANDIDATES if args.sweep else SAMPLER_CANDIDATES)
    wpol = ThresholdPolicy(mode=args.threshold_mode, budget=args.window_budget, min_recall=args.min_recall)
    upol = ThresholdPolicy(mode="budget", budget=args.user_budget)
    uagg = {"method": args.user_agg, "k": 3, "n": 3}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    use_db = not (args.csv_dir or args.features)
    kind = {"compare": "sampler_comparison", "train": "train", "evaluate": "evaluation"}[args.mode]
    out_dir = Path(args.out_dir)
    db = None

    if use_db:
        from app.config import ARTIFACT_DIR
        from app.database import SessionLocal, engine
        from app.migrations import ensure_schema
        from app.models import DetectionRun
        ensure_schema(engine)
        db = SessionLocal()
        run = db.query(DetectionRun).get(args.run_id) if args.run_id else None
        if run is None:
            run = DetectionRun(sampling_technique="pending", detector="LSTM", window_days=args.window_days)
            db.add(run)
            db.commit()
        rec = _DbRecorder(db, run)
        rec.update(run_kind=kind, model_type="LSTM", detector="LSTM", dataset=args.dataset,
                   window_days=args.window_days, status="running", notes=None)
        art_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_DIR / f"run_{run.id}"
    else:
        rec = _FileRecorder(out_dir, kind)
        rec.update(run_kind=kind, model_type="LSTM", detector="LSTM", dataset=args.dataset,
                   window_days=args.window_days, status="running")
        art_dir = Path(args.artifact_dir) if args.artifact_dir else out_dir / "model"

    try:
        prep, extra = _acquire_data(args, out_dir, use_db, db)
        try:
            import torch
            extra["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        log.info("prepared %d samples, %d users (%d insiders), %d positive samples",
                 len(prep.y), len(prep.unique_users), int(prep.user_is_insider.sum()), int(prep.y.sum()))
        base_cfg = {"dataset": args.dataset, "window_days": args.window_days, "seq_len": args.seq_len,
                    "seeds": args.seeds, "candidates": args.candidates, "selection_metric": args.selection_metric,
                    "min_recall": args.min_recall, "answers": args.answers, **extra}

        if args.mode == "evaluate":
            from app.ml.evaluation import render_markdown, run_full_evaluation
            report = run_full_evaluation(prep, args.seeds, args.candidates, args.selection_metric, args.min_recall, progress=log.info,
                                          window_policy=wpol, user_policy=upol)
            art_dir.mkdir(parents=True, exist_ok=True)
            (art_dir / f"evaluation_{args.dataset}.md").write_text(render_markdown(report))
            rec.update(status="completed", finished_at=datetime.utcnow(),
                       sampling_technique=report["layer1"].get("winner") or "n/a",
                       metrics=to_jsonable(report), config=to_jsonable({**base_cfg, "report_dir": str(art_dir)}),
                       notes=f"full evaluation over seeds {report['layer1']['seeds_used']}")
            log.info("report written to %s", art_dir / f"evaluation_{args.dataset}.md")
            return 0

        comp = run_sampler_comparison(prep, args.seeds, args.candidates, selection_metric=args.selection_metric,
                                      min_recall=args.min_recall, progress=log.info,
                                      window_policy=wpol, user_policy=upol, evaluator=args.evaluator)
        winner = comp["winner"]
        if args.mode == "compare":
            rec.update(status="completed", finished_at=datetime.utcnow(), sampling_technique=winner or "n/a",
                       metrics=to_jsonable({"layer1": comp}), config=to_jsonable(base_cfg),
                       notes=f"winner: {winner} ({comp['winner_reason']})")
            return 0

        if winner is None:
            raise RuntimeError("No eligible sampler in Layer 1: " + str(comp["excluded"]))
        final_seed = comp["seeds_used"][0]
        log.info("[layer2] training production LSTM with sampler=%s seed=%d", winner, final_seed)
        art, m, _ = train_production(prep, winner, final_seed, min_recall=args.min_recall, artifact_dir=art_dir,
                                  window_policy=wpol, user_policy=upol, user_agg=uagg)
        t = m["test"]
        rec.update(status="completed", finished_at=datetime.utcnow(), sampling_technique=winner,
                   precision=t["precision"], recall=t["recall"], f_score=t["f1"], accuracy=t["accuracy"],
                   metrics=to_jsonable({"production": m, "layer1": comp}),
                   config=to_jsonable({**base_cfg, "sampler": winner, "final_seed": final_seed,
                                       "lstm_config": art.config, "threshold": art.threshold,
                                       "artifact_dir": str(art_dir)}),
                   notes=f"LSTM + {winner}: test recall={t['recall']} precision={t['precision']} pr_auc={t['pr_auc']}")
        log.info("artifact saved to %s", art_dir)
        return 0
    except Exception as e:
        rec.rollback()
        rec.update(status="failed", finished_at=datetime.utcnow(), notes=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    sys.exit(main())