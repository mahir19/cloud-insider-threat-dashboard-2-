"""Streamlit-free training pipeline shared by dashboard.py and kaggle_run.py."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.ml import hgb_detector, offline_features, sequences, train
from app.ml.metrics import to_jsonable


def release_value(answer_path: Path, release: str) -> str | None:
    raw = pd.read_csv(answer_path)
    if "dataset" not in raw.columns:
        return None
    norm = lambda value: str(value).strip().lower().lstrip("r")
    hits = [value for value in raw["dataset"].astype(str).unique() if norm(value) == norm(release)]
    if not hits:
        available = ", ".join(sorted(raw["dataset"].astype(str).unique()))
        raise ValueError(f"The labels file has no {release} rows. Available releases: {available}")
    return hits[0]


def feature_cache(data_dir: Path, run_dir: Path, window_days: int, domain: str,
                   chunksize: int, max_benign: int, rebuild: bool, labels: pd.DataFrame,
                   report) -> tuple[pd.DataFrame, dict[int, str], dict]:
    cache = run_dir / f"features_w{window_days}.pkl"
    requested_benign = max_benign or None
    if cache.exists() and not rebuild:
        try:
            features_df, user_codes, meta = offline_features.load_features(cache)
            same = (
                meta.get("source_dir") == str(data_dir.resolve())
                and meta.get("window_days") == window_days
                and meta.get("internal_domain") == domain
                and meta.get("max_benign_users") == requested_benign
            )
            if same:
                report(f"Loaded cached features from {cache}")
                return features_df, user_codes, meta
        except (OSError, ValueError, EOFError, KeyError):
            pass

    insiders = set(labels.loc[labels["malicious"] == 1, "user"].astype(str))
    all_users = offline_features.list_users(data_dir, chunksize=chunksize)
    keep_users = None
    if requested_benign is not None:
        keep_users = offline_features.select_users(all_users, insiders, requested_benign)
    report(f"Building features for {len(all_users):,} source users")
    features_df, user_codes = offline_features.build_features_from_csvs(
        data_dir,
        window_days=window_days,
        internal_domain=domain,
        keep_users=keep_users,
        chunksize=chunksize,
        progress=report,
    )
    meta = {
        "source_dir": str(data_dir.resolve()),
        "window_days": window_days,
        "internal_domain": domain,
        "max_benign_users": requested_benign,
        "n_users_source": len(all_users),
    }
    offline_features.save_features(cache, features_df, user_codes, meta)
    report(f"Feature cache saved to {cache}")
    return features_df, user_codes, meta


def train_run(config: dict, report) -> dict:
    data_dir = offline_features.find_dataset_dir(config["data_dir"])
    answers = Path(config["answers"])
    if not answers.is_file():
        raise FileNotFoundError(f"Labels file not found: {answers}")
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_release = release_value(answers, config["release"])
    labels = sequences.load_labels(str(answers), dataset=labels_release)

    features_df, user_codes, meta = feature_cache(
        data_dir, run_dir, config["window_days"], config["internal_domain"],
        config["chunksize"], config["max_benign"], config["rebuild"], labels, report,
    )
    features_df, feature_columns = sequences.engineer_features(features_df)
    report(f"Preparing {len(features_df):,} user-window rows with {len(feature_columns)} model features")
    prep = train.prepare_from_features(
        features_df,
        user_codes,
        str(answers),
        "cert",
        config["window_days"],
        config["seq_len"],
        labels_release,
        feature_columns=feature_columns,
    )

    artifact_dir = run_dir / "model"
    artifact, metrics, split_data = hgb_detector.train_production_hgb(
        prep,
        sampler=config["sampler"],
        seed=config["seed"],
        window_policy={"mode": "budget", "budget": config["window_budget"]},
        user_policy={"mode": "budget", "budget": config["user_budget"]},
        user_agg={"method": "topk", "k": 3, "n": 3},
        artifact_dir=str(artifact_dir),
    )
    detector = hgb_detector.HGBDetector(artifact)
    test_user_ids = prep.features_df["user_id"].to_numpy()[split_data.idx_test]
    ranked = detector.score_users(
        test_user_ids,
        prep.seq.window_starts[split_data.idx_test],
        split_data.X_test,
    )
    ranked.index.name = "user_id"
    ranked = ranked.reset_index()
    ranked["user"] = ranked["user_id"].map(user_codes)
    insider_codes = set(labels.loc[labels["malicious"] == 1, "user"].astype(str))
    ranked["is_labeled_insider"] = ranked["user"].isin(insider_codes)
    ranked = ranked.sort_values("user_score", ascending=False)
    score_path = run_dir / "test_user_scores.csv"
    ranked.to_csv(score_path, index=False)

    model_config = {
        "window_days": config["window_days"],
        "seq_len": config["seq_len"],
        "internal_domain": config["internal_domain"],
        "feature_columns": feature_columns,
        "sampler": config["sampler"],
        "seed": config["seed"],
    }
    (artifact_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    result = {
        "config": {**model_config, "release": config["release"], "dataset_dir": str(data_dir),
                   "user_budget": config["user_budget"], "window_budget": config["window_budget"],
                   "n_users_source": meta.get("n_users_source", len(user_codes)),
                   "n_users_used": len(user_codes)},
        "metrics": to_jsonable(metrics),
        "model_path": str(artifact_dir / "hgb.pkl"),
        "scores_path": str(score_path),
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {**result, "ranked": ranked}
