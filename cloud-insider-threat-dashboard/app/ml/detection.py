"""
Detection pipeline.

Detector interface (`BaseDetector`, `get_detector`) + the five classical
detectors. The LSTM detector (app/ml/lstm_detector.py) implements the same
interface. Two things live here, mirroring the two layers from the paper this project
extends ("Malicious Insider Threat Detection Using Variation of Sampling
Methods for Anomaly Detection in Cloud Environment"):

1. `run_unsupervised_detection` — production path. No ground-truth labels
   needed. Runs one of the paper's Layer-2 anomaly detectors (OCSVM, IF,
   Robust Covariance, LOF, KNN-distance) over the feature windows and
   returns a normalized anomaly score per row. This is what powers the
   live dashboard.

2. `run_sampling_benchmark` — evaluation path. If you have labeled data
   (CERT ships an answer key identifying which users were scripted as
   malicious), this reproduces the paper's Layer-1 comparison: try each
   sampling technique (ROS/SMOTE/ADASYN/RUS/NM-1/NM-2/T-L/ENN), score with
   an SVM, and report precision/recall/f-score/accuracy so you can compare
   your results against the paper's reported numbers (NM-2 + OCSVM: recall
   100%, f-score 78.72%, accuracy 82.46%).
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from app.ml.feature_core import FEATURE_COLUMNS

CLASSICAL_DETECTORS = {"OCSVM", "IF", "RC", "LOF", "KNN"}
DETECTORS = CLASSICAL_DETECTORS | {"LSTM"}   # LSTM lives in app/ml/lstm_detector.py (needs PyTorch)
SAMPLING_TECHNIQUES = {"ROS", "SMOTE", "ADASYN", "RUS", "NM-1", "NM-2", "T-L", "ENN", "none"}


def _scale(features_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    X = features_df[FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


class BaseDetector(ABC):
    """
    Common detector interface. Every detector -- the five classical ones and
    the LSTM -- takes the output of build_feature_windows and returns a copy
    with three added columns:

        raw_score     detector's native output (higher = more anomalous)
        score         normalized 0-1, higher = more anomalous
        is_anomalous  0/1 flag

    so they are swappable at runtime (the pipeline router only ever calls
    `get_detector(name).score(features_df)`).
    """
    name: str = "base"

    @abstractmethod
    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        ...


class ClassicalDetector(BaseDetector):
    """OCSVM / IF / RC / LOF / KNN. Unsupervised: no labels needed."""

    def __init__(self, name: str, contamination: float = 0.05):
        if name not in CLASSICAL_DETECTORS:
            raise ValueError(f"Unknown detector '{name}'. Choose from {sorted(DETECTORS)}")
        self.name = name
        self.contamination = contamination
        self._model = None

    # -- transductive path: fit and score the same data (what powers the live dashboard)
    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        detector, contamination = self.name, self.contamination
        if features_df.empty:
            return features_df.assign(raw_score=[], score=[], is_anomalous=[])

        X, _ = _scale(features_df)
        out = features_df.copy()

        if detector == "OCSVM":
            model = OneClassSVM(kernel="rbf", gamma="auto", nu=max(contamination, 0.01))
            model.fit(X)
            raw = -model.decision_function(X)   # flip sign: higher = more anomalous
            pred = model.predict(X)             # -1 anomaly, 1 normal

        elif detector == "IF":
            model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            model.fit(X)
            raw = -model.decision_function(X)
            pred = model.predict(X)

        elif detector == "RC":
            model = EllipticEnvelope(contamination=contamination, random_state=42)
            model.fit(X)
            raw = -model.decision_function(X)
            pred = model.predict(X)

        elif detector == "LOF":
            n_neighbors = min(35, max(2, len(X) - 1))
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            pred = model.fit_predict(X)
            raw = -model.negative_outlier_factor_

        else:  # KNN distance-based
            n_neighbors = min(10, max(2, len(X) - 1))
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            raw = distances.mean(axis=1)
            threshold = np.quantile(raw, 1 - contamination)
            pred = np.where(raw >= threshold, -1, 1)

        # normalize raw score to 0-1 for display
        raw = np.asarray(raw, dtype=float)
        rmin, rmax = raw.min(), raw.max()
        score = (raw - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(raw)

        out["raw_score"] = raw
        out["score"] = score
        out["is_anomalous"] = (pred == -1).astype(int)
        return out

    # -- inductive path: fit on train rows, score unseen rows (used by the evaluation
    #    harness so classical detectors are compared on the same user-level split)
    def fit(self, X_train: np.ndarray) -> "ClassicalDetector":
        n, c = len(X_train), self.contamination
        if self.name == "OCSVM":
            self._model = OneClassSVM(kernel="rbf", gamma="auto", nu=max(c, 0.01)).fit(X_train)
        elif self.name == "IF":
            self._model = IsolationForest(n_estimators=100, contamination=c, random_state=42).fit(X_train)
        elif self.name == "RC":
            self._model = EllipticEnvelope(contamination=c, random_state=42).fit(X_train)
        elif self.name == "LOF":
            self._model = LocalOutlierFactor(n_neighbors=min(35, max(2, n - 1)), contamination=c, novelty=True).fit(X_train)
        else:
            self._model = NearestNeighbors(n_neighbors=min(10, max(2, n - 1))).fit(X_train)
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score for unseen rows, higher = more anomalous."""
        if self._model is None:
            raise RuntimeError("call fit() first")
        if self.name in ("OCSVM", "IF", "RC"):
            return -self._model.decision_function(X)
        if self.name == "LOF":
            return -self._model.score_samples(X)
        dist, _ = self._model.kneighbors(X)
        return dist.mean(axis=1)


def get_detector(name: str, contamination: float = 0.05, **kwargs) -> BaseDetector:
    """Registry: name -> detector instance. 'LSTM' requires `artifact_dir=`."""
    if name == "LSTM":
        from app.ml.lstm_detector import LSTMDetector    # lazy: keeps PyTorch optional
        return LSTMDetector(**kwargs)
    return ClassicalDetector(name, contamination=contamination)


def run_unsupervised_detection(
    features_df: pd.DataFrame,
    detector: str = "OCSVM",
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Scores each row of `features_df` (output of build_feature_windows) with
    the chosen classical anomaly detector. Adds `raw_score`, `score`
    (normalized 0-1, higher = more anomalous) and `is_anomalous` (0/1).
    Kept as a thin wrapper for backwards compatibility.
    """
    return ClassicalDetector(detector, contamination).score(features_df)


def compute_top_reasons(features_df: pd.DataFrame, row_index: int, top_n: int = 3) -> dict:
    """
    Lightweight explainability: for the given row, compute each feature's
    z-score relative to the whole population in `features_df` and return
    the top_n features with the highest positive deviation. This is what
    populates "flagged due to: ..." in the alert UI.
    """
    X = features_df[FEATURE_COLUMNS].fillna(0)
    means, stds = X.mean(), X.std().replace(0, 1)
    z = (X.loc[row_index] - means) / stds
    top = z.sort_values(ascending=False).head(top_n)
    return {feat: round(float(val), 2) for feat, val in top.items() if val > 0}


def _apply_sampler(technique: str, X: np.ndarray, y: np.ndarray):
    if technique == "none":
        return X, y

    if technique == "ROS":
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=42)
    elif technique == "SMOTE":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=42, k_neighbors=min(5, y.sum() - 1) if y.sum() > 1 else 1)
    elif technique == "ADASYN":
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(random_state=42, n_neighbors=min(5, y.sum() - 1) if y.sum() > 1 else 1)
    elif technique == "RUS":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=42)
    elif technique == "NM-1":
        from imblearn.under_sampling import NearMiss
        sampler = NearMiss(version=1)
    elif technique == "NM-2":
        from imblearn.under_sampling import NearMiss
        sampler = NearMiss(version=2)
    elif technique == "T-L":
        from imblearn.under_sampling import TomekLinks
        sampler = TomekLinks()
    elif technique == "ENN":
        from imblearn.under_sampling import EditedNearestNeighbours
        sampler = EditedNearestNeighbours()
    else:
        raise ValueError(f"Unknown sampling technique '{technique}'. Choose from {sorted(SAMPLING_TECHNIQUES)}")

    return sampler.fit_resample(X, y)


def run_sampling_benchmark(
    features_df: pd.DataFrame,
    labels: pd.Series,
    sampling_technique: str = "NM-2",
) -> dict:
    """
    Evaluation-only path (needs ground-truth `labels`, aligned by index with
    `features_df`, 1 = known malicious insider, 0 = genuine). Reproduces the
    paper's Layer 1: apply a sampling technique, train an SVM, report
    precision/recall/f-score/accuracy on a held-out split.
    """
    X, _ = _scale(features_df)
    y = labels.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.sum() > 1 else None
    )

    X_res, y_res = _apply_sampler(sampling_technique, X_train, y_train)

    # class_weight="balanced" compensates for any residual imbalance the
    # sampler didn't fully correct; gamma="scale" adapts to this project's
    # feature count/variance rather than reusing the paper's fixed gamma,
    # which was tuned for their specific feature set.
    clf = SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced")
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)

    return {
        "sampling_technique": sampling_technique,
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "train_size": len(y_res),
        "test_size": len(y_test),
    }