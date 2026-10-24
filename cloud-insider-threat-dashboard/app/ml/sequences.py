"""
Sequence builder, labels, user-level splits and the train-only scaler
(architecture.md sections 3.3 and 3.4).

Sample i <-> row i of the feature-window DataFrame: sample i is the T x F
matrix of the user's last T *time windows* ending at row i's window. Windows
in which the user had no events are real zero-activity windows (filled with
0), not skipped -- so the sequence is time-aligned. Windows before the
user's first observed window are unknown history: NaN here, 0 after scaling
(the population-mean equivalent).

Leakage rules enforced by this module:
  * splits are by USER (overlapping windows of one user never straddle splits)
  * the scaler is fit on train-split users' rows only
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.ml.feature_core import FEATURE_COLUMNS


@dataclass
class SequenceData:
    X: np.ndarray            # (N, T, F) raw features, NaN = no history yet
    lengths: np.ndarray      # (N,) number of real windows in the sequence (<= T)
    user_ids: np.ndarray     # (N,)
    window_starts: np.ndarray
    window_ends: np.ndarray
    seq_len: int
    feature_columns: List[str]


def _window_grid(features_df: pd.DataFrame):
    starts = pd.to_datetime(features_df["window_start"])
    ends = pd.to_datetime(features_df["window_end"])
    step = (ends - starts).iloc[0]
    origin = starts.min()
    idx = ((starts - origin) / step).round().astype("int64").to_numpy()
    return idx, step


def build_sequences(features_df: pd.DataFrame, seq_len: int = 7,
                    feature_columns: Optional[List[str]] = None) -> SequenceData:
    cols = list(feature_columns or FEATURE_COLUMNS)
    n, F = len(features_df), len(cols)
    X = np.full((n, seq_len, F), np.nan, dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    if n == 0:
        return SequenceData(X, lengths, np.array([]), np.array([]), np.array([]), seq_len, cols)

    win_idx, _ = _window_grid(features_df)
    feats = features_df[cols].fillna(0).to_numpy(dtype=np.float32)
    users = features_df["user_id"].to_numpy()

    order = np.lexsort((win_idx, users))
    su = users[order]
    bounds = np.flatnonzero(np.r_[True, su[1:] != su[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        pos = order[a:b]
        w = win_idx[pos]
        first = w[0]
        dense = np.zeros((w[-1] - first + 1, F), dtype=np.float32)   # gaps = zero activity
        dense[w - first] = feats[pos]
        for p, wi in zip(pos, w):
            hi = wi - first + 1
            chunk = dense[max(0, hi - seq_len):hi]
            X[p, seq_len - len(chunk):] = chunk
            lengths[p] = len(chunk)

    return SequenceData(
        X=X, lengths=lengths, user_ids=users,
        window_starts=pd.to_datetime(features_df["window_start"]).to_numpy(),
        window_ends=pd.to_datetime(features_df["window_end"]).to_numpy(),
        seq_len=seq_len, feature_columns=cols,
    )


# ---------------------------------------------------------------- labels

def load_labels(path: str, dataset: Optional[str] = None) -> pd.DataFrame:
    """
    Loads a ground-truth file into columns: user, malicious (0/1), start, end.

    Accepts either
      * `user,malicious[,start,end]`  (this project's answers.csv), or
      * CERT's own insiders.csv (`dataset,scenario,details,user,start,end`) --
        every listed user is an insider; pass `dataset` (e.g. "r4.2") to keep
        only that release's rows.

    If start/end are present a sample is malicious only when its window
    overlaps the insider's malicious period ("labeled malicious if the user
    is a known insider in that period"); without them the whole user is
    labeled malicious.
    """
    df = pd.read_csv(path)
    if "user" not in df.columns:
        raise ValueError("labels file must have a 'user' column")
    if dataset and "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == dataset]
    df = df.copy()
    df["user"] = df["user"].astype(str)
    if "malicious" in df.columns:
        df["malicious"] = df["malicious"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
    else:
        df["malicious"] = 1
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], errors="coerce") if c in df.columns else pd.NaT
    return df[["user", "malicious", "start", "end"]].reset_index(drop=True)


def label_samples(features_df: pd.DataFrame, labels: pd.DataFrame, user_code_by_id: Dict[int, str]) -> np.ndarray:
    codes = features_df["user_id"].map(user_code_by_id).to_numpy()
    ws = pd.to_datetime(features_df["window_start"]).to_numpy()
    we = pd.to_datetime(features_df["window_end"]).to_numpy()
    y = np.zeros(len(features_df), dtype=np.int64)
    for row in labels.itertuples(index=False):
        if not row.malicious:
            continue
        m = codes == row.user
        if pd.notna(row.start):
            m &= we > np.datetime64(row.start)
        if pd.notna(row.end):
            m &= ws <= np.datetime64(row.end)
        y[m] = 1
    return y


def insider_flags(unique_user_ids: np.ndarray, labels: pd.DataFrame, user_code_by_id: Dict[int, str]) -> np.ndarray:
    insiders = set(labels.loc[labels["malicious"] == 1, "user"])
    return np.array([user_code_by_id.get(int(u)) in insiders for u in unique_user_ids])


# ---------------------------------------------------------------- splits

@dataclass
class UserSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int

    def masks(self, user_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.isin(user_ids, self.train), np.isin(user_ids, self.val), np.isin(user_ids, self.test))


def _allocate(n: int, val_frac: float, test_frac: float, what: str) -> Tuple[int, int]:
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n - n_val - n_test < 1:
        raise ValueError(
            f"Need at least 3 {what} users to give train/val/test each at least one "
            f"(got {n}). Add more users/insiders or use a larger dataset."
        )
    return n_val, n_test


def user_level_split(user_ids: np.ndarray, insider_mask: np.ndarray, val_frac: float = 0.15,
                     test_frac: float = 0.15, seed: int = 0) -> UserSplit:
    """
    Stratified split over *users*: insiders and non-insiders are each divided
    so every split contains at least one insider. No user appears in more than
    one split. Val/test keep the population's natural class ratio.
    """
    rng = np.random.RandomState(seed)
    parts = {"train": [], "val": [], "test": []}
    for flag, what in ((True, "insider"), (False, "non-insider")):
        ids = np.array(user_ids)[insider_mask == flag]
        ids = ids[rng.permutation(len(ids))]
        n_val, n_test = _allocate(len(ids), val_frac, test_frac, what)
        parts["val"].append(ids[:n_val])
        parts["test"].append(ids[n_val:n_val + n_test])
        parts["train"].append(ids[n_val + n_test:])
    return UserSplit(*(np.concatenate(parts[k]) for k in ("train", "val", "test")), seed=seed)


# ---------------------------------------------------------------- scaler

class FeatureScaler:
    """Per-feature standardization, fit on train-split rows only."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean, self.std = mean, std

    def fit(self, features_df: pd.DataFrame, feature_columns: List[str]) -> "FeatureScaler":
        X = features_df[feature_columns].fillna(0).to_numpy(dtype=np.float64)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Works on (N, F) or (N, T, F); NaN (no-history padding) becomes 0."""
        out = (X - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def unflatten(X: np.ndarray, seq_len: int, n_features: int) -> np.ndarray:
    return X.reshape(len(X), seq_len, n_features)

# ---------------------------------------------------------------- per-user baseline features (added)

COUNT_COLUMNS = [c for c in FEATURE_COLUMNS if c != "email_external_ratio"]


def engineer_features(features_df: pd.DataFrame, span: int = 15, min_periods: int = 5):
    """
    Adds one `<col>_z` column per count feature: how unusual this window is versus the SAME user's own
    previous `span` windows (log1p scale; strictly past data, shift(1) -> no leakage, independent of the
    train/val/test split). Then log1p-compresses the raw counts. Returns (df, feature_columns).
    Call it on the cached features_df BEFORE prepare_from_features, and again before scoring in the app.
    """
    df = features_df.copy()
    ws = pd.to_datetime(df["window_start"])
    step = (pd.to_datetime(df["window_end"]) - ws).iloc[0]
    df["_w"] = ((ws - ws.min()) / step).round().astype("int64")
    zcols = [f"{c}_z" for c in COUNT_COLUMNS]
    z_all = pd.DataFrame(0.0, index=df.index, columns=zcols)
    for _, g in df.groupby("user_id", sort=False):
        g = g.sort_values("_w")
        grid = range(int(g["_w"].iloc[0]), int(g["_w"].iloc[-1]) + 1)
        dense = np.log1p(g.set_index("_w")[COUNT_COLUMNS].reindex(grid, fill_value=0).astype(float))   # gaps = zero activity
        past = dense.shift(1)
        mu = past.rolling(span, min_periods=min_periods).mean()
        sd = past.rolling(span, min_periods=min_periods).std()
        z = ((dense - mu) / (sd.fillna(0) + 0.5)).clip(-5, 5).fillna(0.0)    # +0.5 keeps near-constant users from exploding
        z_all.loc[g.index, zcols] = z.loc[g["_w"].to_numpy()].to_numpy()
    df[zcols] = z_all
    df[COUNT_COLUMNS] = np.log1p(df[COUNT_COLUMNS].astype(float))
    return df.drop(columns="_w"), list(FEATURE_COLUMNS) + zcols


# ---------------------------------------------------------------- k-fold user splits (added)

def kfold_user_splits(user_ids: np.ndarray, insider_mask: np.ndarray, k: int = 7, seed: int = 0) -> List[UserSplit]:
    """
    Stratified k-fold over USERS. Fold i is the TEST set, fold i+1 is VALIDATION, the rest is TRAIN, so across the
    k splits every user (and every insider) is a test user exactly once. k=7 with 70 insiders -> 10 insiders/fold.
    """
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for flag in (True, False):
        ids = np.array(user_ids)[insider_mask == flag]
        ids = ids[rng.permutation(len(ids))]
        for i, chunk in enumerate(np.array_split(ids, k)):
            folds[i].append(chunk)
    folds = [np.concatenate(f) for f in folds]
    out = []
    for i in range(k):
        v = (i + 1) % k
        train = np.concatenate([folds[j] for j in range(k) if j not in (i, v)])
        out.append(UserSplit(train=train, val=folds[v], test=folds[i], seed=i))
    return out
