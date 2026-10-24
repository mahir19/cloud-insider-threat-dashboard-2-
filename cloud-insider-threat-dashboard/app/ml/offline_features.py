"""
Streaming CSV -> per-window features, with NO database (for Kaggle / big data).

The CERT release is many GB across logon/device/email/file/http CSVs. This
module reads each CSV in chunks (only the columns it needs -- e.g. it never
loads http.csv's huge `content` column), computes the same 12 features as the
DB path (it calls the same app/ml/feature_core.py), and keeps only small
aggregates in memory:

  * additive counts  -> summed per (window, user) chunk by chunk
  * distinct PCs / distinct URLs -> the set of distinct (window, user, value)
    triples (URLs are hashed to 64-bit ints), counted at the end

Peak memory is therefore ~ one chunk + the aggregates, not the dataset.
The result is a table with one row per (user, window) -- typically a few
hundred MB at most -- which is cached to disk so the slow pass runs once.

Optional user subsampling (`keep_users`) drops rows for other users while
streaming, cutting both time and memory; see select_users().
"""
import json
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from app.ml.feature_core import (
    ADDITIVE, DEFAULT_INTERNAL_DOMAIN, EVENT_COLUMNS, event_indicators, finalize_features,
)

# file -> (source, detail column). Same mapping as app/ml/ingestion.py.
SOURCE_FILES = {
    "logon.csv": ("logon", None),
    "device.csv": ("device", None),
    "email.csv": ("email", "to"),
    "file.csv": ("file", "filename"),
    "http.csv": ("http", "url"),
}
CERT_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def find_dataset_dir(root: str) -> Path:
    """Finds the directory holding the CERT CSVs (Kaggle nests them under /kaggle/input/<name>/...)."""
    root_p = Path(root)
    for p in [root_p, *sorted(root_p.rglob("*"))]:
        if p.is_dir() and any((p / f).exists() for f in SOURCE_FILES):
            return p
    raise FileNotFoundError(f"No CERT CSVs ({', '.join(SOURCE_FILES)}) found under {root}")


def _parse_dates(s: pd.Series, date_format: Optional[str]) -> pd.Series:
    if date_format:
        out = pd.to_datetime(s, format=date_format, errors="coerce")
        if out.isna().mean() <= 0.5:
            return out
    return pd.to_datetime(s, errors="coerce")      # slower fallback: let pandas infer


def _iter_files(directory: Path):
    for fname, (source, detail_col) in SOURCE_FILES.items():
        path = directory / fname
        if path.exists():
            yield path, source, detail_col


def scan_min_timestamp(directory, chunksize: int = 2_000_000, date_format: Optional[str] = CERT_DATE_FORMAT,
                       progress: Callable[[str], None] = print) -> pd.Timestamp:
    """Cheap pass over just the `date` column to anchor the window grid (same origin the DB path uses)."""
    best = None
    for path, _, _ in _iter_files(Path(directory)):
        for chunk in pd.read_csv(path, usecols=["date"], chunksize=chunksize, dtype=str):
            m = _parse_dates(chunk["date"], date_format).min()
            if pd.notna(m) and (best is None or m < best):
                best = m
        progress(f"[origin] {path.name}: earliest so far {best}")
    if best is None:
        raise ValueError("no parseable timestamps found")
    return best


def list_users(directory, chunksize: int = 2_000_000) -> List[str]:
    """All user codes (from the first available CSV -- logon.csv in CERT)."""
    users: Set[str] = set()
    for path, _, _ in _iter_files(Path(directory)):
        for chunk in pd.read_csv(path, usecols=["user"], chunksize=chunksize, dtype=str):
            users.update(chunk["user"].dropna().unique())
        break
    return sorted(users)


def select_users(all_users: Iterable[str], insider_users: Iterable[str],
                 max_benign: Optional[int] = None, seed: int = 0) -> Set[str]:
    """
    Keep every insider + a random sample of `max_benign` non-insiders (all if None).
    CAUTION: dropping benign users raises the insider prevalence in val/test, so
    precision / PR-AUC become optimistic relative to the real population. The
    training job records the sampling rate so this can be reported.
    """
    all_users, insiders = set(all_users), set(insider_users)
    benign = sorted(all_users - insiders)
    if max_benign is not None and max_benign < len(benign):
        rng = np.random.RandomState(seed)
        benign = [benign[i] for i in rng.choice(len(benign), size=max_benign, replace=False)]
    return (insiders & all_users) | set(benign)


class _Aggregator:
    """Accumulates per-chunk partial aggregates and compacts them periodically."""

    def __init__(self, compact_every: int = 8):
        self.sums: List[pd.DataFrame] = []
        self.pcs: List[pd.DataFrame] = []
        self.urls: List[pd.DataFrame] = []
        self.compact_every, self._n = compact_every, 0

    def add(self, work: pd.DataFrame):
        self.sums.append(work.groupby(["widx", "user_id"])[ADDITIVE].sum())
        pcs = work.loc[work["pc"].notna(), ["widx", "user_id", "pc"]].drop_duplicates()
        self.pcs.append(pcs)
        http = work.loc[work["_http_detail"].notna(), ["widx", "user_id", "_http_detail"]]
        if len(http):
            h = pd.util.hash_pandas_object(http["_http_detail"], index=False).to_numpy()
            self.urls.append(pd.DataFrame({"widx": http["widx"].to_numpy(), "user_id": http["user_id"].to_numpy(),
                                           "h": h}).drop_duplicates())
        self._n += 1
        if self._n % self.compact_every == 0:
            self.compact()

    def compact(self):
        if len(self.sums) > 1:
            self.sums = [pd.concat(self.sums).groupby(level=[0, 1]).sum()]
        if len(self.pcs) > 1:
            self.pcs = [pd.concat(self.pcs).drop_duplicates()]
        if len(self.urls) > 1:
            self.urls = [pd.concat(self.urls).drop_duplicates()]

    def result(self, origin: pd.Timestamp, window_days: int) -> pd.DataFrame:
        self.compact()
        if not self.sums:
            return pd.DataFrame()
        sums = self.sums[0].sort_index()
        dpc = self.pcs[0].groupby(["widx", "user_id"]).size() if self.pcs else pd.Series(dtype=int)
        dht = self.urls[0].groupby(["widx", "user_id"]).size() if self.urls else pd.Series(dtype=int)
        return finalize_features(sums, dpc, dht, origin, window_days)


def build_features_from_csvs(
    directory, window_days: int = 2, internal_domain: str = DEFAULT_INTERNAL_DOMAIN,
    keep_users: Optional[Set[str]] = None, chunksize: int = 2_000_000,
    origin: Optional[pd.Timestamp] = None, date_format: Optional[str] = CERT_DATE_FORMAT,
    progress: Callable[[str], None] = print,
):
    """
    Streams the CERT CSVs in `directory` and returns (features_df, user_codes)
    where features_df has the same columns/semantics as build_feature_windows()
    and user_codes maps the integer user_id -> user code (e.g. "AAM0658").
    """
    directory = Path(directory)
    files = list(_iter_files(directory))
    if not files:
        raise FileNotFoundError(f"No CERT CSVs found in {directory}")
    if origin is None:
        origin = scan_min_timestamp(directory, chunksize, date_format, progress)
    progress(f"[features] window grid origin = {origin}, window_days = {window_days}")

    agg = _Aggregator()
    user_ids: Dict[str, int] = {}
    t0 = time.time()
    for path, source, detail_col in files:
        usecols = ["date", "user", "pc", "activity"] + ([detail_col] if detail_col else []) + (["attachments"] if source == "email" else [])
        rows_seen = rows_kept = 0
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, dtype=str):
            rows_seen += len(chunk)
            if keep_users is not None:
                chunk = chunk[chunk["user"].isin(keep_users)]
            ts = _parse_dates(chunk["date"], date_format)
            ok = ts.notna() & chunk["user"].notna()
            chunk, ts = chunk[ok], ts[ok]
            if chunk.empty:
                continue
            for u in chunk["user"].unique():
                if u not in user_ids:
                    user_ids[u] = len(user_ids) + 1
            events = pd.DataFrame({
                "user_id": chunk["user"].map(user_ids).to_numpy(),
                "source": source,
                "pc": chunk["pc"].to_numpy() if "pc" in chunk else None,
                "activity": chunk["activity"].to_numpy() if "activity" in chunk else None,
                "detail": chunk[detail_col].to_numpy() if detail_col and detail_col in chunk else None,
                "timestamp": ts.to_numpy(),
            })[EVENT_COLUMNS]
            if "attachments" in chunk:
                events["n_att"] = chunk["attachments"].to_numpy()
            events = events[events["timestamp"] >= origin]
            rows_kept += len(events)
            agg.add(event_indicators(events, window_days, origin, internal_domain))
            progress(f"[features] {path.name}: {rows_seen:,} rows read, {rows_kept:,} kept ({time.time() - t0:.0f}s)")
        progress(f"[features] finished {path.name}")

    features = agg.result(origin, window_days)
    if features.empty:
        raise ValueError("No events survived filtering (check keep_users / date format).")
    user_codes = {i: u for u, i in user_ids.items()}
    progress(f"[features] done: {len(features):,} (user, window) rows for {len(user_codes):,} users in {time.time() - t0:.0f}s")
    return features, user_codes


# ------------------------------------------------------------- caching

def save_features(path, features_df: pd.DataFrame, user_codes: Dict[int, str], meta: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"features": features_df, "user_codes": user_codes, "meta": meta or {}}, f, protocol=4)


def load_features(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["features"], {int(k): v for k, v in d["user_codes"].items()}, d.get("meta", {})