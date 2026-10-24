"""
DB-free feature engineering core.

Everything that turns raw events into one feature vector per (user, window)
lives here with NO database / SQLAlchemy / FastAPI imports, so the same code
serves both the Postgres-backed API (app/ml/features.py) and the streaming
Kaggle/offline path (app/ml/offline_features.py). Sharing one implementation
is what guarantees the two paths produce identical features.

Two stages, so aggregation can be done chunk-by-chunk on data too big for RAM:
  event_indicators()   per-event 0/1 indicator columns (pure row-wise)
  finalize_features()  sums + distinct counts -> the 12 feature columns
Additive columns (ADDITIVE) can be summed across chunks; the two DISTINCT
features (distinct PCs, distinct URLs) cannot, so chunked callers collect the
distinct (user, window, value) triples and count them at the end.
"""
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

AFTER_HOURS_START = 19   # 7pm
AFTER_HOURS_END = 7      # 7am
DEFAULT_INTERNAL_DOMAIN = "@company.com"

FEATURE_COLUMNS = [
    "logon_count", "after_hours_logon_count", "weekend_logon_count",
    "distinct_pc_count", "device_connect_count",
    "email_count", "email_external_ratio", "email_attachment_count",
    "file_count", "file_usb_count",
    "http_count", "http_distinct_domains",
    # --- added ---
    "after_hours_device_count", "after_hours_file_count", "after_hours_http_count", "after_hours_email_count",
    "http_leak_count", "http_job_count", "http_cloud_count",
]

ADDITIVE = [
    "logon_count", "after_hours_logon_count", "weekend_logon_count", "device_connect_count",
    "email_count", "_ext", "email_attachment_count", "file_count", "file_usb_count", "http_count",
    "after_hours_device_count", "after_hours_file_count", "after_hours_http_count", "after_hours_email_count",
    "http_leak_count", "http_job_count", "http_cloud_count",
]

# starting lists -- tune to the domains that actually occur in your http.csv
LEAK_RE = r"wikileaks|pastebin|anonfiles|keylogger"
JOB_RE = r"monster\.com|careerbuilder|indeed\.com|linkedin\.com|simplyhired|jobhuntersbible|hotjobs|job-?hunt|glassdoor"
CLOUD_RE = r"dropbox|mega\.nz|wetransfer|drive\.google|onedrive|box\.com|mediafire|4shared|sendspace"


def _has_external(cell, dom: str) -> bool:
    """True if ANY recipient in a ';'-separated list is outside the internal domain."""
    if not isinstance(cell, str):
        return False
    return any(r.strip() and not r.strip().lower().endswith(dom) for r in cell.split(";"))

EVENT_COLUMNS = ["user_id", "source", "pc", "activity", "detail", "timestamp"]


def event_indicators(events: pd.DataFrame, window_days: int, origin: pd.Timestamp,
                     internal_domain: str = DEFAULT_INTERNAL_DOMAIN) -> pd.DataFrame:
    """Per-event indicator columns + window index (widx = whole windows since `origin`)."""
    ts = events["timestamp"]
    window = timedelta(days=window_days)
    widx = ((ts - origin) // window).astype("int64")

    src = events["source"].astype("category")
    src = src.cat.rename_categories([getattr(c, "value", c) for c in src.cat.categories])
    is_logon, is_dev = src == "logon", src == "device"
    is_email, is_file, is_http = src == "email", src == "file", src == "http"

    detail = events["detail"]
    ext = pd.Series(False, index=events.index)
    if is_email.any():
        dom = internal_domain.lower()
        ext.loc[is_email] = detail[is_email].map(lambda x: _has_external(x, dom)).to_numpy(dtype=bool)

    # real attachment flag (offline_features passes the e-mail `attachments` column as `n_att`)
    if "n_att" in events.columns:
        a = events["n_att"].astype("string").str.strip()
        num = pd.to_numeric(a, errors="coerce")
        has_att = np.asarray(((num > 0) | (num.isna() & a.notna() & (a.str.len() > 0))).fillna(False), dtype=bool)
    else:
        has_att = detail.notna().to_numpy()          # legacy fallback (NOT a real attachment signal)

    hour = ts.dt.hour
    ah = (hour >= AFTER_HOURS_START) | (hour < AFTER_HOURS_END)
    url = detail.where(is_http)
    return pd.DataFrame({
        "widx": widx.to_numpy(),
        "user_id": events["user_id"].to_numpy(),
        "logon_count": is_logon.to_numpy().astype(int),
        "after_hours_logon_count": (is_logon & ((hour >= AFTER_HOURS_START) | (hour < AFTER_HOURS_END))).to_numpy().astype(int),
        "weekend_logon_count": (is_logon & (ts.dt.weekday >= 5)).to_numpy().astype(int),
        "pc": events["pc"].to_numpy(),
        "device_connect_count": (is_dev & (events["activity"] == "Connect")).to_numpy().astype(int),
        "email_count": is_email.to_numpy().astype(int),
        "_ext": ext.to_numpy().astype(int),
        "email_attachment_count": (is_email.to_numpy() & has_att).astype(int),
        "file_count": is_file.to_numpy().astype(int),
        "file_usb_count": (is_file & events["pc"].astype(str).str.contains("usb", case=False, na=False)).to_numpy().astype(int),
        "http_count": is_http.to_numpy().astype(int),
        "after_hours_device_count": (is_dev & (events["activity"] == "Connect") & ah).to_numpy().astype(int),
        "after_hours_file_count": (is_file & ah).to_numpy().astype(int),
        "after_hours_http_count": (is_http & ah).to_numpy().astype(int),
        "after_hours_email_count": (is_email & ah).to_numpy().astype(int),
        "http_leak_count": url.str.contains(LEAK_RE, case=False, regex=True, na=False).to_numpy().astype(int),
        "http_job_count": url.str.contains(JOB_RE, case=False, regex=True, na=False).to_numpy().astype(int),
        "http_cloud_count": url.str.contains(CLOUD_RE, case=False, regex=True, na=False).to_numpy().astype(int),
        "_http_detail": url.to_numpy(),
    })


def finalize_features(sums: pd.DataFrame, distinct_pc: pd.Series, distinct_http: pd.Series,
                      origin: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """
    sums: index (widx, user_id), columns ADDITIVE (already summed over all events).
    distinct_pc / distinct_http: Series on the same (widx, user_id) key -> distinct counts.
    """
    out = sums.copy()
    out["distinct_pc_count"] = distinct_pc.reindex(out.index).fillna(0).astype(int)
    out["http_distinct_domains"] = distinct_http.reindex(out.index).fillna(0).astype(int)
    out["email_external_ratio"] = np.where(out["email_count"] > 0, out["_ext"] / out["email_count"].clip(lower=1), 0.0)
    out = out.reset_index()
    window = timedelta(days=window_days)
    out["window_start"] = origin + out["widx"] * window
    out["window_end"] = out["window_start"] + window
    return out[["user_id", "window_start", "window_end"] + FEATURE_COLUMNS]


def compute_feature_windows(events: pd.DataFrame, window_days: int = 7,
                            internal_domain: str = DEFAULT_INTERNAL_DOMAIN,
                            origin: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Vectorized feature engineering: assigns every event to its window with
    integer division and aggregates with one groupby (no per-window rescan).
    Output is identical to the original per-window implementation (verified in
    tests/test_features.py). `origin` defaults to the earliest event.
    `internal_domain`: recipients ending with it count as internal e-mail
    (default keeps the original behavior; real CERT uses "@dtaa.com").
    """
    cols = ["user_id", "window_start", "window_end"] + FEATURE_COLUMNS
    if events.empty:
        return pd.DataFrame(columns=cols)
    origin = events["timestamp"].min() if origin is None else origin
    work = event_indicators(events, window_days, origin, internal_domain)
    g = work.groupby(["widx", "user_id"], sort=True)
    return finalize_features(g[ADDITIVE].sum(), g["pc"].nunique(), g["_http_detail"].nunique(), origin, window_days)