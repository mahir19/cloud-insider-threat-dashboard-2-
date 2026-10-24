"""
LSTM classifier + detector (architecture.md sections 3.6 and 4.1).

Supervised sequence model: input is a T x F matrix (last T time windows of
F behavioral features), output is the probability that the user is
malicious in that period. It is trained OFFLINE (app/ml/train.py) and saved
as an artifact; the API only ever loads the artifact and scores.

Why supervised rather than an autoencoder: Layer 1 exists to fix the class
imbalance a classifier trains on. An autoencoder trained on normal behavior
never sees the minority class, which would make the sampling stage largely
irrelevant.

Implementation note: the architecture doc says "sigmoid output, binary
cross-entropy loss". We train with BCEWithLogitsLoss (sigmoid + BCE fused, the
numerically stable form) and apply the sigmoid at inference.
"""
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.ml.detection import BaseDetector
from app.ml.metrics import pr_auc
from app.ml.sequences import FeatureScaler, build_sequences
from app.ml.torch_utils import resolve_device, set_seed

ARTIFACT_FORMAT_VERSION = 1


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    batch_size: int = 128
    patience: Optional[int] = 8     # early stopping on validation PR-AUC; None = fixed epochs

    def scaled_for(self, n_train: int) -> "LSTMConfig":
        """Larger batches for very large train sets (>~100k samples) so epochs stay tractable; unchanged for small data."""
        import dataclasses
        return dataclasses.replace(self, batch_size=max(self.batch_size, min(1024, n_train // 500)))

    @classmethod
    def evaluator(cls) -> "LSTMConfig":
        """Small LSTM used to compare samplers in Layer 1 (1 layer, ~32 units, few epochs)."""
        return cls(hidden_size=32, num_layers=1, dropout=0.2, epochs=20, batch_size=64, patience=4)

    @classmethod
    def production(cls) -> "LSTMConfig":
        return cls()


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (N, T, F) -> logits (N,)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


class BiLSTMAttnClassifier(nn.Module):
    """Bidirectional LSTM + additive attention over the T windows (ensemble member; see app/ml/ensemble.py)."""
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (N, T, F) -> logits (N,)
        out, _ = self.lstm(x)                                  # (N, T, 2H)
        w = torch.softmax(self.attn(out).squeeze(-1), dim=1)   # (N, T) attention over windows
        return self.head(self.drop((out * w.unsqueeze(-1)).sum(1))).squeeze(-1)


def build_model(arch: str, n_features: int, cfg: "LSTMConfig") -> nn.Module:
    if arch == "lstm":
        return LSTMClassifier(n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    if arch == "bilstm_attn":
        return BiLSTMAttnClassifier(n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    raise ValueError(f"unknown architecture '{arch}'")


def predict_proba(model: nn.Module, X: np.ndarray, device=None, batch_size: int = 4096) -> np.ndarray:
    device = resolve_device(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size], dtype=np.float32)).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_lstm(X: np.ndarray, y: np.ndarray, cfg: LSTMConfig, seed: int = 0,
               pos_weight: Optional[float] = None, X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None, device=None, arch: str = "lstm"):
    """
    Trains on (N, T, F) scaled sequences. If a validation set is given, the
    epoch with the best validation PR-AUC is restored (and `cfg.patience`
    enables early stopping). Returns (model, history).
    """
    set_seed(seed)
    device = resolve_device(device)
    model = build_model(arch, X.shape[2], cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pw = torch.tensor(float(pos_weight), device=device) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    if device.type == "cuda" and Xt.numel() * 4 < 2_000_000_000:   # whole train set fits on the GPU: no per-batch copies
        Xt, yt = Xt.to(device), yt.to(device)
    gen = torch.Generator().manual_seed(seed)
    use_val = X_val is not None and y_val is not None and int(np.sum(y_val)) > 0

    best_state, best, bad, history = None, -np.inf, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen).to(Xt.device)
        total = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
        rec = {"epoch": epoch + 1, "train_loss": round(total / len(Xt), 5)}
        if use_val:
            m = pr_auc(y_val, predict_proba(model, X_val, device))
            rec["val_pr_auc"] = round(m, 4)
            if m > best + 1e-6:
                best, bad = m, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        history.append(rec)
        if use_val and cfg.patience and bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ------------------------------------------------------------------ artifact

@dataclass
class LSTMArtifact:
    """Everything the API needs to score: weights, scaler, threshold, sampler choice."""
    state_dict: dict
    config: dict
    scaler: FeatureScaler
    threshold: float
    sampler: str
    seq_len: int
    window_days: int
    feature_columns: list
    dataset: Optional[str] = None
    seed: int = 0
    metrics: Optional[dict] = None
    user_threshold: Optional[float] = None     # user-level alert threshold (see user_level.py)
    user_agg: Optional[dict] = None            # how window probabilities are aggregated per user

    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, d / "model.pt")
        meta = {
            "format_version": ARTIFACT_FORMAT_VERSION, "config": self.config,
            "scaler": self.scaler.to_dict(), "threshold": self.threshold, "sampler": self.sampler,
            "seq_len": self.seq_len, "window_days": self.window_days,
            "feature_columns": self.feature_columns, "dataset": self.dataset,
            "seed": self.seed, "metrics": self.metrics, "torch_version": torch.__version__,
            "user_threshold": self.user_threshold, "user_agg": self.user_agg,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return d

    @classmethod
    def load(cls, directory, map_location="cpu") -> "LSTMArtifact":
        d = Path(directory)
        if not (d / "meta.json").exists() or not (d / "model.pt").exists():
            raise FileNotFoundError(f"No LSTM artifact found in {d}")
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"Unsupported artifact format {meta.get('format_version')}")
        state = torch.load(d / "model.pt", map_location=map_location, weights_only=True)
        return cls(
            state_dict=state, config=meta["config"], scaler=FeatureScaler.from_dict(meta["scaler"]),
            threshold=float(meta["threshold"]), sampler=meta["sampler"], seq_len=int(meta["seq_len"]),
            window_days=int(meta["window_days"]), feature_columns=list(meta["feature_columns"]),
            dataset=meta.get("dataset"), seed=int(meta.get("seed", 0)), metrics=meta.get("metrics"),
            user_threshold=meta.get("user_threshold"), user_agg=meta.get("user_agg"),
        )

    def build_model(self, device=None) -> LSTMClassifier:
        m = LSTMClassifier(len(self.feature_columns), self.config["hidden_size"],
                           self.config["num_layers"], self.config["dropout"])
        m.load_state_dict(self.state_dict)
        return m.to(resolve_device(device)).eval()


# ------------------------------------------------------------------ detector

class LSTMDetector(BaseDetector):
    """
    Drop-in detector: same interface as the five classical ones (see
    BaseDetector). Scores REAL, non-resampled data -- resampling only ever
    touches the training split.

      score        = malicious probability (0-1)
      raw_score    = logit
      is_anomalous = probability >= the threshold tuned on the validation set
    """
    name = "LSTM"

    def __init__(self, artifact_dir=None, artifact: Optional[LSTMArtifact] = None, device=None):
        if artifact is None:
            if artifact_dir is None:
                raise ValueError("LSTMDetector needs artifact_dir= (a trained model directory)")
            artifact = LSTMArtifact.load(artifact_dir)
        self.artifact = artifact
        self.device = resolve_device(device)
        self.model = artifact.build_model(self.device)

    @property
    def threshold(self) -> float:
        return self.artifact.threshold

    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        a = self.artifact
        if features_df.empty:
            return features_df.assign(raw_score=[], score=[], is_anomalous=[])
        step_days = (pd.to_datetime(features_df["window_end"].iloc[0]) - pd.to_datetime(features_df["window_start"].iloc[0])).days
        if step_days != a.window_days:
            raise ValueError(
                f"This model was trained on {a.window_days}-day windows but the data uses "
                f"{step_days}-day windows. Rebuild features with window_days={a.window_days}."
            )
        seq = build_sequences(features_df, a.seq_len, a.feature_columns)
        prob = predict_proba(self.model, a.scaler.transform(seq.X), self.device)
        p = np.clip(prob.astype(float), 1e-7, 1 - 1e-7)
        out = features_df.copy()
        out["raw_score"] = np.log(p / (1 - p))
        out["score"] = prob.astype(float)
        out["is_anomalous"] = (prob >= a.threshold).astype(int)
        return out
    def score_users(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """One row per user: aggregated score + flagged, using the artifact's stored aggregation and user threshold."""
        from app.ml.user_level import flag_users
        a = self.artifact
        if a.user_threshold is None or a.user_agg is None:
            raise ValueError("This artifact has no user-level threshold; retrain with the current train.py")
        return flag_users(self.score(features_df), a.user_agg, a.user_threshold, a.window_days, user_col="user_id")
