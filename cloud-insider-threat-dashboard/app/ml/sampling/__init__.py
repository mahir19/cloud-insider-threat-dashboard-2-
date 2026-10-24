"""
Layer-1 sampling stage (architecture.md section 3.5).

A candidate is a *spec string*:

    base[@ratio][+cw][^power]

    base   none | class_weight | NM-2 | RUS | DeepSMOTE
    @ratio target minority:majority ratio AFTER resampling (0.05 = 1 insider
           window per 20 normal ones). Omitted = 1.0 (fully balanced, the
           original behaviour). Ignored by none / class_weight.
    +cw    hybrid: after resampling, ALSO weight the positive class in the loss
           to cover the residual imbalance.
    ^power softens a class weight: pos_weight = (n_neg / n_pos) ** power.
           Default 1.0 for the class_weight base, 0.5 for +cw.

    "NM-2"               full 1:1 balance (old behaviour)
    "NM-2@0.05"          keep majority windows until minority:majority = 1:20
    "DeepSMOTE@0.05+cw"  oversample to 1:20, then half-strength class weight
    "class_weight^0.5"   no resampling, square-root-softened loss weight

Candidates
    NM-2          undersampler (paper's best; baseline to beat)
    RUS           random undersampler (diagnostic: is NM-2's *selection* the
                  problem, or undersampling itself?)
    DeepSMOTE     encoder/decoder latent-space oversampler
    none          control -- no imbalance handling
    class_weight  control -- imbalance handled in the loss, data untouched

All samplers take FLATTENED (N, T*F) training vectors; callers reshape back
to (N, T, F). Only ever call these on the TRAIN split.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SAMPLER_CANDIDATES = ("NM-2", "DeepSMOTE", "none", "class_weight")     # original four (full balance)

# Default sweep used by the notebook: partial ratios + hybrids.
SWEEP_CANDIDATES = (
    "none", "class_weight", "class_weight^0.5",
    "NM-2@0.02", "NM-2@0.05", "NM-2@0.1", "NM-2@0.05+cw",
    "RUS@0.05",
    "DeepSMOTE@0.02", "DeepSMOTE@0.05", "DeepSMOTE@0.05+cw",
)

_BASES = ("none", "class_weight", "NM-2", "RUS", "DeepSMOTE")
_SPEC = re.compile(r"^(?P<base>NM-2|DeepSMOTE|RUS|none|class_weight)"
                   r"(?:@(?P<ratio>[0-9]*\.?[0-9]+))?(?P<cw>\+cw)?(?:\^(?P<pow>[0-9]*\.?[0-9]+))?$")


@dataclass
class ResampleResult:
    X: np.ndarray
    y: np.ndarray
    pos_weight: Optional[float] = None     # loss weight for the positive class (class_weight / +cw)
    info: dict = field(default_factory=dict)


def parse_spec(spec: str) -> dict:
    m = _SPEC.match(spec.strip())
    if not m:
        raise ValueError(f"Unknown sampler spec '{spec}'. Format: base[@ratio][+cw][^power] with base in {list(_BASES)}")
    ratio = float(m["ratio"]) if m["ratio"] else None
    if ratio is not None and not (0 < ratio <= 1):
        raise ValueError(f"ratio in '{spec}' must be in (0, 1]")
    return {"base": m["base"], "ratio": ratio, "cw": bool(m["cw"]), "power": float(m["pow"]) if m["pow"] else None}


def base_of(spec: str) -> str:
    return parse_spec(spec)["base"]


def kwargs_for(spec: str, sampler_kwargs: Optional[dict]) -> dict:
    """Per-candidate overrides: looked up by the full spec first, then by its base name."""
    sampler_kwargs = sampler_kwargs or {}
    return dict(sampler_kwargs.get(spec) or sampler_kwargs.get(base_of(spec)) or {})


def _pos_weight(y: np.ndarray, power: float) -> float:
    n_pos = max(int((y == 1).sum()), 1)
    return float((int((y == 0).sum()) / n_pos) ** power)


def resample(name: str, X: np.ndarray, y: np.ndarray, seed: int = 0, **kwargs) -> ResampleResult:
    p = parse_spec(name)
    base, ratio = p["base"], p["ratio"]
    cache = kwargs.pop("cache", None)                      # DeepSMOTE only: reuse the trained autoencoder across ratios
    y = np.asarray(y).astype(int)

    if base == "none":
        return ResampleResult(X, y, None, {"kept": int(len(y)), "spec": name})
    if base == "class_weight":
        pw = _pos_weight(y, p["power"] if p["power"] is not None else 1.0)
        return ResampleResult(X, y, pw, {"kept": int(len(y)), "pos_weight": round(pw, 3), "spec": name})

    if base == "NM-2":
        from app.ml.sampling.nm2 import resample_nm2
        Xr, yr, info = resample_nm2(X, y, ratio=ratio or 1.0, **kwargs)
    elif base == "RUS":
        from app.ml.sampling.rus import resample_rus
        Xr, yr, info = resample_rus(X, y, seed=seed, ratio=ratio or 1.0, **kwargs)
    else:  # DeepSMOTE
        from app.ml.sampling.deepsmote import resample_deepsmote   # imports torch lazily
        Xr, yr, info = resample_deepsmote(X, y, seed=seed, target_ratio=ratio or 1.0, cache=cache, **kwargs)

    pw = None
    if p["cw"]:
        pw = _pos_weight(yr, p["power"] if p["power"] is not None else 0.5)
        info = {**info, "pos_weight": round(pw, 3)}
    return ResampleResult(Xr, yr, pw, {**info, "spec": name, "ratio_after": round(float((yr == 1).sum() / max((yr == 0).sum(), 1)), 4)})
