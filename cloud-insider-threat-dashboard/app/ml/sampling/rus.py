"""
Random undersampling (RUS). Diagnostic control for NM-2: same number of majority
windows kept, but chosen at random instead of "closest to the minority class".
If RUS@r beats NM-2@r, the problem is NM-2's biased selection; if both are
worse than `none`, discarding normal data is the problem.
"""
import numpy as np


def resample_rus(X: np.ndarray, y: np.ndarray, seed: int = 0, ratio: float = 1.0):
    y = np.asarray(y).astype(int)
    min_idx, maj_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_min, n_maj = len(min_idx), len(maj_idx)
    if n_min == 0 or n_maj == 0:
        raise ValueError("RUS needs both classes present in the training split.")
    n_keep = min(n_maj, max(n_min, int(round(n_min / ratio))))
    keep = np.random.RandomState(seed).choice(maj_idx, size=n_keep, replace=False)
    idx = np.sort(np.concatenate([keep, min_idx]))
    return X[idx], y[idx], {"kept": int(len(idx)), "minority": int(n_min), "majority": int(n_keep)}
