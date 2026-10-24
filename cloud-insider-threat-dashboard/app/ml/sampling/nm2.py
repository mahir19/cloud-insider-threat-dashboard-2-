"""
NM-2 (NearMiss version 2) -- the paper's best sampler and the baseline to beat.
Distance-based undersampler: keeps the majority samples whose summed distance
to their `n_neighbors` FARTHEST minority samples is smallest, until the
minority:majority ratio reaches `ratio` (1.0 = balanced, 0.05 = 1:20). Works directly on flattened T x F vectors.

This is a memory-bounded reimplementation of imbalanced-learn's
NearMiss(version=2). imblearn materializes the full (n_majority x n_minority)
distance matrix (twice: distances and indices); on a large dataset that is
tens of GB. Here majority rows are processed in chunks so peak memory is
bounded by `max_chunk_elements` regardless of dataset size. It selects the
same samples as imblearn (verified in tests/test_sampling.py).
"""
import numpy as np
from sklearn.metrics import pairwise_distances


def resample_nm2(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3, max_chunk_elements: int = 20_000_000,
                 ratio: float = 1.0):
    y = np.asarray(y).astype(int)
    min_idx, maj_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_min, n_maj = len(min_idx), len(maj_idx)
    if n_min == 0 or n_maj == 0:
        raise ValueError("NM-2 needs both classes present in the training split.")
    k = min(n_neighbors, n_min)
    X_min = np.asarray(X[min_idx], dtype=np.float64)

    chunk = max(1, max_chunk_elements // n_min)
    scores = np.empty(n_maj, dtype=np.float64)
    for i in range(0, n_maj, chunk):
        d = pairwise_distances(np.asarray(X[maj_idx[i:i + chunk]], dtype=np.float64), X_min)   # (chunk, n_min)
        scores[i:i + chunk] = np.partition(d, n_min - k, axis=1)[:, n_min - k:].sum(axis=1)   # sum of k farthest

    # keep the n_min/ratio majority samples nearest to the minority class (ratio=1.0 -> fully balanced)
    n_keep = min(n_maj, max(n_min, int(round(n_min / ratio))))
    keep_maj = maj_idx[np.argsort(scores, kind="stable")[:n_keep]]
    idx = np.concatenate([keep_maj, min_idx])
    info = {"kept": int(len(idx)), "minority": int(n_min), "majority": int(len(keep_maj)), "n_neighbors": int(k), "ratio": ratio}
    return X[idx], y[idx], info