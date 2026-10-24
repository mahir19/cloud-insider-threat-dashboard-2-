"""Small torch helpers shared by the LSTM and DeepSMOTE modules."""
import random

import numpy as np


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:   # pragma: no cover
        raise ImportError(
            "PyTorch is required for the LSTM / DeepSMOTE features. "
            "Install it with `pip install torch` (see README)."
        ) from e


def resolve_device(device=None):
    import torch
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)