"""
DeepSMOTE for flattened behavioral sequences (architecture.md section 3.5).

DeepSMOTE (Dablain, Krawczyk & Chawla, 2022) oversamples in a learned latent
space: train an encoder/decoder, run SMOTE-style interpolation between
minority latents, then decode the new latents back to input space. The
original targets images (convolutional AE); here the inputs are flattened
T x F vectors so the encoder/decoder are MLPs.

Training loss = reconstruction MSE
              + mix_weight * latent-mix consistency loss

The second term is this project's adaptation, NOT a verbatim copy of the
paper's permutation loss: for pairs of same-class samples we interpolate in
latent space, decode, re-encode, and require the re-encoded latent to land
where we interpolated. It plays the same role -- keeping in-class latent
interpolations decodable into plausible samples.

Batches are class-balanced (minority drawn as often as majority) because the
autoencoder would otherwise barely see the few insider samples.

Stability: DeepSMOTE can be unreliable on very small minority classes. After
fitting we compute `recon_ratio` (reconstruction MSE / MSE of predicting the
feature mean, overall and on the minority class). If it is non-finite or too
high, `info["converged"]` is False. Callers treat that as a *finding*
("did not train reliably") rather than forcing the sampler to win.
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from app.ml.torch_utils import resolve_device, set_seed


class _AutoEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: Tuple[int, int], latent: int):
        super().__init__()
        h1, h2 = hidden
        self.enc = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(), nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, d_in))


def _same_class_partner(y: torch.Tensor) -> torch.Tensor:
    """For each batch element, the index of a random same-class element (itself if alone)."""
    partner = torch.arange(len(y), device=y.device)
    for c in torch.unique(y):
        idx = torch.nonzero(y == c, as_tuple=True)[0]
        partner[idx] = idx[torch.randint(len(idx), (len(idx),), device=y.device)]
    return partner


def resample_deepsmote(
    X: np.ndarray, y: np.ndarray, seed: int = 0, latent_dim: int = 16,
    hidden: Tuple[int, int] = (128, 64), epochs: int = 60, batch_size: int = 64,
    lr: float = 1e-3, k_neighbors: int = 5, target_ratio: float = 1.0,
    mix_weight: float = 0.1, max_recon_ratio: float = 0.7,
    max_synthetic: Optional[int] = 200_000, max_steps_per_epoch: int = 200, device=None,
    cache: Optional[dict] = None,
):
    """
    Returns (X_resampled, y_resampled, info). `target_ratio` is the desired
    minority:majority ratio after oversampling (1.0 = fully balanced).

    `cache`: optional dict. The autoencoder depends only on (X, y, seed), not on
    `target_ratio`, so when several ratios are compared on the same split pass
    the same dict each time and the autoencoder is trained once.
    """
    y = np.asarray(y).astype(int)
    n_min, n_maj = int((y == 1).sum()), int((y == 0).sum())
    if n_min == 0 or n_maj == 0:
        raise ValueError("DeepSMOTE needs both classes present in the training split.")

    set_seed(seed)
    device = resolve_device(device)

    if cache is not None and "state" in cache:
        Z, health = cache["state"]["Z"], cache["state"]["health"]
        model = cache["state"]["model"]
        last_loss, recon_ratio, recon_ratio_min, converged = health
    else:
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
        yt = torch.from_numpy(y).to(device)
        n, d = Xt.shape

        model = _AutoEncoder(d, hidden, latent_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        mse = nn.MSELoss()

        # class-balanced batch sampling
        w = np.where(y == 1, 0.5 / n_min, 0.5 / n_maj)
        rng = np.random.RandomState(seed)
        steps = max(1, min(max_steps_per_epoch, int(np.ceil(n / batch_size))))   # cap keeps huge datasets tractable
        last_loss = float("nan")
        for _ in range(epochs):
            model.train()
            for _ in range(steps):
                idx = torch.from_numpy(rng.choice(n, size=min(batch_size, n), p=w)).to(device)
                xb, yb = Xt[idx], yt[idx]
                z = model.enc(xb)
                loss = mse(model.dec(z), xb)
                if mix_weight > 0 and len(idx) > 1:
                    partner = _same_class_partner(yb)
                    lam = torch.rand(len(idx), 1, device=device)
                    z_mix = z + lam * (z[partner] - z)
                    loss = loss + mix_weight * mse(model.enc(model.dec(z_mix)), z_mix.detach())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                last_loss = float(loss.detach().cpu())

        # ---- health check + latent codes
        model.eval()
        with torch.no_grad():
            Z = model.enc(Xt)
            recon = model.dec(Z)
            err = ((recon - Xt) ** 2).mean(dim=1).cpu().numpy()
        Xn = Xt.cpu().numpy()
        base = ((Xn - Xn.mean(axis=0)) ** 2).mean(axis=1)
        recon_ratio = float(err.mean() / max(base.mean(), 1e-12))
        recon_ratio_min = float(err[y == 1].mean() / max(base[y == 1].mean(), 1e-12)) if n_min else float("nan")
        converged = bool(np.isfinite(last_loss) and np.isfinite(recon_ratio) and np.isfinite(recon_ratio_min)
                         and recon_ratio <= max_recon_ratio and recon_ratio_min <= max_recon_ratio + 0.2)
        if cache is not None:
            cache["state"] = {"Z": Z, "model": model, "health": (last_loss, recon_ratio, recon_ratio_min, converged)}

    # ---- SMOTE in latent space over the minority class
    n_new = max(0, int(round(target_ratio * n_maj)) - n_min)
    if max_synthetic is not None:
        n_new = min(n_new, max_synthetic)
    info = {
        "converged": converged, "recon_ratio": round(recon_ratio, 4),
        "recon_ratio_minority": round(recon_ratio_min, 4), "final_loss": round(last_loss, 6),
        "n_synthetic": n_new, "epochs": epochs, "latent_dim": latent_dim,
    }
    if n_new == 0:
        return X, y, info

    Zm = Z[torch.from_numpy(y == 1).to(device)].cpu().numpy()
    gen = np.random.RandomState(seed + 1)
    if len(Zm) >= 2:
        k = min(k_neighbors, len(Zm) - 1)
        nn_idx = NearestNeighbors(n_neighbors=k + 1).fit(Zm).kneighbors(Zm, return_distance=False)[:, 1:]
        base_i = gen.randint(len(Zm), size=n_new)
        nb_i = nn_idx[base_i, gen.randint(k, size=n_new)]
        lam = gen.rand(n_new, 1)
        z_new = Zm[base_i] + lam * (Zm[nb_i] - Zm[base_i])
    else:                                    # a single insider sample: jitter it
        info["single_minority_fallback"] = True
        z_new = Zm[np.zeros(n_new, dtype=int)] + gen.normal(0, 0.05 * float(Z.std().cpu()), size=(n_new, Zm.shape[1]))

    with torch.no_grad():
        X_new = model.dec(torch.from_numpy(z_new.astype(np.float32)).to(device)).cpu().numpy()
    if not np.isfinite(X_new).all():
        info["converged"] = False
        X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)

    X_res = np.concatenate([np.asarray(X, dtype=np.float32), X_new.astype(np.float32)])
    y_res = np.concatenate([y, np.ones(n_new, dtype=int)])
    return X_res, y_res, info