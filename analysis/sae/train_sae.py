"""Train sparse autoencoders on real-data layer-7 Plant-DnaGemma activations.

Two variants per seed:
    L1-SAE   : decoder-normalized SAE with L1 sparsity penalty (Bricken et al.).
    TopK-SAE : encoder with per-sample top-k gating (Gao et al., OpenAI).

Inputs:
    data/real/activations/region_type/trained.npz    (hidden: N×13×768)

Outputs (data/real/results/sae/):
    <tag>_L<layer>_exp<ratio>_seed<S>.pt   — model state + meta
    <tag>_L<layer>_exp<ratio>_seed<S>.json — metrics: recon MSE, L0, dead rate

Use the same inputs that probing uses so we can causally tie SAE features back
to probing predictions later.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ACT_ROOT = REPO / "data" / "real" / "activations"
OUT = REPO / "data" / "real" / "results" / "sae"
OUT.mkdir(parents=True, exist_ok=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class L1SAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden, bias=True)
        self.dec = nn.Linear(d_hidden, d_in, bias=True)
        # Normalize decoder columns to unit norm.
        with torch.no_grad():
            w = torch.randn(d_in, d_hidden) / math.sqrt(d_in)
            w /= w.norm(dim=0, keepdim=True).clamp(min=1e-6)
            self.dec.weight.copy_(w)
            self.enc.weight.copy_(w.T.clone())
            self.enc.bias.zero_()
            self.dec.bias.zero_()

    def forward(self, x):
        pre = self.enc(x)
        z = F.relu(pre)
        recon = self.dec(z)
        return recon, z, pre

    def normalize_decoder(self):
        with torch.no_grad():
            w = self.dec.weight
            norm = w.norm(dim=0, keepdim=True).clamp(min=1e-6)
            self.dec.weight.div_(norm)


class TopKSAE(nn.Module):
    """Top-k SAE à la Gao et al. 2024."""

    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d_in, d_hidden, bias=True)
        self.dec = nn.Linear(d_hidden, d_in, bias=True)
        with torch.no_grad():
            w = torch.randn(d_in, d_hidden) / math.sqrt(d_in)
            w /= w.norm(dim=0, keepdim=True).clamp(min=1e-6)
            self.dec.weight.copy_(w)
            self.enc.weight.copy_(w.T.clone())
            self.enc.bias.zero_()
            self.dec.bias.zero_()

    def forward(self, x):
        pre = self.enc(x)
        # Top-k gating per sample
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        gate = torch.zeros_like(pre)
        gate.scatter_(-1, topk_idx, topk_vals)
        z = F.relu(gate)
        recon = self.dec(z)
        return recon, z, pre

    def normalize_decoder(self):
        with torch.no_grad():
            w = self.dec.weight
            norm = w.norm(dim=0, keepdim=True).clamp(min=1e-6)
            self.dec.weight.div_(norm)


def train_sae(
    X: np.ndarray,
    *,
    variant: str,
    d_hidden: int,
    seed: int = 0,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    l1_coef: float = 5e-4,
    topk: int = 32,
    device: torch.device,
    val_frac: float = 0.1,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N, D = X.shape
    idx = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    # Standardize using training mean/std.
    mu = X[tr_idx].mean(axis=0)
    sigma = X[tr_idx].std(axis=0) + 1e-6
    Xn = (X - mu) / sigma
    Xn = torch.from_numpy(Xn.astype(np.float32)).to(device)

    if variant == "l1":
        model = L1SAE(D, d_hidden).to(device)
    elif variant == "topk":
        model = TopKSAE(D, d_hidden, k=topk).to(device)
    else:
        raise ValueError(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    hist = []
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr_idx))
        tot_loss = tot_recon = tot_l1 = 0.0
        n_batches = 0
        for i in range(0, len(tr_idx), batch_size):
            sel = tr_idx[order[i : i + batch_size]]
            xb = Xn[sel]
            recon, z, pre = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            if variant == "l1":
                l1 = z.abs().mean()
                loss = recon_loss + l1_coef * l1
            else:
                l1 = torch.tensor(0.0, device=device)
                loss = recon_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_decoder()
            tot_loss += loss.item()
            tot_recon += recon_loss.item()
            tot_l1 += float(l1.item())
            n_batches += 1
        # Eval
        model.eval()
        with torch.no_grad():
            xv = Xn[val_idx]
            recon, z, _ = model(xv)
            val_recon = F.mse_loss(recon, xv).item()
            l0 = float((z.abs() > 1e-8).float().sum(dim=-1).mean().item())
            sparsity = 1.0 - l0 / d_hidden
        hist.append(
            dict(
                epoch=ep,
                train_loss=tot_loss / n_batches,
                train_recon=tot_recon / n_batches,
                train_l1=tot_l1 / n_batches,
                val_recon=val_recon,
                val_L0=l0,
                val_sparsity=sparsity,
            )
        )
        if val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics over full data
    model.eval()
    with torch.no_grad():
        Xall = Xn
        recon, z, _ = model(Xall)
        full_recon = F.mse_loss(recon, Xall).item()
        # Feature usage / dead features
        active_fracs = (z > 1e-8).float().mean(dim=0).cpu().numpy()
        # Fraction of variance explained
        var_explained = 1.0 - (Xall - recon).var(dim=0).mean().item() / Xall.var(dim=0).mean().item()
        l0 = float((z.abs() > 1e-8).float().sum(dim=-1).mean().item())

    meta = dict(
        variant=variant,
        d_in=D,
        d_hidden=d_hidden,
        expansion=d_hidden / D,
        seed=seed,
        topk=topk if variant == "topk" else None,
        l1_coef=l1_coef if variant == "l1" else None,
        epochs=epochs,
        full_recon=float(full_recon),
        var_explained=float(var_explained),
        L0_mean=l0,
        dead_fraction=float((active_fracs < 1e-6).mean()),
        sparsity=1.0 - l0 / d_hidden,
        history=hist,
        mu=mu.tolist(),
        sigma=sigma.tolist(),
    )
    return model, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="region_type")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument(
        "--expansions", type=int, nargs="+", default=[4, 16],
        help="d_hidden / d_in multipliers.",
    )
    ap.add_argument("--variants", nargs="+", default=["l1", "topk"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--l1-coef", type=float, default=5e-4)
    args = ap.parse_args()

    act_path = ACT_ROOT / args.dataset / "trained.npz"
    print(f"[load] {act_path}")
    cache = np.load(act_path, allow_pickle=True)
    H = cache["hidden"]
    X = H[:, args.layer, :].astype(np.float32)
    D = X.shape[1]
    print(f"  X: {X.shape}  layer={args.layer}")

    device = pick_device()
    print(f"[dev ] {device}")

    for variant, exp, seed in itertools.product(args.variants, args.expansions, args.seeds):
        d_hidden = D * exp
        tag = f"{variant}_exp{exp}_L{args.layer}_seed{seed}"
        json_path = OUT / f"{tag}.json"
        pt_path = OUT / f"{tag}.pt"
        if json_path.exists() and pt_path.exists():
            print(f"[skip] {tag}")
            continue
        t0 = time.time()
        model, meta = train_sae(
            X,
            variant=variant,
            d_hidden=d_hidden,
            seed=seed,
            epochs=args.epochs,
            l1_coef=args.l1_coef,
            topk=args.topk,
            device=device,
        )
        meta["time"] = time.time() - t0
        meta["dataset"] = args.dataset
        meta["tag"] = tag
        # Save
        torch.save(model.state_dict(), pt_path)
        with json_path.open("w") as f:
            json.dump(meta, f, indent=2)
        print(
            f"[done] {tag} in {meta['time']:.1f}s "
            f"recon={meta['full_recon']:.4f} L0={meta['L0_mean']:.1f} "
            f"varexp={meta['var_explained']:.3f} dead={meta['dead_fraction']*100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
