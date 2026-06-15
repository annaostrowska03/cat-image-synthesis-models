"""Post-hoc training vs validation loss curves.

Loads each saved checkpoint and evaluates loss on a fixed 90/10 train/val
split (seed 42).  Generates combined plots that are dropped into the report.

Usage:
    uv run python scripts/compute_val_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CatDataset, build_transform
from src.models.vqvae import VQVAE, PixelCNN
from src.models.ddpm import GaussianDiffusion, UNet
from src.utils.seed import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_FRAC = 0.10
SEED = 42
N_EVAL_BATCHES = 30  # limit eval batches for speed (~960 images for bs=32)


def split_dataset(data_dir: str, image_size: int):
    transform = build_transform(image_size, augment=False)
    full = CatDataset(root=data_dir, image_size=image_size, transform=transform)
    n_val = int(len(full) * VAL_FRAC)
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    return train_ds, val_ds


def eval_loss_batches(loader: DataLoader, loss_fn, n_batches: int) -> float:
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        with torch.no_grad():
            total += loss_fn(batch.to(DEVICE)).item()
        count += 1
    return total / count if count > 0 else float("nan")


def plot_train_val(epochs, train_vals, val_vals, ylabel, title, out_path,
                   train_label="Train", val_label="Validation"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_vals, label=train_label, color="steelblue")
    ax.plot(epochs, val_vals, label=val_label, color="tomato", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def vqvae_curves(cfg_path: str = "configs/vqvae_config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    data = cfg["data"]
    mc = cfg["model"]
    pc = cfg["prior"]

    train_ds, val_ds = split_dataset(data["cats_dir"], data["image_size"])
    bs = data["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

    model = VQVAE(
        in_channels=mc["in_channels"],
        hidden_channels=mc["hidden_channels"],
        embedding_dim=mc["embedding_dim"],
        num_embeddings=mc["num_embeddings"],
        commitment_cost=mc["commitment_cost"],
        decay=mc["decay"],
    ).to(DEVICE)

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    out_dir   = Path(cfg["logging"]["output_dir"])

    ae_epochs, tr_recon, va_recon = [], [], []
    ae_ckpts = sorted(ckpt_dir.glob("vqvae_epoch*.pt"))
    for ckpt in ae_ckpts:
        epoch = int(ckpt.stem.split("epoch")[1])
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        def recon_loss(batch):
            x_recon, _, _ = model(batch)
            return F.mse_loss(x_recon, batch)

        tr_recon.append(eval_loss_batches(train_loader, recon_loss, N_EVAL_BATCHES))
        va_recon.append(eval_loss_batches(val_loader, recon_loss, N_EVAL_BATCHES))
        ae_epochs.append(epoch)
        print(f"  VQ-VAE AE epoch {epoch}: train={tr_recon[-1]:.4f} val={va_recon[-1]:.4f}")

    plot_train_val(
        ae_epochs, tr_recon, va_recon,
        ylabel="Reconstruction MSE",
        title="VQ-VAE Autoencoder: Train vs Validation Loss",
        out_path=out_dir / "losses_val.png",
    )

    # Prior curves
    model.eval()
    final_ae = sorted(ckpt_dir.glob("vqvae_epoch*.pt"))[-1]
    state = torch.load(final_ae, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])

    prior = PixelCNN(mc["num_embeddings"], pc["n_layers"], pc["n_filters"]).to(DEVICE)

    prior_epochs, tr_prior, va_prior = [], [], []
    prior_ckpts = sorted(ckpt_dir.glob("prior_epoch*.pt"))
    for ckpt in prior_ckpts:
        epoch = int(ckpt.stem.split("epoch")[1])
        ps = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        prior.load_state_dict(ps["prior"])
        prior.eval()

        def prior_loss(batch):
            with torch.no_grad():
                indices = model.encode_indices(batch)
            logits = prior(indices)
            return F.cross_entropy(logits, indices)

        tr_prior.append(eval_loss_batches(train_loader, prior_loss, N_EVAL_BATCHES))
        va_prior.append(eval_loss_batches(val_loader, prior_loss, N_EVAL_BATCHES))
        prior_epochs.append(epoch)
        print(f"  VQ-VAE prior epoch {epoch}: train={tr_prior[-1]:.4f} val={va_prior[-1]:.4f}")

    plot_train_val(
        prior_epochs, tr_prior, va_prior,
        ylabel="Cross-Entropy (NLL)",
        title="VQ-VAE PixelCNN Prior: Train vs Validation Loss",
        out_path=out_dir / "prior_losses_val.png",
    )


def ddpm_curves(cfg_path: str = "configs/ddpm_config.yaml"):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    data = cfg["data"]
    mc = cfg["model"]
    dc = cfg["diffusion"]

    train_ds, val_ds = split_dataset(data["cats_dir"], data["image_size"])
    bs = data["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=DEVICE,
    )

    model = UNet(
        image_channels=mc["image_channels"],
        base_channels=mc["base_channels"],
        channel_mults=tuple(mc["channel_mults"]),
        attention_resolutions=tuple(mc["attention_resolutions"]),
        num_res_blocks=mc["num_res_blocks"],
        dropout=mc["dropout"],
    ).to(DEVICE)

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    out_dir   = Path(cfg["logging"]["output_dir"])

    ddpm_epochs, tr_loss, va_loss = [], [], []
    ckpts = sorted(ckpt_dir.glob("ddpm_epoch*.pt"))
    for ckpt in ckpts:
        epoch = int(ckpt.stem.split("epoch")[1])
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        torch.manual_seed(SEED)

        def noise_mse(batch):
            return diffusion.training_loss(model, batch)

        tr_loss.append(eval_loss_batches(train_loader, noise_mse, N_EVAL_BATCHES))
        va_loss.append(eval_loss_batches(val_loader, noise_mse, N_EVAL_BATCHES))
        ddpm_epochs.append(epoch)
        print(f"  DDPM epoch {epoch}: train={tr_loss[-1]:.4f} val={va_loss[-1]:.4f}")

    plot_train_val(
        ddpm_epochs, tr_loss, va_loss,
        ylabel="Noise MSE",
        title="DDPM: Train vs Validation Loss",
        out_path=out_dir / "losses_val.png",
    )


if __name__ == "__main__":
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Val fraction: {VAL_FRAC*100:.0f}%  (seed {SEED})")

    print("\n--- VQ-VAE ---")
    vqvae_curves()

    print("\n--- DDPM ---")
    ddpm_curves()

    print("\nDone. Plots saved to outputs/vqvae/ and outputs/ddpm/")
