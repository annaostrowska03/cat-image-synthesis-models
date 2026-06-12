"""VQ-VAE training script (autoencoder + PixelCNN prior).

Usage:
    python -m src.training.train_vqvae --config configs/vqvae_config.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.data.preprocessing import get_cat_loader
from src.models.vqvae import VQVAE, PixelCNN
from src.utils.seed import set_seed
from src.utils.visualization import plot_losses, save_image_grid


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_autoencoder(cfg: dict, device: torch.device) -> VQVAE:
    mc = cfg["model"]
    tc = cfg["training"]
    lc = cfg["logging"]

    loader = get_cat_loader(
        data_dir=cfg["data"]["cats_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"].get("augment", False),
    )

    model = VQVAE(
        in_channels=mc["in_channels"],
        hidden_channels=mc["hidden_channels"],
        embedding_dim=mc["embedding_dim"],
        num_embeddings=mc["num_embeddings"],
        commitment_cost=mc["commitment_cost"],
        decay=mc["decay"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=tc["lr"], betas=(tc["beta1"], tc["beta2"])
    )

    out_dir = Path(lc["output_dir"])
    ckpt_dir = Path(lc["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    recon_fn = F.mse_loss if tc.get("recon_loss", "mse") == "mse" else F.l1_loss
    recon_losses, vq_losses = [], []

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        ep_recon, ep_vq = 0.0, 0.0

        for real in loader:
            real = real.to(device)
            optimizer.zero_grad()
            x_recon, _, vq_loss = model(real)
            recon_loss = recon_fn(x_recon, real)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            ep_recon += recon_loss.item()
            ep_vq += vq_loss.item()

        n = len(loader)
        recon_losses.append(ep_recon / n)
        vq_losses.append(ep_vq / n)
        perplexity = model.vq.perplexity_last
        print(
            f"Epoch [{epoch:>4}/{tc['epochs']}]  "
            f"Recon: {recon_losses[-1]:.4f}  VQ: {vq_losses[-1]:.4f}  "
            f"Perplexity: {perplexity:.1f}"
        )

        if epoch % lc["save_every_epochs"] == 0 or epoch == tc["epochs"]:
            model.eval()
            with torch.no_grad():
                sample_real = next(iter(loader))[:lc["sample_size"]].to(device)
                x_recon, _, _ = model(sample_real)
            combined = torch.cat([sample_real, x_recon], dim=0)
            save_image_grid(combined, out_dir / f"recon_epoch{epoch:04d}.png",
                            nrow=lc["sample_size"], title=f"Top: real  Bottom: recon  Epoch {epoch}")
            model.train()
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_dir / f"vqvae_epoch{epoch:04d}.pt",
            )

    plot_losses(
        {"Reconstruction": recon_losses, "VQ": vq_losses},
        out_dir / "losses.png",
        title="VQ-VAE Training Losses",
    )
    return model

def train_prior(cfg: dict, vqvae: VQVAE, device: torch.device) -> PixelCNN:
    mc = cfg["model"]
    pc = cfg["prior"]
    tc = cfg["training"]
    lc = cfg["logging"]

    # augment=False: prior learns the real image index distribution, not augmented variants
    loader = get_cat_loader(
        data_dir=cfg["data"]["cats_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=False,
    )

    prior = PixelCNN(
        num_embeddings=mc["num_embeddings"],
        n_layers=pc["n_layers"],
        n_filters=pc["n_filters"],
    ).to(device)

    optimizer = torch.optim.Adam(prior.parameters(), lr=tc["lr"])
    vqvae.eval()

    out_dir = Path(lc["output_dir"])
    prior_losses = []

    print("Training PixelCNN prior...")
    for epoch in range(1, tc["epochs"] + 1):
        prior.train()
        ep_loss = 0.0
        for real in loader:
            real = real.to(device)
            with torch.no_grad():
                indices = vqvae.encode_indices(real)  # (B, H_z, W_z)
            optimizer.zero_grad()
            logits = prior(indices)                    # (B, K, H_z, W_z)
            loss = F.cross_entropy(logits, indices)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        prior_losses.append(ep_loss / len(loader))
        print(f"PixelCNN Epoch [{epoch:>4}/{tc['epochs']}]  Loss: {prior_losses[-1]:.4f}")

        if epoch % lc["save_every_epochs"] == 0 or epoch == tc["epochs"]:
            prior.eval()
            with torch.no_grad():
                B, H_z, W_z = lc["sample_size"], indices.shape[1], indices.shape[2]
                sampled_indices = prior.sample((B, H_z, W_z), device=device)
                samples = vqvae.decode_indices(sampled_indices)
            save_image_grid(samples, out_dir / f"prior_samples_epoch{epoch:04d}.png",
                            title=f"PixelCNN samples, Epoch {epoch}")
            torch.save(
                {"epoch": epoch, "prior": prior.state_dict()},
                Path(lc["checkpoint_dir"]) / f"prior_epoch{epoch:04d}.pt",
            )

    plot_losses({"PixelCNN": prior_losses}, out_dir / "prior_losses.png", title="PixelCNN Prior Loss")
    return prior

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vqvae_config.yaml")
    parser.add_argument("--skip_prior", action="store_true",
                        help="Train only the autoencoder, skip PixelCNN.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vqvae = train_autoencoder(cfg, device)

    if cfg["prior"].get("train_prior", True) and not args.skip_prior:
        train_prior(cfg, vqvae, device)

    print("VQ-VAE training complete.")

if __name__ == "__main__":
    main()
