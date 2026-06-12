"""DCGAN training script.

Usage:
    python -m src.training.train_dcgan --config configs/dcgan_config.yaml

Supports:
- Label smoothing for real samples
- WGAN-GP gradient penalty (optional)
- Checkpoint resumption
- Periodic sample grids saved to outputs/dcgan/
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data.preprocessing import get_cat_loader
from src.models.dcgan import Discriminator, Generator, compute_gradient_penalty
from src.utils.seed import set_seed
from src.utils.visualization import plot_losses, save_image_grid


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(cfg: dict) -> None:
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = get_cat_loader(
        data_dir=cfg["data"]["cats_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"].get("augment", False),
    )

    mc = cfg["model"]
    tc = cfg["training"]
    lc = cfg["logging"]

    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(device)

    D = Discriminator(
        base_channels=mc["d_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=tc["lr_g"], betas=(tc["beta1"], tc["beta2"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=tc["lr_d"], betas=(tc["beta1"], tc["beta2"]))

    use_gp = tc.get("use_gradient_penalty", True)
    gp_lambda = tc.get("gp_lambda", 10.0)
    smooth_real = tc.get("label_smooth_real", 0.9)
    n_critic = tc.get("n_critic", 1)

    out_dir = Path(lc["output_dir"])
    ckpt_dir = Path(lc["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fixed_z = torch.randn(lc["sample_size"], mc["z_dim"], device=device)

    g_losses, d_losses = [], []

    for epoch in range(1, tc["epochs"] + 1):
        G.train()
        D.train()
        epoch_g, epoch_d = 0.0, 0.0

        data_iter = iter(loader)
        n_batches = len(loader)
        g_steps = 0

        while True:
            d_loss_accum = 0.0
            for _ in range(n_critic):
                try:
                    real = next(data_iter)
                except StopIteration:
                    data_iter = None
                    break
                real = real.to(device)
                B = real.size(0)

                opt_D.zero_grad()
                z = torch.randn(B, mc["z_dim"], device=device)
                fake = G(z).detach()

                real_logit = D(real)
                fake_logit = D(fake)

                if use_gp:
                    d_loss = fake_logit.mean() - real_logit.mean()
                    gp = compute_gradient_penalty(D, real, fake.requires_grad_(True), device)
                    d_loss = d_loss + gp_lambda * gp
                else:
                    real_labels = torch.full((B, 1), smooth_real, device=device)
                    fake_labels = torch.zeros(B, 1, device=device)
                    criterion = nn.BCEWithLogitsLoss()
                    d_loss = criterion(real_logit, real_labels) + criterion(fake_logit, fake_labels)

                d_loss.backward()
                opt_D.step()
                d_loss_accum += d_loss.item()

            if data_iter is None:
                break

            epoch_d += d_loss_accum / n_critic

            opt_G.zero_grad()
            z = torch.randn(B, mc["z_dim"], device=device)
            fake = G(z)
            fake_logit = D(fake)

            if use_gp:
                g_loss = -fake_logit.mean()
            else:
                g_loss = nn.BCEWithLogitsLoss()(fake_logit, torch.ones(B, 1, device=device))

            g_loss.backward()
            opt_G.step()
            epoch_g += g_loss.item()
            g_steps += 1

        g_losses.append(epoch_g / max(g_steps, 1))
        d_losses.append(epoch_d / max(g_steps, 1))
        print(f"Epoch [{epoch:>4}/{tc['epochs']}]  G: {g_losses[-1]:.4f}  D: {d_losses[-1]:.4f}")

        if epoch % lc["save_every_epochs"] == 0 or epoch == tc["epochs"]:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z)
            save_image_grid(samples, out_dir / f"samples_epoch{epoch:04d}.png", title=f"Epoch {epoch}")
            G.train()

            torch.save(
                {"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                 "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict()},
                ckpt_dir / f"ckpt_epoch{epoch:04d}.pt",
            )

    plot_losses({"G": g_losses, "D": d_losses}, out_dir / "losses.png", title="DCGAN Losses")
    print("Training complete.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dcgan_config.yaml")
    parser.add_argument("--z_dim", type=int, default=None)
    parser.add_argument("--lr_g", type=float, default=None)
    parser.add_argument("--lr_d", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.z_dim is not None:
        cfg["model"]["z_dim"] = args.z_dim
    if args.lr_g is not None:
        cfg["training"]["lr_g"] = args.lr_g
    if args.lr_d is not None:
        cfg["training"]["lr_d"] = args.lr_d
    if args.beta1 is not None:
        cfg["training"]["beta1"] = args.beta1

    train(cfg)

if __name__ == "__main__":
    main()
