"""DDPM training script with EMA and DDIM sampling.

Usage:
    python -m src.training.train_ddpm --config configs/ddpm_config.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.data.preprocessing import get_cat_loader
from src.models.ddpm import EMA, GaussianDiffusion, UNet
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

    dc = cfg["diffusion"]
    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=device,
    )

    mc = cfg["model"]
    model = UNet(
        image_channels=mc["image_channels"],
        base_channels=mc["base_channels"],
        channel_mults=tuple(mc["channel_mults"]),
        attention_resolutions=tuple(mc["attention_resolutions"]),
        num_res_blocks=mc["num_res_blocks"],
        dropout=mc["dropout"],
    ).to(device)

    tc = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tc["lr"], betas=(tc["beta1"], tc["beta2"])
    )
    ema = EMA(model, decay=tc.get("ema_decay", 0.9999))
    grad_clip = tc.get("grad_clip", 1.0)

    sc = cfg["sampling"]
    lc = cfg["logging"]
    out_dir = Path(lc["output_dir"])
    ckpt_dir = Path(lc["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    img_size = cfg["data"]["image_size"]
    sample_shape = (lc["sample_size"], mc["image_channels"], img_size, img_size)

    losses = []

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        ep_loss = 0.0

        for real in loader:
            real = real.to(device)
            optimizer.zero_grad()
            loss = diffusion.training_loss(model, real)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update()
            ep_loss += loss.item()

        losses.append(ep_loss / len(loader))
        print(f"Epoch [{epoch:>4}/{tc['epochs']}]  Loss: {losses[-1]:.4f}")

        if epoch % lc["save_every_epochs"] == 0 or epoch == tc["epochs"]:
            original_weights = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            ema.apply_shadow()
            model.eval()
            with torch.no_grad():
                if sc.get("use_ddim", True):
                    samples = diffusion.ddim_sample(
                        model, sample_shape,
                        ddim_steps=sc.get("ddim_steps", 50),
                        eta=sc.get("eta", 0.0),
                    )
                else:
                    samples = diffusion.p_sample_loop(model, sample_shape)
            save_image_grid(samples, out_dir / f"samples_epoch{epoch:04d}.png",
                            nrow=4, title=f"DDPM (DDIM) Epoch {epoch}")
            ema.restore(original_weights)
            model.train()

            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "ema_shadow": ema.shadow, "optimizer": optimizer.state_dict()},
                ckpt_dir / f"ddpm_epoch{epoch:04d}.pt",
            )

    plot_losses({"DDPM": losses}, out_dir / "losses.png", title="DDPM Training Loss")
    print("Training complete.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ddpm_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.timesteps is not None:
        cfg["diffusion"]["timesteps"] = args.timesteps
    if args.base_channels is not None:
        cfg["model"]["base_channels"] = args.base_channels

    train(cfg)

if __name__ == "__main__":
    main()
