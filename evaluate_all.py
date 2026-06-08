"""Run full evaluation: FID computation + latent interpolation for all three models.

Usage:
    python evaluate_all.py [--n_fid N] [--real_dir DIR]

Checkpoint paths are derived from each model's config (checkpoint_dir + final epoch),
so there are no hardcoded paths in this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.data.preprocessing import get_cat_loader
from src.evaluation.fid import compute_fid, save_real_samples, save_samples
from src.evaluation.interpolation import (
    dcgan_interpolation,
    ddpm_interpolation,
    vqvae_interpolation,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FID_BATCH = 64


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _last_ckpt(cfg: dict, prefix: str = "ckpt_epoch") -> Path:
    """Return path to the final-epoch checkpoint derived from config."""
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    epochs = cfg["training"]["epochs"]
    return ckpt_dir / f"{prefix}{epochs:04d}.pt"


def save_real(cfg_path: str, real_dir: str, n_samples: int) -> None:
    cfg = load_yaml(cfg_path)
    loader = get_cat_loader(
        data_dir=cfg["data"]["cats_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=64,
        num_workers=0,
        shuffle=False,
    )
    save_real_samples(loader, real_dir, n_samples=n_samples)


def eval_dcgan(cfg_path: str, real_dir: str, n_fid: int) -> float:
    from src.models.dcgan import Generator

    cfg = load_yaml(cfg_path)
    mc = cfg["model"]
    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(DEVICE)
    ckpt = _last_ckpt(cfg)
    G.load_state_dict(torch.load(ckpt, map_location=DEVICE)["G"])

    fake_dir = Path(cfg["logging"]["output_dir"]) / "fid_samples"
    save_samples(G, fake_dir, n_samples=n_fid, batch_size=FID_BATCH,
                 device=DEVICE, z_dim=mc["z_dim"])
    return compute_fid(real_dir, str(fake_dir))


def eval_vqvae(cfg_path: str, real_dir: str, n_fid: int) -> float:
    from src.models.vqvae import VQVAE, PixelCNN

    cfg = load_yaml(cfg_path)
    mc = cfg["model"]
    pc = cfg["prior"]

    vqvae = VQVAE(
        in_channels=mc["in_channels"],
        hidden_channels=mc["hidden_channels"],
        embedding_dim=mc["embedding_dim"],
        num_embeddings=mc["num_embeddings"],
        commitment_cost=mc["commitment_cost"],
        decay=mc["decay"],
    ).to(DEVICE)
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    tc = cfg["training"]
    ae_epochs = tc.get("ae_epochs", tc["epochs"])
    prior_epochs = tc.get("prior_epochs", tc["epochs"])
    vqvae.load_state_dict(
        torch.load(ckpt_dir / f"vqvae_epoch{ae_epochs:04d}.pt", map_location=DEVICE)["model"]
    )

    prior = PixelCNN(
        num_embeddings=mc["num_embeddings"],
        n_layers=pc["n_layers"],
        n_filters=pc["n_filters"],
    ).to(DEVICE)
    prior.load_state_dict(
        torch.load(ckpt_dir / f"prior_epoch{prior_epochs:04d}.pt", map_location=DEVICE)["prior"]
    )

    img_size = cfg["data"]["image_size"]
    latent_hw = (img_size // 4, img_size // 4)
    fake_dir = Path(cfg["logging"]["output_dir"]) / "fid_samples"
    save_samples(vqvae, fake_dir, n_samples=n_fid, batch_size=FID_BATCH,
                 device=DEVICE, prior=prior, latent_hw=latent_hw)
    return compute_fid(real_dir, str(fake_dir))


def eval_ddpm(cfg_path: str, real_dir: str, n_fid: int, use_ema: bool = True) -> float:
    from src.models.ddpm import GaussianDiffusion, UNet

    cfg = load_yaml(cfg_path)
    mc = cfg["model"]
    dc = cfg["diffusion"]
    sc = cfg["sampling"]

    model = UNet(
        image_channels=mc["image_channels"],
        base_channels=mc["base_channels"],
        channel_mults=tuple(mc["channel_mults"]),
        attention_resolutions=tuple(mc["attention_resolutions"]),
        num_res_blocks=mc["num_res_blocks"],
        dropout=mc["dropout"],
    ).to(DEVICE)
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    epochs = cfg["training"]["epochs"]
    ckpt = ckpt_dir / f"ddpm_epoch{epochs:04d}.pt"
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"])
    if use_ema and "ema_shadow" in state:
        shadow = state["ema_shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in shadow:
                    param.data.copy_(shadow[name])

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=DEVICE,
    )
    img_size = cfg["data"]["image_size"]
    sample_shape = (1, mc["image_channels"], img_size, img_size)
    fake_dir = Path(cfg["logging"]["output_dir"]) / "fid_samples"
    save_samples(model, fake_dir, n_samples=n_fid, batch_size=16,
                 device=DEVICE, diffusion=diffusion, sample_shape=sample_shape,
                 ddim_steps=sc.get("ddim_steps", 50), eta=sc.get("eta", 0.0))
    return compute_fid(real_dir, str(fake_dir))


def run_interpolations(dcgan_cfg: str, vqvae_cfg: str, ddpm_cfg: str) -> None:
    cfg = load_yaml(dcgan_cfg)
    dcgan_interpolation(str(_last_ckpt(cfg)), cfg,
                        str(Path(cfg["logging"]["output_dir"]) / "interpolation.png"), DEVICE)

    cfg = load_yaml(vqvae_cfg)
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    _tc = cfg["training"]
    _ae = _tc.get("ae_epochs", _tc["epochs"])
    _pr = _tc.get("prior_epochs", _tc["epochs"])
    vqvae_interpolation(
        str(ckpt_dir / f"vqvae_epoch{_ae:04d}.pt"),
        str(ckpt_dir / f"prior_epoch{_pr:04d}.pt"),
        cfg,
        str(Path(cfg["logging"]["output_dir"]) / "interpolation.png"),
        DEVICE,
    )

    cfg = load_yaml(ddpm_cfg)
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ddpm_interpolation(
        str(ckpt_dir / f"ddpm_epoch{cfg['training']['epochs']:04d}.pt"),
        cfg,
        str(Path(cfg["logging"]["output_dir"]) / "interpolation.png"),
        DEVICE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full evaluation: FID + interpolation.")
    parser.add_argument("--dcgan_cfg", default="configs/dcgan_config.yaml")
    parser.add_argument("--vqvae_cfg", default="configs/vqvae_config.yaml")
    parser.add_argument("--ddpm_cfg", default="configs/ddpm_config.yaml")
    parser.add_argument("--real_dir", default="outputs/real_64")
    parser.add_argument("--n_fid", type=int, default=5000,
                        help="Number of generated samples for FID (real set uses same count).")
    parser.add_argument("--no_ema", action="store_true",
                        help="Load raw model weights instead of EMA shadow for DDPM.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}  |  n_fid={args.n_fid}\n")

    print("Saving real images ...")
    save_real(args.dcgan_cfg, args.real_dir, args.n_fid)

    print("\nDCGAN: generating FID samples ...")
    fid_dcgan = eval_dcgan(args.dcgan_cfg, args.real_dir, args.n_fid)
    print(f"DCGAN FID: {fid_dcgan:.4f}")

    print("\nVQ-VAE: generating FID samples ...")
    fid_vqvae = eval_vqvae(args.vqvae_cfg, args.real_dir, args.n_fid)
    print(f"VQ-VAE FID: {fid_vqvae:.4f}")

    print("\nDDPM: generating FID samples (slow) ...")
    fid_ddpm = eval_ddpm(args.ddpm_cfg, args.real_dir, args.n_fid, use_ema=not args.no_ema)
    print(f"DDPM FID: {fid_ddpm:.4f}")

    print("\nLatent interpolations ...")
    run_interpolations(args.dcgan_cfg, args.vqvae_cfg, args.ddpm_cfg)

    print("\nResults:")
    print(f"  DCGAN   FID: {fid_dcgan:.4f}")
    print(f"  VQ-VAE  FID: {fid_vqvae:.4f}")
    print(f"  DDPM    FID: {fid_ddpm:.4f}")
    print("Interpolation strips saved to outputs/*/interpolation.png")


if __name__ == "__main__":
    main()
