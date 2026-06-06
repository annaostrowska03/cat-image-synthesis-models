"""Run full evaluation: FID computation + latent interpolation for all three models.

Usage:
    python evaluate_all.py
"""

from __future__ import annotations

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
N_FID = 5000
FID_BATCH = 64   # larger batch OK for inference (no gradient graph)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_real(cfg_path: str = "configs/dcgan_config.yaml") -> None:
    cfg = load_yaml(cfg_path)
    loader = get_cat_loader(
        data_dir=cfg["data"]["cats_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=64,
        num_workers=0,
    )
    save_real_samples(loader, "outputs/real_64", n_samples=N_FID)


def eval_dcgan() -> float:
    from src.models.dcgan import Generator

    cfg = load_yaml("configs/dcgan_config.yaml")
    mc = cfg["model"]
    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(DEVICE)
    state = torch.load("checkpoints/dcgan/ckpt_epoch0100.pt", map_location=DEVICE)
    G.load_state_dict(state["G"])

    save_samples(
        G,
        "outputs/dcgan/fid_samples",
        n_samples=N_FID,
        batch_size=FID_BATCH,
        device=DEVICE,
        z_dim=mc["z_dim"],
    )
    return compute_fid("outputs/real_64", "outputs/dcgan/fid_samples")


def eval_vqvae() -> float:
    from src.models.vqvae import VQVAE, PixelCNN

    cfg = load_yaml("configs/vqvae_config.yaml")
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
    state = torch.load("checkpoints/vqvae/vqvae_epoch0100.pt", map_location=DEVICE)
    vqvae.load_state_dict(state["model"])

    prior = PixelCNN(
        num_embeddings=mc["num_embeddings"],
        n_layers=pc["n_layers"],
        n_filters=pc["n_filters"],
    ).to(DEVICE)
    ps = torch.load("checkpoints/vqvae/prior_epoch0100.pt", map_location=DEVICE)
    prior.load_state_dict(ps["prior"])

    img_size = cfg["data"]["image_size"]
    latent_hw = (img_size // 4, img_size // 4)

    save_samples(
        vqvae,
        "outputs/vqvae/fid_samples",
        n_samples=N_FID,
        batch_size=FID_BATCH,
        device=DEVICE,
        prior=prior,
        latent_hw=latent_hw,
    )
    return compute_fid("outputs/real_64", "outputs/vqvae/fid_samples")


def eval_ddpm() -> float:
    from src.models.ddpm import GaussianDiffusion, UNet

    cfg = load_yaml("configs/ddpm_config.yaml")
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
    state = torch.load("checkpoints/ddpm/ddpm_epoch0200.pt", map_location=DEVICE)
    model.load_state_dict(state["model"])

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=DEVICE,
    )

    img_size = cfg["data"]["image_size"]
    sample_shape = (1, mc["image_channels"], img_size, img_size)

    save_samples(
        model,
        "outputs/ddpm/fid_samples",
        n_samples=N_FID,
        batch_size=16,
        device=DEVICE,
        diffusion=diffusion,
        sample_shape=sample_shape,
        ddim_steps=sc.get("ddim_steps", 50),
        eta=sc.get("eta", 0.0),
    )
    return compute_fid("outputs/real_64", "outputs/ddpm/fid_samples")


def run_interpolations() -> None:
    dcgan_interpolation(
        "checkpoints/dcgan/ckpt_epoch0100.pt",
        load_yaml("configs/dcgan_config.yaml"),
        "outputs/dcgan/interpolation.png",
        DEVICE,
    )

    vqvae_interpolation(
        "checkpoints/vqvae/vqvae_epoch0100.pt",
        "checkpoints/vqvae/prior_epoch0100.pt",
        load_yaml("configs/vqvae_config.yaml"),
        "outputs/vqvae/interpolation.png",
        DEVICE,
    )

    ddpm_interpolation(
        "checkpoints/ddpm/ddpm_epoch0200.pt",
        load_yaml("configs/ddpm_config.yaml"),
        "outputs/ddpm/interpolation.png",
        DEVICE,
    )


def main() -> None:
    print(f"Device: {DEVICE}\n")

    print("Saving real images ...")
    save_real()

    print("\nDCGAN: generating FID samples ...")
    fid_dcgan = eval_dcgan()
    print(f"DCGAN FID: {fid_dcgan:.4f}")

    print("\nVQ-VAE: generating FID samples ...")
    fid_vqvae = eval_vqvae()
    print(f"VQ-VAE FID: {fid_vqvae:.4f}")

    print("\nDDPM: generating FID samples (slow) ...")
    fid_ddpm = eval_ddpm()
    print(f"DDPM FID: {fid_ddpm:.4f}")

    print("\nLatent interpolations ...")
    run_interpolations()

    print("\nResults:")
    print(f"  DCGAN   FID: {fid_dcgan:.4f}")
    print(f"  VQ-VAE  FID: {fid_vqvae:.4f}")
    print(f"  DDPM    FID: {fid_ddpm:.4f}")
    print("Interpolation strips saved to outputs/*/interpolation.png")


if __name__ == "__main__":
    main()
