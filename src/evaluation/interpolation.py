"""Latent space interpolation for all three generative models.

Generates the required 10-image strip:
  [image_A, interp_1, ..., interp_8, image_B]

where interp_i = decode(lerp(z_A, z_B, i/9)) for i in 1..8.

Usage:
    python -m src.evaluation.interpolation \\
        --model dcgan \\
        --checkpoint checkpoints/dcgan/ckpt_epoch0100.pt \\
        --config configs/dcgan_config.yaml \\
        --out outputs/dcgan/interpolation.png
"""

from __future__ import annotations

import argparse

import torch
import yaml

from src.utils.visualization import show_interpolation_strip


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two flat latent vectors."""
    z0_n = z0 / z0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z1_n = z1 / z1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    omega = torch.acos((z0_n * z1_n).sum(dim=-1, keepdim=True).clamp(-1, 1))
    sin_omega = omega.sin().clamp(min=1e-8)
    return (torch.sin((1 - t) * omega) / sin_omega) * z0 + (torch.sin(t * omega) / sin_omega) * z1

def lerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation."""
    return (1 - t) * z0 + t * z1

def interpolation_strip(
    z0: torch.Tensor,
    z1: torch.Tensor,
    decode_fn,
    n_steps: int = 10,
    use_slerp: bool = False,
) -> torch.Tensor:
    """Return a strip of *n_steps* decoded images interpolating from z0 to z1.

    Args:
        z0:        Starting latent (1, ...).
        z1:        Ending latent (1, ...).
        decode_fn: Callable mapping z to an image tensor of shape (1, C, H, W).
        n_steps:   Total frames including endpoints (default 10).
        use_slerp: Use spherical interpolation (better for unit-norm spaces).

    Returns:
        Tensor of shape (n_steps, C, H, W) in [-1, 1].
    """
    interp = slerp if use_slerp else lerp
    images = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        z_i = interp(z0, z1, t)
        with torch.no_grad():
            img = decode_fn(z_i)
        images.append(img.squeeze(0))
    return torch.stack(images)

def dcgan_interpolation(checkpoint: str, cfg: dict, out: str, device: torch.device) -> None:
    from src.models.dcgan import Generator

    mc = cfg["model"]
    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    G.load_state_dict(state["G"])
    G.eval()

    z0 = torch.randn(1, mc["z_dim"], device=device)
    z1 = torch.randn(1, mc["z_dim"], device=device)
    strip = interpolation_strip(z0, z1, lambda z: G(z), use_slerp=True)
    show_interpolation_strip(strip, out, title="DCGAN Latent Interpolation (slerp)")
    print(f"DCGAN interpolation strip saved to {out}")

def vqvae_interpolation(checkpoint: str, prior_ckpt: str | None, cfg: dict, out: str, device: torch.device) -> None:
    from src.models.vqvae import VQVAE, PixelCNN

    mc = cfg["model"]
    pc = cfg["prior"]
    vqvae = VQVAE(
        in_channels=mc["in_channels"],
        hidden_channels=mc["hidden_channels"],
        embedding_dim=mc["embedding_dim"],
        num_embeddings=mc["num_embeddings"],
        commitment_cost=mc["commitment_cost"],
        decay=mc["decay"],
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    vqvae.load_state_dict(state["model"])
    vqvae.eval()

    if prior_ckpt is not None:
        prior = PixelCNN(mc["num_embeddings"], pc["n_layers"], pc["n_filters"]).to(device)
        ps = torch.load(prior_ckpt, map_location=device)
        prior.load_state_dict(ps["prior"])
        prior.eval()
        img_size = cfg["data"]["image_size"]
        H_z = W_z = img_size // 4
        idx0 = prior.sample((1, H_z, W_z), device=device)
        idx1 = prior.sample((1, H_z, W_z), device=device)
        emb = vqvae.vq.embedding
        z0 = emb[idx0.view(-1)].view(1, H_z, W_z, mc["embedding_dim"]).permute(0, 3, 1, 2)
        z1 = emb[idx1.view(-1)].view(1, H_z, W_z, mc["embedding_dim"]).permute(0, 3, 1, 2)
    else:
        H_z = W_z = cfg["data"]["image_size"] // 4
        z0 = torch.randn(1, mc["embedding_dim"], H_z, W_z, device=device)
        z1 = torch.randn(1, mc["embedding_dim"], H_z, W_z, device=device)

    def decode(z: torch.Tensor) -> torch.Tensor:
        return vqvae.decoder(z)

    strip = interpolation_strip(z0, z1, decode, use_slerp=False)
    show_interpolation_strip(strip, out, title="VQ-VAE Latent Interpolation (lerp in embedding space)")
    print(f"VQ-VAE interpolation strip saved to {out}")

def ddpm_interpolation(checkpoint: str, cfg: dict, out: str, device: torch.device) -> None:
    from src.models.ddpm import GaussianDiffusion, UNet

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
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=device,
    )

    img_size = cfg["data"]["image_size"]
    z0 = torch.randn(1, mc["image_channels"], img_size, img_size, device=device)
    z1 = torch.randn(1, mc["image_channels"], img_size, img_size, device=device)

    images = []
    for i in range(10):
        t = i / 9
        z_i = lerp(z0, z1, t)
        with torch.no_grad():
            img = diffusion.ddim_sample(
                model, z_i.shape,
                ddim_steps=sc.get("ddim_steps", 50),
                eta=sc.get("eta", 0.0),
                x_T=z_i,
            )
        images.append(img.squeeze(0))

    strip = torch.stack(images)
    show_interpolation_strip(strip, out, title="DDPM Noise-Space Interpolation (lerp)")
    print(f"DDPM interpolation strip saved to {out}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dcgan", "vqvae", "ddpm"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--prior_ckpt", default=None, help="VQ-VAE only: path to PixelCNN checkpoint.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or f"outputs/{args.model}/interpolation.png"

    if args.model == "dcgan":
        dcgan_interpolation(args.checkpoint, cfg, out, device)
    elif args.model == "vqvae":
        vqvae_interpolation(args.checkpoint, args.prior_ckpt, cfg, out, device)
    elif args.model == "ddpm":
        ddpm_interpolation(args.checkpoint, cfg, out, device)

if __name__ == "__main__":
    main()
