"""Generate side-by-side slerp vs. linear interpolation strips for DCGAN and DDPM.

Outputs:
    outputs/dcgan/interpolation_linear.png - linear only
    outputs/dcgan/interpolation_compare.png - slerp (top) vs linear (bottom)
    outputs/ddpm/interpolation_linear.png - linear only
    outputs/ddpm/interpolation_compare.png - slerp (top) vs linear (bottom)

Run from project root:
    python scripts/generate_linear_interp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np

from src.evaluation.interpolation import interpolation_strip, lerp, slerp
from src.models.dcgan import Generator
from src.utils.visualization import show_interpolation_strip


def to_uint8_grid(strip: torch.Tensor) -> np.ndarray:
    """Convert (N, C, H, W) tensor in [-1,1] to (H, N*W, C) uint8 numpy."""
    imgs = ((strip.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return np.concatenate(list(imgs), axis=1)


def save_compare(slerp_strip: torch.Tensor, lerp_strip: torch.Tensor, out: Path, title: str) -> None:
    top = to_uint8_grid(slerp_strip)
    bot = to_uint8_grid(lerp_strip)
    gap = np.ones((4, top.shape[1], 3), dtype=np.uint8) * 200

    combined = np.concatenate([top, gap, bot], axis=0)
    h, w = combined.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 80, h / 80 + 0.8))
    ax.imshow(combined)
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=4)

    n = slerp_strip.shape[0]
    row_h = top.shape[0]
    fig.text(0.01, 1 - (row_h / 2) / (h + 20) * 0.88, "slerp", va="center",
             fontsize=8, color="#333", fontweight="bold")
    fig.text(0.01, 1 - (row_h + 4 + row_h / 2) / (h + 20) * 0.88, "linear", va="center",
             fontsize=8, color="#c0392b", fontweight="bold")

    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def run_dcgan(device: torch.device) -> None:

    with open("configs/dcgan_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]

    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(device)
    state = torch.load("checkpoints/dcgan/ckpt_epoch0100.pt", map_location=device)
    G.load_state_dict(state["G"])
    G.eval()

    torch.manual_seed(0)
    z0 = torch.randn(1, mc["z_dim"], device=device)
    z1 = torch.randn(1, mc["z_dim"], device=device)

    def decode(z):
        return G(z)
    s_strip = interpolation_strip(z0, z1, decode, n_steps=10, use_slerp=True)
    l_strip = interpolation_strip(z0, z1, decode, n_steps=10, use_slerp=False)

    show_interpolation_strip(l_strip, "outputs/dcgan/interpolation_linear.png",
                             title="DCGAN Latent Interpolation (linear)")
    save_compare(s_strip, l_strip,
                 Path("outputs/dcgan/interpolation_compare.png"),
                 "DCGAN: slerp (top) vs. linear interpolation (bottom)")
    print("DCGAN done.")


def run_ddpm(device: torch.device) -> None:
    from src.models.ddpm import GaussianDiffusion, UNet

    with open("configs/ddpm_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
    state = torch.load("checkpoints/ddpm/ddpm_epoch0200.pt", map_location=device)
    model.load_state_dict(state["model"])
    if "ema_shadow" in state:
        shadow = state["ema_shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in shadow:
                    param.data.copy_(shadow[name])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=dc["timesteps"],
        beta_start=dc["beta_start"],
        beta_end=dc["beta_end"],
        schedule=dc["schedule"],
        device=device,
    )

    img_size = cfg["data"]["image_size"]

    torch.manual_seed(0)
    z0 = torch.randn(1, mc["image_channels"], img_size, img_size, device=device)
    z1 = torch.randn(1, mc["image_channels"], img_size, img_size, device=device)

    ddim_steps = sc.get("ddim_steps", 50)
    eta = sc.get("eta", 0.0)

    def decode_slerp(i: int, n: int) -> torch.Tensor:
        t = i / (n - 1)
        z_i = slerp(z0, z1, t)
        with torch.no_grad():
            return diffusion.ddim_sample(model, z_i.shape, ddim_steps=ddim_steps, eta=eta, x_T=z_i)

    def decode_lerp(i: int, n: int) -> torch.Tensor:
        t = i / (n - 1)
        z_i = lerp(z0, z1, t)
        with torch.no_grad():
            return diffusion.ddim_sample(model, z_i.shape, ddim_steps=ddim_steps, eta=eta, x_T=z_i)

    n = 10
    print("  DDPM slerp strip...")
    s_imgs = [decode_slerp(i, n).squeeze(0) for i in range(n)]
    s_strip = torch.stack(s_imgs)

    print("  DDPM linear strip...")
    l_imgs = [decode_lerp(i, n).squeeze(0) for i in range(n)]
    l_strip = torch.stack(l_imgs)

    show_interpolation_strip(l_strip, "outputs/ddpm/interpolation_linear.png",
                             title="DDPM Noise-Space Interpolation (linear)")
    save_compare(s_strip, l_strip,
                 Path("outputs/ddpm/interpolation_compare.png"),
                 "DDPM: slerp (top) vs. linear interpolation (bottom)")
    print("DDPM done.")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("DCGAN")
    run_dcgan(device)
    print("DDPM")
    run_ddpm(device)
    print("\nAll done. Files saved to outputs/dcgan/ and outputs/ddpm/")


if __name__ == "__main__":
    main()
