"""FID (Fréchet Inception Distance) computation using torch-fidelity.

Usage from Python:
    from src.evaluation.fid import compute_fid
    score = compute_fid(real_dir="outputs/real_64", fake_dir="outputs/dcgan/fid_samples")

Usage from CLI:
    python -m src.evaluation.fid \\
        --real_dir data/cats_64 \\
        --fake_dir outputs/dcgan/fid_samples \\
        --n_samples 5000

The script also provides save_samples() to dump a model's generated images
to disk in preparation for FID calculation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def save_samples(
    generator,
    out_dir: str | Path,
    n_samples: int = 5000,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    z_dim: int | None = None,
    diffusion=None,
    sample_shape: tuple | None = None,
    ddim_steps: int = 50,
    eta: float = 0.0,
    prior=None,
    latent_hw: tuple[int, int] | None = None,
) -> None:
    """Generate *n_samples* images and save them as PNGs to *out_dir*.

    Supports:
    - DCGAN            (pass generator + z_dim)
    - VQ-VAE + prior   (pass generator as VQVAE + prior + latent_hw)
    - DDPM             (pass generator as UNet + diffusion object + sample_shape)

    Images are saved normalized to [0, 1].
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    generator.eval()
    if prior is not None:
        prior.eval()

    saved = 0
    while saved < n_samples:
        n = min(batch_size, n_samples - saved)
        with torch.no_grad():
            if diffusion is not None:
                shape = (n, *sample_shape[1:])
                imgs = diffusion.ddim_sample(generator, shape, ddim_steps=ddim_steps, eta=eta)
            elif prior is not None:
                H_z, W_z = latent_hw
                indices = prior.sample((n, H_z, W_z), device=device)
                imgs = generator.decode_indices(indices)
            else:
                z = torch.randn(n, z_dim, device=device)
                imgs = generator(z)

        imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0
        for j, img in enumerate(imgs):
            save_image(img.cpu(), out_dir / f"{saved + j:06d}.png")
        saved += n

    print(f"Saved {saved} images to {out_dir}")

def compute_fid(real_dir: str | Path, fake_dir: str | Path) -> float:
    """Compute FID between two directories of PNG/JPEG images.

    Requires ``torch-fidelity`` (``pip install torch-fidelity``).

    Args:
        real_dir: Directory of reference (real) images.
        fake_dir: Directory of generated images.

    Returns:
        FID score (lower is better).
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError as exc:
        raise ImportError(
            "torch-fidelity is required for FID computation. "
            "Install it with: pip install torch-fidelity"
        ) from exc

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        fid=True,
        verbose=False,
        cuda=torch.cuda.is_available(),
    )
    return metrics["frechet_inception_distance"]

def save_real_samples(
    data_loader: DataLoader,
    out_dir: str | Path,
    n_samples: int = 5000,
) -> None:
    """Save reference real images to disk for FID computation."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    saved = 0
    for batch in data_loader:
        if saved >= n_samples:
            break
        imgs = (batch.clamp(-1, 1) + 1.0) / 2.0
        for img in imgs:
            if saved >= n_samples:
                break
            save_image(img.cpu(), out_dir / f"{saved:06d}.png")
            saved += 1
    print(f"Saved {saved} real images to {out_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID between two image directories.")
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dir", required=True)
    args = parser.parse_args()

    score = compute_fid(args.real_dir, args.fake_dir)
    print(f"FID: {score:.4f}")

if __name__ == "__main__":
    main()
