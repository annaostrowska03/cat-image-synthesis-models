"""Visualization helpers for generated images and training curves."""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Rescale from [-1, 1] to [0, 1] for display."""
    return (tensor.clamp(-1, 1) + 1.0) / 2.0


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    title: Optional[str] = None,
) -> None:
    """Save a grid of images (values in [-1, 1]) to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(denormalize(images.cpu()), nrow=nrow, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(nrow * 1.5, max(1, images.shape[0] // nrow) * 1.5))
    ax.imshow(np_grid)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_losses(
    losses: dict[str, Sequence[float]],
    path: str | Path,
    title: str = "Training Loss",
) -> None:
    """Plot one or more loss curves and save to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, vals in losses.items():
        ax.plot(vals, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def show_interpolation_strip(
    images: torch.Tensor,
    path: str | Path,
    title: str = "Latent Space Interpolation",
) -> None:
    """Display the 10-image interpolation strip and save to *path*.

    Args:
        images: Tensor of shape (10, C, H, W) in [-1, 1].
        path:   Output file path.
        title:  Figure title.
    """
    assert images.shape[0] == 10, "Expected exactly 10 images for the interpolation strip."
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = denormalize(images.cpu())
    fig, axes = plt.subplots(1, 10, figsize=(20, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
        if i == 0:
            ax.set_title("z₁", fontsize=8)
        elif i == 9:
            ax.set_title("z₂", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
