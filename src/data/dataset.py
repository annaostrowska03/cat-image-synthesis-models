"""Cat image dataset: handles multi-resolution images."""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size: int = 64, augment: bool = False) -> transforms.Compose:
    """Center-crop + normalize transform, optionally with data augmentation.

    Args:
        image_size: Target spatial resolution (square).
        augment:    If True enables random augmentation:
                    - RandomResizedCrop (scale 85-100 %)
                    - RandomHorizontalFlip
                    - ColorJitter (brightness/contrast/saturation/hue)
                    - RandomGrayscale (5 % probability)
                    - RandomRotation ±10°
                    - RandomErasing (Cutout, 30 % probability)
    """
    if augment:
        base = [
            transforms.Resize(int(image_size * 1.1)),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(degrees=10),
        ]
    else:
        base = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]

    base += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    if augment:
        base.append(
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0.0)
        )
    return transforms.Compose(base)


class CatDataset(Dataset):
    """Dataset of cat images.

    Scans *root* recursively for JPEG/PNG files.  All images are converted
    to RGB so grayscale or RGBA images are handled transparently.

    Args:
        root:        Root directory of the Cat Dataset (e.g. ``data/cats``).
        image_size:  Target spatial resolution (square).
        transform:   Override the default transform.  If *None* the default
                     center-crop + normalize pipeline is used.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: str | Path,
        image_size: int = 64,
        augment: bool = False,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or build_transform(image_size, augment=augment)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No images found under '{self.root}'. "
                "Download the dataset first: "
                "kaggle datasets download crawford/cat-dataset -p data/"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> "torch.Tensor":
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
