"""Cat image dataset — handles multi-resolution images."""

from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size: int = 64) -> transforms.Compose:
    """Standard transform: center-crop to square, resize, normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                          # → [0, 1]
        transforms.Normalize([0.5, 0.5, 0.5],          # → [-1, 1]
                             [0.5, 0.5, 0.5]),
    ])


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
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or build_transform(image_size)
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
