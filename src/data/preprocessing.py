"""DataLoader factories for cats-only and cats+dogs datasets."""

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import CatDataset, build_transform


def get_cat_loader(
    data_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    augment: bool = False,
) -> DataLoader:
    """Return a DataLoader for the Cat Dataset.

    Args:
        augment: If True, applies RandomHorizontalFlip, ColorJitter and
                 RandomRotation during training.
    """
    dataset = CatDataset(root=data_dir, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def get_cats_and_dogs_loader(
    cats_dir: str | Path,
    dogs_dir: str | Path,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Return a DataLoader for a merged cats + dogs dataset.

    Both splits use the same transform (center-crop + normalize).
    Labels are intentionally discarded; this is an unconditional generative task.
    """
    transform = build_transform(image_size)
    cats = CatDataset(root=cats_dir, image_size=image_size, transform=transform)
    dogs = CatDataset(root=dogs_dir, image_size=image_size, transform=transform)
    combined = ConcatDataset([cats, dogs])
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
