"""Evaluate DCGAN trained on cats+dogs.

Computes:
  1. FID vs real cats  (uses existing outputs/real_64/ or regenerates)
  2. FID vs real dogs  (saves dog reference images from data/cats_and_dogs/train/)
  3. Saves generated samples grid

Usage:
    python evaluate_cats_and_dogs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.data.preprocessing import get_cat_loader
from src.evaluation.fid import compute_fid, save_real_samples, save_samples
from src.models.dcgan import Generator
from src.utils.seed import set_seed

CHECKPOINT  = "checkpoints/cats_and_dogs/ckpt_epoch0100.pt"
TRAIN_DIR   = Path("data/cats_and_dogs/train")
REAL_CATS   = Path("outputs/real_64")
REAL_DOGS   = Path("outputs/real_dogs_64")
FAKE_DIR    = Path("outputs/cats_and_dogs/fid_samples")
RESULTS_FILE = Path("outputs/cats_and_dogs/fid_results.json")

N_REAL  = 5000
N_FAKE  = 5000
IMAGE_SIZE = 64
BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_real_dogs(out_dir: Path, n: int = 5000) -> None:
    """Copy/resize real dog images from the Kaggle train folder."""
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= n:
        print(f"Real dogs already saved ({out_dir})")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dog_files = sorted(TRAIN_DIR.glob("dog.*.jpg"))
    print(f"Found {len(dog_files)} dog images, saving {n} …")
    for i, fpath in enumerate(dog_files[:n]):
        img = Image.open(fpath).convert("RGB")
        t = transform(img)
        save_image(t, out_dir / f"dog_{i:05d}.png")
    print(f"Saved {min(n, len(dog_files))} real dog images to {out_dir}")


def save_real_cats_if_needed(out_dir: Path, cats_dir: Path, n: int = 5000) -> None:
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= n:
        print(f"Real cats already saved ({out_dir})")
        return
    loader = get_cat_loader(
        data_dir=cats_dir, image_size=IMAGE_SIZE,
        batch_size=BATCH, num_workers=0, shuffle=False, pin_memory=False,
    )
    save_real_samples(loader, out_dir, n_samples=n)



def main() -> None:
    set_seed(42)

    save_real_cats_if_needed(REAL_CATS, Path("data/cats"), N_REAL)
    save_real_dogs(REAL_DOGS, N_REAL)

    cfg_path = "configs/dcgan_cats_and_dogs.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=IMAGE_SIZE,
        image_channels=mc["image_channels"],
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    G.load_state_dict(ckpt["G"])
    G.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    save_samples(
        G,
        out_dir=FAKE_DIR,
        n_samples=N_FAKE,
        batch_size=BATCH,
        device=DEVICE,
        z_dim=mc["z_dim"],
    )

    print("Computing FID vs cats …")
    fid_cats = compute_fid(str(REAL_CATS), str(FAKE_DIR))
    print(f"  FID vs cats : {fid_cats:.2f}")

    print("Computing FID vs dogs …")
    fid_dogs = compute_fid(str(REAL_DOGS), str(FAKE_DIR))
    print(f"  FID vs dogs : {fid_dogs:.2f}")

    results = {"fid_vs_cats": fid_cats, "fid_vs_dogs": fid_dogs}
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
