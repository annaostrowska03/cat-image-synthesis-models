"""DCGAN ablation study: trains multiple configurations sequentially and computes FID.

Ablation groups:
  1. z_dim:  64, 128 (baseline), 256
  2. lr:   1e-4, 2e-4 (baseline), 4e-4

Each config trains for 50 epochs (configurable in the YAML).
The baseline (z_dim=128, lr=2e-4) is run fresh so the comparison is fair.

Usage:
    python ablate_dcgan.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.evaluation.fid import compute_fid, save_samples
from src.models.dcgan import Generator
from src.training.train_dcgan import train as _train_dcgan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DIR = "outputs/real_64"
RESULTS_FILE = "outputs/ablation/dcgan_ablation_results.json"

if not Path(REAL_DIR).exists():
    raise FileNotFoundError(
        f"Real images directory not found: '{REAL_DIR}'. "
        "Run evaluate_all.py (or notebook 02) first to populate it."
    )

CONFIGS = [
    ("z_dim=64",        "configs/dcgan_ablation_zdim64.yaml"),
    ("z_dim=128 (base)","configs/dcgan_ablation_zdim128_base.yaml"),
    ("z_dim=256",       "configs/dcgan_ablation_zdim256.yaml"),
    ("lr=1e-4",         "configs/dcgan_ablation_lr1e4.yaml"),
    ("lr=2e-4 (base)",  "configs/dcgan_ablation_lr2e4_base.yaml"),
    ("lr=4e-4",         "configs/dcgan_ablation_lr4e4.yaml"),
    ("augmented",       "configs/dcgan_ablation_augmented.yaml"),
]

def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_dcgan(cfg: dict) -> Generator:
    """Train DCGAN with cfg, return the loaded Generator from its checkpoint."""
    _train_dcgan(cfg)
    mc = cfg["model"]
    G = Generator(
        z_dim=mc["z_dim"],
        base_channels=mc["g_channels"],
        image_size=cfg["data"]["image_size"],
        image_channels=mc["image_channels"],
    ).to(DEVICE)
    ckpt_path = (
        Path(cfg["logging"]["checkpoint_dir"])
        / f"ckpt_epoch{cfg['training']['epochs']:04d}.pt"
    )
    G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["G"])
    return G

def eval_fid(G: Generator, cfg: dict, label: str) -> float:
    mc = cfg["model"]
    lc = cfg["logging"]
    fake_dir = Path(lc["output_dir"]) / "fid_samples"
    save_samples(G, fake_dir, n_samples=5000, batch_size=64,
                 device=DEVICE, z_dim=mc["z_dim"])
    fid = compute_fid(REAL_DIR, str(fake_dir))
    print(f"  FID ({label}): {fid:.2f}")
    return fid

def main() -> None:
    print(f"Device: {DEVICE}\n")
    Path("outputs/ablation").mkdir(parents=True, exist_ok=True)

    results = {}
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing result(s) from {RESULTS_FILE}")

    for label, config_path in CONFIGS:
        if not Path(config_path).exists():
            print(f"[SKIP] Config not found: {config_path}")
            continue
        if label in results:
            print(f"[SKIP] Already computed: {label}  (FID={results[label]['fid']:.2f})")
            continue
        print(f"  Config: {label}  ({config_path})")
        cfg = load_yaml(config_path)
        G = train_dcgan(cfg)
        fid = eval_fid(G, cfg, label)
        results[label] = {"config": config_path, "fid": fid,
                          "z_dim": cfg["model"]["z_dim"],
                          "lr_g": cfg["training"]["lr_g"]}
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("ABLATION RESULTS")
    for label, r in results.items():
        print(f"  {label:<22}  FID: {r['fid']:.2f}")
    print(f"\nFull results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
