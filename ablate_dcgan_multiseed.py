"""3-seed DCGAN ablation: each config trained with seeds {0, 42, 123}.

Saves per-run FID to outputs/ablation/multiseed_results.json, resuming
where it left off if interrupted.  Prints mean ± std summary at the end.

Usage:
    python -u ablate_dcgan_multiseed.py 2>&1 | tee logs/ablation_multiseed.log
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.evaluation.fid import compute_fid, save_samples
from src.models.dcgan import Generator
from src.training.train_dcgan import train as _train_dcgan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DIR = "outputs/real_64"
RESULTS_FILE = "outputs/ablation/multiseed_results.json"
SEEDS = [0, 42, 123]

CONFIGS = [
    ("z_dim=64", "configs/dcgan_ablation_zdim64.yaml"),
    ("z_dim=128 (base)", "configs/dcgan_ablation_zdim128_base.yaml"),
    ("z_dim=256", "configs/dcgan_ablation_zdim256.yaml"),
    ("lr=1e-4", "configs/dcgan_ablation_lr1e4.yaml"),
    ("lr=2e-4 (base)", "configs/dcgan_ablation_lr2e4_base.yaml"),
    ("lr=4e-4", "configs/dcgan_ablation_lr4e4.yaml"),
    ("augmented", "configs/dcgan_ablation_augmented.yaml"),
]


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single(base_cfg: dict, seed: int, label: str) -> float:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    suffix = f"_seed{seed}"
    cfg["logging"]["checkpoint_dir"] = cfg["logging"]["checkpoint_dir"] + suffix
    cfg["logging"]["output_dir"] = cfg["logging"]["output_dir"] + suffix

    print(f"  Training {label} | seed={seed} ...", flush=True)
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
    G.eval()

    fake_dir = Path(cfg["logging"]["output_dir"]) / "fid_samples"
    save_samples(G, fake_dir, n_samples=5000, batch_size=64,
                 device=DEVICE, z_dim=mc["z_dim"])
    fid = compute_fid(REAL_DIR, str(fake_dir))
    print(f"  FID ({label}, seed={seed}): {fid:.2f}", flush=True)
    return fid


def main() -> None:
    if not Path(REAL_DIR).exists():
        raise FileNotFoundError(
            f"Real images not found at '{REAL_DIR}'. "
            "Run evaluate_all.py first to populate it."
        )

    Path("outputs/ablation").mkdir(parents=True, exist_ok=True)
    results: dict = {}
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resumed — {sum(len(v['fids']) for v in results.values())} "
              f"runs already done.\n", flush=True)

    print(f"Device: {DEVICE}   Seeds: {SEEDS}\n", flush=True)

    for label, config_path in CONFIGS:
        if not Path(config_path).exists():
            print(f"[SKIP] Config not found: {config_path}", flush=True)
            continue

        if label not in results:
            results[label] = {"config": config_path, "fids": {}}

        base_cfg = load_yaml(config_path)
        for seed in SEEDS:
            key = str(seed)
            if key in results[label]["fids"]:
                print(f"[SKIP] {label} seed={seed}  "
                      f"FID={results[label]['fids'][key]:.2f}", flush=True)
                continue

            fid = run_single(base_cfg, seed, label)
            results[label]["fids"][key] = fid

            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    print(f"{'Config':<22}  {'FID mean':>10}  {'FID std':>8}", flush=True)
    for label, r in results.items():
        fids = list(r["fids"].values())
        if fids:
            mean, std = np.mean(fids), np.std(fids, ddof=1) if len(fids) > 1 else 0.0
            print(f"{label:<22}  {mean:>10.2f}  {std:>8.2f}", flush=True)
    print(f"\nFull results: {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
